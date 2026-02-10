#!/usr/bin/env python3
"""
Context Manager for NATNA AI System
Handles token counting, conversation history, and context limits
"""

import re
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    """A single conversation message"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)


# Model context limits (in tokens)
MODEL_CONTEXT_LIMITS = {
    # Small models
    'qwen2.5:0.5b': 8192,
    'smollm2:360m': 8192,
    'qwen3:0.6b': 8192,
    'alibayram/smollm3': 8192,
    'phi4-mini': 8192,

    # General uncensored
    'deepseek-r1-abliterated-4gb': 4096,
    'deepseek-r1-uncensored-16gb': 8192,

    # Coding models (8GB)
    'qwen2.5-coder:7b': 8192,
    'deepseek-coder:6.7b': 8192,
    'codegemma:7b': 8192,

    # Coding models (16GB)
    'qwen2.5-coder:14b': 16384,
    'deepseek-coder-v2:16b': 16384,
    'codellama:13b': 16384,
}

# Default limit for unknown models
DEFAULT_CONTEXT_LIMIT = 4096

# Reserve tokens for response generation
RESPONSE_RESERVE = 512


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    Uses a hybrid approach:
    - ~4 characters per token for English
    - ~2 characters per token for non-ASCII (Tigrinya, etc.)
    """
    if not text:
        return 0

    # Count ASCII vs non-ASCII characters
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii_chars = len(text) - ascii_chars

    # Estimate tokens (English ~4 chars/token, non-ASCII ~2 chars/token)
    ascii_tokens = ascii_chars / 4
    non_ascii_tokens = non_ascii_chars / 2

    # Add overhead for whitespace and special tokens
    total = int(ascii_tokens + non_ascii_tokens)

    # Minimum 1 token for non-empty text
    return max(1, total)


def get_context_limit(model: str) -> int:
    """Get context limit for a model"""
    return MODEL_CONTEXT_LIMITS.get(model, DEFAULT_CONTEXT_LIMIT)


def get_available_context(model: str, used_tokens: int) -> int:
    """Get remaining available context for a model"""
    limit = get_context_limit(model)
    available = limit - used_tokens - RESPONSE_RESERVE
    return max(0, available)


class ConversationHistory:
    """
    Manages conversation history with automatic summarization.
    Keeps track of messages and token usage.
    """

    def __init__(self, max_messages: int = 20):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.total_tokens = 0
        self.summary: Optional[str] = None
        self.summary_tokens = 0

    def add_message(self, role: str, content: str) -> Message:
        """Add a message to history"""
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        self.total_tokens += msg.token_count

        # Trim old messages if too many
        if len(self.messages) > self.max_messages:
            self._trim_history()

        return msg

    def _trim_history(self):
        """Remove oldest messages, keeping summary"""
        # Keep last 10 messages, summarize the rest
        if len(self.messages) > 10:
            old_messages = self.messages[:-10]
            self.messages = self.messages[-10:]

            # Create summary of old messages
            old_content = "\n".join([f"{m.role}: {m.content[:100]}..." for m in old_messages])
            self.summary = f"[Previous conversation summary: {len(old_messages)} messages about various topics]"
            self.summary_tokens = estimate_tokens(self.summary)

            # Recalculate total tokens
            self.total_tokens = sum(m.token_count for m in self.messages) + self.summary_tokens

    def get_context_messages(self, model: str, system_prompt: str = "") -> Tuple[List[Dict], int, bool]:
        """
        Get messages that fit within context limit.

        Returns:
            - List of message dicts for API
            - Total tokens used
            - Whether context was truncated
        """
        limit = get_context_limit(model)
        system_tokens = estimate_tokens(system_prompt)
        available = limit - system_tokens - RESPONSE_RESERVE

        result = []
        used_tokens = 0
        truncated = False

        # Add summary if exists
        if self.summary:
            if self.summary_tokens <= available:
                result.append({'role': 'system', 'content': self.summary})
                used_tokens += self.summary_tokens
                available -= self.summary_tokens

        # Add messages from most recent, working backwards
        for msg in reversed(self.messages):
            if msg.token_count <= available:
                result.insert(0 if not self.summary else 1, {
                    'role': msg.role,
                    'content': msg.content
                })
                used_tokens += msg.token_count
                available -= msg.token_count
            else:
                truncated = True
                break

        return result, used_tokens + system_tokens, truncated

    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.total_tokens = 0
        self.summary = None
        self.summary_tokens = 0

    def get_token_count(self) -> int:
        """Get total tokens in history"""
        return self.total_tokens


class ContextManager:
    """
    Main context manager that handles all context-related operations.
    """

    def __init__(self):
        self.history = ConversationHistory()
        self.current_model = 'qwen2.5:0.5b'
        self.wikipedia_context = ""
        self.wikipedia_tokens = 0

    def set_model(self, model: str):
        """Set the current model"""
        self.current_model = model

    def set_wikipedia_context(self, context: str):
        """Set Wikipedia context for current query"""
        self.wikipedia_context = context
        self.wikipedia_tokens = estimate_tokens(context)

    def clear_wikipedia_context(self):
        """Clear Wikipedia context"""
        self.wikipedia_context = ""
        self.wikipedia_tokens = 0

    def add_user_message(self, content: str) -> Message:
        """Add user message to history"""
        return self.history.add_message('user', content)

    def add_assistant_message(self, content: str) -> Message:
        """Add assistant message to history"""
        return self.history.add_message('assistant', content)

    def get_context_stats(self) -> Dict:
        """Get current context usage statistics"""
        limit = get_context_limit(self.current_model)
        history_tokens = self.history.get_token_count()
        wiki_tokens = self.wikipedia_tokens

        total_used = history_tokens + wiki_tokens
        available = limit - total_used - RESPONSE_RESERVE

        return {
            'model': self.current_model,
            'limit': limit,
            'history_tokens': history_tokens,
            'wikipedia_tokens': wiki_tokens,
            'total_used': total_used,
            'available': max(0, available),
            'response_reserve': RESPONSE_RESERVE,
            'usage_percent': min(100, int((total_used / limit) * 100)),
            'message_count': len(self.history.messages),
            'has_summary': self.history.summary is not None
        }

    def prepare_prompt(self, user_input: str, system_prompt: str = "") -> Tuple[str, Dict, bool]:
        """
        Prepare the full prompt for the AI, respecting context limits.

        Returns:
            - Full prompt string
            - Context stats
            - Whether context was truncated
        """
        limit = get_context_limit(self.current_model)
        system_tokens = estimate_tokens(system_prompt)
        user_tokens = estimate_tokens(user_input)

        # Calculate what we can fit
        available = limit - system_tokens - user_tokens - RESPONSE_RESERVE

        # Truncate Wikipedia context if needed
        wiki_context = self.wikipedia_context
        wiki_tokens = self.wikipedia_tokens
        truncated = False

        if wiki_tokens > available * 0.6:  # Wiki gets max 60% of available
            max_wiki_tokens = int(available * 0.6)
            # Estimate chars to keep (rough: 4 chars per token)
            max_chars = max_wiki_tokens * 4
            wiki_context = wiki_context[:max_chars] + "..."
            wiki_tokens = estimate_tokens(wiki_context)
            truncated = True

        available -= wiki_tokens

        # Get conversation history that fits
        history_msgs, history_tokens, history_truncated = self.history.get_context_messages(
            self.current_model,
            system_prompt + wiki_context
        )
        truncated = truncated or history_truncated

        # Build final prompt
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")

        if wiki_context:
            prompt_parts.append(f"\nReference Information:\n{wiki_context}")

        # Add conversation history
        for msg in history_msgs:
            if msg['role'] == 'user':
                prompt_parts.append(f"\nUser: {msg['content']}")
            elif msg['role'] == 'assistant':
                prompt_parts.append(f"\nAssistant: {msg['content']}")

        # Add current user input
        prompt_parts.append(f"\nUser: {user_input}")
        prompt_parts.append("\nAssistant:")

        full_prompt = "\n".join(prompt_parts)

        stats = self.get_context_stats()
        stats['prompt_tokens'] = estimate_tokens(full_prompt)
        stats['truncated'] = truncated

        return full_prompt, stats, truncated

    def prepare_messages(self, user_input: str, system_prompt: str = "") -> Tuple[List[Dict], Dict, bool]:
        """
        Prepare structured messages for Ollama /api/chat endpoint.

        Returns:
            - List of message dicts [{role: "system"/"user"/"assistant", content: ...}]
            - Context stats
            - Whether context was truncated
        """
        limit = get_context_limit(self.current_model)
        system_tokens = estimate_tokens(system_prompt)
        user_tokens = estimate_tokens(user_input)

        # Calculate what we can fit
        available = limit - system_tokens - user_tokens - RESPONSE_RESERVE

        # Truncate Wikipedia context if needed
        wiki_context = self.wikipedia_context
        wiki_tokens = self.wikipedia_tokens
        truncated = False

        if wiki_tokens > available * 0.6:  # Wiki gets max 60% of available
            max_wiki_tokens = int(available * 0.6)
            max_chars = max_wiki_tokens * 4
            wiki_context = wiki_context[:max_chars] + "..."
            wiki_tokens = estimate_tokens(wiki_context)
            truncated = True

        available -= wiki_tokens

        # Get conversation history that fits
        history_msgs, history_tokens, history_truncated = self.history.get_context_messages(
            self.current_model,
            system_prompt + wiki_context
        )
        truncated = truncated or history_truncated

        # Build structured messages array
        messages = []

        # 1. System message: domain prompt + wiki context combined
        system_content = system_prompt
        if wiki_context:
            system_content += f"\n\nReference Information:\n{wiki_context}"
        if system_content:
            messages.append({"role": "system", "content": system_content})

        # 2. Conversation history messages (already role-tagged)
        for msg in history_msgs:
            # Skip any system messages from history (we already have the system message)
            if msg['role'] in ('user', 'assistant'):
                messages.append({"role": msg['role'], "content": msg['content']})

        # 3. Current user input
        messages.append({"role": "user", "content": user_input})

        stats = self.get_context_stats()
        stats['prompt_tokens'] = system_tokens + wiki_tokens + history_tokens + user_tokens
        stats['truncated'] = truncated

        return messages, stats, truncated

    def check_overflow(self, additional_text: str = "") -> Tuple[bool, str]:
        """
        Check if adding text would overflow context.

        Returns:
            - Whether overflow would occur
            - Warning message if applicable
        """
        stats = self.get_context_stats()
        additional_tokens = estimate_tokens(additional_text)

        if stats['available'] < additional_tokens:
            overflow_amount = additional_tokens - stats['available']
            return True, f"Context overflow: need {additional_tokens} tokens but only {stats['available']} available. {overflow_amount} tokens over limit."

        if stats['usage_percent'] > 80:
            return False, f"Warning: Context {stats['usage_percent']}% full. Consider starting a new conversation."

        return False, ""

    def clear_history(self):
        """Clear conversation history"""
        self.history.clear()
        self.clear_wikipedia_context()

    def summarize_for_new_context(self) -> str:
        """Create a summary of current conversation for a fresh start"""
        if not self.history.messages:
            return ""

        topics = []
        for msg in self.history.messages[-5:]:  # Last 5 messages
            # Extract key topics (simple keyword extraction)
            words = msg.content.lower().split()
            # Filter common words
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                        'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                        'from', 'as', 'into', 'through', 'during', 'before', 'after',
                        'above', 'below', 'between', 'under', 'again', 'further',
                        'then', 'once', 'here', 'there', 'when', 'where', 'why',
                        'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                        'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                        'because', 'until', 'while', 'about', 'what', 'which', 'who',
                        'this', 'that', 'these', 'those', 'i', 'me', 'my', 'you',
                        'your', 'it', 'its', 'we', 'our', 'they', 'their', 'them'}
            keywords = [w for w in words if len(w) > 3 and w not in stopwords][:3]
            topics.extend(keywords)

        unique_topics = list(set(topics))[:5]
        if unique_topics:
            return f"Previous discussion topics: {', '.join(unique_topics)}"
        return ""


# Global context manager instance (thread-safe singleton)
_context_manager = None
_context_lock = threading.Lock()


def get_context_manager() -> ContextManager:
    """Get or create the global context manager (thread-safe)"""
    global _context_manager
    with _context_lock:
        if _context_manager is None:
            _context_manager = ContextManager()
        return _context_manager


# Convenience functions
def count_tokens(text: str) -> int:
    """Count tokens in text"""
    return estimate_tokens(text)


def get_model_limit(model: str) -> int:
    """Get context limit for model"""
    return get_context_limit(model)


def format_context_status(stats: Dict) -> str:
    """Format context stats for display"""
    bar_length = 20
    filled = int((stats['usage_percent'] / 100) * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)

    status = f"Context: [{bar}] {stats['usage_percent']}%"
    status += f" | {stats['total_used']}/{stats['limit']} tokens"
    status += f" | {stats['message_count']} messages"

    if stats.get('truncated'):
        status += " | [WARN] Truncated"
    if stats.get('has_summary'):
        status += " | [NOTE] Summarized"

    return status


if __name__ == "__main__":
    # Test the context manager
    print("=== Context Manager Test ===\n")

    cm = get_context_manager()
    cm.set_model('smollm2:360m')

    print(f"Model: {cm.current_model}")
    print(f"Context limit: {get_context_limit(cm.current_model)} tokens\n")

    # Add some messages
    cm.add_user_message("What is malaria?")
    cm.add_assistant_message("Malaria is a disease caused by parasites transmitted through mosquito bites.")
    cm.add_user_message("How can I prevent it?")

    # Set some Wikipedia context
    cm.set_wikipedia_context("Malaria is a mosquito-borne infectious disease...")

    # Get stats
    stats = cm.get_context_stats()
    print(format_context_status(stats))
    print(f"\nDetailed stats: {stats}")

    # Prepare a prompt
    prompt, stats, truncated = cm.prepare_prompt(
        "What are the symptoms?",
        "You are a helpful medical assistant."
    )
    print(f"\nPrepared prompt ({stats['prompt_tokens']} tokens):")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
