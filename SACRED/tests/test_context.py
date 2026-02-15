#!/usr/bin/env python3
"""
Tests for context management functionality
"""

import pytest
import sys
from pathlib import Path

# Add APP_Production to path
sys.path.insert(0, str(Path(__file__).parent.parent / "APP_Production"))


class TestContextManager:
    """Tests for context management"""

    @pytest.fixture
    def context_manager(self):
        """Create context manager instance"""
        try:
            from context_manager import ContextManager
            return ContextManager()
        except ImportError:
            pytest.skip("context_manager module not available")

    def test_token_estimation_english(self, context_manager):
        """Verify token estimation works for English text"""
        from context_manager import estimate_tokens
        text = "This is a simple English sentence for testing."
        tokens = estimate_tokens(text)

        # ~4 chars per token for English
        expected = len(text) / 4
        assert 0.5 * expected < tokens < 2 * expected

    def test_token_estimation_tigrinya(self, context_manager):
        """Verify token estimation works for Tigrinya text"""
        from context_manager import estimate_tokens
        text = "ሰላም ከመይ ኣለኻ"  # Tigrinya text
        tokens = estimate_tokens(text)

        # ~2 chars per token for Tigrinya (each character is more semantic)
        assert tokens > 0

    def test_model_limits_defined(self, context_manager):
        """Verify model context limits are defined"""
        from context_manager import get_context_limit
        expected_models = ["smollm2:360m", "qwen3:0.6b"]

        for model in expected_models:
            limit = get_context_limit(model)
            assert limit > 0, f"Model {model} should have a limit"
            assert limit <= 16384, f"Model {model} limit seems too high: {limit}"

    def test_history_management(self, context_manager):
        """Verify conversation history management works"""
        # Add messages using proper API
        context_manager.add_user_message("Hello")
        context_manager.add_assistant_message("Hi there!")

        history = context_manager.history.messages
        assert len(history) >= 2

    def test_history_trimming(self, context_manager):
        """Verify history trimming when limit exceeded"""
        from context_manager import estimate_tokens
        # Add many messages
        for i in range(100):
            context_manager.add_user_message(f"Message {i} " * 50)
            context_manager.add_assistant_message(f"Response {i} " * 50)

        # History should be trimmed
        history = context_manager.history.messages
        total_tokens = sum(estimate_tokens(m.content) for m in history)

        # Should have been trimmed by ConversationHistory
        assert len(history) <= 20, "History should be trimmed"

    def test_clear_history(self, context_manager):
        """Verify history clearing works"""
        context_manager.add_user_message("Test message")
        context_manager.clear_history()

        history = context_manager.history.messages
        assert len(history) == 0, "History should be empty after clear"

    def test_context_percentage(self, context_manager):
        """Verify context usage percentage calculation"""
        context_manager.add_user_message("Test " * 100)

        stats = context_manager.get_context_stats()
        percentage = stats['usage_percent']
        assert 0 <= percentage <= 100, f"Percentage should be 0-100, got {percentage}"


class TestModelLimits:
    """Tests for model context limits"""

    def test_smollm2_limit(self):
        """Verify SmolLM2 360M has expected context limit"""
        from context_manager import get_context_limit
        limit = get_context_limit("smollm2:360m")
        assert limit == 8192, f"SmolLM2 360M should have 8192 limit, got {limit}"

    def test_qwen_limit(self):
        """Verify Qwen 0.6B has expected context limit"""
        from context_manager import get_context_limit
        limit = get_context_limit("qwen3:0.6b")
        assert limit == 8192, f"Qwen 0.6B should have 8192 limit, got {limit}"

    def test_unknown_model_default(self):
        """Verify unknown models get default limit"""
        from context_manager import get_context_limit, DEFAULT_CONTEXT_LIMIT
        limit = get_context_limit("unknown-model-xyz")
        assert limit == DEFAULT_CONTEXT_LIMIT, "Unknown models should get default limit"


class TestOverflowHandling:
    """Tests for context overflow handling"""

    @pytest.fixture
    def context_manager(self):
        """Create context manager instance"""
        try:
            from context_manager import ContextManager
            return ContextManager()
        except ImportError:
            pytest.skip("context_manager module not available")

    def test_overflow_check(self, context_manager):
        """Verify overflow check works"""
        # Fill context near limit
        for i in range(50):
            context_manager.add_user_message("Test message " * 100)
            context_manager.add_assistant_message("Response " * 100)

        # Check overflow
        overflow, message = context_manager.check_overflow("more text " * 1000)
        assert isinstance(overflow, bool)
        assert isinstance(message, str)

    def test_graceful_overflow(self, context_manager):
        """Verify overflow is handled gracefully"""
        # Try to overflow
        huge_message = "x" * 100000

        # Should not raise exception
        try:
            context_manager.add_user_message(huge_message)
        except Exception as e:
            pytest.fail(f"Overflow should be handled gracefully: {e}")
