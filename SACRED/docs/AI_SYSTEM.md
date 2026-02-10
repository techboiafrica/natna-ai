# AI System Documentation

How NATNA's AI and translation systems work.

## Overview

NATNA uses a local AI inference system powered by Ollama. All processing happens on your device - no internet required after initial setup.

```
User Query → Language Detection → Domain Routing → AI Model → Response
                                       ↓
                              Wikipedia Context (optional)
                                       ↓
                              Tigrinya Translation
                                       ↓
                              Bilingual Response
```

## Ollama Integration

### What is Ollama?

Ollama is a local AI model runner. NATNA includes portable Ollama binaries for each platform:

| Platform | Binary Location |
|----------|-----------------|
| Mac | `SACRED/runtime/ollama/ollama` |
| Windows | `SACRED/runtime/ollama-windows/ollama.exe` |
| Linux | `SACRED/runtime/ollama-linux/bin/ollama` |

### Model Storage

Models are stored in `SACRED/runtime/models/` (approximately 62GB total). This allows NATNA to work on any computer without downloading models.

### Communication

NATNA communicates with Ollama via HTTP on port 11434:

```
POST http://localhost:11434/api/generate
{
    "model": "qwen3:0.6b",
    "prompt": "...",
    "stream": false
}
```

---

## Available Models

### General Purpose (Uncensored)

| Model | Size | Description |
|-------|------|-------------|
| `deepseek-r1-abliterated-4gb` | 4 GB | Uncensored for sensitive topics |
| `deepseek-r1-uncensored-16gb` | 16 GB | Advanced uncensored reasoning |

### Small/Fast Models (4GB RAM)

| Model | Size | Best For |
|-------|------|----------|
| `qwen3:0.6b` | 522 MB | **Default** - fast, good quality |
| `smollm2:360m` | 726 MB | Ultra-light fallback |
| `alibayram/smollm3` | 1.8 GB | Good reasoning, 64K context |
| `phi4-mini` | 2.5 GB | Better reasoning, 128K context |

### Coding Models (8GB RAM)

| Model | Size | Best For |
|-------|------|----------|
| `qwen2.5-coder:7b` | 4.7 GB | General coding |
| `deepseek-coder:6.7b` | 3.8 GB | Code generation |
| `codegemma:7b` | 5 GB | Google's coding model |

### Coding Models (16GB RAM)

| Model | Size | Best For |
|-------|------|----------|
| `qwen2.5-coder:14b` | 9 GB | Complex code tasks |
| `deepseek-coder-v2:16b` | 8.9 GB | Advanced code reasoning |
| `codellama:13b` | 7.4 GB | Meta's coding model |

---

## Translation Pipeline

### Language Detection

The system detects input language by checking for:
- Ethiopic Unicode characters (U+1200 to U+137F)
- Common Tigrinya patterns

### Tigrinya Translation

NATNA uses a 265,000+ entry SQL database for translation:

1. **Word-by-word lookup** - Direct dictionary matches
2. **Phrase matching** - Common expressions
3. **Domain-specific terms** - Medical, agricultural, technical vocabulary
4. **Academic translations** - Scientific terminology

**Database sources:**
- `massive_tigrinya_database.db` (181MB) - Main dictionary
- `academic_translations_v2.db` (122MB) - Scientific terms
- `cultural_translations_v2.db` (130MB) - Cultural terms

### Response Generation

```python
# Simplified flow
1. User input received
2. Language detected (English or Tigrinya)
3. Domain determined (medical, educational, etc.)
4. Wikipedia context gathered (automatically)
5. Prompt constructed with context
6. Ollama generates English response (plain text, no markdown)
7. Response translated to Tigrinya via NATNA-TI model
8. Both versions returned to UI
```

### Text Formatting

All AI system prompts include instructions to output plain text without markdown formatting (no asterisks for bold/italic). This ensures clean display in both English and Tigrinya without parsing issues.

---

## Context Management

### Token Limits

Each model has a context window limit:

| Model | Context Tokens |
|-------|----------------|
| Qwen 3 0.6B | 2,048 |
| SmolLM2 360M, SmolLM3 3B | 8,192 |
| Phi-4 Mini | 8,192 |
| DeepSeek Abliterated 4GB | 4,096 |
| Coding models (7B) | 8,192 |
| Coding models (14B+) | 16,384 |

### Context Tracking

The UI shows context usage as a percentage. When approaching the limit:
- Older messages are summarized
- Context is trimmed to fit

### Clearing Context

Click "Clear History" to reset the conversation and free up context space.

---

## Wikipedia Integration

### How It Works

1. User asks a question
2. System automatically searches Wikipedia databases using FTS5 full-text search
3. Relevant articles appear in Research Panel (Sources)
4. User can click "Expand Answer" to enhance response with Wikipedia context

### Automatic Search

Wikipedia search happens automatically with each query - no manual search button needed. Results appear in the Sources panel.

### Article Viewer

Click any article in the Research Panel to read the full content. From there, you can:
- Read the complete article
- Click "Learn More" to ask the AI about that topic (response includes Tigrinya translation)

### Enhanced Answers

Click "Expand Answer" after receiving a response to get a more detailed answer using Wikipedia context. Enhanced answers are also translated to Tigrinya when bilingual mode is enabled.

---

## Performance Optimization

### Model Preloading

On startup, NATNA preloads the default Qwen 3 0.6B model so first queries are fast.

### Memory Management

- Models are kept loaded for quick responses
- Only one model is loaded at a time (OLLAMA_MAX_LOADED_MODELS=1)
- RAM usage is displayed in the status bar

### Auto-Restart

If Ollama disconnects, NATNA attempts automatic restart with a 30-second cooldown.

---

## Mobile Access

The AI system works identically when accessed from phone/tablet. All processing still happens on the host computer - mobile devices are just displaying the interface.

See [QUICK_START.md](QUICK_START.md#mobile-access-phonetablet) for setup instructions.

---

## Troubleshooting AI Issues

| Issue | Solution |
|-------|----------|
| "Rhizome Server: Disconnected" | Ollama not running - check launcher terminal |
| Slow first response | Model loading - wait for preload to complete |
| Out of memory | Switch to smaller model (Qwen 3 0.6B) |
| No Tigrinya translation | Check database paths in path_config.py |

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more solutions.
