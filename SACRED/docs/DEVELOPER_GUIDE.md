# Developer Guide

Architecture and code structure for continued development.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Browser                             │
│                   localhost:8080                             │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                NatnaUI.py                        │
│                  HTTP Server (8080)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Static HTML  │  │  API Router  │  │ Status APIs  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Intelligent │   │  Wikipedia  │   │   Ollama    │
│ Translator  │   │   Search    │   │   Server    │
│             │   │             │   │  :11434     │
└──────┬──────┘   └──────┬──────┘   └─────────────┘
       │                 │
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│  Tigrinya   │   │  Wikipedia  │
│  Databases  │   │  Databases  │
└─────────────┘   └─────────────┘
```

---

## Key Files

### APP_Production/

| File | Lines | Purpose |
|------|-------|---------|
| `NatnaUI.py` | ~3000 | Main HTTP server, UI, all API endpoints |
| `intelligent_translator.py` | ~800 | AI orchestration, translation, domain routing |
| `wikipedia_search.py` | ~1400 | FTS5 search, connection pooling, warming |
| `context_manager.py` | ~200 | Token counting, history management |
| `path_config.py` | ~50 | Central path configuration |

### educational_archive/knowledge/

| File | Purpose |
|------|---------|
| `wikipedia_search.py` | WikipediaKnowledgeSearch class |
| `massive_wikipedia.db` | Main Wikipedia database |
| `*_wikipedia.db` | Domain-specific databases |

---

## Core Classes

### TerminalHandler (NatnaUI.py)

HTTP request handler serving the UI and API.

```python
class TerminalHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Serves HTML UI at /
        # API endpoints: /api/status/*, /api/article/*

    def do_POST(self):
        # /api/chat - Main query endpoint
        # /api/expand - Wikipedia-enhanced response
        # /api/search_wikipedia - Manual search
        # /api/more_articles - Pagination
```

### IntelligentTigrinyaTranslator (intelligent_translator.py)

Core AI and translation logic.

```python
class IntelligentTigrinyaTranslator:
    def __init__(self):
        # Loads Tigrinya database
        # Initializes Wikipedia search
        # Checks Ollama availability

    def process_query(self, query, domain, model):
        # Main entry point for queries
        # Returns bilingual response

    def _query_local_model(self, prompt, model):
        # Sends prompt to Ollama
        # Returns AI response

    def search_wikipedia(self, query, domain, limit):
        # Searches Wikipedia databases
        # Returns ranked results
```

### WikipediaKnowledgeSearch (wikipedia_search.py)

Wikipedia search with FTS5.

```python
class WikipediaKnowledgeSearch:
    def __init__(self, db_path):
        # Opens database connection
        # Starts background warming

    def search(self, query, limit, min_words):
        # FTS5 BM25 search
        # Returns ranked articles

    def get_article(self, article_id):
        # Fetches full article content
```

---

## API Endpoints

### GET Endpoints

| Endpoint | Returns |
|----------|---------|
| `/` | Main HTML UI |
| `/api/status/ollama` | Ollama connection status |
| `/api/status/wikipedia` | Wikipedia DB status + article count |
| `/api/status/context` | Current context usage |
| `/api/status/memory` | RAM usage |
| `/api/status/all` | Combined status |
| `/api/warming_status` | Database warming progress |
| `/api/article/{id}` | Full Wikipedia article |
| `/api/clear_history` | Clears conversation |

### POST Endpoints

| Endpoint | Payload | Returns |
|----------|---------|---------|
| `/api/chat` | `{message, domain, model, english_only}` | Bilingual response + sources |
| `/api/expand` | `{original_answer, query, wikipedia_sources, english_only}` | Enhanced response (EN + TI) |
| `/api/search_wikipedia` | `{query}` | Search results |
| `/api/more_articles` | `{query, offset}` | Additional results |
| `/api/start-ollama` | - | Starts Ollama server |
| `/api/verify_admin` | `{password}` | Admin authentication for mobile model changes (default: `Mekelle8080`, override with `NATNA_ADMIN_PW` env var, rate limited to 5/min/IP) |

---

## Adding a New Feature

### Example: Adding a New Domain

1. **Update UI dropdown** (NatnaUI.py ~line 1810):
```html
<option value="newdomain">New Domain / ትግርኛ</option>
```

2. **Update mobile picker** (~line 1900):
```html
<div class="picker-option" data-value="newdomain" onclick="pickDomain(this)">New Domain</div>
```

3. **Add domain-specific database** (optional):
   - Create `newdomain_wikipedia.db` in knowledge folder
   - Follow existing schema

4. **Update intelligent_translator.py** (if special handling needed):
```python
if domain == 'newdomain':
    # Custom logic
```

### Example: Adding a New API Endpoint

In `NatnaUI.py`:

```python
# In do_GET or do_POST method
elif self.path == '/api/your_endpoint':
    try:
        # Your logic here
        result = {"data": "your data"}

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode('utf-8'))
    except Exception as e:
        self.send_response(500)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
```

---

## Debugging

### Log Locations

| Log | Location |
|-----|----------|
| Startup log | Terminal output |
| Wikipedia debug | `/tmp/natna_wiki_debug.txt` |
| Conversations | `organized_data/config/natna_conversations.log` |

### Enable Verbose Logging

In `intelligent_translator.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Ollama Connection

```bash
curl http://localhost:11434/api/tags
```

### Test Wikipedia Search

```python
from wikipedia_search import WikipediaKnowledgeSearch
ws = WikipediaKnowledgeSearch("path/to/massive_wikipedia.db")
results = ws.search("photosynthesis", limit=5)
print(results)
```

---

## Process Management

### Singleton Lock

NATNA uses file locking to prevent multiple instances:

```python
lock_file = open('/tmp/natna.lock', 'w')
fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
```

### Signal Handling

Clean shutdown on Ctrl+C or kill:

```python
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
```

### Resource Cleanup

Registered via `atexit`:
- Release singleton lock
- Close database connections
- Stop HTTP server

---

## Performance Considerations

### Memory

- Default model (Qwen 3 0.6B) uses ~522MB
- Wikipedia search uses adaptive caching based on system RAM
- Monitor via `/api/status/memory`

### Startup Time

- Model preloading happens asynchronously
- Database warming runs in background thread
- UI is usable during warming (slower queries)

### Database Queries

- Use FTS5 MATCH for text search (fast)
- Avoid LIKE '%..%' patterns (full table scan)
- Limit result sets appropriately

---

## Testing

### Run Tests

```bash
cd SACRED
python -m pytest tests/ -v
```

### Manual Testing Checklist

- [ ] Startup without errors
- [ ] Query in English returns bilingual response
- [ ] Query in Tigrinya detected and processed
- [ ] Domain switching works
- [ ] Model switching works
- [ ] Wikipedia search returns results
- [ ] Article viewer opens
- [ ] Mobile access works
- [ ] Clean shutdown with Ctrl+C

---

## Common Pitfalls

1. **Hardcoded paths** - Always use `path_config.py`
2. **Blocking the main thread** - Use threading for long operations
3. **Database connection leaks** - Return connections to pool
4. **Large responses** - Truncate content appropriately
5. **Unicode handling** - Always use UTF-8 encoding

