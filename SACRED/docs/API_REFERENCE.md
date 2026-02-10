# API Reference

NATNA exposes a REST API on `http://localhost:8080`.

## Main Endpoints

### Chat (Main Query)

Send a message to the AI.

```
POST /api/chat
Content-Type: application/json

{
    "message": "What is photosynthesis?",
    "domain": "education",
    "model": "qwen3:0.6b"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| message | string | Yes | The question or message |
| domain | string | No | Knowledge domain (default: "general") |
| model | string | No | AI model to use (default: "qwen3:0.6b") |

**Valid Domains:** `general`, `medical`, `education`, `college`, `technical`, `agriculture`, `programming`

**Response:**

```json
{
    "english": "Photosynthesis is the process by which plants...",
    "tigrinya": "ጽማረ ብርሃን ማለት...",
    "sources": [
        {
            "id": 12345,
            "title": "Photosynthesis",
            "snippet": "Process used by plants...",
            "word_count": 2500
        }
    ],
    "query_time": 2.5
}
```

### Expand Answer

Enhance a response with Wikipedia context.

```
POST /api/expand
Content-Type: application/json

{
    "original_answer": "Photosynthesis is...",
    "query": "photosynthesis",
    "wikipedia_sources": [...],
    "english_only": false
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| original_answer | string | Yes | The original AI response to enhance |
| query | string | Yes | The original user query |
| wikipedia_sources | array | Yes | Wikipedia sources from the original response |
| english_only | boolean | No | If true, skip Tigrinya translation (default: false) |

**Response:**

```json
{
    "enhanced_answer": "Enhanced answer with Wikipedia context...",
    "enhanced_answer_ti": "ዝተመሓየሸ መልሲ...",
    "source": "Enhanced with Wikipedia: Article Title"
}
```

### Search Wikipedia

Manually search Wikipedia articles.

```
POST /api/search_wikipedia
Content-Type: application/json

{
    "query": "Ethiopian history"
}
```

**Response:**

```json
{
    "results": [
        {
            "id": 54321,
            "title": "Ethiopian Empire",
            "snippet": "The Ethiopian Empire was...",
            "word_count": 5420,
            "score": 15.7
        }
    ],
    "total": 45,
    "query": "Ethiopian history"
}
```

### More Articles (Pagination)

Get additional Wikipedia results.

```
POST /api/more_articles
Content-Type: application/json

{
    "query": "Ethiopian history",
    "offset": 5
}
```

---

## Status Endpoints

### Ollama Status

```
GET /api/status/ollama
```

**Response:**

```json
{
    "status": "connected"
}
```
or
```json
{
    "status": "disconnected"
}
```

### Wikipedia Status

```
GET /api/status/wikipedia
```

**Response:**

```json
{
    "available": true,
    "article_count": 475154,
    "database_size_gb": 4.4
}
```

### Context Status

```
GET /api/status/context
```

**Response:**

```json
{
    "tokens_used": 512,
    "tokens_limit": 2048,
    "percentage": 25,
    "model": "qwen3:0.6b",
    "messages": 4
}
```

### Memory Status

```
GET /api/status/memory
```

**Response:**

```json
{
    "rss": 450.5,
    "vms": 1200.3,
    "percent": 5.2,
    "model": "qwen3:0.6b"
}
```

### All Status (Combined)

```
GET /api/status/all
```

**Response:**

```json
{
    "ollama": {"status": "connected"},
    "wikipedia": {"available": true, "article_count": 475154},
    "context": {"percentage": 25},
    "memory": {"rss": 450.5, "percent": 5.2}
}
```

### Warming Status

```
GET /api/warming_status
```

**Response:**

```json
{
    "phase": 2,
    "complete": false,
    "progress": 65,
    "message": "Loading domain terms..."
}
```

---

## Article Endpoints

### Get Full Article

```
GET /api/article/{id}
```

**Response:**

```json
{
    "id": 12345,
    "title": "Photosynthesis",
    "content": "Full article content here...",
    "url": "https://en.wikipedia.org/wiki/Photosynthesis",
    "word_count": 2500,
    "char_count": 15000
}
```

---

## Control Endpoints

### Clear History

```
GET /api/clear_history
```

**Response:**

```json
{
    "success": true,
    "message": "History cleared"
}
```

### Start Ollama Server

```
POST /api/start-ollama
```

**Response:**

```json
{
    "success": true,
    "message": "Ollama server started"
}
```
or
```json
{
    "success": false,
    "error": "Failed to start Ollama"
}
```

### Verify Admin Password

Used for mobile model changes. Desktop users do not require authentication.

The default password is `Mekelle8080`. Override it by setting the `NATNA_ADMIN_PW` environment variable before launching NATNA.

Rate limited to 5 attempts per minute per IP. Exceeding the limit returns `429 Too Many Requests`.

```
POST /api/verify_admin
Content-Type: application/json

{
    "password": "Mekelle8080"
}
```

**Response:**

```json
{
    "success": true
}
```
or
```json
{
    "success": false
}
```
or (rate limited)
```json
{
    "error": "Too many attempts. Try again later."
}
```

---

## Error Responses

All errors return JSON with an `error` field:

```json
{
    "error": "Error description",
    "code": "ERROR_CODE"
}
```

**Common Error Codes:**

| Code | Description |
|------|-------------|
| `OLLAMA_UNAVAILABLE` | Ollama server not running |
| `MODEL_NOT_FOUND` | Requested model not available |
| `DATABASE_ERROR` | Database connection failed |
| `INVALID_REQUEST` | Missing or invalid parameters |
| `TIMEOUT` | Request exceeded time limit |

---

## Request Headers

| Header | Value |
|--------|-------|
| Content-Type | application/json |
| Accept | application/json |

---

## Code Examples

### Python

```python
import requests

# Chat query
response = requests.post(
    "http://localhost:8080/api/chat",
    json={
        "message": "What is DNA?",
        "domain": "education",
        "model": "qwen3:0.6b"
    }
)
data = response.json()
print(data["english"])
print(data["tigrinya"])
```

### JavaScript

```javascript
// Chat query
fetch("http://localhost:8080/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        message: "What is DNA?",
        domain: "education",
        model: "qwen3:0.6b"
    })
})
.then(res => res.json())
.then(data => {
    console.log(data.english);
    console.log(data.tigrinya);
});
```

### cURL

```bash
# Chat query
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is DNA?", "domain": "education"}'

# Check status
curl http://localhost:8080/api/status/all

# Search Wikipedia
curl -X POST http://localhost:8080/api/search_wikipedia \
  -H "Content-Type: application/json" \
  -d '{"query": "photosynthesis"}'
```

---

## Mobile Access

All API endpoints work identically when accessed from mobile devices via the LAN IP address:

```
http://192.168.x.x:8080/api/chat
```

The mobile device sends requests to the host computer, which processes them and returns responses.
