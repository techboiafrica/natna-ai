# Database System Documentation

How NATNA's knowledge databases work.

## Overview

NATNA uses SQLite databases for all knowledge storage. This enables:
- Fast full-text search (FTS5)
- Offline operation
- Portable storage on USB drive

---

## Wikipedia Databases

Located in `SACRED/educational_archive/knowledge/`

### Main Database

| Database | Size | Articles | Description |
|----------|------|----------|-------------|
| `massive_wikipedia.db` | 4.4 GB | 475,000+ | Main English Wikipedia content |

### Domain-Specific Databases

| Database | Size | Articles | Domain |
|----------|------|----------|--------|
| `medical_wikipedia.db` | 20 MB | 78,353 | Health, diseases, treatments |
| `college_education_wikipedia.db` | 4.6 MB | 24,741 | University-level content |
| `technical_education_wikipedia.db` | 3.7 MB | 26,576 | Engineering, vocational |
| `mathematics_wikipedia.db` | 1.8 MB | 9,945 | Math concepts and formulas |
| `k12_education_wikipedia.db` | 2.6 MB | 2,459 | K-12 educational content |
| `technical_enhanced_wikipedia.db` | 532 KB | 3,459 | Additional technical content |
| `agriculture_wikipedia.db` | 332 KB | 2,263 | Farming, crops, livestock |
| `african_arab_world_massive_wikipedia.db` | 1.8 MB | 240 | African/Arab regional content |

### Schema

```sql
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    url TEXT,
    word_count INTEGER,
    char_count INTEGER,
    category TEXT,
    domain TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Full-text search index
CREATE VIRTUAL TABLE articles_fts USING fts5(
    title, content, tokenize='porter'
);
```

### Search Algorithm

1. **Primary**: FTS5 MATCH with BM25 scoring
   ```
   final_score = (bm25_score * 0.7) + (title_match_score * 0.3)
   ```
2. **Fallback**: LIKE search on title (slower)

### Search Behavior

- 2-word queries use AND logic (e.g., "Julius Caesar")
- Multi-word queries use OR logic (better recall)
- Results filtered to articles with 25+ words
- Only first 500 characters fetched for preview

---

## Tigrinya Translation Database

Located in `SACRED/organized_data/databases/`

### Main Database

| Database | Size | Description |
|----------|------|-------------|
| `massive_tigrinya_database.db` | 181 MB | 265,972+ translation entries |

### Supporting Databases

| Database | Size | Description |
|----------|------|-------------|
| `academic_translations_v2.db` | 122 MB | Scientific/technical terms |
| `cultural_translations_v2.db` | 130 MB | Cultural expressions |
| `word_dictionary_v2.db` | 5.8 MB | Basic word dictionary |
| `hot_cache_translations_v2.db` | 172 KB | Frequently used terms |

### Schema (Main Translation Table)

```sql
CREATE TABLE translations (
    id INTEGER PRIMARY KEY,
    english TEXT,
    tigrinya TEXT,
    confidence REAL,
    frequency INTEGER,
    domain TEXT
);
```

### Translation Lookup

1. Exact match lookup
2. Phrase pattern matching
3. Word-by-word fallback
4. Domain-specific term lookup

---

## Database Warming

On startup, NATNA "warms" the Wikipedia database for faster queries:

### 3-Phase Warming

1. **Phase 1**: Load database indexes into memory
2. **Phase 2**: Cache domain-specific terms
3. **Phase 3**: Preload common article metadata

### Warming Status

The UI shows warming progress in the status bar. Queries work during warming but may be slower.

---

## SQLite Optimizations

### Connection Settings

```python
PRAGMA journal_mode = WAL;      # Write-ahead logging
PRAGMA cache_size = -100000;    # 100MB cache (adaptive)
PRAGMA mmap_size = 268435456;   # 256MB memory mapping
```

### Connection Pooling

- Maximum 3 concurrent connections
- 30-second connection timeout
- Automatic connection return

---

## Database Paths

Configured in `APP_Production/path_config.py`:

```python
TIGRINYA_DB = "organized_data/databases/massive_tigrinya_database.db"
WIKIPEDIA_DB = "educational_archive/knowledge/massive_wikipedia.db"
KNOWLEDGE_DIR = "educational_archive/knowledge"
```

---

## Mobile Access

Database queries work identically from mobile devices. The databases remain on the host computer - only query results are sent to the mobile browser.

---

## Maintenance

### Checking Database Integrity

```bash
sqlite3 massive_wikipedia.db "PRAGMA integrity_check;"
# Expected output: ok
```

### Rebuilding FTS Index

If search stops working:

```bash
sqlite3 massive_wikipedia.db "INSERT INTO articles_fts(articles_fts) VALUES('rebuild');"
```

### Updating Statistics

```bash
sqlite3 massive_wikipedia.db "ANALYZE;"
```

---

## Adding Content

### Adding Wikipedia Articles

New articles should match the existing schema:

```sql
INSERT INTO articles (title, content, url, word_count, char_count, domain)
VALUES ('Article Title', 'Full content...', 'https://...', 500, 2500, 'medical');

-- Update FTS index
INSERT INTO articles_fts (rowid, title, content)
SELECT id, title, content FROM articles WHERE id = last_insert_rowid();
```

### Adding Translations

```sql
INSERT INTO translations (english, tigrinya, confidence, domain)
VALUES ('hello', 'ሰላም', 1.0, 'general');
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No such table: articles" | Database corrupted - restore from backup |
| Search returns nothing | FTS index may need rebuild |
| Slow queries | Run ANALYZE, check cache settings |
| "Database is locked" | Stop other NATNA instances |

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more solutions.
