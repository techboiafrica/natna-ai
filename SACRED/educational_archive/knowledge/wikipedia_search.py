#!/usr/bin/env python3
"""
Wikipedia Knowledge Search System for PARABL NATNA
Provides educational article search from 84,500 Wikipedia articles

Features:
- Quality-aware filtering (minimum word count)
- First paragraph extraction for context injection
- Wiki markup cleaning
- Multi-keyword relevance scoring
- Compatible with existing medical/agricultural search patterns
"""

import sqlite3
import re
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any
import time
import math


class WikipediaKnowledgeSearch:
    """Wikipedia article search for NATNA educational support"""

    # Minimum word count for quality filtering
    DEFAULT_MIN_WORDS = 25

    # Maximum words to include in summary
    MAX_SUMMARY_WORDS = 150

    # Wiki markup patterns to clean
    WIKI_PATTERNS = [
        # Remove markdown image syntax and URLs FIRST (highest priority)
        (r'!\[[^\]]*\]\([^)]*\)', ''),               # ![alt](url) -> remove markdown images
        (r'\\\[[^\]]*\\\]\([^)]*\)', ''),            # \[text\]\(url\) -> remove escaped markdown links
        (r'\\\[[^\]]*\\\]\\\([^)]*\\\)', ''),        # \[text\]\\\(url\\\) -> remove double-escaped
        (r'\([Hh]ttps?://[^)]+\)', ''),              # (https://...) -> remove URLs in parentheses
        (r'[Hh]ttps?://\S+', ''),                    # https://... -> remove standalone URLs
        (r'\bSpecial:Redirect/File/[^\s)]*', ''),   # Special:Redirect/File/... -> remove
        (r'\bWiki/Special:[^\s)]*', ''),             # Wiki/Special:... -> remove
        (r'[Ww]idth:\*?\*?\s*\d+\)?', ''),           # Width:** 300) -> remove
        (r'[Hh]eight:\*?\*?\s*\d+\)?', ''),          # Height:** 300) -> remove
        # Remove wiki-style image and file references
        (r'\[\[File:[^\]]*\]\]', ''),                # [[File:...]] -> remove completely
        (r'\[\[Image:[^\]]*\]\]', ''),               # [[Image:...]] -> remove completely
        (r'\[\[Media:[^\]]*\]\]', ''),               # [[Media:...]] -> remove completely
        (r'\[\[[^:]*\.(?:jpg|jpeg|png|gif|svg|webp|tiff)[^\]]*\]\]', ''), # Image files by extension
        (r'[^\s]*\.(?:jpg|jpeg|png|gif|svg|webp|tiff)\b[^\s]*', ''),  # Standalone image filenames
        (r'thumb\|[^|]*\|?', ''),                    # thumb|... -> remove
        (r'frame\|[^|]*\|?', ''),                    # frame|... -> remove
        (r'frameless\|[^|]*\|?', ''),                # frameless|... -> remove
        (r'right\|', ''),                            # right| -> remove
        (r'left\|', ''),                             # left| -> remove
        (r'center\|', ''),                           # center| -> remove
        (r'upright\|?', ''),                         # upright| -> remove
        (r'\d+px\|?', ''),                           # 180px| -> remove
        (r'alt=[^|]*\|?', ''),                       # alt=text| -> remove
        # Process links after image removal
        (r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2'),     # [[link|text]] -> text
        (r'\[\[([^\]]+)\]\]', r'\1'),                # [[link]] -> link
        # Templates and other markup
        (r'\{\{[^}]*\}\}', ''),                      # {{template}} -> remove (greedy)
        (r'\{\{[^}]*$', ''),                         # {{template without closing -> remove to end of line
        (r'^\}\}', ''),                              # Standalone }} at start of line -> remove
        (r'\}\}', ''),                               # Any remaining }} -> remove
        (r"'''(.*?)'''", r'\1'),                     # '''bold''' -> 'bold' (preserve original spacing)
        (r"''(.*?)''", r'\1'),                       # ''italic'' -> 'italic' (preserve original spacing)
        (r"'''", ''),                                # Stray ''' -> remove
        (r"''", ''),                                 # Stray '' -> remove
        (r'<ref[^>]*>.*?</ref>', ''),                # <ref>...</ref> -> remove
        (r'<ref[^/>]*/>', ''),                       # <ref .../> -> remove
        (r'<[^>]+>', ''),                            # Other HTML tags -> remove
        # Preserve raw infobox patterns for specialized processing (remove old cleanup patterns)
        # Enhanced readability patterns
        (r'([.!?])\s*([A-Z])', r'\1\n\n\2'),        # Add paragraph breaks after sentences
        (r'^\*\s*(.+)$', r'• \1'),                   # Convert * lists to bullets
        (r'^\#\s*(.+)$', r'1. \1'),                  # Convert # lists to numbers
        (r'\s*-{2,}\s*', '\n\n'),                    # Convert multiple dashes to paragraph breaks
        (r'\s+', ' '),                               # Multiple spaces -> single (keep last)
    ]

    # Field mapping for biographical/infobox data to human-readable labels
    INFOBOX_FIELD_MAPPING = {
        'birth_place': 'Birth Place',
        'birth_date': 'Birth Date',
        'death_place': 'Death Place',
        'death_date': 'Death Date',
        'field': 'Field',
        'fields': 'Fields',
        'work_institutions': 'Institutions',
        'alma_mater': 'Alma Mater',
        'thesis_title': 'Thesis',
        'thesis_url': 'Thesis URL',
        'thesis_year': 'Thesis Year',
        'doctoral_advisor': 'Doctoral Advisor',
        'doctoral_advisors': 'Doctoral Advisors',
        'notable_students': 'Notable Students',
        'prizes': 'Awards',
        'awards': 'Awards',
        'spouse': 'Spouse',
        'children': 'Children',
        'nationality': 'Nationality',
        'citizenship': 'Citizenship',
        'education': 'Education',
        'occupation': 'Occupation',
        'known_for': 'Known For',
        'signature': 'Signature',
        'website': 'Website',
        'influences': 'Influences',
        'influenced': 'Influenced',
        'era': 'Era',
        'region': 'Region',
        'school_tradition': 'School/Tradition',
        'main_interests': 'Main Interests',
        'notable_ideas': 'Notable Ideas'
    }

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize Wikipedia search with multi-database support.

        Args:
            db_path: Path to massive_wikipedia.db. Auto-detects if not provided.
        """
        # Always set base_path first
        self.base_path = Path(__file__).parent

        if db_path is None:
            self.db_path = self.base_path / "massive_wikipedia.db"
        else:
            self.db_path = Path(db_path)

        # Setup domain-specific database mapping
        self.domain_databases = {
            'education': [
                self.base_path / "k12_education_wikipedia.db",
                self.base_path / "college_education_wikipedia.db",
                self.base_path / "mathematics_wikipedia.db"
            ],
            'medical': [
                self.base_path / "medical_wikipedia.db"
            ],
            'agriculture': [
                self.base_path / "agriculture_wikipedia.db"
            ],
            'technical': [
                self.base_path / "technical_education_wikipedia.db",
                self.base_path / "technical_enhanced_wikipedia.db"
            ]
        }

        self.conn = None
        self.fts_available = False
        self.stats = {}

        # Connection pool for performance (thread-safe)
        self._connection_pool = []
        self._max_pool_size = 3
        self._pool_lock = threading.Lock()

        # Database warming system for 7.6GB Wikipedia database
        self._master_connection = None
        self._warming_thread = None
        self._warming_complete = False
        self._warm_cache_ready = False
        self._warming_phase = 0  # 0=not started, 1=indexes, 2=domains, 3=full, 4=complete
        self._warming_progress = 0  # 0-100 percentage

        self._connect()
        self._check_capabilities()
        self._start_database_warming()

    def _get_thread_safe_conn(self):
        """Get a thread-safe database connection from pool."""
        # Try to reuse a connection from pool (thread-safe)
        with self._pool_lock:
            if self._connection_pool:
                return self._connection_pool.pop()

        # Create new optimized connection if pool is empty
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row

        # Adaptive SQLite optimizations based on system resources
        cache_size, mmap_size, description = self._get_adaptive_memory_settings()
        # Use smaller cache for connection pool (half of master connection)
        pool_cache_size = max(cache_size // 2, 5000)  # Minimum 20MB

        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(f"PRAGMA cache_size = {pool_cache_size}")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute(f"PRAGMA mmap_size = {mmap_size // 2}")  # Half mmap for pool connections
        conn.execute("PRAGMA page_size = 4096")  # Standard pages for compatibility

        return conn

    def _return_connection(self, conn):
        """Return a connection to the pool for reuse."""
        with self._pool_lock:
            if len(self._connection_pool) < self._max_pool_size:
                self._connection_pool.append(conn)
                return
        conn.close()

    def _connect(self):
        """Establish database connection."""
        try:
            if self.db_path.exists():
                self.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
                self.conn.row_factory = sqlite3.Row

                # Adaptive SQLite optimizations based on system resources
                try:
                    cache_size, mmap_size, description = self._get_adaptive_memory_settings()
                except (OSError, ValueError, AttributeError):
                    # Fallback if adaptive settings fail
                    cache_size, mmap_size, description = 25000, 134217728, "100MB fallback"

                self.conn.execute("PRAGMA journal_mode = WAL")
                self.conn.execute("PRAGMA synchronous = NORMAL")
                self.conn.execute(f"PRAGMA cache_size = {cache_size}")
                self.conn.execute("PRAGMA temp_store = MEMORY")
                self.conn.execute(f"PRAGMA mmap_size = {mmap_size}")
                self.conn.execute("PRAGMA page_size = 4096")
                print(f"Connected to Wikipedia database: {self.db_path}")
            else:
                print(f"Wikipedia database not found: {self.db_path}")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            self.conn = None

    def _check_capabilities(self):
        """Check what search capabilities are available (LAZY LOADING VERSION)."""
        if not self.conn:
            return

        conn = self._get_thread_safe_conn()
        cursor = conn.cursor()

        # LAZY LOADING: Skip expensive COUNT operations at startup
        # Instead, do fast table existence checks
        try:
            # Quick check if FTS table exists without counting
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles_fts_data'")
            fts_table_exists = cursor.fetchone() is not None
            self.fts_available = fts_table_exists

            # Set placeholder stats that will be computed lazily when needed
            self.stats['total_articles'] = '475k+' # Display estimate without slow COUNT
            self.stats['quality_articles'] = '400k+' # Estimate
            self.stats['indexed_keywords'] = '50k+' # Estimate
            self.stats['lazy_loaded'] = True

            print(f"[WIKI] Wikipedia search ready - 475k+ articles available")

        except Exception as e:
            print(f"[WARN] Database capability check failed: {e}")
            self.fts_available = False
            self.stats['total_articles'] = 'unknown'

    def search(self, query: str, max_results: int = 5,
               min_words: int = None, domain: str = None) -> List[Dict[str, Any]]:
        """
        Search Wikipedia articles for relevant information.

        Args:
            query: Search query (English)
            max_results: Maximum number of results to return
            min_words: Minimum word count for quality filtering (default: 100)
            domain: Optional domain filter (medical, education, agriculture, etc.)

        Returns:
            List of result dicts with: type, title, content, summary, score, word_count, url
        """
        if not self.conn:
            return []

        if min_words is None:
            min_words = self.DEFAULT_MIN_WORDS

        start_time = time.time()

        try:
            # Extract keywords from query
            keyword_data = self._extract_keywords(query)

            if not keyword_data['all']:
                return []

            # Multi-database search: Main database + Domain-specific databases
            all_results = []

            # Search main database (always) with user keywords only
            print(f"Searching main database for: {' '.join(keyword_data['original'])}")
            main_results = self._search_single_database(
                self.db_path, keyword_data, min_words, max_results, use_original_only=True
            )
            for result in main_results:
                result['source'] = 'main'
            all_results.extend(main_results)

            # Search domain databases if domain specified
            if domain and domain in self.domain_databases:
                for domain_db_path in self.domain_databases[domain]:
                    if domain_db_path.exists():
                        print(f"Searching {domain} database: {domain_db_path.name}")
                        domain_results = self._search_single_database(
                            domain_db_path, keyword_data, min_words, max_results // 2, use_original_only=False
                        )
                        for result in domain_results:
                            result['source'] = domain
                            result['score'] += 0.1  # Slight boost for domain relevance
                        all_results.extend(domain_results)

            # Deduplicate by title and merge scores
            unique_results = {}
            for result in all_results:
                title = result['title']
                if title in unique_results:
                    # Merge results, keeping higher score
                    if result['score'] > unique_results[title]['score']:
                        unique_results[title] = result
                else:
                    unique_results[title] = result

            # Convert back to list and sort
            results = list(unique_results.values())
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:max_results]

            # Add summaries and clean content for final results
            for result in results:
                result['content'] = self._clean_wiki_markup(result['content'])  # Clean raw content
                result['summary'] = self._extract_summary(result['content'])
                result['type'] = 'wikipedia'

            # TITLE FALLBACK: Check for exact title matches not found in FTS results
            # This addresses BM25 limitations with biographical articles
            exact_query = query.strip()
            has_exact_match = any(result['title'].lower() == exact_query.lower() for result in results)

            if not has_exact_match and len(keyword_data['original']) == 2:
                # For 2-word queries (likely proper names), try title search as fallback
                print(f"[SEARCH] No exact match in FTS results. Trying title fallback for: '{exact_query}'")
                title_results = self.search_by_title(exact_query, max_results=1, min_words=min_words)

                if title_results:
                    # Found exact title match via title search
                    title_match = title_results[0]
                    if title_match['title'].lower() == exact_query.lower():
                        # Boost the exact title match to highest priority
                        title_match['score'] = 100.0
                        title_match['source'] = 'title_fallback'
                        print(f"[TARGET] Title fallback found: '{title_match['title']}' (Score: {title_match['score']})")

                        # Add to results and re-sort
                        results.insert(0, title_match)
                        results.sort(key=lambda x: x['score'], reverse=True)
                        results = results[:max_results]  # Trim to requested size

            elapsed = time.time() - start_time
            print(f"Search completed in {elapsed:.3f}s - found {len(results)} results")

            return results

        except Exception as e:
            print(f"Search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """Extract searchable keywords from query with domain expansion.

        Returns:
            Dict with 'original' and 'all' keyword lists
        """
        # Lowercase and split
        words = query.lower().split()

        # Remove common stop words (keep some educational terms)
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'although',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'it', 'its', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them'
        }

        # Filter and clean - be more aggressive with filtering
        base_keywords = []
        important_terms = {'malaria', 'medicine', 'disease', 'treatment', 'health', 'medical', 'fever', 'infection',
                          'mathematics', 'euler', 'equation', 'formula', 'theory', 'science', 'physics',
                          'agriculture', 'farming', 'crop', 'plant', 'soil', 'harvest',
                          'technology', 'engineering', 'computer', 'machine', 'system', 'electric'}

        for word in words:
            # Remove punctuation
            clean = re.sub(r'[^\w]', '', word)
            if clean and clean not in stop_words and len(clean) > 2:
                base_keywords.append(clean)

        # CRITICAL FIX: Limit to most important keywords (max 3-4 for performance)
        if len(base_keywords) > 4:
            # Prioritize important domain terms
            prioritized = []
            remaining = []

            for keyword in base_keywords:
                if keyword in important_terms:
                    prioritized.append(keyword)
                else:
                    remaining.append(keyword)

            # Take up to 2 important terms + 2 other terms
            base_keywords = prioritized[:2] + remaining[:2]

            print(f"⚡ Limited keywords for performance: {' '.join(base_keywords)}")

        # Apply domain-specific expansion (restored natural behavior)
        expanded_keywords = self._expand_domain_terms(base_keywords)

        # PERFORMANCE: Cap total expanded keywords
        if len(expanded_keywords) > 6:
            expanded_keywords = expanded_keywords[:6]
            print(f"⚡ Capped expanded keywords: {len(expanded_keywords)} terms")

        # Debug logging
        if len(expanded_keywords) != len(base_keywords):
            print(f"Query expansion: '{' '.join(base_keywords)}' → {len(expanded_keywords)} terms")

        return {
            'original': base_keywords,
            'all': expanded_keywords
        }

    def _expand_domain_terms(self, keywords: List[str]) -> List[str]:
        """Expand keywords with domain-specific terms and synonyms."""

        # Optimized domain expansion mappings (limited to 2-3 most relevant terms each)
        domain_expansions = {
            # Mathematics (Enhanced)
            'math': ['mathematics', 'calculus', 'euler'],
            'mathematics': ['calculus', 'algebra', 'euler'],
            'calc': ['calculus', 'derivative'],
            'calculus': ['derivative', 'integral', 'euler'],
            'algebra': ['equation', 'linear', 'polynomial'],
            'geometry': ['triangle', 'circle', 'euclidean'],
            'stats': ['statistics', 'probability'],
            'statistics': ['probability', 'distribution'],
            'euler': ['mathematics', 'formula', 'constant'],
            'number': ['theory', 'mathematics', 'prime'],
            'formula': ['equation', 'mathematics', 'theorem'],

            # Physics
            'physics': ['mechanics', 'quantum'],
            'mechanics': ['force', 'motion'],
            'quantum': ['particle', 'atom'],
            'electricity': ['current', 'voltage'],

            # Chemistry
            'chemistry': ['molecule', 'atom'],
            'molecule': ['atom', 'bond'],
            'reaction': ['chemical', 'catalyst'],

            # Biology
            'biology': ['cell', 'organism'],
            'cell': ['nucleus', 'dna'],
            'evolution': ['species', 'darwin'],
            'genetics': ['dna', 'gene'],

            # Computer Science
            'programming': ['algorithm', 'code'],
            'algorithm': ['sorting', 'search'],
            'computer': ['processor', 'memory'],

            # Medicine
            'medicine': ['disease', 'treatment'],
            'disease': ['symptom', 'infection'],
            'treatment': ['therapy', 'medication'],

            # Agriculture
            'agriculture': ['farming', 'crop'],
            'farming': ['crop', 'livestock'],
            'crop': ['plant', 'harvest']
        }

        # Optimized synonym mappings (limited to 2 most relevant synonyms)
        synonyms = {
            'learn': ['study', 'education'],
            'teach': ['education', 'instruction'],
            'explain': ['describe', 'clarify'],
            'understand': ['comprehend', 'knowledge'],
            'help': ['assist', 'aid'],
            'show': ['demonstrate', 'display']
        }

        expanded = set(keywords)  # Start with original keywords

        # Add domain-specific expansions
        for keyword in keywords:
            if keyword in domain_expansions:
                expanded.update(domain_expansions[keyword])

            # Add synonyms
            if keyword in synonyms:
                expanded.update(synonyms[keyword])

        # Natural expansion - no artificial limits (restored original behavior)
        result = list(expanded)

        return result

    def _search_by_keywords(self, keyword_data: Dict[str, List[str]], min_words: int,
                           limit: int, domain: str = None) -> List[Dict[str, Any]]:
        """Enhanced search with BM25 scoring and connection pooling."""
        import time

        # PROFILING: Track timing for each step
        step_times = {}
        total_start = time.time()

        step_start = time.time()
        conn = self._get_thread_safe_conn()
        cursor = conn.cursor()
        step_times['connection'] = time.time() - step_start

        # FIXED: Prioritize original user keywords over expanded terms
        original_keywords = keyword_data['original']
        all_keywords = keyword_data['all']
        expanded_only = [kw for kw in all_keywords if kw not in original_keywords]

        if len(all_keywords) <= 2:
            # Small queries: use simple OR with all terms
            search_term = ' OR '.join(all_keywords)
        elif len(all_keywords) <= 4:
            # Medium queries: prioritize original terms with AND, expanded with OR
            if original_keywords:
                primary = ' AND '.join(original_keywords)
                secondary = ' OR '.join(expanded_only[:2]) if expanded_only else ''
                search_term = f"({primary})" + (f" OR ({secondary})" if secondary else '')
            else:
                search_term = ' OR '.join(all_keywords)
        else:
            # Large queries: use original + top expanded terms only
            priority_terms = original_keywords + expanded_only[:1]  # Original + 1 expansion
            search_term = ' AND '.join(priority_terms)
            print(f"⚡ Large query optimization: prioritizing original terms {original_keywords}")

        # Build domain filtering if specified
        domain_filter = ""
        params = [search_term, min_words]

        # Domain filtering is now handled by multi-database architecture
        # No need for restrictive AND logic here

        # OPTIMIZED: Join to slim articles_meta instead of fat articles table
        query = '''
            SELECT
                m.id,
                m.title,
                m.summary as content_preview,
                m.word_count,
                m.url,
                bm25(articles_fts) as bm25_score
            FROM articles_meta m
            JOIN articles_fts fts ON m.id = fts.rowid
            WHERE articles_fts MATCH ?
            AND m.word_count >= ?
            ORDER BY bm25(articles_fts)
            LIMIT ?
        '''

        params.append(limit)  # Add limit parameter

        # PROFILING: Time SQL execution
        step_start = time.time()
        cursor.execute(query, params)
        step_times['sql_execute'] = time.time() - step_start

        # PROFILING: Time result processing
        step_start = time.time()

        results = []
        for row in cursor.fetchall():
            # Simple fast scoring (complex calculations removed for performance)
            base_score = abs(row['bm25_score'])  # BM25 score (negative, so abs)

            # Simple title boost
            title_score = 0
            title_lower = row['title'].lower()
            for keyword in all_keywords:
                if keyword.lower() in title_lower:
                    title_score += 25

            # Fast final score: 70% BM25 + 30% title boost
            final_score = (base_score * 0.7) + (title_score * 0.3)

            results.append({
                'id': row['id'],
                'title': row['title'],
                'content': row['content_preview'],  # Use optimized 500-char preview
                'word_count': row['word_count'],
                'url': row['url'] or '',
                'score': final_score
            })

        step_times['result_processing'] = time.time() - step_start

        self._return_connection(conn)  # Return connection to pool instead of closing

        # Performance monitoring (optional - comment out for production)
        # step_times['total'] = time.time() - total_start
        # print(f"[TIME] Search profiling breakdown:")
        # for step, duration in step_times.items():
        #     print(f"   {step}: {duration*1000:.1f}ms")

        # Results already ordered by BM25, just return them
        return results

    def _search_single_database(self, db_path: Path, keyword_data: Dict[str, List[str]],
                               min_words: int, limit: int, use_original_only: bool = False) -> List[Dict[str, Any]]:
        """Search a single database with enhanced keyword handling."""
        try:
            # Choose keywords based on strategy
            if use_original_only:
                keywords = keyword_data['original']
            else:
                keywords = keyword_data['all']

            if not keywords:
                return []

            # Connect to specific database with optimizations
            conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row

            try:
                # Research-backed SQLite performance optimizations
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA cache_size = 10000")
                conn.execute("PRAGMA temp_store = MEMORY")
                conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory map

                cursor = conn.cursor()

                # Build FTS query with intelligent AND/OR logic for proper names
                # Detect proper names (exactly 2 keywords suggests a person's name)
                if len(keywords) == 2:
                    # For 2-word queries like "Julius Caesar", use AND for precision
                    search_term = ' AND '.join(keywords)
                    print(f"[SEARCH] 2-word query detected: using AND logic for '{' '.join(keywords)}'")
                else:
                    # For general queries, use OR for better recall
                    search_term = ' OR '.join(keywords)

                # Detect schema: main DB uses articles_meta, domain DBs use articles
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles_meta'")
                has_meta = cursor.fetchone() is not None

                if has_meta:
                    query = '''
                        SELECT
                            m.id,
                            m.title,
                            m.summary as content_preview,
                            m.word_count,
                            m.url,
                            bm25(articles_fts) as bm25_score
                        FROM articles_meta m
                        JOIN articles_fts fts ON m.id = fts.rowid
                        WHERE articles_fts MATCH ?
                        AND m.word_count >= ?
                        ORDER BY bm25(articles_fts)
                        LIMIT ?
                    '''
                else:
                    query = '''
                        SELECT
                            a.id,
                            a.title,
                            substr(a.content, 1, 500) as content_preview,
                            a.word_count,
                            a.url,
                            bm25(articles_fts) as bm25_score
                        FROM articles a
                        JOIN articles_fts fts ON a.id = fts.rowid
                        WHERE articles_fts MATCH ?
                        AND a.word_count >= ?
                        ORDER BY bm25(articles_fts)
                        LIMIT ?
                    '''

                cursor.execute(query, [search_term, min_words, limit])
                rows = cursor.fetchall()

                results = []
                original_query = ' '.join(keywords)
                for row in rows:
                    # Calculate title bonus for exact matches (like existing system)
                    base_score = abs(row['bm25_score']) if row['bm25_score'] else 0.0
                    title_score = 0

                    # Big bonus for exact title match
                    if row['title'].lower() == original_query:
                        title_score += 50
                        print(f"[SEARCH] Exact title match found: '{row['title']}'")

                    # Smaller bonus if title contains all query words
                    title_lower = row['title'].lower()
                    if all(keyword in title_lower for keyword in keywords):
                        title_score += 25

                    # Simple balanced scoring: 70% BM25 + 30% title boost
                    final_score = (base_score * 0.7) + (title_score * 0.3)

                    results.append({
                        'id': row['id'],
                        'title': row['title'],
                        'content': row['content_preview'],
                        'word_count': row['word_count'],
                        'url': row['url'],
                        'score': final_score
                    })

                # Sort by final score (highest first)
                results.sort(key=lambda x: x['score'], reverse=True)

                return results
            finally:
                conn.close()

        except Exception as e:
            print(f"Error searching database {db_path}: {e}")
            return []

    def _filter_by_domain(self, results: List[Dict], domain: str) -> List[Dict]:
        """Filter results by domain-specific keywords."""
        domain_keywords = {
            'medical': [
                'disease', 'treatment', 'symptom', 'medicine', 'health', 'medical',
                'patient', 'doctor', 'hospital', 'drug', 'therapy', 'diagnosis',
                'infection', 'virus', 'bacteria', 'fever', 'pain', 'surgery',
                'vaccine', 'illness', 'condition', 'syndrome', 'disorder'
            ],
            'education': [
                'school', 'education', 'university', 'student', 'learning', 'teach',
                'science', 'mathematics', 'history', 'geography', 'language', 'study',
                'research', 'theory', 'concept', 'knowledge', 'curriculum', 'academic'
            ],
            'agriculture': [
                'farm', 'crop', 'plant', 'soil', 'harvest', 'agriculture', 'seed',
                'irrigation', 'livestock', 'cattle', 'wheat', 'maize', 'drought',
                'fertilizer', 'cultivation', 'yield', 'growing', 'farming'
            ],
            'technical': [
                'technology', 'engineering', 'machine', 'system', 'computer', 'electric',
                'power', 'energy', 'construction', 'building', 'design', 'mechanical',
                'software', 'hardware', 'equipment', 'tool', 'device', 'technical'
            ]
        }

        if domain not in domain_keywords:
            return results

        keywords = set(domain_keywords[domain])
        filtered = []

        for result in results:
            content_lower = result['content'].lower()
            title_lower = result['title'].lower()

            # Check if article matches domain
            domain_score = 0
            for kw in keywords:
                if kw in title_lower:
                    domain_score += 3
                if kw in content_lower:
                    domain_score += 1

            if domain_score > 0:
                result['score'] += domain_score
                filtered.append(result)

        return filtered

    def _extract_summary(self, content: str) -> str:
        """Extract and clean first paragraph for context injection."""
        # Get content before first section header
        # Handle both "==Section==" and "\n==Section==" patterns
        # Also handle cases where == appears without newline (e.g., "text. ==Section==")
        parts = re.split(r'(?:\n\s*|\s+)==[^=]', content, maxsplit=1)
        first_section = parts[0].strip()

        # Clean wiki markup
        cleaned = self._clean_wiki_markup(first_section)

        # Truncate to max words
        words = cleaned.split()
        if len(words) > self.MAX_SUMMARY_WORDS:
            cleaned = ' '.join(words[:self.MAX_SUMMARY_WORDS]) + '...'

        return cleaned.strip()

    def _clean_wiki_markup(self, text: str) -> str:
        """Remove wiki markup from text and format for readability."""
        result = text

        # FIRST: Process infobox/biographical data BEFORE other wiki cleanup
        # This preserves the || structure needed for infobox detection
        result = self._process_infobox_data(result)

        for pattern, replacement in self.WIKI_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.DOTALL | re.MULTILINE)

        # Enhanced post-processing for better readability
        result = result.strip()

        # AGGRESSIVE INFOBOX CLEANUP - Remove raw infobox data patterns
        # Pattern: sequences of "field_name = value |" or "| field_name = value"
        # These are Wikipedia infobox fields that weren't properly processed
        result = re.sub(r'\b\w+_\w+\s*=\s*[^|]*\|', '', result)  # field_name = value | -> remove
        result = re.sub(r'\|\s*\w+_\w+\s*=\s*[^|]*', '', result)  # | field_name = value -> remove
        result = re.sub(r'\b(?:birth|death|spouse|children|parents|relatives|nationality|office|known_for|occupation|alma_mater|field|fields|awards|prizes)\w*\s*=\s*[^|.!?\n]*\|?', '', result, flags=re.IGNORECASE)  # Common infobox fields

        # Remove any remaining template markers
        result = re.sub(r'\}\}', '', result)  # Stray }} markers
        result = re.sub(r'\{\{', '', result)  # Stray {{ markers

        # Remove empty parentheses and brackets
        result = re.sub(r'\(\s*\)', '', result)
        result = re.sub(r'\[\s*\]', '', result)

        # Clean up excessive punctuation and formatting artifacts
        result = re.sub(r'[|]{2,}', '', result)  # Remove || artifacts
        result = re.sub(r'\s*\|\s*', ' ', result)  # Convert remaining single pipes to spaces
        result = re.sub(r'[-]{3,}', '\n\n', result)  # Convert long dashes to breaks
        result = re.sub(r'={2,}[^=]*={2,}', '', result)  # Remove wiki section headers

        # Remove remaining image-related content that might have slipped through
        result = re.sub(r'\b(?:thumb|frame|frameless|upright)\s*\|[^.!?]*', '', result)
        result = re.sub(r'\b(?:left|right|center|none)\s*\|', '', result)
        result = re.sub(r'caption[^.!?]*[.!?]', '', result, flags=re.IGNORECASE)
        result = re.sub(r'\b\d+\s*×\s*\d+\s*(?:px|pixels?)\b', '', result)  # Remove dimensions
        result = re.sub(r'\bfile\s*:\s*[^.!?]*', '', result, flags=re.IGNORECASE)  # Remove file: references

        # Additional URL and markdown cleanup (catch anything that slipped through)
        result = re.sub(r'!\[[^\]]*\]', '', result)  # Stray ![alt] without URL
        result = re.sub(r'\\\[(?:File|Image|Media)[^\]]*\\\]', '', result)  # Escaped file/image bracket refs only
        result = re.sub(r'\([Hh]ttps?://[^)]*\)', '', result)  # URLs in parentheses
        result = re.sub(r'[Hh]ttps?://\S+', '', result)  # Any remaining URLs
        result = re.sub(r'\bWikipedia\.\s*[Oo]rg\S*', '', result)  # Wikipedia.Org references
        result = re.sub(r'\b[A-Z][a-z]+\s+[Cc]rater\s+[A-Z][a-z0-9-]+\.?\s*[Jj]pg\S*', '', result)  # Crater image refs
        result = re.sub(r'[Ww]idth:\s*\*?\*?\s*\d+\)?', '', result)  # Width params
        result = re.sub(r'\\{3,}', '', result)  # Remove runs of 3+ backslashes (wiki artifacts) but keep singles
        result = re.sub(r'\*\*+', '', result)  # Remove stray ** markdown
        result = re.sub(r'^\*\s*$', '', result, flags=re.MULTILINE)  # Remove lone asterisks on lines
        result = re.sub(r'##\s*External\s+links.*$', '', result, flags=re.IGNORECASE)  # Remove external links section header

        # Remove repeated phrases (deduplicate consecutive identical phrases)
        result = re.sub(r'(\b\w+(?:\s+\w+){2,5}\b)(?:\s*[-–]\s*\1){2,}', r'\1', result)  # Repeated 3-6 word phrases

        # Clean up coordinate/measurement data dumps
        result = re.sub(r'(?:[A-Z]\s+\d+\.?\d*°\s*[NS]\s+\d+\.?\d*°\s*[EW]\s+\d+\s*km\s*)+', ' ', result)  # Lat/long/km sequences

        # Improve spacing around punctuation
        result = re.sub(r'([.!?])\s*\n+\s*', r'\1\n\n', result)  # Normalize paragraph breaks
        result = re.sub(r'\n{3,}', '\n\n', result)  # Limit consecutive line breaks
        result = re.sub(r'[ \t]+', ' ', result)  # Clean up spaces but preserve newlines

        # Ensure proper sentence spacing
        result = re.sub(r'([.!?])([A-Z])', r'\1 \2', result)

        # Fix concatenated words from italic/bold markup (e.g., "PhysicalDissertation" -> "Physical Dissertation")
        # This handles edge cases where lowercase letter directly precedes uppercase letter
        result = re.sub(r'([a-z])([A-Z])', r'\1 \2', result)

        # Final cleanup
        result = result.strip()

        return result

    def _process_infobox_data(self, text: str) -> str:
        """
        Process Wikipedia infobox/biographical data into clean, readable format.

        Now handles multiple patterns:
        1. Title|| field = value | field = value (double pipe format)
        2. value | field = value | field = value (single pipe format at start)
        3. Raw infobox data scattered in text

        Args:
            text: Raw Wikipedia text containing infobox data

        Returns:
            Cleaned text with infobox data removed or formatted
        """
        if not text:
            return text

        result = text

        # PATTERN 1: Handle || format (Title|| field = value)
        if re.search(r'\|\|.*?\|.*?=', text):
            infobox_pattern = r'([^\|\n]*)\|\|([^|\n]*\|[^|]*=[^|\n]*(?:\|[^|]*=[^|\n]*)*)'
            matches = re.findall(infobox_pattern, result)

            for title_part, fields_part in matches:
                title = title_part.strip()
                formatted_bio = self._parse_infobox_fields_only(fields_part)

                if formatted_bio:
                    original_line = f"{title_part}||{fields_part}"
                    if title:
                        replacement = f"{title}\n\n**Biographical Information:**\n{formatted_bio}\n"
                    else:
                        replacement = f"**Biographical Information:**\n{formatted_bio}\n"
                    result = result.replace(original_line, replacement)
                else:
                    original_line = f"{title_part}||{fields_part}"
                    result = result.replace(original_line, title_part.strip())

        # PATTERN 2: Strip infobox prefix from lines containing field=value patterns
        # For single-line articles, infobox + article text are on the same line,
        # so we must keep content AFTER the closing }}
        cleaned_lines = []
        for line in result.split('\n'):
            stripped = line.strip()
            if '}}' in stripped and '=' in stripped and '|' in stripped:
                if stripped.count('|') >= 2 and stripped.count('=') >= 2:
                    # Find the last }} — everything after it is article content
                    brace_idx = stripped.rfind('}}')
                    after_braces = stripped[brace_idx + 2:].strip()
                    if after_braces:
                        cleaned_lines.append(after_braces)
                    continue
            cleaned_lines.append(line)
        result = '\n'.join(cleaned_lines)

        # PATTERN 3: Clean up any remaining sequences of "field = value |"
        # Bounded quantifier {0,200} prevents backtracking on long content
        field_sequence_pattern = r'(?:\w+\s*=\s*[^|\n]{0,200}\s*\|){3,}'
        result = re.sub(field_sequence_pattern, '', result)

        return result

    def _parse_infobox_fields_only(self, fields_text: str) -> str:
        """
        Parse just the fields part of an infobox into formatted biographical information.

        Args:
            fields_text: String like " birth_place = Basel | death_date = 1783 | field = Math"

        Returns:
            Formatted biographical information lines
        """
        if not fields_text:
            return ""

        formatted_fields = []

        # Split by | and process each field=value pair
        fields = fields_text.split('|')

        for field in fields:
            field = field.strip()
            if '=' in field:
                try:
                    key, value = field.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    if key and value:
                        formatted_field = self._format_infobox_field(key, value)
                        if formatted_field:
                            formatted_fields.append(formatted_field)
                except (ValueError, TypeError):
                    continue

        return '\n'.join(formatted_fields) if formatted_fields else ""

    def _parse_infobox_fields(self, infobox_block: str) -> str:
        """
        Parse an infobox block into formatted fields.

        Args:
            infobox_block: Raw infobox string like "|| birth_place = Basel | death_date = 1783"

        Returns:
            Formatted string with biographical information or empty string if parsing fails
        """
        if not infobox_block:
            return ""

        # Remove leading || and split on |
        clean_block = infobox_block.lstrip('|').strip()
        fields = []

        # Split on | but be careful of nested content
        field_parts = re.split(r'\s*\|\s*', clean_block)

        formatted_fields = []
        for part in field_parts:
            if '=' in part:
                try:
                    field, value = part.split('=', 1)
                    field = field.strip()
                    value = value.strip()

                    if field and value:
                        formatted_field = self._format_infobox_field(field, value)
                        if formatted_field:
                            formatted_fields.append(formatted_field)
                except ValueError:
                    continue

        if formatted_fields:
            return '\n\n**Biographical Information:**\n' + '\n'.join(formatted_fields) + '\n\n'

        return ""

    def _format_infobox_field(self, field: str, value: str) -> str:
        """
        Format a single infobox field into readable form.

        Args:
            field: Field name (e.g., 'birth_place')
            value: Field value (e.g., 'Basel, Switzerland')

        Returns:
            Formatted string like "**Birth Place:** Basel, Switzerland"
        """
        if not field or not value:
            return ""

        # Clean up the field name
        field = field.lower().strip()

        # Skip technical or less useful fields
        skip_fields = {
            'image', 'caption', 'signature', 'website', 'thesis_url',
            'homepage', 'url', 'portal', 'commons', 'wikiquote'
        }
        if field in skip_fields:
            return ""

        # Get human-readable label
        label = self.INFOBOX_FIELD_MAPPING.get(field, field.replace('_', ' ').title())

        # Clean up the value
        value = value.strip()

        # Handle bracketed content (like [OS: 7 September 1783])
        if value.startswith('[') and value.endswith(']'):
            value = value[1:-1]  # Remove brackets

        # Handle OS (Old Style) dates
        if 'OS:' in value:
            value = value.replace('OS:', '(Old Style)').strip()

        # Clean up multiple spaces and line breaks
        value = re.sub(r'\s+', ' ', value).strip()

        # Limit value length for readability
        if len(value) > 100:
            value = value[:97] + "..."

        return f"**{label}:** {value}"


    def search_by_title(self, title_query: str, max_results: int = 5,
                        min_words: int = None) -> List[Dict[str, Any]]:
        """
        Search articles by title match.

        Args:
            title_query: Title to search for
            max_results: Maximum results
            min_words: Minimum word count

        Returns:
            List of matching articles
        """
        if not self.conn:
            return []

        if min_words is None:
            min_words = self.DEFAULT_MIN_WORDS

        conn = self._get_thread_safe_conn()
        cursor = conn.cursor()

        # Use FTS MATCH on title column instead of LIKE (avoids full table scan)
        # Escape FTS special characters in query
        safe_query = re.sub(r'[^\w\s]', '', title_query).strip()
        if not safe_query:
            self._return_connection(conn)
            return []

        query = '''
            SELECT m.id, m.title, m.summary, m.word_count, m.url
            FROM articles_meta m
            JOIN articles_fts fts ON m.id = fts.rowid
            WHERE articles_fts MATCH ?
            AND m.word_count >= ?
            ORDER BY
                CASE WHEN m.title = ? THEN 0 ELSE 1 END,
                bm25(articles_fts)
            LIMIT ?
        '''

        fts_title_query = 'title:' + ' '.join(safe_query.split())
        cursor.execute(query, (fts_title_query, min_words, title_query, max_results))

        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'title': row['title'],
                'content': row['summary'] or '',
                'summary': row['summary'] or '',
                'word_count': row['word_count'],
                'url': row['url'] or '',
                'score': 100 if row['title'].lower() == title_query.lower() else 50,
                'type': 'wikipedia'
            })

        self._return_connection(conn)
        return results

    def get_article(self, article_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific article by ID.

        Args:
            article_id: Article ID

        Returns:
            Article dict or None
        """
        if not self.conn:
            return None

        conn = self._get_thread_safe_conn()
        cursor = conn.cursor()

        # Fetch metadata + full content from split tables
        cursor.execute('''
            SELECT m.id, m.title, c.content, m.word_count, m.url, m.summary
            FROM articles_meta m
            JOIN articles_content c ON m.id = c.article_id
            WHERE m.id = ?
        ''', (article_id,))

        row = cursor.fetchone()
        if row:
            result = {
                'id': row[0],
                'title': row[1],
                'content': self._clean_wiki_markup(row[2]),
                'summary': row[5] or self._extract_summary(row[2]),
                'word_count': row[3],
                'url': row[4] or '',
                'type': 'wikipedia'
            }
            self._return_connection(conn)
            return result

        self._return_connection(conn)
        return None

    def get_random_quality_articles(self, count: int = 5,
                                    min_words: int = None) -> List[Dict[str, Any]]:
        """Get random quality articles for testing or exploration."""
        if not self.conn:
            return []

        if min_words is None:
            min_words = self.DEFAULT_MIN_WORDS

        conn = self._get_thread_safe_conn()
        cursor = conn.cursor()

        # Random ID probe instead of ORDER BY RANDOM() (avoids sorting 475k rows)
        cursor.execute('''
            SELECT m.id, m.title, m.summary, m.word_count, m.url
            FROM articles_meta m
            WHERE m.id >= (ABS(RANDOM()) % (SELECT MAX(id) FROM articles_meta))
            AND m.word_count >= ?
            LIMIT ?
        ''', (min_words, count))

        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'title': row['title'],
                'content': row['summary'] or '',
                'summary': row['summary'] or '',
                'word_count': row['word_count'],
                'url': row['url'] or '',
                'type': 'wikipedia'
            })

        self._return_connection(conn)
        return results

    def populate_fts_index(self, batch_size: int = 1000) -> bool:
        """
        Populate FTS5 index for faster full-text search.
        This is a one-time operation that can take several minutes.

        Returns:
            True if successful
        """
        if not self.conn:
            return False

        print("Populating FTS5 index... This may take a few minutes.")
        start_time = time.time()

        try:
            conn = self._get_thread_safe_conn()
            cursor = conn.cursor()

            # External content mode: use rebuild command
            cursor.execute("INSERT INTO articles_fts(articles_fts) VALUES('rebuild')")
            conn.commit()

            elapsed = time.time() - start_time
            print(f"FTS5 index populated in {elapsed:.1f}s")

            self.fts_available = True
            self._return_connection(conn)
            return True

        except Exception as e:
            print(f"Error populating FTS index: {e}")
            self.conn.rollback()
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.conn:
            return {}

        conn = self._get_thread_safe_conn()
        cursor = conn.cursor()

        # Word count distribution
        cursor.execute('''
            SELECT
                CASE
                    WHEN word_count < 50 THEN 'tiny (<50)'
                    WHEN word_count < 100 THEN 'short (50-99)'
                    WHEN word_count < 200 THEN 'medium (100-199)'
                    WHEN word_count < 500 THEN 'good (200-499)'
                    ELSE 'excellent (500+)'
                END as category,
                COUNT(*) as count
            FROM articles_meta
            GROUP BY category
            ORDER BY category
        ''')

        distribution = {row[0]: row[1] for row in cursor.fetchall()}

        self._return_connection(conn)
        return {
            **self.stats,
            'word_count_distribution': distribution,
            'fts_available': self.fts_available
        }

    def _start_database_warming(self):
        """Start database warming system for 7.6GB Wikipedia database."""
        if not self.conn:
            print("[WARN] Cannot start database warming - no connection")
            return

        print("[FIRE] Starting Wikipedia database warming system...")

        import threading
        def warming_worker():
            try:
                # Create master connection that stays warm
                self._warming_phase = 1
                self._warming_progress = 5
                self._create_master_connection()
                self._warming_progress = 15

                # Phase 1: Critical indexes
                self._warm_critical_indexes()

                # Phase 2: Medical/agricultural terms
                self._warming_phase = 2
                self._warm_domain_specific_terms()

                # Phase 3: Full database optimization
                self._warming_phase = 3
                self._warm_full_database()

                self._warming_phase = 4
                self._warming_progress = 100
                self._warming_complete = True
                print("[OK] Wikipedia database loaded and ready")

            except Exception as e:
                print(f"[ERROR] Database warming failed: {e}")
                self._warming_phase = -1  # Error state

        self._warming_thread = threading.Thread(target=warming_worker, daemon=True)
        self._warming_thread.start()
        print("[LAUNCH] Database warming thread started")

    def _create_master_connection(self):
        """Create master connection with adaptive settings based on system memory."""
        try:
            print("[FIRE] Creating master warm connection...")
            self._master_connection = sqlite3.connect(str(self.db_path), timeout=60.0)
            self._master_connection.row_factory = sqlite3.Row

            # ADAPTIVE memory configuration based on system resources
            cache_size, mmap_size, description = self._get_adaptive_memory_settings()

            optimizations = [
                ("PRAGMA journal_mode = WAL", "WAL mode for concurrent reads"),
                ("PRAGMA synchronous = NORMAL", "Balanced performance/safety"),
                (f"PRAGMA cache_size = {cache_size}", f"{description} cache"),
                ("PRAGMA page_size = 4096", "Standard pages for compatibility"),
                ("PRAGMA temp_store = MEMORY", "Memory temp storage"),
                (f"PRAGMA mmap_size = {mmap_size}", f"{mmap_size//1024//1024}MB memory mapping"),
                ("PRAGMA optimize", "Query optimizer tuning")
            ]

            for pragma, desc in optimizations:
                self._master_connection.execute(pragma)
                print(f"   ✓ {desc}")

            self._master_connection.commit()
            print("[OK] Master connection optimized")

        except Exception as e:
            print(f"[ERROR] Master connection failed: {e}")

    def _get_adaptive_memory_settings(self):
        """Get memory settings based on available system RAM."""
        try:
            import psutil
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            available_ram_gb = psutil.virtual_memory().available / (1024**3)

            print(f"   [STATS] System RAM: {total_ram_gb:.1f}GB total, {available_ram_gb:.1f}GB available")

            if total_ram_gb <= 4.5:  # 4GB systems (accounting for reported vs actual)
                # CONSERVATIVE: ~100MB cache, 128MB mmap for field laptops
                cache_size = 25000  # ~100MB
                mmap_size = 134217728  # 128MB
                description = "100MB field-optimized"
                print(f"   [PC] Field laptop mode: Conservative memory usage")

            elif total_ram_gb <= 8:  # 8GB systems
                # MODERATE: ~200MB cache, 256MB mmap
                cache_size = 50000  # ~200MB
                mmap_size = 268435456  # 256MB
                description = "200MB balanced"
                print(f"   [HOME] Home system mode: Balanced memory usage")

            else:  # 16GB+ systems
                # PERFORMANCE: ~400MB cache, 512MB mmap
                cache_size = 100000  # ~400MB
                mmap_size = 536870912  # 512MB
                description = "400MB high-performance"
                print(f"   [LAUNCH] High-end mode: Performance-optimized memory usage")

            return cache_size, mmap_size, description

        except ImportError:
            print("   [WARN] psutil not available, using conservative settings")
            return 25000, 134217728, "100MB conservative"  # Safe defaults
        except Exception as e:
            print(f"   [WARN] Memory detection failed: {e}, using conservative settings")
            return 25000, 134217728, "100MB fallback"

    def _warm_critical_indexes(self):
        """Warm critical database indexes for immediate search performance."""
        if not self._master_connection:
            return

        print("[FIRE] Phase 1: Warming critical indexes...")
        critical_queries = [
            ("SELECT COUNT(*) FROM articles_meta", "Article count"),
            ("SELECT title FROM articles_meta LIMIT 100", "Metadata index"),
            ("SELECT id, title, word_count FROM articles_meta WHERE word_count > 50 LIMIT 50", "Quality filter index"),
            ("SELECT rowid FROM articles_fts WHERE articles_fts MATCH 'test' LIMIT 5", "FTS5 search index"),
        ]

        progress_start = 15
        progress_end = 35
        step = (progress_end - progress_start) / len(critical_queries)

        for i, (query, label) in enumerate(critical_queries):
            try:
                cursor = self._master_connection.cursor()
                start = time.time()
                cursor.execute(query)
                results = cursor.fetchall()
                elapsed = time.time() - start
                self._warming_progress = int(progress_start + step * (i + 1))
                print(f"   ✓ {label} ({elapsed:.3f}s)")
            except Exception as e:
                self._warming_progress = int(progress_start + step * (i + 1))
                print(f"   [ERROR] {label} failed: {e}")

        self._warm_cache_ready = True
        print("[OK] Phase 1 complete - basic searches now fast")

    def _warm_domain_specific_terms(self):
        """Warm common medical and agricultural search terms."""
        if not self._master_connection:
            return

        print("[FIRE] Phase 2: Warming domain-specific terms...")

        # Medical terms (high priority for NATNA)
        medical_terms = ["headache", "fever", "malaria", "dengue", "tuberculosis",
                        "pneumonia", "diabetes", "hypertension", "infection", "treatment"]

        # Agricultural terms
        agricultural_terms = ["agriculture", "farming", "crop", "irrigation", "fertilizer",
                             "pesticide", "harvest", "soil", "seeds", "livestock"]

        all_terms = medical_terms + agricultural_terms
        progress_start = 35
        progress_end = 75
        step = (progress_end - progress_start) / len(all_terms)

        for i, term in enumerate(all_terms):
            try:
                cursor = self._master_connection.cursor()
                cursor.execute("""
                    SELECT m.title, m.summary, m.word_count
                    FROM articles_meta m
                    JOIN articles_fts fts ON m.id = fts.rowid
                    WHERE articles_fts MATCH ?
                    LIMIT 10
                """, (term,))
                results = cursor.fetchall()
            except Exception as e:
                print(f"   [ERROR] Term '{term}' warming failed: {e}")

            self._warming_progress = int(progress_start + step * (i + 1))
            if i % 5 == 0:
                print(f"   ✓ Warmed {i+1}/{len(all_terms)} domain terms")

        print("[OK] Phase 2 complete - medical/agricultural searches optimized")

    def _warm_full_database(self):
        """Final database warming - comprehensive optimization."""
        if not self._master_connection:
            return

        print("[FIRE] Phase 3: Full database optimization...")

        try:
            cursor = self._master_connection.cursor()

            cursor.execute("SELECT COUNT(*) FROM articles_meta WHERE word_count > 25")
            count = cursor.fetchone()[0]
            print(f"   [STATS] {count:,} quality articles available")
            self._warming_progress = 80

            cursor.execute("""
                SELECT title, summary, word_count
                FROM articles_meta
                WHERE word_count BETWEEN 50 AND 1000
                ORDER BY word_count DESC
                LIMIT 1000
            """)
            results = cursor.fetchall()
            print(f"   ✓ Pre-loaded {len(results)} high-quality article metadata")
            self._warming_progress = 90

            self._master_connection.execute("PRAGMA optimize")
            print("   ✓ Database structure optimized")
            self._warming_progress = 95

        except Exception as e:
            print(f"[ERROR] Full database warming error: {e}")

        print("[OK] Phase 3 complete - database fully loaded")

    def is_database_warmed(self):
        """Check if database warming is complete."""
        return self._warm_cache_ready

    def get_warming_status(self):
        """Get current warming status with progress details."""
        phase_names = {
            0: "Initializing",
            1: "Warming indexes",
            2: "Warming domain terms",
            3: "Full database optimization",
            4: "Complete",
            -1: "Error"
        }

        return {
            "phase": self._warming_phase,
            "phase_name": phase_names.get(self._warming_phase, "Unknown"),
            "progress": self._warming_progress,
            "complete": self._warming_complete,
            "cache_ready": self._warm_cache_ready
        }

    def close(self):
        """Close database connections including master connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

        if self._master_connection:
            try:
                self._master_connection.close()
            except Exception:
                pass  # Master connection created in warming thread
            self._master_connection = None

        # Close pooled connections
        for conn in self._connection_pool:
            try:
                conn.close()
            except Exception:
                pass
        self._connection_pool.clear()

        print("Database connections closed")


def format_for_context(results: List[Dict], max_chars: int = 2000) -> str:
    """
    Format search results for injection into DeepSeek context.

    Args:
        results: List of search results
        max_chars: Maximum character count for context

    Returns:
        Formatted context string
    """
    if not results:
        return ""

    context_parts = ["Relevant Wikipedia information:"]
    char_count = len(context_parts[0])

    for result in results:
        entry = f"\n\n{result['title']}: {result['summary']}"

        if char_count + len(entry) > max_chars:
            break

        context_parts.append(entry)
        char_count += len(entry)

    return ''.join(context_parts)


def main():
    """Test the Wikipedia search system."""
    print("=" * 60)
    print("Wikipedia Knowledge Search - Test Suite")
    print("=" * 60)

    searcher = WikipediaKnowledgeSearch()

    # Print statistics
    print("\n--- Database Statistics ---")
    stats = searcher.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


    # Test queries across domains
    test_queries = [
        # Mathematical (testing enhanced relevance)
        ("math calculus", "education"),
        ("algebra equations", "education"),
        ("geometry theorems", "education"),

        # Medical
        ("fever treatment", "medical"),
        ("malaria symptoms", "medical"),
        ("heart disease", "medical"),

        # Educational
        ("mitochondria cell", "education"),
        ("world war history", "education"),
        ("mathematics algebra", "education"),

        # Agricultural
        ("crop irrigation", "agriculture"),
        ("soil fertility", "agriculture"),

        # General
        ("solar system planets", None),
        ("climate change", None),
    ]

    print("\n--- Search Tests ---")
    for query, domain in test_queries:
        print(f"\nQuery: '{query}'" + (f" [domain: {domain}]" if domain else ""))
        print("-" * 40)

        start = time.time()
        results = searcher.search(query, max_results=3, domain=domain)
        elapsed = time.time() - start

        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (score: {result['score']}, words: {result['word_count']})")
                print(f"   Summary: {result['summary'][:100]}...")
        else:
            print("   No results found")

        print(f"   Time: {elapsed:.3f}s")

    # Test context formatting
    print("\n--- Context Formatting Test ---")
    results = searcher.search("dengue fever symptoms treatment", max_results=3)
    context = format_for_context(results, max_chars=1500)
    print(f"Context ({len(context)} chars):")
    print(context[:500] + "..." if len(context) > 500 else context)

    # Test summary extraction quality
    print("\n--- Summary Extraction Quality ---")
    test_articles = searcher.search_by_title("Dengue fever", max_results=1)
    if test_articles:
        article = test_articles[0]
        print(f"Article: {article['title']}")
        print(f"Word count: {article['word_count']}")
        print(f"Summary ({len(article['summary'].split())} words):")
        print(article['summary'])

    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)

    searcher.close()


if __name__ == "__main__":
    main()
