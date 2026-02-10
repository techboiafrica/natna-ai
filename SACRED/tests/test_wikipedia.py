#!/usr/bin/env python3
"""
Tests for Wikipedia search functionality
"""

import pytest
import sys
from pathlib import Path
import time

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "APP_Production"))
sys.path.insert(0, str(Path(__file__).parent.parent / "educational_archive" / "knowledge"))


class TestWikipediaSearch:
    """Tests for Wikipedia search functionality"""

    @pytest.fixture
    def wikipedia_searcher(self, wikipedia_db_path):
        """Create Wikipedia searcher instance"""
        if not wikipedia_db_path.exists():
            pytest.skip("Wikipedia database not found")

        try:
            from wikipedia_search import WikipediaKnowledgeSearch
            return WikipediaKnowledgeSearch(str(wikipedia_db_path))
        except ImportError:
            pytest.skip("wikipedia_search module not available")

    def test_search_returns_results(self, wikipedia_searcher):
        """Verify basic search returns results"""
        results = wikipedia_searcher.search("science", limit=5)
        assert len(results) > 0, "Search should return results"

    def test_search_result_structure(self, wikipedia_searcher):
        """Verify search results have expected structure"""
        results = wikipedia_searcher.search("Ethiopia", limit=1)
        if not results:
            pytest.skip("No results for test query")

        result = results[0]
        # Check for expected fields
        assert hasattr(result, 'title') or 'title' in result
        assert hasattr(result, 'content') or 'content' in result

    def test_search_performance(self, wikipedia_searcher):
        """Verify search completes in acceptable time"""
        start = time.time()
        results = wikipedia_searcher.search("photosynthesis", limit=10)
        elapsed = time.time() - start

        assert elapsed < 2.0, f"Search took too long: {elapsed:.2f}s"

    def test_empty_query_handling(self, wikipedia_searcher):
        """Verify empty query is handled gracefully"""
        results = wikipedia_searcher.search("", limit=5)
        # Should return empty list or handle gracefully
        assert isinstance(results, list)

    def test_special_characters(self, wikipedia_searcher):
        """Verify queries with special characters work"""
        # Should not raise an exception
        results = wikipedia_searcher.search("DNA & RNA", limit=5)
        assert isinstance(results, list)

    def test_tigrinya_query(self, wikipedia_searcher):
        """Verify Tigrinya text queries don't crash"""
        # Ethiopian script query
        results = wikipedia_searcher.search("ኢትዮጵያ", limit=5)
        assert isinstance(results, list)


class TestDomainFiltering:
    """Tests for domain filtering in Wikipedia search"""

    @pytest.fixture
    def wikipedia_searcher(self, wikipedia_db_path):
        """Create Wikipedia searcher instance"""
        if not wikipedia_db_path.exists():
            pytest.skip("Wikipedia database not found")

        try:
            from wikipedia_search import WikipediaKnowledgeSearch
            return WikipediaKnowledgeSearch(str(wikipedia_db_path))
        except ImportError:
            pytest.skip("wikipedia_search module not available")

    def test_science_domain_filter(self, wikipedia_searcher):
        """Test filtering by science domain"""
        if not hasattr(wikipedia_searcher, 'search_by_domain'):
            pytest.skip("Domain filtering not implemented")

        results = wikipedia_searcher.search_by_domain("cell", domain="science", limit=5)
        assert isinstance(results, list)

    def test_medicine_domain_filter(self, wikipedia_searcher):
        """Test filtering by medicine domain"""
        if not hasattr(wikipedia_searcher, 'search_by_domain'):
            pytest.skip("Domain filtering not implemented")

        results = wikipedia_searcher.search_by_domain("fever", domain="medicine", limit=5)
        assert isinstance(results, list)


class TestSimpleWikipediaIntegration:
    """Tests for Simple English Wikipedia content"""

    def test_simple_articles_present(self, wikipedia_db_conn):
        """Verify Simple English Wikipedia articles are in database"""
        cursor = wikipedia_db_conn.execute(
            "SELECT COUNT(*) FROM articles WHERE url LIKE '%simple.wikipedia.org%'"
        )
        count = cursor.fetchone()[0]

        # Should have ~206K Simple English articles
        assert count > 100000, f"Expected >100K Simple articles, found {count}"

    def test_article_readability(self, wikipedia_db_conn):
        """Verify Simple articles have readable content"""
        cursor = wikipedia_db_conn.execute(
            """SELECT content FROM articles
               WHERE url LIKE '%simple.wikipedia.org%'
               LIMIT 1"""
        )
        result = cursor.fetchone()
        if result:
            content = result[0]
            # Simple Wikipedia should have shorter sentences on average
            assert len(content) > 50, "Article should have content"
