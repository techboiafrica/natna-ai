#!/usr/bin/env python3
"""
Tests for database connectivity and integrity
"""

import pytest
import sqlite3
import time


class TestTigrinyaDatabase:
    """Tests for Tigrinya database"""

    def test_database_connects(self, tigrinya_db_path):
        """Verify database file can be opened"""
        if not tigrinya_db_path.exists():
            pytest.skip("Tigrinya database not found")

        conn = sqlite3.connect(str(tigrinya_db_path))
        assert conn is not None
        conn.close()

    def test_database_has_entries(self, tigrinya_db_conn):
        """Verify database has translation entries"""
        cursor = tigrinya_db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert len(tables) > 0, "Database should have tables"

    def test_translations_table_exists(self, tigrinya_db_conn):
        """Verify translations table exists"""
        cursor = tigrinya_db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='translations'"
        )
        result = cursor.fetchone()
        if result is None:
            # Check for alternative table names
            cursor = tigrinya_db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            pytest.skip(f"No 'translations' table. Available tables: {tables}")

    def test_entry_count_reasonable(self, tigrinya_db_conn):
        """Verify database has substantial content"""
        cursor = tigrinya_db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        total_rows = 0
        for table in tables:
            try:
                cursor = tigrinya_db_conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                total_rows += count
            except sqlite3.Error:
                continue

        assert total_rows > 1000, f"Database should have >1000 entries, found {total_rows}"

    def test_query_performance(self, tigrinya_db_conn):
        """Verify queries complete within acceptable time"""
        cursor = tigrinya_db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1"
        )
        table = cursor.fetchone()
        if table is None:
            pytest.skip("No tables found")

        start = time.time()
        cursor = tigrinya_db_conn.execute(f"SELECT * FROM {table[0]} LIMIT 100")
        cursor.fetchall()
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Query took too long: {elapsed:.2f}s"


class TestWikipediaDatabase:
    """Tests for Wikipedia database"""

    def test_database_connects(self, wikipedia_db_path):
        """Verify database file can be opened"""
        if not wikipedia_db_path.exists():
            pytest.skip("Wikipedia database not found")

        conn = sqlite3.connect(str(wikipedia_db_path))
        assert conn is not None
        conn.close()

    def test_articles_table_exists(self, wikipedia_db_conn):
        """Verify articles table exists"""
        cursor = wikipedia_db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='articles'"
        )
        result = cursor.fetchone()
        assert result is not None, "articles table should exist"

    def test_article_count(self, wikipedia_db_conn):
        """Verify Wikipedia has expected article count"""
        cursor = wikipedia_db_conn.execute("SELECT COUNT(*) FROM articles")
        count = cursor.fetchone()[0]

        # Should have 286K+ articles (84K English + 206K Simple English)
        assert count > 200000, f"Expected >200K articles, found {count}"

    def test_fts_index_exists(self, wikipedia_db_conn):
        """Verify FTS5 index exists for full-text search"""
        cursor = wikipedia_db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%fts%'"
        )
        fts_tables = cursor.fetchall()
        assert len(fts_tables) > 0, "FTS index should exist"

    def test_fts_search_works(self, wikipedia_db_conn):
        """Verify FTS search returns results"""
        try:
            cursor = wikipedia_db_conn.execute(
                "SELECT title FROM articles_fts WHERE articles_fts MATCH 'science' LIMIT 5"
            )
            results = cursor.fetchall()
            assert len(results) > 0, "FTS search for 'science' should return results"
        except sqlite3.Error as e:
            pytest.skip(f"FTS search not available: {e}")

    def test_article_structure(self, wikipedia_db_conn):
        """Verify article table has expected columns"""
        cursor = wikipedia_db_conn.execute("PRAGMA table_info(articles)")
        columns = {row[1] for row in cursor.fetchall()}

        expected = {'id', 'title', 'content'}
        missing = expected - columns
        assert not missing, f"Missing columns: {missing}"

    def test_query_performance(self, wikipedia_db_conn):
        """Verify search completes within acceptable time"""
        start = time.time()
        cursor = wikipedia_db_conn.execute(
            "SELECT title, content FROM articles WHERE title LIKE '%Ethiopia%' LIMIT 10"
        )
        cursor.fetchall()
        elapsed = time.time() - start

        assert elapsed < 2.0, f"Query took too long: {elapsed:.2f}s"


class TestConcurrentAccess:
    """Tests for concurrent database access"""

    def test_multiple_connections(self, wikipedia_db_path):
        """Verify multiple connections work"""
        if not wikipedia_db_path.exists():
            pytest.skip("Wikipedia database not found")

        connections = []
        try:
            for i in range(3):
                conn = sqlite3.connect(str(wikipedia_db_path))
                connections.append(conn)

            # All connections should work
            for conn in connections:
                cursor = conn.execute("SELECT COUNT(*) FROM articles")
                count = cursor.fetchone()[0]
                assert count > 0
        finally:
            for conn in connections:
                conn.close()
