#!/usr/bin/env python3
"""
Shared pytest fixtures for NATNA test suite
"""

import pytest
import sqlite3
import sys
import platform
from pathlib import Path
from datetime import datetime

# Add APP_Production to path for imports
SACRED_DIR = Path(__file__).parent.parent
APP_DIR = SACRED_DIR / "APP_Production"
ERROR_LOG_DIR = SACRED_DIR / "error_logs"
sys.path.insert(0, str(APP_DIR))

# Ensure error_logs directory exists
ERROR_LOG_DIR.mkdir(exist_ok=True)


# Error logging hooks
_failed_tests = []


def pytest_runtest_makereport(item, call):
    """Capture test failures for error logging."""
    if call.when == "call" and call.excinfo is not None:
        _failed_tests.append({
            "test": item.nodeid,
            "error": str(call.excinfo.value),
            "traceback": str(call.excinfo.getrepr()),
            "timestamp": datetime.now().isoformat()
        })


def pytest_sessionfinish(session, exitstatus):
    """Write error log if any tests failed."""
    if _failed_tests:
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = ERROR_LOG_DIR / f"test_errors_{date_str}.log"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"Test Run: {datetime.now().isoformat()}\n")
            f.write(f"Platform: {platform.system()} {platform.release()}\n")
            f.write(f"Python: {platform.python_version()}\n")
            f.write(f"Failed: {len(_failed_tests)} test(s)\n")
            f.write("="*60 + "\n\n")

            for failure in _failed_tests:
                f.write(f"TEST: {failure['test']}\n")
                f.write(f"TIME: {failure['timestamp']}\n")
                f.write(f"ERROR: {failure['error']}\n")
                f.write(f"TRACEBACK:\n{failure['traceback']}\n")
                f.write("-"*40 + "\n\n")


@pytest.fixture
def sacred_dir():
    """Return the SACRED directory path"""
    return Path(__file__).parent.parent


@pytest.fixture
def app_dir(sacred_dir):
    """Return the APP_Production directory path"""
    return sacred_dir / "APP_Production"


@pytest.fixture
def tigrinya_db_path(sacred_dir):
    """Return the Tigrinya database path"""
    return sacred_dir / "organized_data" / "databases" / "massive_tigrinya_database.db"


@pytest.fixture
def wikipedia_db_path(sacred_dir):
    """Return the Wikipedia database path"""
    return sacred_dir / "educational_archive" / "knowledge" / "massive_wikipedia.db"


@pytest.fixture
def curriculum_db_path(sacred_dir):
    """Return the curriculum database path"""
    return sacred_dir / "organized_data" / "curriculum_cache" / "curriculum_database.db"


@pytest.fixture
def tigrinya_db_conn(tigrinya_db_path):
    """Provide a connection to the Tigrinya database"""
    if not tigrinya_db_path.exists():
        pytest.skip(f"Tigrinya database not found: {tigrinya_db_path}")

    conn = sqlite3.connect(str(tigrinya_db_path))
    yield conn
    conn.close()


@pytest.fixture
def wikipedia_db_conn(wikipedia_db_path):
    """Provide a connection to the Wikipedia database"""
    if not wikipedia_db_path.exists():
        pytest.skip(f"Wikipedia database not found: {wikipedia_db_path}")

    conn = sqlite3.connect(str(wikipedia_db_path))
    yield conn
    conn.close()


@pytest.fixture
def sample_english_queries():
    """Sample English queries for testing"""
    return [
        "What is photosynthesis?",
        "How do plants grow?",
        "Explain gravity",
        "What is DNA?",
        "Tell me about Ethiopia"
    ]


@pytest.fixture
def sample_tigrinya_text():
    """Sample Tigrinya text for testing"""
    return [
        "ሰላም",  # Hello
        "ከመይ ኣለኻ",  # How are you
        "ጥዕና",  # Health
    ]


@pytest.fixture
def ollama_url():
    """Return the Ollama API base URL"""
    return "http://localhost:11434"
