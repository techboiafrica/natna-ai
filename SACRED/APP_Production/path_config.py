#!/usr/bin/env python3
"""
Cross-Platform Path Configuration for NATNA

All paths are computed relative to this file's location, ensuring
the application works on Windows, Linux, and macOS without modification.
"""

import os
import platform
from pathlib import Path

# Base paths computed from this file's location
APP_DIR = Path(__file__).parent.resolve()
SACRED_DIR = APP_DIR.parent

# Data directories
ORGANIZED_DATA_DIR = SACRED_DIR / "organized_data"
EDUCATIONAL_ARCHIVE_DIR = SACRED_DIR / "educational_archive"

# Database paths
DATABASES_DIR = ORGANIZED_DATA_DIR / "databases"
TIGRINYA_DB = DATABASES_DIR / "massive_tigrinya_database.db"
ACADEMIC_DB = DATABASES_DIR / "academic_translations_v2.db"
CULTURAL_DB = DATABASES_DIR / "cultural_translations_v2.db"
DICTIONARY_DB = DATABASES_DIR / "word_dictionary_v2.db"

# Knowledge paths
KNOWLEDGE_DIR = EDUCATIONAL_ARCHIVE_DIR / "knowledge"
WIKIPEDIA_DB = KNOWLEDGE_DIR / "massive_wikipedia.db"

# Config paths
CONFIG_DIR = ORGANIZED_DATA_DIR / "config"
CURRICULUM_DIR = ORGANIZED_DATA_DIR / "curriculum_cache"
CURRICULUM_DB = CURRICULUM_DIR / "curriculum_database.db"

# Log file
LOG_FILE = CONFIG_DIR / "natna_conversations.log"

# Error logs
ERROR_LOG_DIR = SACRED_DIR / "error_logs"

# Runtime paths
RUNTIME_DIR = SACRED_DIR / "runtime"
MODELS_DIR = RUNTIME_DIR / "models"


def get_platform_info():
    """Return current platform information for debugging."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "app_dir": str(APP_DIR),
        "sacred_dir": str(SACRED_DIR)
    }


def ensure_directories():
    """Create required directories if they do not exist."""
    directories = [CONFIG_DIR, CURRICULUM_DIR, ERROR_LOG_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def validate_paths():
    """
    Validate that critical paths exist.
    Returns list of missing paths with descriptions.
    """
    required = [
        ("APP_Production directory", APP_DIR),
        ("SACRED directory", SACRED_DIR),
        ("Databases directory", DATABASES_DIR),
    ]

    optional_databases = [
        ("Tigrinya Database", TIGRINYA_DB),
        ("Wikipedia Database", WIKIPEDIA_DB),
    ]

    missing = []

    # Check required paths
    for name, path in required:
        if not path.exists():
            missing.append(f"REQUIRED - {name}: {path}")

    # Check optional databases
    for name, path in optional_databases:
        if not path.exists():
            missing.append(f"OPTIONAL - {name}: {path}")

    return missing


def log_error(error_type, message, traceback_str=None, query=None):
    """
    Log a runtime error to the error_logs directory.

    Args:
        error_type: Type of error (e.g., "DatabaseError", "OllamaError")
        message: Error message
        traceback_str: Full traceback string (optional)
        query: Query that caused the error (optional)
    """
    from datetime import datetime

    ensure_directories()
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = ERROR_LOG_DIR / f"runtime_errors_{date_str}.log"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "-"*40 + "\n")
        f.write(f"TIME: {datetime.now().isoformat()}\n")
        f.write(f"TYPE: {error_type}\n")
        f.write(f"MESSAGE: {message}\n")
        if query:
            f.write(f"QUERY: {query}\n")
        if traceback_str:
            f.write(f"TRACEBACK:\n{traceback_str}\n")


def get_database_path(db_name):
    """
    Get path to a database by name.
    Supports: tigrinya, wikipedia, academic, cultural, curriculum, dictionary
    """
    db_map = {
        "tigrinya": TIGRINYA_DB,
        "wikipedia": WIKIPEDIA_DB,
        "academic": ACADEMIC_DB,
        "cultural": CULTURAL_DB,
        "curriculum": CURRICULUM_DB,
        "dictionary": DICTIONARY_DB,
    }
    return db_map.get(db_name.lower())


if __name__ == "__main__":
    # Self-test when run directly
    print("NATNA Path Configuration")
    print("=" * 50)

    info = get_platform_info()
    print(f"Platform: {info['system']} ({info['machine']})")
    print(f"Python: {info['python_version']}")
    print(f"APP_DIR: {info['app_dir']}")
    print(f"SACRED_DIR: {info['sacred_dir']}")
    print()

    print("Path Validation:")
    missing = validate_paths()
    if missing:
        for item in missing:
            print(f"  MISSING: {item}")
    else:
        print("  All paths valid")
    print()

    print("Database Paths:")
    print(f"  Tigrinya: {TIGRINYA_DB}")
    print(f"  Wikipedia: {WIKIPEDIA_DB}")
    print(f"  Curriculum: {CURRICULUM_DB}")
