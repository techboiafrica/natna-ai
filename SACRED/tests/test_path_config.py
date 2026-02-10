#!/usr/bin/env python3
"""
Tests for cross-platform path configuration
"""

import pytest
import sys
from pathlib import Path

# Add APP_Production to path
sys.path.insert(0, str(Path(__file__).parent.parent / "APP_Production"))


class TestPathConfig:
    """Test path configuration module"""

    def test_path_config_imports(self):
        """Verify path_config module imports without error"""
        from path_config import (
            APP_DIR, SACRED_DIR, DATABASES_DIR, TIGRINYA_DB,
            WIKIPEDIA_DB, CONFIG_DIR, LOG_FILE, CURRICULUM_DB
        )

        assert APP_DIR is not None
        assert SACRED_DIR is not None

    def test_app_dir_exists(self):
        """Verify APP_DIR points to existing directory"""
        from path_config import APP_DIR
        assert APP_DIR.exists(), f"APP_DIR does not exist: {APP_DIR}"
        assert APP_DIR.is_dir(), f"APP_DIR is not a directory: {APP_DIR}"

    def test_sacred_dir_exists(self):
        """Verify SACRED_DIR points to existing directory"""
        from path_config import SACRED_DIR
        assert SACRED_DIR.exists(), f"SACRED_DIR does not exist: {SACRED_DIR}"
        assert SACRED_DIR.is_dir(), f"SACRED_DIR is not a directory: {SACRED_DIR}"

    def test_databases_dir_exists(self):
        """Verify DATABASES_DIR exists"""
        from path_config import DATABASES_DIR
        assert DATABASES_DIR.exists(), f"DATABASES_DIR does not exist: {DATABASES_DIR}"

    def test_tigrinya_db_path_valid(self):
        """Verify Tigrinya database path is valid"""
        from path_config import TIGRINYA_DB
        assert TIGRINYA_DB.suffix == ".db", "TIGRINYA_DB should have .db extension"
        assert TIGRINYA_DB.exists(), f"Tigrinya database not found: {TIGRINYA_DB}"

    def test_wikipedia_db_path_valid(self):
        """Verify Wikipedia database path is valid"""
        from path_config import WIKIPEDIA_DB
        assert WIKIPEDIA_DB.suffix == ".db", "WIKIPEDIA_DB should have .db extension"
        assert WIKIPEDIA_DB.exists(), f"Wikipedia database not found: {WIKIPEDIA_DB}"

    def test_config_dir_exists(self):
        """Verify CONFIG_DIR exists"""
        from path_config import CONFIG_DIR
        assert CONFIG_DIR.exists(), f"CONFIG_DIR does not exist: {CONFIG_DIR}"

    def test_no_hardcoded_volumes_path(self, app_dir):
        """Verify no hardcoded /Volumes/ paths in production code"""
        production_files = [
            "intelligent_translator.py",
            "minimal_translator.py",
            "NatnaUI.py",
            "path_config.py"
        ]

        for filename in production_files:
            filepath = app_dir / filename
            if not filepath.exists():
                continue

            content = filepath.read_text()
            # Check for hardcoded paths (excluding comments and strings in path_config itself)
            if filename != "path_config.py":
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    # Skip comments
                    if line.strip().startswith('#'):
                        continue
                    # Check for hardcoded /Volumes/ paths
                    if '/Volumes/' in line and 'PATH_CONFIG_AVAILABLE' not in line:
                        # Allow if it's in a fallback block
                        if 'else:' not in content[max(0, content.find(line)-200):content.find(line)]:
                            pytest.fail(f"Hardcoded /Volumes/ path in {filename}:{i}: {line.strip()}")

    def test_paths_are_absolute(self):
        """Verify all paths are absolute, not relative"""
        from path_config import (
            APP_DIR, SACRED_DIR, DATABASES_DIR, TIGRINYA_DB,
            WIKIPEDIA_DB, CONFIG_DIR, LOG_FILE, CURRICULUM_DB
        )

        paths = [APP_DIR, SACRED_DIR, DATABASES_DIR, TIGRINYA_DB,
                 WIKIPEDIA_DB, CONFIG_DIR, LOG_FILE, CURRICULUM_DB]

        for path in paths:
            assert path.is_absolute(), f"Path is not absolute: {path}"

    def test_ensure_directories_function(self):
        """Test ensure_directories creates necessary directories"""
        from path_config import ensure_directories, CONFIG_DIR

        ensure_directories()
        assert CONFIG_DIR.exists(), "ensure_directories should create CONFIG_DIR"
