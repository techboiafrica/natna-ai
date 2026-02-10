#!/usr/bin/env python3
"""
Tests for cross-platform compatibility
"""

import pytest
import sys
import platform
import subprocess
import os
from pathlib import Path


class TestPythonEnvironment:
    """Tests for Python environment"""

    def test_python_version(self):
        """Verify Python version is 3.9+"""
        version = sys.version_info
        assert version.major == 3, "Python 3 required"
        assert version.minor >= 9, f"Python 3.9+ required, got 3.{version.minor}"

    def test_required_modules_available(self):
        """Verify required standard library modules"""
        required = [
            "sqlite3",
            "json",
            "pathlib",
            "logging",
            "http.server",
            "urllib.request",
            "subprocess",
            "threading"
        ]

        for module in required:
            try:
                __import__(module)
            except ImportError:
                pytest.fail(f"Required module not available: {module}")

    def test_optional_modules(self):
        """Check optional modules (informational)"""
        optional = {
            "requests": "HTTP client",
            "torch": "Helsinki translation",
            "transformers": "Helsinki translation"
        }

        available = []
        missing = []

        for module, purpose in optional.items():
            try:
                __import__(module)
                available.append(module)
            except ImportError:
                missing.append(f"{module} ({purpose})")

        # Just informational - doesn't fail
        if missing:
            print(f"\nOptional modules not installed: {missing}")


class TestEncodingSupport:
    """Tests for Unicode/UTF-8 encoding"""

    def test_utf8_default(self):
        """Verify UTF-8 is default encoding"""
        import locale
        encoding = locale.getpreferredencoding(False)
        # Should be UTF-8 or compatible
        assert encoding.lower() in ['utf-8', 'utf8', 'cp65001'], \
            f"Expected UTF-8 encoding, got {encoding}"

    def test_tigrinya_characters(self):
        """Verify Tigrinya characters can be processed"""
        tigrinya = "ሰላም ከመይ ኣለኻ"

        # Should be able to encode/decode
        encoded = tigrinya.encode('utf-8')
        decoded = encoded.decode('utf-8')
        assert decoded == tigrinya

    def test_ethiopic_range(self):
        """Verify Ethiopic Unicode range detection works"""
        tigrinya_char = 'ሀ'  # First Ethiopic character
        code_point = ord(tigrinya_char)

        assert 0x1200 <= code_point <= 0x137F, \
            f"Ethiopic detection range incorrect: {hex(code_point)}"


class TestFileSystem:
    """Tests for file system compatibility"""

    def test_pathlib_works(self, sacred_dir):
        """Verify pathlib works correctly"""
        assert sacred_dir.exists()
        assert sacred_dir.is_dir()

    def test_spaces_in_path(self, sacred_dir):
        """Verify paths with spaces work (common on macOS/Windows)"""
        # The SACRED directory path may contain spaces
        path_str = str(sacred_dir)
        # Should be able to resolve
        resolved = Path(path_str).resolve()
        assert resolved.exists()

    def test_relative_paths(self, sacred_dir):
        """Verify relative path resolution works"""
        app_dir = sacred_dir / "APP_Production"
        if app_dir.exists():
            relative = Path("APP_Production")
            # Can create relative paths
            assert relative.name == "APP_Production"

    def test_file_permissions(self, sacred_dir):
        """Verify file permissions allow read access"""
        # Should be able to list directory
        try:
            list(sacred_dir.iterdir())
        except PermissionError:
            pytest.fail("Cannot read SACRED directory - permission denied")


class TestSubprocessExecution:
    """Tests for subprocess execution"""

    def test_subprocess_works(self):
        """Verify subprocess execution works"""
        result = subprocess.run(
            [sys.executable, "-c", "print('test')"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "test" in result.stdout

    def test_python_executable_found(self):
        """Verify Python executable can be found"""
        assert Path(sys.executable).exists()


class TestPlatformDetection:
    """Tests for platform detection"""

    def test_platform_identified(self):
        """Verify platform can be identified"""
        system = platform.system()
        assert system in ["Darwin", "Linux", "Windows"], \
            f"Unexpected platform: {system}"

    def test_path_separator(self):
        """Verify correct path separator for platform"""
        system = platform.system()

        if system == "Windows":
            assert os.sep == "\\"
        else:
            assert os.sep == "/"


class TestNetworkCapabilities:
    """Tests for network capabilities"""

    def test_localhost_resolution(self):
        """Verify localhost can be resolved"""
        import socket
        try:
            socket.gethostbyname("localhost")
        except socket.gaierror:
            pytest.fail("Cannot resolve localhost")

    def test_port_availability_check(self):
        """Verify can check if port is available"""
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Check if port 8080 can be checked (not necessarily available)
            result = sock.connect_ex(('localhost', 8080))
            # 0 = connected (server running), anything else = not connected
            assert isinstance(result, int)
        finally:
            sock.close()


class TestEnvironmentVariables:
    """Tests for environment variable handling"""

    def test_ollama_models_path(self):
        """Check OLLAMA_MODELS environment variable"""
        ollama_models = os.environ.get("OLLAMA_MODELS")

        if ollama_models:
            path = Path(ollama_models)
            assert path.exists(), f"OLLAMA_MODELS path does not exist: {ollama_models}"

    def test_path_env_exists(self):
        """Verify PATH environment variable exists"""
        path = os.environ.get("PATH")
        assert path is not None
        assert len(path) > 0


class TestDatabaseCompatibility:
    """Tests for SQLite compatibility across platforms"""

    def test_sqlite_version(self):
        """Verify SQLite version supports required features"""
        import sqlite3
        version = sqlite3.sqlite_version_info

        # Need SQLite 3.9+ for FTS5
        assert version[0] >= 3
        if version[0] == 3:
            assert version[1] >= 9, f"SQLite 3.9+ needed for FTS5, got {sqlite3.sqlite_version}"

    def test_sqlite_fts5_support(self):
        """Verify FTS5 extension is available"""
        import sqlite3

        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content)")
            conn.execute("DROP TABLE test_fts")
        except sqlite3.Error as e:
            pytest.fail(f"FTS5 not supported: {e}")
        finally:
            conn.close()
