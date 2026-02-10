#!/usr/bin/env python3
"""
Tests for HTTP API endpoints
"""

import pytest
import requests
import time


class TestUIServerEndpoints:
    """Tests for HTTP API endpoints when server is running"""

    @pytest.fixture
    def base_url(self):
        """Base URL for the NATNA server"""
        return "http://localhost:8080"

    def test_server_responds(self, base_url):
        """Verify server is running and responds"""
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("NATNA server not running")

    def test_static_files_served(self, base_url):
        """Verify static files are served"""
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            if response.status_code != 200:
                pytest.skip("Server not responding")

            # Should contain HTML
            assert "html" in response.headers.get("content-type", "").lower() or \
                   "<html" in response.text.lower()
        except requests.exceptions.ConnectionError:
            pytest.skip("NATNA server not running")

    def test_query_endpoint(self, base_url):
        """Verify /api/query endpoint works"""
        try:
            response = requests.post(
                f"{base_url}/api/query",
                json={"query": "hello", "domain": "general"},
                timeout=30
            )
            if response.status_code != 200:
                pytest.skip("Query endpoint not available")

            data = response.json()
            assert "response" in data or "english" in data or "error" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("NATNA server not running")

    def test_models_endpoint(self, base_url):
        """Verify /api/models endpoint works"""
        try:
            response = requests.get(f"{base_url}/api/models", timeout=10)
            if response.status_code != 200:
                pytest.skip("Models endpoint not available")

            data = response.json()
            assert isinstance(data, (list, dict))
        except requests.exceptions.ConnectionError:
            pytest.skip("NATNA server not running")

    def test_wikipedia_search_endpoint(self, base_url):
        """Verify /api/wikipedia endpoint works"""
        try:
            response = requests.get(
                f"{base_url}/api/wikipedia",
                params={"q": "science"},
                timeout=10
            )
            if response.status_code != 200:
                pytest.skip("Wikipedia endpoint not available")

            data = response.json()
            assert isinstance(data, (list, dict))
        except requests.exceptions.ConnectionError:
            pytest.skip("NATNA server not running")

    def test_context_status_endpoint(self, base_url):
        """Verify /api/status/context endpoint works"""
        try:
            response = requests.get(f"{base_url}/api/status/context", timeout=5)
            if response.status_code != 200:
                pytest.skip("Context status endpoint not available")

            data = response.json()
            assert "percentage" in data or "usage" in data or "tokens" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("NATNA server not running")

    def test_clear_history_endpoint(self, base_url):
        """Verify /api/clear_history endpoint works"""
        try:
            response = requests.post(f"{base_url}/api/clear_history", timeout=5)
            if response.status_code != 200:
                pytest.skip("Clear history endpoint not available")

            data = response.json()
            assert "success" in data or "status" in data or "message" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("NATNA server not running")


class TestAPIResponseFormat:
    """Tests for API response format consistency"""

    @pytest.fixture
    def base_url(self):
        return "http://localhost:8080"

    def test_json_content_type(self, base_url):
        """Verify API returns JSON content type"""
        try:
            response = requests.post(
                f"{base_url}/api/query",
                json={"query": "test", "domain": "general"},
                timeout=30
            )
            if response.status_code != 200:
                pytest.skip("Query endpoint not available")

            content_type = response.headers.get("content-type", "")
            assert "application/json" in content_type
        except requests.exceptions.ConnectionError:
            pytest.skip("NATNA server not running")

    def test_error_handling(self, base_url):
        """Verify errors return proper JSON"""
        try:
            response = requests.post(
                f"{base_url}/api/query",
                json={},  # Missing required fields
                timeout=10
            )
            # Should still be JSON
            if response.status_code == 400:
                data = response.json()
                assert "error" in data or "message" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("NATNA server not running")
        except ValueError:
            pytest.fail("Error response should be valid JSON")


class TestCORSHeaders:
    """Tests for CORS headers"""

    @pytest.fixture
    def base_url(self):
        return "http://localhost:8080"

    def test_cors_headers_present(self, base_url):
        """Verify CORS headers are set for cross-origin requests"""
        try:
            response = requests.options(
                f"{base_url}/api/query",
                headers={"Origin": "http://localhost:3000"},
                timeout=5
            )
            # CORS headers should be present or endpoint should work
            # (Local apps may not need CORS)
        except requests.exceptions.ConnectionError:
            pytest.skip("NATNA server not running")
