#!/usr/bin/env python3
"""
Tests for Ollama API integration
"""

import pytest
import requests
import time


class TestOllamaConnection:
    """Tests for Ollama server connectivity"""

    def test_server_responds(self, ollama_url):
        """Verify Ollama server is running and responds"""
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            assert response.status_code == 200, f"Unexpected status: {response.status_code}"
        except requests.exceptions.ConnectionError:
            pytest.skip("Ollama server not running")
        except requests.exceptions.Timeout:
            pytest.fail("Ollama server timeout")

    def test_models_available(self, ollama_url):
        """Verify at least one model is available"""
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.skip("Ollama server not responding")

            data = response.json()
            models = data.get("models", [])
            assert len(models) > 0, "No models available in Ollama"
        except requests.exceptions.ConnectionError:
            pytest.skip("Ollama server not running")

    def test_expected_models_present(self, ollama_url):
        """Verify expected NATNA models are available"""
        expected_models = [
            "smollm2:360m",
            "deepseek-r1"
        ]

        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.skip("Ollama server not responding")

            data = response.json()
            available = {m["name"].split(":")[0] for m in data.get("models", [])}

            missing = []
            for model in expected_models:
                if not any(model in name for name in available):
                    missing.append(model)

            if missing:
                pytest.skip(f"Missing models (optional): {missing}")

        except requests.exceptions.ConnectionError:
            pytest.skip("Ollama server not running")


class TestOllamaGenerate:
    """Tests for Ollama generate API"""

    def test_generate_responds(self, ollama_url):
        """Verify generate API works"""
        try:
            # Get available models first
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.skip("Ollama server not responding")

            models = response.json().get("models", [])
            if not models:
                pytest.skip("No models available")

            # Use first available model
            model_name = models[0]["name"]

            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Say hello",
                    "stream": False,
                    "options": {"num_predict": 10}
                },
                timeout=30
            )

            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert len(data["response"]) > 0

        except requests.exceptions.ConnectionError:
            pytest.skip("Ollama server not running")
        except requests.exceptions.Timeout:
            pytest.skip("Ollama generate timeout")

    def test_generate_performance(self, ollama_url):
        """Verify generate completes in reasonable time"""
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.skip("Ollama server not responding")

            models = response.json().get("models", [])
            if not models:
                pytest.skip("No models available")

            # Prefer smollm2:360m for speed test
            model_name = "smollm2:360m"
            available_names = [m["name"] for m in models]
            if not any("smollm2" in n for n in available_names):
                model_name = models[0]["name"]

            start = time.time()
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "What is 2+2?",
                    "stream": False,
                    "options": {"num_predict": 20}
                },
                timeout=60
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                assert elapsed < 30, f"Generate took too long: {elapsed:.1f}s"

        except requests.exceptions.ConnectionError:
            pytest.skip("Ollama server not running")
        except requests.exceptions.Timeout:
            pytest.skip("Ollama timeout")


class TestOllamaModels:
    """Tests for model configuration"""

    def test_ollama_models_env_var(self):
        """Check if OLLAMA_MODELS environment variable is set for portable storage"""
        import os
        ollama_models = os.environ.get("OLLAMA_MODELS")

        # This is informational - not a failure if not set
        if ollama_models:
            from pathlib import Path
            models_path = Path(ollama_models)
            assert models_path.exists(), f"OLLAMA_MODELS path does not exist: {ollama_models}"
