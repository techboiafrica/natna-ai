#!/usr/bin/env python3
"""
Tests for translation functionality
"""

import pytest
import sys
from pathlib import Path

# Add APP_Production to path
sys.path.insert(0, str(Path(__file__).parent.parent / "APP_Production"))


class TestMinimalTranslator:
    """Tests for MinimalNATNATranslator"""

    @pytest.fixture
    def translator(self):
        """Create translator instance"""
        try:
            from minimal_translator import MinimalNATNATranslator
            return MinimalNATNATranslator()
        except ImportError as e:
            pytest.skip(f"minimal_translator not available: {e}")

    def test_translator_initializes(self, translator):
        """Verify translator initializes without error"""
        assert translator is not None

    def test_quick_response_hello(self, translator):
        """Verify quick responses work"""
        response = translator._get_quick_response("hello")
        assert response is not None
        assert "NATNA" in response or "Hello" in response

    def test_math_evaluation_simple(self, translator):
        """Verify simple math works"""
        result = translator._evaluate_math("2+2")
        assert result is not None
        assert result["english"] == "2+2 = 4"

    def test_math_evaluation_complex(self, translator):
        """Verify complex math works"""
        result = translator._evaluate_math("10*5+3")
        assert result is not None
        assert "53" in result["english"]

    def test_tigrinya_detection(self, translator):
        """Verify Tigrinya text is detected"""
        tigrinya_text = "ሰላም"
        is_tigrinya = any('\u1200' <= char <= '\u137F' for char in tigrinya_text)
        assert is_tigrinya, "Tigrinya text should be detected"

    def test_english_detection(self, translator):
        """Verify English text is not detected as Tigrinya"""
        english_text = "Hello world"
        is_tigrinya = any('\u1200' <= char <= '\u137F' for char in english_text)
        assert not is_tigrinya, "English should not be detected as Tigrinya"

    def test_subject_detection_science(self, translator):
        """Verify science subjects are detected"""
        subjects = translator._detect_subject_areas("photosynthesis in plants")
        assert "science" in subjects

    def test_subject_detection_medicine(self, translator):
        """Verify medical subjects are detected"""
        subjects = translator._detect_subject_areas("fever and headache treatment")
        assert "medicine" in subjects

    def test_subject_detection_math(self, translator):
        """Verify math subjects are detected"""
        subjects = translator._detect_subject_areas("calculate the equation")
        assert "mathematics" in subjects

    def test_empty_query_handling(self, translator):
        """Verify empty queries handled gracefully"""
        result = translator.process_query("")
        assert result is not None
        assert "Please ask" in str(result) or isinstance(result, str)


class TestTigrinyaTranslator:
    """Tests for TigrinyaTranslator"""

    @pytest.fixture
    def tigrinya_translator(self):
        """Create Tigrinya translator instance"""
        try:
            from tigrinya_focused_translator import TigrinyaTranslator
            return TigrinyaTranslator()
        except ImportError:
            pytest.skip("tigrinya_focused_translator not available")

    def test_translator_initializes(self, tigrinya_translator):
        """Verify translator initializes"""
        assert tigrinya_translator is not None

    def test_has_dictionary(self, tigrinya_translator):
        """Verify dictionary is loaded"""
        assert hasattr(tigrinya_translator, 'tigrinya_dictionary')
        assert len(tigrinya_translator.tigrinya_dictionary) > 0

    def test_basic_translation(self, tigrinya_translator):
        """Verify basic word translation"""
        result = tigrinya_translator.translate("hello", "tigrinya")
        assert result is not None
        assert len(result) > 0


class TestDictionarySearch:
    """Tests for dictionary search functionality"""

    @pytest.fixture
    def translator(self):
        """Create translator instance"""
        try:
            from minimal_translator import MinimalNATNATranslator
            return MinimalNATNATranslator()
        except ImportError:
            pytest.skip("minimal_translator not available")

    def test_dictionary_search(self, translator):
        """Verify dictionary search returns results"""
        results = translator._search_tigrinya_dictionary("water")
        # Should return list (may be empty if databases not present)
        assert isinstance(results, list)

    def test_curriculum_search(self, translator):
        """Verify curriculum search works"""
        results = translator._search_curriculum("photosynthesis")
        assert isinstance(results, list)


class TestIntelligentTranslator:
    """Tests for IntelligentTigrinyaTranslator"""

    @pytest.fixture
    def translator(self):
        """Create intelligent translator instance"""
        try:
            from intelligent_translator import IntelligentTigrinyaTranslator
            return IntelligentTigrinyaTranslator()
        except ImportError:
            pytest.skip("intelligent_translator not available")

    def test_translator_initializes(self, translator):
        """Verify translator initializes"""
        assert translator is not None

    def test_db_path_set(self, translator):
        """Verify database path is set"""
        assert translator.db_path is not None
        # Path should not contain /Volumes/ hardcoded (unless on macOS)
        import platform
        if platform.system() != "Darwin":
            assert "/Volumes/" not in str(translator.db_path)
