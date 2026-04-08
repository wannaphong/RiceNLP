"""Tests for Tagalog NLP module – backed by calamanCy."""

import unittest

# Check if the calamancy library is importable.
try:
    import calamancy as _calamancy_lib  # noqa: F401

    _CALAMANCY_LIB_AVAILABLE = True
except ImportError:
    _CALAMANCY_LIB_AVAILABLE = False

# Check if the default model package is already installed (no network needed).
try:
    import spacy as _spacy

    _MODEL_PKG = "tl_calamancy_md"
    _CALAMANCY_MODEL_AVAILABLE = _CALAMANCY_LIB_AVAILABLE and _spacy.util.is_package(
        _MODEL_PKG
    )
except ImportError:
    _CALAMANCY_MODEL_AVAILABLE = False


@unittest.skipUnless(_CALAMANCY_MODEL_AVAILABLE, "calamanCy model not installed")
class TestTagalogWordTokenize(unittest.TestCase):
    def test_basic(self):
        from ricenlp import word_tokenize, LANG_TL

        result = word_tokenize("Ako si Juan de la Cruz", lang=LANG_TL)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_returns_strings(self):
        from ricenlp import word_tokenize, LANG_TL

        for token in word_tokenize("Mabuhay ang Pilipinas", lang=LANG_TL):
            self.assertIsInstance(token, str)

    def test_empty_string(self):
        from ricenlp import word_tokenize, LANG_TL

        result = word_tokenize("", lang=LANG_TL)
        self.assertEqual(result, [])


@unittest.skipUnless(_CALAMANCY_MODEL_AVAILABLE, "calamanCy model not installed")
class TestTagalogSentTokenize(unittest.TestCase):
    def test_basic(self):
        from ricenlp import sent_tokenize, LANG_TL

        result = sent_tokenize("Ako si Juan. Mahal ko ang Pilipinas.", lang=LANG_TL)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_returns_strings(self):
        from ricenlp import sent_tokenize, LANG_TL

        for sent in sent_tokenize("Mabuhay.", lang=LANG_TL):
            self.assertIsInstance(sent, str)

    def test_empty_string(self):
        from ricenlp import sent_tokenize, LANG_TL

        result = sent_tokenize("", lang=LANG_TL)
        self.assertEqual(result, [])


@unittest.skipUnless(_CALAMANCY_MODEL_AVAILABLE, "calamanCy model not installed")
class TestTagalogPosTag(unittest.TestCase):
    def test_basic(self):
        from ricenlp import pos_tag, LANG_TL

        result = pos_tag("Ako si Juan de la Cruz", lang=LANG_TL)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_returns_tuples(self):
        from ricenlp import pos_tag, LANG_TL

        for item in pos_tag("Mabuhay ang Pilipinas", lang=LANG_TL):
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_empty_string(self):
        from ricenlp import pos_tag, LANG_TL

        result = pos_tag("", lang=LANG_TL)
        self.assertEqual(result, [])


class TestTagalogMissingDependency(unittest.TestCase):
    @unittest.skipIf(_CALAMANCY_LIB_AVAILABLE, "calamanCy is installed")
    def test_word_tokenize_raises_import_error(self):
        from ricenlp import word_tokenize, LANG_TL

        with self.assertRaises(ImportError):
            word_tokenize("Mabuhay", lang=LANG_TL)


class TestTagalogLanguageConstant(unittest.TestCase):
    def test_lang_constant(self):
        import ricenlp

        self.assertEqual(ricenlp.LANG_TL, "tl")
        self.assertIn("tl", ricenlp.SUPPORTED_LANGUAGES)


if __name__ == "__main__":
    unittest.main()
