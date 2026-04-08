"""Tests for Vietnamese NLP (requires underthesea)."""

import unittest

try:
    import underthesea  # noqa: F401

    _UNDERTHESEA_AVAILABLE = True
except ImportError:
    _UNDERTHESEA_AVAILABLE = False


@unittest.skipUnless(_UNDERTHESEA_AVAILABLE, "underthesea not installed")
class TestVietnameseWordTokenize(unittest.TestCase):
    def test_basic(self):
        from ricenlp import word_tokenize, LANG_VI

        result = word_tokenize("Tôi yêu Việt Nam", lang=LANG_VI)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_returns_strings(self):
        from ricenlp import word_tokenize, LANG_VI

        result = word_tokenize("Xin chào", lang=LANG_VI)
        for token in result:
            self.assertIsInstance(token, str)

    def test_empty_string(self):
        from ricenlp import word_tokenize, LANG_VI

        result = word_tokenize("", lang=LANG_VI)
        self.assertIsInstance(result, list)


@unittest.skipUnless(_UNDERTHESEA_AVAILABLE, "underthesea not installed")
class TestVietnameseSentTokenize(unittest.TestCase):
    def test_basic(self):
        from ricenlp import sent_tokenize, LANG_VI

        text = "Tôi yêu Việt Nam. Việt Nam rất đẹp."
        result = sent_tokenize(text, lang=LANG_VI)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)


@unittest.skipUnless(_UNDERTHESEA_AVAILABLE, "underthesea not installed")
class TestVietnamesePosTag(unittest.TestCase):
    def test_basic(self):
        from ricenlp import pos_tag, LANG_VI

        result = pos_tag("Tôi yêu Việt Nam", lang=LANG_VI)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_returns_tuples(self):
        from ricenlp import pos_tag, LANG_VI

        result = pos_tag("Xin chào", lang=LANG_VI)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)


class TestVietnameseMissingDependency(unittest.TestCase):
    """Ensure ImportError is raised gracefully when underthesea is absent."""

    @unittest.skipIf(_UNDERTHESEA_AVAILABLE, "underthesea is installed")
    def test_word_tokenize_raises_import_error(self):
        from ricenlp import word_tokenize, LANG_VI

        with self.assertRaises(ImportError):
            word_tokenize("Xin chào", lang=LANG_VI)


if __name__ == "__main__":
    unittest.main()
