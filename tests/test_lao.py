"""Tests for Lao NLP (requires laonlp)."""

import unittest

try:
    import laonlp  # noqa: F401

    _LAONLP_AVAILABLE = True
except ImportError:
    _LAONLP_AVAILABLE = False


@unittest.skipUnless(_LAONLP_AVAILABLE, "laonlp not installed")
class TestLaoWordTokenize(unittest.TestCase):
    def test_basic(self):
        from ricenlp import word_tokenize, LANG_LO

        result = word_tokenize("ສະບາຍດີ", lang=LANG_LO)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_returns_strings(self):
        from ricenlp import word_tokenize, LANG_LO

        result = word_tokenize("ພາສາລາວ", lang=LANG_LO)
        for token in result:
            self.assertIsInstance(token, str)

    def test_empty_string(self):
        from ricenlp import word_tokenize, LANG_LO

        result = word_tokenize("", lang=LANG_LO)
        self.assertIsInstance(result, list)


@unittest.skipUnless(_LAONLP_AVAILABLE, "laonlp not installed")
class TestLaoSentTokenize(unittest.TestCase):
    def test_basic(self):
        from ricenlp import sent_tokenize, LANG_LO

        text = "ສະບາຍດີ. ຂອບໃຈ."
        result = sent_tokenize(text, lang=LANG_LO)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_multiple_sentences(self):
        from ricenlp import sent_tokenize, LANG_LO

        text = "ສະບາຍດີ. ຂອບໃຈ. ລາກ່ອນ."
        result = sent_tokenize(text, lang=LANG_LO)
        self.assertEqual(len(result), 3)

    def test_no_punctuation(self):
        from ricenlp import sent_tokenize, LANG_LO

        text = "ສະບາຍດີ"
        result = sent_tokenize(text, lang=LANG_LO)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], text)


@unittest.skipUnless(_LAONLP_AVAILABLE, "laonlp not installed")
class TestLaoPosTag(unittest.TestCase):
    def test_raises_not_implemented(self):
        from ricenlp import pos_tag, LANG_LO

        with self.assertRaises(NotImplementedError):
            pos_tag("ສະບາຍດີ", lang=LANG_LO)


class TestLaoMissingDependency(unittest.TestCase):
    """Ensure ImportError is raised gracefully when laonlp is absent."""

    @unittest.skipIf(_LAONLP_AVAILABLE, "laonlp is installed")
    def test_word_tokenize_raises_import_error(self):
        from ricenlp import word_tokenize, LANG_LO

        with self.assertRaises(ImportError):
            word_tokenize("ສະບາຍດີ", lang=LANG_LO)


if __name__ == "__main__":
    unittest.main()
