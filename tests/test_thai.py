"""Tests for Thai NLP (requires pythainlp)."""

import unittest

try:
    import pythainlp  # noqa: F401

    _PYTHAINLP_AVAILABLE = True
except ImportError:
    _PYTHAINLP_AVAILABLE = False


@unittest.skipUnless(_PYTHAINLP_AVAILABLE, "pythainlp not installed")
class TestThaiWordTokenize(unittest.TestCase):
    def test_basic(self):
        from ricenlp import word_tokenize, LANG_TH

        result = word_tokenize("สวัสดีครับ", lang=LANG_TH)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_returns_strings(self):
        from ricenlp import word_tokenize, LANG_TH

        result = word_tokenize("ฉันรักประเทศไทย", lang=LANG_TH)
        for token in result:
            self.assertIsInstance(token, str)

    def test_empty_string(self):
        from ricenlp import word_tokenize, LANG_TH

        result = word_tokenize("", lang=LANG_TH)
        self.assertIsInstance(result, list)


@unittest.skipUnless(_PYTHAINLP_AVAILABLE, "pythainlp not installed")
class TestThaiSentTokenize(unittest.TestCase):
    def test_basic(self):
        from ricenlp import sent_tokenize, LANG_TH

        text = "ฉันรักประเทศไทย ประเทศไทยสวยงาม"
        result = sent_tokenize(text, lang=LANG_TH)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)


@unittest.skipUnless(_PYTHAINLP_AVAILABLE, "pythainlp not installed")
class TestThaiPosTag(unittest.TestCase):
    def test_basic(self):
        from ricenlp import pos_tag, LANG_TH

        result = pos_tag("ฉันรักประเทศไทย", lang=LANG_TH)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_returns_tuples(self):
        from ricenlp import pos_tag, LANG_TH

        result = pos_tag("ฉันรัก", lang=LANG_TH)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)


class TestThaiMissingDependency(unittest.TestCase):
    """Ensure ImportError is raised gracefully when pythainlp is absent."""

    @unittest.skipIf(_PYTHAINLP_AVAILABLE, "pythainlp is installed")
    def test_word_tokenize_raises_import_error(self):
        from ricenlp import word_tokenize, LANG_TH

        with self.assertRaises(ImportError):
            word_tokenize("สวัสดี", lang=LANG_TH)


if __name__ == "__main__":
    unittest.main()
