"""Tests for Khmer NLP module (pure Python, no external dependency)."""

import unittest


class TestKCCTokenize(unittest.TestCase):
    def test_basic(self):
        from ricenlp.khmer import kcc_tokenize

        result = kcc_tokenize("ភាសាខ្មែរ")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_returns_strings(self):
        from ricenlp.khmer import kcc_tokenize

        for token in kcc_tokenize("ភាសាខ្មែរ"):
            self.assertIsInstance(token, str)

    def test_empty_string(self):
        from ricenlp.khmer import kcc_tokenize

        self.assertEqual(kcc_tokenize(""), [])

    def test_non_khmer_passthrough(self):
        from ricenlp.khmer import kcc_tokenize

        result = kcc_tokenize("hello")
        self.assertEqual(result, list("hello"))


class TestDictWordTokenizeFMM(unittest.TestCase):
    """Forward Maximum Matching (dict_word_tokenize)."""

    def setUp(self):
        self.dictionary = {"ភាសា", "ខ្មែរ", "ជាតិ"}

    def test_basic(self):
        from ricenlp.khmer import dict_word_tokenize

        result = dict_word_tokenize("ភាសាខ្មែរ", self.dictionary)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_known_words_matched(self):
        from ricenlp.khmer import dict_word_tokenize

        result = dict_word_tokenize("ភាសាខ្មែរ", self.dictionary)
        self.assertIn("ភាសា", result)
        self.assertIn("ខ្មែរ", result)

    def test_empty_string(self):
        from ricenlp.khmer import dict_word_tokenize

        self.assertEqual(dict_word_tokenize("", self.dictionary), [])

    def test_empty_dictionary(self):
        from ricenlp.khmer import dict_word_tokenize

        result = dict_word_tokenize("abc", set())
        self.assertEqual(result, list("abc"))


class TestDictWordTokenizeMin(unittest.TestCase):
    """Forward Minimum Matching (dict_word_tokenize_min)."""

    def setUp(self):
        self.dictionary = {"ភាសា", "ខ្មែរ", "ភាសាខ្មែរ"}

    def test_basic(self):
        from ricenlp.khmer import dict_word_tokenize_min

        result = dict_word_tokenize_min("ភាសាខ្មែរ", self.dictionary)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_prefers_shorter_match(self):
        from ricenlp.khmer import dict_word_tokenize_min

        # FMinM picks the shortest match first, so "ភាសា" before "ភាសាខ្មែរ"
        result = dict_word_tokenize_min("ភាសាខ្មែរ", self.dictionary)
        self.assertEqual(result[0], "ភាសា")


class TestDictWordTokenizeRev(unittest.TestCase):
    """Backward Maximum Matching (dict_word_tokenize_rev)."""

    def setUp(self):
        self.dictionary = {"ភាសា", "ខ្មែរ", "ជាតិ"}

    def test_basic(self):
        from ricenlp.khmer import dict_word_tokenize_rev

        result = dict_word_tokenize_rev("ភាសាខ្មែរ", self.dictionary)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_known_words_matched(self):
        from ricenlp.khmer import dict_word_tokenize_rev

        result = dict_word_tokenize_rev("ភាសាខ្មែរ", self.dictionary)
        self.assertIn("ភាសា", result)
        self.assertIn("ខ្មែរ", result)

    def test_empty_string(self):
        from ricenlp.khmer import dict_word_tokenize_rev

        self.assertEqual(dict_word_tokenize_rev("", self.dictionary), [])


class TestDictWordTokenizeRevMin(unittest.TestCase):
    """Backward Minimum Matching (dict_word_tokenize_rev_min)."""

    def setUp(self):
        self.dictionary = {"ភាសា", "ខ្មែរ", "ភាសាខ្មែរ"}

    def test_basic(self):
        from ricenlp.khmer import dict_word_tokenize_rev_min

        result = dict_word_tokenize_rev_min("ភាសាខ្មែរ", self.dictionary)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_empty_string(self):
        from ricenlp.khmer import dict_word_tokenize_rev_min

        self.assertEqual(dict_word_tokenize_rev_min("", self.dictionary), [])


class TestKhmerWordTokenize(unittest.TestCase):
    """Top-level word_tokenize uses KCC (no external dep)."""

    def test_basic(self):
        from ricenlp import word_tokenize, LANG_KM

        result = word_tokenize("ភាសាខ្មែរ", lang=LANG_KM)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_returns_strings(self):
        from ricenlp import word_tokenize, LANG_KM

        for token in word_tokenize("ភាសាខ្មែរ", lang=LANG_KM):
            self.assertIsInstance(token, str)

    def test_empty_string(self):
        from ricenlp import word_tokenize, LANG_KM

        result = word_tokenize("", lang=LANG_KM)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])


class TestKhmerSentTokenize(unittest.TestCase):
    def test_basic(self):
        from ricenlp import sent_tokenize, LANG_KM

        result = sent_tokenize("ភាសាខ្មែរ", lang=LANG_KM)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_multiple_sentences(self):
        from ricenlp import sent_tokenize, LANG_KM

        text = "ខ្ញុំស្រលាញ់ប្រទេសខ្មែរ។ ប្រទេសខ្មែរល្អណាស់។"
        result = sent_tokenize(text, lang=LANG_KM)
        self.assertEqual(len(result), 2)

    def test_no_punctuation(self):
        from ricenlp import sent_tokenize, LANG_KM

        text = "ភាសាខ្មែរ"
        result = sent_tokenize(text, lang=LANG_KM)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], text)


class TestKhmerPosTag(unittest.TestCase):
    def test_raises_not_implemented(self):
        from ricenlp import pos_tag, LANG_KM

        with self.assertRaises(NotImplementedError):
            pos_tag("ភាសាខ្មែរ", lang=LANG_KM)


if __name__ == "__main__":
    unittest.main()
