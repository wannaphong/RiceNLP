"""Tests for Burmese NLP module (pure Python, no external dependency)."""

import unittest


class TestSyllableTokenize(unittest.TestCase):
    def test_basic(self):
        from ricenlp.burmese import syllable_tokenize

        result = syllable_tokenize("မြန်မာဘာသာ")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_returns_strings(self):
        from ricenlp.burmese import syllable_tokenize

        for token in syllable_tokenize("မြန်မာ"):
            self.assertIsInstance(token, str)

    def test_empty_string(self):
        from ricenlp.burmese import syllable_tokenize

        self.assertEqual(syllable_tokenize(""), [])

    def test_non_burmese_passthrough(self):
        from ricenlp.burmese import syllable_tokenize

        result = syllable_tokenize("hello")
        self.assertEqual(result, list("hello"))


class TestDictWordTokenizeFMM(unittest.TestCase):
    """Forward Maximum Matching (dict_word_tokenize)."""

    def setUp(self):
        self.dictionary = {"မြန်မာ", "ဘာသာ", "စကား"}

    def test_basic(self):
        from ricenlp.burmese import dict_word_tokenize

        result = dict_word_tokenize("မြန်မာဘာသာ", self.dictionary)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_known_words_matched(self):
        from ricenlp.burmese import dict_word_tokenize

        result = dict_word_tokenize("မြန်မာဘာသာ", self.dictionary)
        self.assertIn("မြန်မာ", result)
        self.assertIn("ဘာသာ", result)

    def test_empty_string(self):
        from ricenlp.burmese import dict_word_tokenize

        self.assertEqual(dict_word_tokenize("", self.dictionary), [])

    def test_empty_dictionary(self):
        from ricenlp.burmese import dict_word_tokenize

        result = dict_word_tokenize("abc", set())
        self.assertEqual(result, list("abc"))


class TestDictWordTokenizeMin(unittest.TestCase):
    """Forward Minimum Matching (dict_word_tokenize_min)."""

    def setUp(self):
        self.dictionary = {"မြန်မာ", "ဘာသာ", "မြန်မာဘာသာ"}

    def test_basic(self):
        from ricenlp.burmese import dict_word_tokenize_min

        result = dict_word_tokenize_min("မြန်မာဘာသာ", self.dictionary)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_prefers_shorter_match(self):
        from ricenlp.burmese import dict_word_tokenize_min

        # FMinM picks the shortest match first
        result = dict_word_tokenize_min("မြန်မာဘာသာ", self.dictionary)
        self.assertEqual(result[0], "မြန်မာ")


class TestDictWordTokenizeRev(unittest.TestCase):
    """Backward Maximum Matching (dict_word_tokenize_rev)."""

    def setUp(self):
        self.dictionary = {"မြန်မာ", "ဘာသာ", "စကား"}

    def test_basic(self):
        from ricenlp.burmese import dict_word_tokenize_rev

        result = dict_word_tokenize_rev("မြန်မာဘာသာ", self.dictionary)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_known_words_matched(self):
        from ricenlp.burmese import dict_word_tokenize_rev

        result = dict_word_tokenize_rev("မြန်မာဘာသာ", self.dictionary)
        self.assertIn("မြန်မာ", result)
        self.assertIn("ဘာသာ", result)

    def test_empty_string(self):
        from ricenlp.burmese import dict_word_tokenize_rev

        self.assertEqual(dict_word_tokenize_rev("", self.dictionary), [])


class TestDictWordTokenizeRevMin(unittest.TestCase):
    """Backward Minimum Matching (dict_word_tokenize_rev_min)."""

    def setUp(self):
        self.dictionary = {"မြန်မာ", "ဘာသာ", "မြန်မာဘာသာ"}

    def test_basic(self):
        from ricenlp.burmese import dict_word_tokenize_rev_min

        result = dict_word_tokenize_rev_min("မြန်မာဘာသာ", self.dictionary)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_empty_string(self):
        from ricenlp.burmese import dict_word_tokenize_rev_min

        self.assertEqual(dict_word_tokenize_rev_min("", self.dictionary), [])


class TestBurmeseWordTokenize(unittest.TestCase):
    """Top-level word_tokenize uses syllable segmentation (no external dep)."""

    def test_basic(self):
        from ricenlp import word_tokenize, LANG_MY

        result = word_tokenize("မြန်မာဘာသာ", lang=LANG_MY)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_returns_strings(self):
        from ricenlp import word_tokenize, LANG_MY

        for token in word_tokenize("မြန်မာ", lang=LANG_MY):
            self.assertIsInstance(token, str)

    def test_empty_string(self):
        from ricenlp import word_tokenize, LANG_MY

        result = word_tokenize("", lang=LANG_MY)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])


class TestBurmeseSentTokenize(unittest.TestCase):
    def test_basic(self):
        from ricenlp import sent_tokenize, LANG_MY

        result = sent_tokenize("မြန်မာဘာသာ", lang=LANG_MY)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_multiple_sentences(self):
        from ricenlp import sent_tokenize, LANG_MY

        text = "မြန်မာဘာသာ။ ငါတို့ဘာသာ။"
        result = sent_tokenize(text, lang=LANG_MY)
        self.assertEqual(len(result), 2)

    def test_no_punctuation(self):
        from ricenlp import sent_tokenize, LANG_MY

        text = "မြန်မာ"
        result = sent_tokenize(text, lang=LANG_MY)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], text)


class TestBurmesePosTag(unittest.TestCase):
    def test_raises_not_implemented(self):
        from ricenlp import pos_tag, LANG_MY

        with self.assertRaises(NotImplementedError):
            pos_tag("မြန်မာ", lang=LANG_MY)


if __name__ == "__main__":
    unittest.main()
