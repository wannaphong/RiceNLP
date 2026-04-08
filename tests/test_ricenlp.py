"""Tests for the top-level ricenlp package."""

import unittest


class TestRiceNLPImport(unittest.TestCase):
    """Ensure the package can be imported and exposes the expected public API."""

    def test_import(self):
        import ricenlp  # noqa: F401

    def test_version(self):
        import ricenlp

        self.assertIsInstance(ricenlp.__version__, str)

    def test_language_constants(self):
        import ricenlp

        self.assertEqual(ricenlp.LANG_TH, "th")
        self.assertEqual(ricenlp.LANG_VI, "vi")
        self.assertEqual(ricenlp.LANG_LO, "lo")

    def test_supported_languages(self):
        import ricenlp

        self.assertIn("th", ricenlp.SUPPORTED_LANGUAGES)
        self.assertIn("vi", ricenlp.SUPPORTED_LANGUAGES)
        self.assertIn("lo", ricenlp.SUPPORTED_LANGUAGES)

    def test_public_api(self):
        import ricenlp

        for attr in ("word_tokenize", "sent_tokenize", "pos_tag"):
            self.assertTrue(
                callable(getattr(ricenlp, attr, None)),
                msg=f"ricenlp.{attr} should be callable",
            )

    def test_unsupported_language_raises(self):
        import ricenlp

        with self.assertRaises(ValueError):
            ricenlp.word_tokenize("hello", lang="xx")


if __name__ == "__main__":
    unittest.main()
