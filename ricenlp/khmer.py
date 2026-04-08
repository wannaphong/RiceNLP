"""Khmer (Cambodian) NLP module.

All functionality is implemented in pure Python with no external dependencies,
ported from SEANLP (https://github.com/zhaoshiyu/SEANLP):

* **KCC segmentation** – rule-based *Khmer Character Cluster* segmentation,
  analogous to Thai TCC.  Groups each Khmer consonant (plus its optional
  subscript coeng-consonant, dependent vowel(s) and diacritics) into an
  indivisible cluster.

* **Dict-based word segmentation** – four matching strategies ported from
  SEANLP's ``maxSegment``, ``minSegment``, ``reMaxSegment`` and
  ``reMinSegment``:

  - :func:`dict_word_tokenize` – Forward Maximum Matching (FMM)
  - :func:`dict_word_tokenize_min` – Forward Minimum Matching (FMinM)
  - :func:`dict_word_tokenize_rev` – Backward Maximum Matching (BMM)
  - :func:`dict_word_tokenize_rev_min` – Backward Minimum Matching (BMinM)

The top-level :func:`word_tokenize` and :func:`sent_tokenize` require no
external package and use KCC segmentation and punctuation-based splitting
respectively.  :func:`pos_tag` raises ``NotImplementedError`` as no
dependency-free POS model is available.
"""

from __future__ import annotations

import importlib.resources
import re
from functools import lru_cache
from typing import Collection

# ---------------------------------------------------------------------------
# Default dictionary (sourced from google/language-resources km/data/lexicon.tsv)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_default_wordlist() -> frozenset[str]:
    """Load the bundled Khmer wordlist extracted from google/language-resources.

    The wordlist is loaded lazily and cached for subsequent calls.

    Returns
    -------
    frozenset of str
        The set of Khmer words from the bundled dictionary.
    """
    pkg = importlib.resources.files("ricenlp.data")
    text = (pkg / "khmer_wordlist.txt").read_text(encoding="utf-8")
    return frozenset(line for line in text.splitlines() if line)


def get_wordlist() -> frozenset[str]:
    """Return the bundled Khmer wordlist from google/language-resources.

    The list is extracted from ``km/data/lexicon.tsv`` in the
    `google/language-resources <https://github.com/google/language-resources>`_
    project (CC-BY 4.0).

    Returns
    -------
    frozenset of str
        Khmer words included in the bundled dictionary.
    """
    return _load_default_wordlist()


# ---------------------------------------------------------------------------
# KCC segmentation (ported from SEANLP's KCCSegmentor logic)
# ---------------------------------------------------------------------------

# Khmer Unicode ranges:
#   Consonants            U+1780–U+17A2
#   Independent vowels    U+17A3–U+17B5
#   Dependent vowels      U+17B6–U+17C5
#   Diacritics            U+17C6–U+17D1, U+17DD
#   Coeng (subscript)     U+17D2  (must precede another consonant)
#   Digits                U+17E0–U+17E9

_KHMER_CONSONANT = r"[\u1780-\u17A2]"
_KHMER_COENG = r"\u17D2" + _KHMER_CONSONANT  # subscript consonant
_KHMER_DEP_VOWEL = r"[\u17B6-\u17C5]"
_KHMER_DIACRITIC = r"[\u17C6-\u17D1\u17DD]"
_KHMER_INDEP_VOWEL = r"[\u17A3-\u17B5]"
_KHMER_DIGIT = r"[\u17E0-\u17E9]"

# A KCC is:
#   1. A base consonant, optionally followed by one subscript consonant
#      (coeng), an optional dependent vowel, and optional diacritics.
#   2. An independent vowel (stands alone).
#   3. A Khmer digit.
#   4. Any other character (pass-through).
_KCC_PATTERN = re.compile(
    _KHMER_CONSONANT
    + r"(?:"
    + _KHMER_COENG
    + r")*"
    + r"(?:"
    + _KHMER_DEP_VOWEL
    + r")?"
    + r"(?:"
    + _KHMER_DIACRITIC
    + r")*"
    + r"|"
    + _KHMER_INDEP_VOWEL
    + r"|"
    + _KHMER_DIGIT
    + r"|.",
    re.DOTALL,
)

_SENT_SPLIT_CHARS = {".", "!", "?", "។", "៕", "៖"}


def kcc_tokenize(text: str) -> list[str]:
    """Segment *text* into Khmer Character Clusters (KCC).

    Rule-based segmentation that groups a Khmer consonant with its dependent
    vowels and diacritics into an indivisible cluster, analogous to Thai TCC.
    Ported from SEANLP's KCC segmentor.

    Parameters
    ----------
    text : str
        Khmer input text.

    Returns
    -------
    list of str
        List of KCC tokens.
    """
    return _KCC_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Dict-based word segmentation
# Ported from SEANLP's maxSegment / minSegment / reMaxSegment / reMinSegment
# ---------------------------------------------------------------------------


def _fmm(text: str, vocab: set[str], max_len: int) -> list[str]:
    """Forward Maximum Matching (greedy longest-match left-to-right)."""
    tokens: list[str] = []
    i = 0
    while i < len(text):
        end = min(i + max_len, len(text))
        matched = False
        while end > i:
            candidate = text[i:end]
            if candidate in vocab:
                tokens.append(candidate)
                i = end
                matched = True
                break
            end -= 1
        if not matched:
            tokens.append(text[i])
            i += 1
    return tokens


def _fminm(text: str, vocab: set[str]) -> list[str]:
    """Forward Minimum Matching (greedy shortest-match left-to-right)."""
    tokens: list[str] = []
    i = 0
    while i < len(text):
        matched = False
        for end in range(i + 1, len(text) + 1):
            candidate = text[i:end]
            if candidate in vocab:
                tokens.append(candidate)
                i = end
                matched = True
                break
        if not matched:
            tokens.append(text[i])
            i += 1
    return tokens


def _bmm(text: str, vocab: set[str], max_len: int) -> list[str]:
    """Backward Maximum Matching (greedy longest-match right-to-left)."""
    tokens: list[str] = []
    i = len(text)
    while i > 0:
        start = max(i - max_len, 0)
        matched = False
        while start < i:
            candidate = text[start:i]
            if candidate in vocab:
                tokens.append(candidate)
                i = start
                matched = True
                break
            start += 1
        if not matched:
            tokens.append(text[i - 1])
            i -= 1
    tokens.reverse()
    return tokens


def _bminm(text: str, vocab: set[str]) -> list[str]:
    """Backward Minimum Matching (greedy shortest-match right-to-left)."""
    tokens: list[str] = []
    i = len(text)
    while i > 0:
        matched = False
        for start in range(i - 1, -1, -1):
            candidate = text[start:i]
            if candidate in vocab:
                tokens.append(candidate)
                i = start
                matched = True
                break
        if not matched:
            tokens.append(text[i - 1])
            i -= 1
    tokens.reverse()
    return tokens


def dict_word_tokenize(
    text: str,
    dictionary: Collection[str] | None = None,
) -> list[str]:
    """Tokenize *text* using Forward Maximum Matching (FMM).

    Ported from SEANLP's ``maxSegment``.

    Parameters
    ----------
    text : str
        Khmer input text.
    dictionary : collection of str, optional
        Set / list of known Khmer words.  When *None* (the default) the
        bundled wordlist from google/language-resources is used.

    Returns
    -------
    list of str
    """
    vocab = set(dictionary) if dictionary is not None else _load_default_wordlist()
    if not vocab:
        return list(text)
    max_len = max(len(w) for w in vocab)
    return _fmm(text, vocab, max_len)


def dict_word_tokenize_min(
    text: str,
    dictionary: Collection[str] | None = None,
) -> list[str]:
    """Tokenize *text* using Forward Minimum Matching (FMinM).

    Ported from SEANLP's ``minSegment``.

    Parameters
    ----------
    text : str
        Khmer input text.
    dictionary : collection of str, optional
        Set / list of known Khmer words.  When *None* (the default) the
        bundled wordlist from google/language-resources is used.

    Returns
    -------
    list of str
    """
    vocab = set(dictionary) if dictionary is not None else _load_default_wordlist()
    if not vocab:
        return list(text)
    return _fminm(text, vocab)


def dict_word_tokenize_rev(
    text: str,
    dictionary: Collection[str] | None = None,
) -> list[str]:
    """Tokenize *text* using Backward Maximum Matching (BMM).

    Ported from SEANLP's ``reMaxSegment``.

    Parameters
    ----------
    text : str
        Khmer input text.
    dictionary : collection of str, optional
        Set / list of known Khmer words.  When *None* (the default) the
        bundled wordlist from google/language-resources is used.

    Returns
    -------
    list of str
    """
    vocab = set(dictionary) if dictionary is not None else _load_default_wordlist()
    if not vocab:
        return list(text)
    max_len = max(len(w) for w in vocab)
    return _bmm(text, vocab, max_len)


def dict_word_tokenize_rev_min(
    text: str,
    dictionary: Collection[str] | None = None,
) -> list[str]:
    """Tokenize *text* using Backward Minimum Matching (BMinM).

    Ported from SEANLP's ``reMinSegment``.

    Parameters
    ----------
    text : str
        Khmer input text.
    dictionary : collection of str, optional
        Set / list of known Khmer words.  When *None* (the default) the
        bundled wordlist from google/language-resources is used.

    Returns
    -------
    list of str
    """
    vocab = set(dictionary) if dictionary is not None else _load_default_wordlist()
    if not vocab:
        return list(text)
    return _bminm(text, vocab)


# ---------------------------------------------------------------------------
# Top-level API (no external dependency)
# ---------------------------------------------------------------------------


def word_tokenize(text: str) -> list[str]:
    """Tokenize Khmer *text* into word-level tokens using KCC segmentation.

    Uses KCC (Khmer Character Cluster) segmentation — a pure-Python rule-based
    algorithm ported from SEANLP.  For dictionary-guided word segmentation use
    :func:`dict_word_tokenize` and its variants.

    Parameters
    ----------
    text : str
        Khmer input text.

    Returns
    -------
    list of str
    """
    return kcc_tokenize(text)


def sent_tokenize(text: str) -> list[str]:
    """Tokenize Khmer *text* into sentences using punctuation-based splitting.

    Parameters
    ----------
    text : str
        Khmer input text.

    Returns
    -------
    list of str
    """
    sentences: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in _SENT_SPLIT_CHARS:
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    remainder = "".join(current).strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def pos_tag(text: str) -> list[tuple[str, str]]:
    """Return POS tags for Khmer *text*.

    No dependency-free POS tagger is currently available; this function raises
    ``NotImplementedError``.

    Parameters
    ----------
    text : str
        Khmer input text.
    """
    raise NotImplementedError(
        "POS tagging for Khmer is not yet supported without an external model."
    )
