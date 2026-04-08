"""Burmese (Myanmar) NLP module.

All functionality is implemented in pure Python with no external dependencies,
ported from SEANLP (https://github.com/zhaoshiyu/SEANLP):

* **Syllable segmentation** – rule-based Myanmar orthographic syllable
  segmentation using Unicode character-class patterns.  Each base consonant
  (or independent vowel) is grouped with its following subscript consonant,
  medials, vowel signs, and diacritics to form an indivisible syllable cluster.

* **Dict-based word segmentation** – four matching strategies ported from
  SEANLP's ``maxSegment``, ``minSegment``, ``reMaxSegment`` and
  ``reMinSegment``:

  - :func:`dict_word_tokenize`         – Forward Maximum Matching (FMM)
  - :func:`dict_word_tokenize_min`     – Forward Minimum Matching (FMinM)
  - :func:`dict_word_tokenize_rev`     – Backward Maximum Matching (BMM)
  - :func:`dict_word_tokenize_rev_min` – Backward Minimum Matching (BMinM)

The top-level :func:`word_tokenize` and :func:`sent_tokenize` require no
external package and use syllable segmentation and punctuation-based splitting
respectively.  :func:`pos_tag` raises ``NotImplementedError`` as no
dependency-free POS model is available.
"""

from __future__ import annotations

import re
from typing import Collection

# ---------------------------------------------------------------------------
# Syllable segmentation (ported from SEANLP's syllable segmentor)
# ---------------------------------------------------------------------------

# Myanmar Unicode character classes:
#   Consonants (U+1000–U+1021)
#   Independent vowels (U+1023–U+1027, U+1029–U+102A)
#   Vowel signs (U+102B–U+1032)
#   Subscript marker / stacker (U+1039) – precedes a stacked consonant
#   Medials / ya, ra, wa, ha (U+103B–U+103E)
#   Anusvara (U+1036), dot below (U+1037), visarga (U+1038)
#   Asat / killer (U+103A)
#   Kinzi prefix: U+1004 + U+103A + U+1039
#   Myanmar digits (U+1040–U+1049)

_MY_CONS = r"[\u1000-\u1021]"
_MY_INDEP_VOWEL = r"[\u1023-\u1027\u1029\u102A]"
_MY_SUBSCRIPT = r"\u1039" + _MY_CONS   # stacked consonant
_MY_KINZI = r"\u1004\u103A\u1039"      # kinzi (stacked nga)
_MY_MEDIAL = r"[\u103B-\u103E]"        # ya/ra/wa/ha medials
_MY_VOWEL_SIGN = r"[\u102B-\u1032]"   # dependent vowel signs
_MY_DIACRITIC = r"[\u1036\u1037\u1038\u103A]"  # anusvara, dot, visarga, asat
_MY_DIGIT = r"[\u1040-\u1049]"

# A Burmese syllable is:
#   1. Optional kinzi prefix
#   2. A base consonant or independent vowel
#   3. Optional stacked (subscript) consonant
#   4. Optional medials (ya/ra/wa/ha)
#   5. Optional vowel signs
#   6. Optional diacritics (anusvara, dot, visarga, asat/killer)
#
#   OR a Myanmar digit.
#   OR any other character (pass-through).
_SYLLABLE_PATTERN = re.compile(
    r"(?:"
    + _MY_KINZI
    + r")?"
    + r"(?:"
    + _MY_CONS
    + r"|"
    + _MY_INDEP_VOWEL
    + r")"
    + r"(?:"
    + _MY_SUBSCRIPT
    + r")*"
    + r"(?:"
    + _MY_MEDIAL
    + r")*"
    + r"(?:"
    + _MY_VOWEL_SIGN
    + r")*"
    + _MY_DIACRITIC
    + r"*"
    + r"|"
    + _MY_DIGIT
    + r"|.",
    re.DOTALL,
)

_SENT_SPLIT_CHARS = {".", "!", "?", "။", "၊"}


def syllable_tokenize(text: str) -> list[str]:
    """Segment Burmese *text* into orthographic syllable clusters.

    Rule-based segmentation that groups each Myanmar base consonant with its
    following subscript consonant, medials, vowel signs and diacritics.
    Ported from SEANLP's Burmese syllable segmentor.

    Parameters
    ----------
    text : str
        Burmese input text.

    Returns
    -------
    list of str
        List of syllable tokens.
    """
    return _SYLLABLE_PATTERN.findall(text)


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
    dictionary: Collection[str],
) -> list[str]:
    """Tokenize *text* using Forward Maximum Matching (FMM).

    Ported from SEANLP's ``maxSegment``.

    Parameters
    ----------
    text : str
        Burmese input text.
    dictionary : collection of str
        Set / list of known Burmese words.

    Returns
    -------
    list of str
    """
    vocab = set(dictionary)
    if not vocab:
        return list(text)
    max_len = max(len(w) for w in vocab)
    return _fmm(text, vocab, max_len)


def dict_word_tokenize_min(
    text: str,
    dictionary: Collection[str],
) -> list[str]:
    """Tokenize *text* using Forward Minimum Matching (FMinM).

    Ported from SEANLP's ``minSegment``.

    Parameters
    ----------
    text : str
        Burmese input text.
    dictionary : collection of str
        Set / list of known Burmese words.

    Returns
    -------
    list of str
    """
    vocab = set(dictionary)
    if not vocab:
        return list(text)
    return _fminm(text, vocab)


def dict_word_tokenize_rev(
    text: str,
    dictionary: Collection[str],
) -> list[str]:
    """Tokenize *text* using Backward Maximum Matching (BMM).

    Ported from SEANLP's ``reMaxSegment``.

    Parameters
    ----------
    text : str
        Burmese input text.
    dictionary : collection of str
        Set / list of known Burmese words.

    Returns
    -------
    list of str
    """
    vocab = set(dictionary)
    if not vocab:
        return list(text)
    max_len = max(len(w) for w in vocab)
    return _bmm(text, vocab, max_len)


def dict_word_tokenize_rev_min(
    text: str,
    dictionary: Collection[str],
) -> list[str]:
    """Tokenize *text* using Backward Minimum Matching (BMinM).

    Ported from SEANLP's ``reMinSegment``.

    Parameters
    ----------
    text : str
        Burmese input text.
    dictionary : collection of str
        Set / list of known Burmese words.

    Returns
    -------
    list of str
    """
    vocab = set(dictionary)
    if not vocab:
        return list(text)
    return _bminm(text, vocab)


# ---------------------------------------------------------------------------
# Top-level API (no external dependency)
# ---------------------------------------------------------------------------


def word_tokenize(text: str) -> list[str]:
    """Tokenize Burmese *text* into syllable-level tokens.

    Uses the rule-based syllable segmentation algorithm ported from SEANLP.
    For dictionary-guided word segmentation use :func:`dict_word_tokenize`
    and its variants.

    Parameters
    ----------
    text : str
        Burmese input text.

    Returns
    -------
    list of str
    """
    return syllable_tokenize(text)


def sent_tokenize(text: str) -> list[str]:
    """Tokenize Burmese *text* into sentences using punctuation-based splitting.

    Parameters
    ----------
    text : str
        Burmese input text.

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
    """Return POS tags for Burmese *text*.

    No dependency-free POS tagger is currently available; this function raises
    ``NotImplementedError``.

    Parameters
    ----------
    text : str
        Burmese input text.
    """
    raise NotImplementedError(
        "POS tagging for Burmese is not yet supported without an external model."
    )
