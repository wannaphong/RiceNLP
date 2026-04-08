"""
RiceNLP: Southeast Asia Natural Language Processing library.

Provides a unified API for NLP tasks across Southeast Asian languages:
  - Thai (th)       : powered by pythainlp
  - Vietnamese (vi) : powered by underthesea
  - Lao (lo)        : powered by laonlp
  - Khmer (km)      : pure Python, no external dependency (ported from SEANLP)
  - Burmese (my)    : pure Python, no external dependency (ported from SEANLP)
  - Tagalog (tl)    : powered by calamanCy (spaCy-based)
"""

__version__ = "0.1.0"
__author__ = "wannaphong"

# Language codes
LANG_TH = "th"
LANG_VI = "vi"
LANG_LO = "lo"
LANG_KM = "km"
LANG_MY = "my"
LANG_TL = "tl"

SUPPORTED_LANGUAGES = {LANG_TH, LANG_VI, LANG_LO, LANG_KM, LANG_MY, LANG_TL}


def _get_lang_module(lang: str):
    """Return the language-specific module for *lang*."""
    if lang == LANG_TH:
        from ricenlp import thai as _mod
    elif lang == LANG_VI:
        from ricenlp import vietnamese as _mod
    elif lang == LANG_LO:
        from ricenlp import lao as _mod
    elif lang == LANG_KM:
        from ricenlp import khmer as _mod
    elif lang == LANG_MY:
        from ricenlp import burmese as _mod
    elif lang == LANG_TL:
        from ricenlp import tagalog as _mod
    else:
        raise ValueError(
            f"Unsupported language: '{lang}'. "
            f"Supported languages are: {sorted(SUPPORTED_LANGUAGES)}"
        )
    return _mod


def word_tokenize(text: str, lang: str = LANG_TH) -> list:
    """Tokenize *text* into a list of words.

    Parameters
    ----------
    text : str
        Input text.
    lang : str
        Language code (``'th'``, ``'vi'``, ``'lo'``, ``'km'``, ``'my'``, or ``'tl'``).

    Returns
    -------
    list of str
        List of word tokens.
    """
    return _get_lang_module(lang).word_tokenize(text)


def sent_tokenize(text: str, lang: str = LANG_TH) -> list:
    """Tokenize *text* into a list of sentences.

    Parameters
    ----------
    text : str
        Input text.
    lang : str
        Language code (``'th'``, ``'vi'``, ``'lo'``, ``'km'``, ``'my'``, or ``'tl'``).

    Returns
    -------
    list of str
        List of sentence tokens.
    """
    return _get_lang_module(lang).sent_tokenize(text)


def pos_tag(text: str, lang: str = LANG_TH) -> list:
    """Return part-of-speech tags for each token in *text*.

    Parameters
    ----------
    text : str
        Input text.
    lang : str
        Language code (``'th'``, ``'vi'``, ``'lo'``, ``'km'``, ``'my'``, or ``'tl'``).

    Returns
    -------
    list of tuple (str, str)
        List of (token, POS-tag) pairs.
    """
    return _get_lang_module(lang).pos_tag(text)
