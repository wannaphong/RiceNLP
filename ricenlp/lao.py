"""Lao NLP module – backed by `laonlp`."""

try:
    import laonlp
    from laonlp.tokenize import word_tokenize as _lo_word_tokenize

    _LAONLP_AVAILABLE = True
except ImportError:
    _LAONLP_AVAILABLE = False

_MISSING_MSG = (
    "laonlp is required for Lao NLP. "
    "Install it with: pip install laonlp"
)

# laonlp does not currently provide a sentence tokenizer or POS tagger,
# so we provide basic fallbacks.

_SENT_SPLIT_CHARS = {".", "!", "?", "។", "၊", "။"}


def word_tokenize(text: str) -> list:
    """Tokenize Lao *text* into words using laonlp.

    Parameters
    ----------
    text : str
        Lao input text.

    Returns
    -------
    list of str
    """
    if not _LAONLP_AVAILABLE:
        raise ImportError(_MISSING_MSG)
    return _lo_word_tokenize(text)


def sent_tokenize(text: str) -> list:
    """Tokenize Lao *text* into sentences.

    Uses a simple punctuation-based splitter as laonlp does not provide a
    dedicated sentence tokenizer.

    Parameters
    ----------
    text : str
        Lao input text.

    Returns
    -------
    list of str
    """
    if not _LAONLP_AVAILABLE:
        raise ImportError(_MISSING_MSG)
    sentences = []
    current = []
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


def pos_tag(text: str) -> list:
    """Return POS tags for Lao *text*.

    laonlp does not provide a POS tagger; this function raises
    ``NotImplementedError``.

    Parameters
    ----------
    text : str
        Lao input text.
    """
    if not _LAONLP_AVAILABLE:
        raise ImportError(_MISSING_MSG)
    raise NotImplementedError(
        "POS tagging for Lao is not yet supported by laonlp."
    )
