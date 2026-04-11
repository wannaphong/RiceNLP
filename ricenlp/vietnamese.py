"""Vietnamese NLP module – backed by `underthesea`."""

try:
    import underthesea
    from underthesea import word_tokenize as _vi_word_tokenize
    from underthesea import sent_tokenize as _vi_sent_tokenize
    from underthesea import pos_tag as _vi_pos_tag

    _UNDERTHESEA_AVAILABLE = True
except ImportError:
    _UNDERTHESEA_AVAILABLE = False

_MISSING_MSG = (
    "underthesea is required for Vietnamese NLP. "
    "Install it with: pip install underthesea"
)


def word_tokenize(text: str) -> list:
    """Tokenize Vietnamese *text* into words using underthesea.

    Parameters
    ----------
    text : str
        Vietnamese input text.

    Returns
    -------
    list of str
    """
    if not _UNDERTHESEA_AVAILABLE:
        raise ImportError(_MISSING_MSG)
    return _vi_word_tokenize(text, format="list")


def sent_tokenize(text: str) -> list:
    """Tokenize Vietnamese *text* into sentences using underthesea.

    Parameters
    ----------
    text : str
        Vietnamese input text.

    Returns
    -------
    list of str
    """
    if not _UNDERTHESEA_AVAILABLE:
        raise ImportError(_MISSING_MSG)
    return _vi_sent_tokenize(text)


def pos_tag(text: str) -> list:
    """Return POS tags for Vietnamese *text* using underthesea.

    Parameters
    ----------
    text : str
        Vietnamese input text.

    Returns
    -------
    list of tuple (str, str)
    """
    if not _UNDERTHESEA_AVAILABLE:
        raise ImportError(_MISSING_MSG)
    return _vi_pos_tag(text)
