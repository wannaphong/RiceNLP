"""Thai NLP module – backed by `pythainlp`."""

try:
    import pythainlp
    from pythainlp.tokenize import word_tokenize as _th_word_tokenize
    from pythainlp.tokenize import sent_tokenize as _th_sent_tokenize
    from pythainlp.tag import pos_tag as _th_pos_tag

    _PYTHAINLP_AVAILABLE = True
except ImportError:
    _PYTHAINLP_AVAILABLE = False

_MISSING_MSG = (
    "pythainlp is required for Thai NLP. "
    "Install it with: pip install pythainlp"
)


def word_tokenize(text: str) -> list:
    """Tokenize Thai *text* into words using pythainlp.

    Parameters
    ----------
    text : str
        Thai input text.

    Returns
    -------
    list of str
    """
    if not _PYTHAINLP_AVAILABLE:
        raise ImportError(_MISSING_MSG)
    return _th_word_tokenize(text)


def sent_tokenize(text: str, engine: str = "whitespace+newline") -> list:
    """Tokenize Thai *text* into sentences using pythainlp.

    Parameters
    ----------
    text : str
        Thai input text.
    engine : str
        Sentence tokenization engine passed to pythainlp.  The default
        ``'whitespace+newline'`` requires no extra dependencies.
        Use ``'crfcut'`` for better accuracy (requires ``pycrfsuite``).

    Returns
    -------
    list of str
    """
    if not _PYTHAINLP_AVAILABLE:
        raise ImportError(_MISSING_MSG)
    return _th_sent_tokenize(text, engine=engine)


def pos_tag(text: str) -> list:
    """Return POS tags for Thai *text* using pythainlp.

    Parameters
    ----------
    text : str
        Thai input text.

    Returns
    -------
    list of tuple (str, str)
    """
    if not _PYTHAINLP_AVAILABLE:
        raise ImportError(_MISSING_MSG)
    tokens = _th_word_tokenize(text)
    return _th_pos_tag(tokens)
