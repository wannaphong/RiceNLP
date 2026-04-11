"""Tagalog NLP module – backed by `calamanCy` (spaCy-based).

calamanCy provides spaCy pipelines for Tagalog trained on the TLUnified
dataset.  Models are downloaded from HuggingFace on first use.

Available models (pass as ``model`` argument):
  - ``"tl_calamancy_md-0.2.0"`` – medium, CPU-optimised (default, ~74 MB)
  - ``"tl_calamancy_lg-0.2.0"`` – large, CPU-optimised (~432 MB)
  - ``"tl_calamancy_trf-0.2.0"`` – transformer, GPU-optimised (~776 MB)
"""

from __future__ import annotations

from typing import Any

try:
    import calamancy as _calamancy

    _CALAMANCY_AVAILABLE = True
except ImportError:
    _CALAMANCY_AVAILABLE = False

_MISSING_MSG = (
    "calamanCy is required for Tagalog NLP. "
    "Install it with: pip install calamanCy"
)

_DEFAULT_MODEL = "tl_calamancy_md-0.2.0"

# Lazy-loaded spaCy Language pipeline; populated on first call.
_nlp_cache: dict[str, Any] = {}


def _get_nlp(model: str):
    """Return a cached (or freshly loaded) calamanCy pipeline for *model*."""
    if not _CALAMANCY_AVAILABLE:
        raise ImportError(_MISSING_MSG)
    if model not in _nlp_cache:
        try:
            _nlp_cache[model] = _calamancy.load(model)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load calamanCy model '{model}'. "
                "Ensure the model is downloaded first by running: "
                f"python -c \"import calamancy; calamancy.load('{model}')\" "
                "or by calling calamancy.load() in an environment with internet access."
            ) from exc
    return _nlp_cache[model]


def word_tokenize(text: str, model: str = _DEFAULT_MODEL) -> list[str]:
    """Tokenize Tagalog *text* into words using a calamanCy spaCy pipeline.

    Parameters
    ----------
    text : str
        Tagalog input text.
    model : str
        calamanCy model name.  Defaults to ``"tl_calamancy_md-0.2.0"``.

    Returns
    -------
    list of str
    """
    if not text:
        return []
    nlp = _get_nlp(model)
    doc = nlp(text)
    return [token.text for token in doc]


def sent_tokenize(text: str, model: str = _DEFAULT_MODEL) -> list[str]:
    """Tokenize Tagalog *text* into sentences using a calamanCy spaCy pipeline.

    Parameters
    ----------
    text : str
        Tagalog input text.
    model : str
        calamanCy model name.  Defaults to ``"tl_calamancy_md-0.2.0"``.

    Returns
    -------
    list of str
    """
    if not text:
        return []
    nlp = _get_nlp(model)
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def pos_tag(text: str, model: str = _DEFAULT_MODEL) -> list[tuple[str, str]]:
    """Return part-of-speech tags for Tagalog *text*.

    Uses the Universal Dependencies POS tags provided by calamanCy's tagger
    component (trained on the Tagalog Reference Grammar (TRG) and Ugnayan
    treebanks).

    Parameters
    ----------
    text : str
        Tagalog input text.
    model : str
        calamanCy model name.  Defaults to ``"tl_calamancy_md-0.2.0"``.

    Returns
    -------
    list of tuple (str, str)
        List of ``(token, POS-tag)`` pairs where the POS tag is a Universal
        Dependencies coarse-grained tag (e.g. ``"NOUN"``, ``"VERB"``).
    """
    if not text:
        return []
    nlp = _get_nlp(model)
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]
