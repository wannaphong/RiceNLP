# RiceNLP

[![CI](https://github.com/wannaphong/RiceNLP/actions/workflows/ci.yml/badge.svg)](https://github.com/wannaphong/RiceNLP/actions/workflows/ci.yml)

RiceNLP: Southeast Asia Natural Language Processing

A Python library for NLP tasks across Southeast Asian languages, inspired by [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) and [NLTK](https://www.nltk.org/).

## Supported Languages

| Language   | Code | Backend                                                          |
|------------|------|------------------------------------------------------------------|
| Thai       | `th` | [pythainlp](https://github.com/PyThaiNLP/pythainlp)             |
| Vietnamese | `vi` | [underthesea](https://github.com/undertheseanlp/underthesea)    |
| Lao        | `lo` | [laonlp](https://github.com/wannaphong/lao-nlp)                 |

## Installation

```bash
# Install with all language backends
pip install ricenlp[all]

# Or install only what you need
pip install ricenlp[thai]
pip install ricenlp[vietnamese]
pip install ricenlp[lao]
```

## Quick Start

```python
import ricenlp

# Word tokenization
ricenlp.word_tokenize("สวัสดีครับ", lang="th")   # Thai
ricenlp.word_tokenize("Xin chào", lang="vi")      # Vietnamese
ricenlp.word_tokenize("ສະບາຍດີ", lang="lo")       # Lao

# Sentence tokenization
ricenlp.sent_tokenize("ฉันรักประเทศไทย ประเทศไทยสวยงาม", lang="th")
ricenlp.sent_tokenize("Tôi yêu Việt Nam. Việt Nam rất đẹp.", lang="vi")
ricenlp.sent_tokenize("ສະບາຍດີ. ຂອບໃຈ.", lang="lo")

# POS tagging
ricenlp.pos_tag("ฉันรักประเทศไทย", lang="th")
ricenlp.pos_tag("Tôi yêu Việt Nam", lang="vi")
```

## Language Constants

```python
ricenlp.LANG_TH  # "th"
ricenlp.LANG_VI  # "vi"
ricenlp.LANG_LO  # "lo"
```

## Running Tests

```bash
pip install ricenlp[all,dev]
pytest tests/ -v
```

## License

Apache-2.0

