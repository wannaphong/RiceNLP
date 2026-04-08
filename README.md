# RiceNLP

[![CI](https://github.com/wannaphong/RiceNLP/actions/workflows/ci.yml/badge.svg)](https://github.com/wannaphong/RiceNLP/actions/workflows/ci.yml)

RiceNLP: Southeast Asia Natural Language Processing

A Python library for NLP tasks across Southeast Asian languages, inspired by [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) and [NLTK](https://www.nltk.org/).

## Supported Languages

| Language            | Code | Backend                                                                           | Notes |
|---------------------|------|-----------------------------------------------------------------------------------|-------|
| Thai                | `th` | [pythainlp](https://github.com/PyThaiNLP/pythainlp)                              | |
| Vietnamese          | `vi` | [underthesea](https://github.com/undertheseanlp/underthesea)                     | |
| Lao                 | `lo` | [laonlp](https://github.com/wannaphong/lao-nlp)                                  | |
| Khmer (Cambodian)   | `km` | Pure Python (no extra dependency)                                                 | KCC & dict-based segmentation ported from SEANLP |
| Burmese (Myanmar)   | `my` | Pure Python (no extra dependency)                                                 | Syllable segmentation & dict-based segmentation ported from SEANLP |

## Installation

```bash
# Install with all language backends
pip install ricenlp[all]

# Or install only what you need
pip install ricenlp[thai]
pip install ricenlp[vietnamese]
pip install ricenlp[lao]
# Note: Khmer and Burmese support is built-in (pure Python, no extra install needed)
```

## Quick Start

```python
import ricenlp

# Word tokenization
ricenlp.word_tokenize("สวัสดีครับ", lang="th")        # Thai
ricenlp.word_tokenize("Xin chào", lang="vi")           # Vietnamese
ricenlp.word_tokenize("ສະບາຍດີ", lang="lo")            # Lao
ricenlp.word_tokenize("ភាសាខ្មែរ", lang="km")          # Khmer
ricenlp.word_tokenize("မြန်မာဘာသာ", lang="my")         # Burmese

# Sentence tokenization
ricenlp.sent_tokenize("ฉันรักประเทศไทย ประเทศไทยสวยงาม", lang="th")
ricenlp.sent_tokenize("Tôi yêu Việt Nam. Việt Nam rất đẹp.", lang="vi")
ricenlp.sent_tokenize("ສະບາຍດີ. ຂອບໃຈ.", lang="lo")
ricenlp.sent_tokenize("ភាសាខ្មែរ", lang="km")
ricenlp.sent_tokenize("မြန်မာဘာသာ။ ငါတို့ဘာသာ။", lang="my")

# POS tagging
ricenlp.pos_tag("ฉันรักประเทศไทย", lang="th")
ricenlp.pos_tag("Tôi yêu Việt Nam", lang="vi")
ricenlp.pos_tag("ភាសាខ្មែរ", lang="km")
```

### Khmer-specific: KCC segmentation & dict-based tokenization

```python
from ricenlp.khmer import kcc_tokenize, dict_word_tokenize

# KCC (Khmer Character Cluster) segmentation – ported from SEANLP, no extra dep
kcc_tokenize("ភាសាខ្មែរ")

# Dict-based Forward Maximum Matching – ported from SEANLP
my_dict = {"ភាសា", "ខ្មែរ"}
dict_word_tokenize("ភាសាខ្មែរ", my_dict)
```

### Burmese-specific: syllable tokenization & dict-based tokenization

```python
from ricenlp.burmese import syllable_tokenize, dict_word_tokenize

# Syllable segmentation (ported from SEANLP, no extra dep)
syllable_tokenize("မြန်မာဘာသာ")

# Dict-based Forward Maximum Matching – ported from SEANLP
my_dict = {"မြန်မာ", "ဘာသာ"}
dict_word_tokenize("မြန်မာဘာသာ", my_dict)
```

## Language Constants

```python
ricenlp.LANG_TH  # "th"
ricenlp.LANG_VI  # "vi"
ricenlp.LANG_LO  # "lo"
ricenlp.LANG_KM  # "km"
ricenlp.LANG_MY  # "my"
```

## Running Tests

```bash
pip install ricenlp[all,dev]
pytest tests/ -v
```

## License

Apache-2.0

