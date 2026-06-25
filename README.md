# NLP Text Processing Suite
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


Comprehensive NLP toolkit featuring text preprocessing, sentiment analysis, named entity recognition (NER), text classification, and language processing pipelines.

## Description

Production-ready NLP utilities for text processing including cleaning, tokenization, POS tagging, named entity recognition, and sentiment analysis. Built with NLTK, spaCy, and transformers.

## Skills & Technologies

- Python 3.9+
- spaCy
- NLTK
- Transformers (Hugging Face)
- scikit-learn
- TextBlob
- Text Preprocessing
- Sentiment Analysis
- NER

## Installation

```bash
git clone https://github.com/AmirAsaad/nlp-text-processing-suite.git
cd nlp-text-processing-suite
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Text Preprocessing

```python
from src.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()
cleaned = preprocessor.clean_text("Hello! This is a TEST text with numbers 123.")
```

### Sentiment Analysis

```python
from src.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze("I love this product, it's amazing!")
```

### Named Entity Recognition

```python
from src.ner import NERExtractor

extractor = NERExtractor()
entities = extractor.extract("Apple Inc. was founded by Steve Jobs in California.")
```

## Project Structure

```
nlp-text-processing-suite/
├── src/
│   ├── preprocessor.py     # Text cleaning
│   ├── sentiment.py        # Sentiment analysis
│   ├── ner.py              # Named entity recognition
│   └── tokenizer.py        # Tokenization
├── requirements.txt
└── README.md
```

## References

- [spaCy Documentation](https://spacy.io/docs)
- [NLTK Documentation](https://www.nltk.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## License

MIT License
