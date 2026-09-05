# NLP Text Processing Suite

A practical Python library for common NLP tasks: text preprocessing, sentiment analysis, and named entity recognition. Pluggable backends (TextBlob / VADER / spaCy / Transformers), no forced API keys.

---

## Modules

| Module | What it does |
|---|---|
| `preprocessor.TextPreprocessor` | Cleaning, normalization, stopword removal, tokenization |
| `sentiment.SentimentAnalyzer` | Sentiment via TextBlob, VADER, or transformer models |
| `ner.NERExtractor` | Named-entity extraction via spaCy (`en_core_web_sm` by default) |

---

## Install

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords vader_lexicon
python -m spacy download en_core_web_sm
```

---

## Usage

### Preprocessing

```python
from src.preprocessor import TextPreprocessor

p = TextPreprocessor(lowercase=True)

text = "<p>Hello WORLD! Visit https://example.com</p>"
clean = p.clean_text(text)             # "hello world visit"
tokens = p.tokenize(clean)
tokens = p.remove_stopwords(tokens)    # ["hello", "world", "visit"]
```

### Sentiment

```python
from src.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer(method="textblob")   # or "vader" / "transformer"
result = analyzer.analyze("I love this product, it works perfectly!")
# -> {"polarity": 0.6, "subjectivity": 0.4, "label": "positive"}
```

### Named-Entity Recognition

```python
from src.ner import NERExtractor

ner = NERExtractor(model="en_core_web_sm")
entities = ner.extract("Apple was founded by Steve Jobs in Cupertino in 1976.")
# -> [{"text": "Apple", "label": "ORG", ...},
#     {"text": "Steve Jobs", "label": "PERSON", ...},
#     {"text": "Cupertino", "label": "GPE", ...}]

# Filter by type
people = ner.extract_by_type(text, ["PERSON"])
```

---

## Project Structure

```
nlp-text-processing-suite/
├── src/
│   ├── preprocessor.py
│   ├── sentiment.py
│   └── ner.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## License

MIT — see [LICENSE](LICENSE).
