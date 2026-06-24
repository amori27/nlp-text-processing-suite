"""Text Preprocessing Module.

This module provides comprehensive text preprocessing utilities
including cleaning, normalization, and tokenization.
"""

import re
import nltk
from typing import Any


class TextPreprocessor:
    """Handles text preprocessing operations."""

    def __init__(self, lowercase: bool = True):
        """Initialize the TextPreprocessor.

        Args:
            lowercase: Whether to convert text to lowercase.
        """
        self.lowercase = lowercase
        self.nltk_stopwords = set(nltk.corpus.stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Input text.

        Returns:
            Cleaned text.
        """
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^\w\s\.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text)

        if self.lowercase:
            text = text.lower()

        return text.strip()

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove stopwords from token list.

        Args:
            tokens: List of tokens.

        Returns:
            Filtered token list.
        """
        return [t for t in tokens if t.lower() not in self.nltk_stopwords]

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words.

        Args:
            text: Input text.

        Returns:
            List of tokens.
        """
        cleaned = self.clean_text(text)
        tokens = re.findall(r'\b\w+\b', cleaned)
        return tokens

    def stem(self, tokens: list[str], algorithm: str = "porter") -> list[str]:
        """Apply stemming to tokens.

        Args:
            tokens: List of tokens.
            algorithm: Stemming algorithm (porter, lancaster, snowball).

        Returns:
            Stemmed tokens.
        """
        stemmer = nltk.stem.PorterStemmer()

        if algorithm == "lancaster":
            stemmer = nltk.stem.LancasterStemmer()
        elif algorithm == "snowball":
            stemmer = nltk.stem.SnowballStemmer("english")

        return [stemmer.stem(t) for t in tokens]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Apply lemmatization to tokens.

        Args:
            tokens: List of tokens.

        Returns:
            Lemmatized tokens.
        """
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return [lemmatizer.lemmatize(t) for t in tokens]

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text.

        Args:
            text: Input text.

        Returns:
            Text without punctuation.
        """
        return re.sub(r'[^\w\s]', '', text)

    def remove_numbers(self, text: str) -> str:
        """Remove numerical characters from text.

        Args:
            text: Input text.

        Returns:
            Text without numbers.
        """
        return re.sub(r'\d+', '', text)

    def extract_hashtags(self, text: str) -> list[str]:
        """Extract hashtags from text.

        Args:
            text: Input text.

        Returns:
            List of hashtags.
        """
        return re.findall(r'#(\w+)', text)

    def extract_mentions(self, text: str) -> list[str]:
        """Extract mentions from text.

        Args:
            text: Input text.

        Returns:
            List of mentions.
        """
        return re.findall(r'@(\w+)', text)

    def preprocess_pipeline(self, text: str) -> list[str]:
        """Run full preprocessing pipeline.

        Args:
            text: Input text.

        Returns:
            Preprocessed tokens.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens


def batch_preprocess(texts: list[str]) -> list[list[str]]:
    """Preprocess multiple texts.

    Args:
        texts: List of input texts.

    Returns:
        List of preprocessed token lists.
    """
    preprocessor = TextPreprocessor()
    return [preprocessor.preprocess_pipeline(t) for t in texts]
