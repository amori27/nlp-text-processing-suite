"""Sentiment Analysis Module.

This module provides sentiment analysis using multiple methods
including TextBlob, VADER, and transformer-based models.
"""

from typing import Any


class SentimentAnalyzer:
    """Handles sentiment analysis of text."""

    def __init__(self, method: str = "textblob"):
        """Initialize the SentimentAnalyzer.

        Args:
            method: Analysis method (textblob, vader, transformer).
        """
        self.method = method
        self._setup_analyzer()

    def _setup_analyzer(self) -> None:
        """Setup the chosen analyzer."""
        if self.method == "textblob":
            from textblob import TextBlob
        elif self.method == "vader":
            from nltk.sentiment import SentimentIntensityAnalyzer

    def analyze(self, text: str) -> dict[str, Any]:
        """Analyze sentiment of text.

        Args:
            text: Input text.

        Returns:
            Sentiment analysis results.
        """
        if self.method == "textblob":
            return self._analyze_textblob(text)
        elif self.method == "vader":
            return self._analyze_vader(text)
        else:
            return {"error": "Unknown method"}

    def _analyze_textblob(self, text: str) -> dict[str, Any]:
        """Analyze sentiment using TextBlob.

        Args:
            text: Input text.

        Returns:
            Sentiment results.
        """
        from textblob import TextBlob

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "text": text,
            "polarity": float(polarity),
            "subjectivity": float(subjectivity),
            "sentiment": sentiment,
            "method": "textblob"
        }

    def _analyze_vader(self, text: str) -> dict[str, Any]:
        """Analyze sentiment using VADER.

        Args:
            text: Input text.

        Returns:
            Sentiment results.
        """
        from nltk.sentiment import SentimentIntensityAnalyzer

        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)

        compound = scores["compound"]
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "text": text,
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"],
            "compound": compound,
            "sentiment": sentiment,
            "method": "vader"
        }

    def batch_analyze(self, texts: list[str]) -> list[dict[str, Any]]:
        """Analyze sentiment for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of sentiment results.
        """
        return [self.analyze(text) for text in texts]


def get_sentiment_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate summary statistics from sentiment results.

    Args:
        results: List of sentiment analysis results.

    Returns:
        Summary statistics.
    """
    sentiments = [r["sentiment"] for r in results]

    return {
        "total": len(results),
        "positive": sentiments.count("positive"),
        "negative": sentiments.count("negative"),
        "neutral": sentiments.count("neutral"),
        "positive_pct": sentiments.count("positive") / len(results) * 100,
        "negative_pct": sentiments.count("negative") / len(results) * 100,
        "neutral_pct": sentiments.count("neutral") / len(results) * 100
    }
