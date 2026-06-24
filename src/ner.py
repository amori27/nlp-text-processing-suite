"""Named Entity Recognition Module.

This module provides NER capabilities using spaCy
for extracting named entities from text.
"""

from typing import Any


class NERExtractor:
    """Handles named entity recognition."""

    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize the NERExtractor.

        Args:
            model: spaCy model name.
        """
        self.model = model
        self._load_model()

    def _load_model(self) -> None:
        """Load the spaCy model."""
        import spacy
        self.nlp = spacy.load(self.model)

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from text.

        Args:
            text: Input text.

        Returns:
            List of extracted entities.
        """
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "label_description": self.nlp.entity.labels.get(ent.label_, "Unknown"),
                "start": ent.start_char,
                "end": ent.end_char
            })

        return entities

    def extract_by_type(self, text: str, entity_types: list[str]) -> list[dict[str, Any]]:
        """Extract entities of specific types.

        Args:
            text: Input text.
            entity_types: List of entity types to extract.

        Returns:
            Filtered list of entities.
        """
        all_entities = self.extract(text)
        return [e for e in all_entities if e["label"] in entity_types]

    def get_unique_entities(self, texts: list[str]) -> set[str]:
        """Get unique entities from multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            Set of unique entity texts.
        """
        unique = set()
        for text in texts:
            entities = self.extract(text)
            for ent in entities:
                unique.add(ent["text"])
        return unique

    def visualize_entities(self, text: str) -> dict[str, Any]:
        """Create visualization data for entities.

        Args:
            text: Input text.

        Returns:
            Visualization data.
        """
        entities = self.extract(text)

        entity_types = {}
        for ent in entities:
            label = ent["label"]
            if label not in entity_types:
                entity_types[label] = {"count": 0, "entities": []}
            entity_types[label]["count"] += 1
            entity_types[label]["entities"].append(ent["text"])

        return {
            "text": text,
            "total_entities": len(entities),
            "entity_types": entity_types,
            "entities": entities
        }


def extract_people(text: str, ner: NERExtractor) -> list[str]:
    """Extract person names from text.

    Args:
        text: Input text.
        ner: NERExtractor instance.

    Returns:
        List of person names.
    """
    entities = ner.extract_by_type(text, ["PERSON", "PER"])
    return [e["text"] for e in entities]


def extract_organizations(text: str, ner: NERExtractor) -> list[str]:
    """Extract organization names from text.

    Args:
        text: Input text.
        ner: NERExtractor instance.

    Returns:
        List of organization names.
    """
    entities = ner.extract_by_type(text, ["ORG"])
    return [e["text"] for e in entities]


def extract_locations(text: str, ner: NERExtractor) -> list[str]:
    """Extract location names from text.

    Args:
        text: Input text.
        ner: NERExtractor instance.

    Returns:
        List of location names.
    """
    entities = ner.extract_by_type(text, ["GPE", "LOC", "FAC"])
    return [e["text"] for e in entities]
