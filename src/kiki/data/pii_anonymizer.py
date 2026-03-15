"""PII anonymization for the Kiki SLM pipeline.

Task 3.3: Detect and replace PII using Presidio + spaCy for entity detection
and Faker for synthetic replacement. Order/ticket IDs are preserved.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import datasets

logger = logging.getLogger(__name__)


@dataclass
class DetectedEntity:
    entity_type: str
    original_text: str
    replacement_text: str
    start: int
    end: int


# Custom patterns to KEEP (non-PII business identifiers)
_KEEP_PATTERNS = [
    re.compile(r"ORD-\w+", re.IGNORECASE),
    re.compile(r"TKT-\w+", re.IGNORECASE),
]


class PIIAnonymizer:
    """Anonymize PII in text using Presidio + Faker."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._analyzer: Any = None
        self._anonymizer: Any = None
        self._faker: Any = None
        self._init_engines()

    def _init_engines(self) -> None:
        from faker import Faker
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._faker = Faker()
        Faker.seed(self._seed)

    def _build_replacement(self, entity_type: str) -> str:
        replacements = {
            "PERSON": self._faker.name(),
            "EMAIL_ADDRESS": self._faker.email(),
            "PHONE_NUMBER": self._faker.phone_number(),
            "CREDIT_CARD": "[CARD_XXXX]",
            "US_SSN": "[SSN_REDACTED]",
            "IBAN_CODE": "[IBAN_REDACTED]",
            "IP_ADDRESS": self._faker.ipv4(),
            "LOCATION": self._faker.city(),
        }
        return replacements.get(entity_type, f"[{entity_type}_REDACTED]")

    def _is_kept_pattern(self, text: str) -> bool:
        return any(p.fullmatch(text) for p in _KEEP_PATTERNS)

    def anonymize_text(self, text: str) -> tuple[str, list[DetectedEntity]]:
        """Anonymize PII in text, returning (anonymized_text, detections)."""
        entities_to_detect = [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "US_SSN",
            "IBAN_CODE",
            "IP_ADDRESS",
            "LOCATION",
        ]

        results = self._analyzer.analyze(
            text=text,
            entities=entities_to_detect,
            language="en",
        )

        # Filter out kept patterns and DATE_TIME
        filtered = []
        for r in results:
            original = text[r.start : r.end]
            if self._is_kept_pattern(original):
                continue
            filtered.append(r)

        if not filtered:
            return text, []

        # Sort by position descending so replacements don't shift indices
        filtered.sort(key=lambda r: r.start, reverse=True)

        detections: list[DetectedEntity] = []
        anonymized = text
        for r in filtered:
            original = anonymized[r.start : r.end]
            replacement = self._build_replacement(r.entity_type)
            anonymized = anonymized[: r.start] + replacement + anonymized[r.end :]
            detections.append(
                DetectedEntity(
                    entity_type=r.entity_type,
                    original_text=original,
                    replacement_text=replacement,
                    start=r.start,
                    end=r.end,
                )
            )

        detections.reverse()  # Return in forward order
        return anonymized, detections

    def process_dataset(
        self,
        dataset: datasets.Dataset,
        text_columns: list[str],
        num_proc: int = 4,
    ) -> datasets.Dataset:
        """Anonymize PII in specified columns of a HuggingFace Dataset."""
        all_detections: list[list[DetectedEntity]] = []

        def _anonymize_row(example: dict) -> dict:
            row_detections: list[DetectedEntity] = []
            for col in text_columns:
                if col in example and example[col]:
                    anonymized, dets = self.anonymize_text(str(example[col]))
                    example[col] = anonymized
                    row_detections.extend(dets)
            return example

        logger.info("Anonymizing columns %s across %d rows (num_proc=%d)", text_columns, len(dataset), num_proc)
        result = dataset.map(_anonymize_row, num_proc=num_proc, desc="Anonymizing PII")

        total_entities = sum(len(d) for d in all_detections)
        logger.info("PII anonymization complete. Detected %d entities across the dataset.", total_entities)
        return result

    def generate_audit_report(self, detections: list[DetectedEntity]) -> dict[str, Any]:
        """Produce a summary of what was anonymized."""
        by_type: dict[str, int] = {}
        for d in detections:
            by_type[d.entity_type] = by_type.get(d.entity_type, 0) + 1

        return {
            "total_entities_detected": len(detections),
            "by_entity_type": by_type,
            "sample_replacements": [
                {
                    "type": d.entity_type,
                    "original": d.original_text[:20] + "..." if len(d.original_text) > 20 else d.original_text,
                    "replacement": d.replacement_text,
                }
                for d in detections[:10]
            ],
        }
