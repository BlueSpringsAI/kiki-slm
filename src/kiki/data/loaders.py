"""Data loaders for the Kiki SLM pipeline.

Task 3.1: Unified data loading from HuggingFace Hub, CSV, JSONL, and databases,
with automatic column normalization to the canonical schema.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import datasets
import yaml

from kiki.data.validators import DatasetMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name normalization
# ---------------------------------------------------------------------------

COLUMN_ALIASES: dict[str, list[str]] = {
    "customer_message": ["body", "text", "message", "input", "instruction", "query", "content"],
    "agent_response": ["response", "answer", "reply", "output", "resolution"],
    "intent": ["intent", "category", "label", "type", "class"],
    "urgency": ["urgency", "priority", "severity"],
}

_ALIAS_LOOKUP: dict[str, str] = {}
for canonical, aliases in COLUMN_ALIASES.items():
    for alias in aliases:
        _ALIAS_LOOKUP[alias] = canonical


def normalize_columns(dataset: datasets.Dataset) -> datasets.Dataset:
    """Rename variant column names to their canonical equivalents."""
    rename_map: dict[str, str] = {}
    existing = set(dataset.column_names)

    for col in dataset.column_names:
        canonical = _ALIAS_LOOKUP.get(col)
        if canonical and canonical != col and canonical not in existing:
            rename_map[col] = canonical
            existing.add(canonical)

    if rename_map:
        logger.info("Normalizing columns: %s", rename_map)
        dataset = dataset.rename_columns(rename_map)

    return dataset


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseDataLoader(ABC):
    @abstractmethod
    def load(self) -> datasets.Dataset:
        ...

    @abstractmethod
    def get_metadata(self) -> DatasetMetadata:
        ...


# ---------------------------------------------------------------------------
# HuggingFace Hub loader
# ---------------------------------------------------------------------------


class HuggingFaceLoader(BaseDataLoader):
    def __init__(
        self,
        dataset_id: str,
        split: str = "train",
        subset: str | None = None,
        token: str | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.split = split
        self.subset = subset
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self._dataset: datasets.Dataset | None = None

    def load(self) -> datasets.Dataset:
        logger.info("Loading HuggingFace dataset '%s' (split=%s, subset=%s)", self.dataset_id, self.split, self.subset)
        kwargs: dict[str, Any] = {"path": self.dataset_id, "split": self.split}
        if self.subset:
            kwargs["name"] = self.subset
        if self.token:
            kwargs["token"] = self.token
        self._dataset = datasets.load_dataset(**kwargs)
        self._dataset = normalize_columns(self._dataset)
        return self._dataset

    def get_metadata(self) -> DatasetMetadata:
        ds = self._dataset
        return DatasetMetadata(
            name=self.dataset_id,
            source="huggingface",
            num_examples=len(ds) if ds is not None else 0,
            columns=list(ds.column_names) if ds is not None else [],
        )


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


class CSVLoader(BaseDataLoader):
    def __init__(
        self,
        path: str | Path,
        encoding: str = "utf-8",
        delimiter: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.encoding = encoding
        self.delimiter = delimiter
        self._dataset: datasets.Dataset | None = None

    def _detect_delimiter(self) -> str:
        if self.delimiter:
            return self.delimiter
        with open(self.path, encoding=self.encoding, newline="") as f:
            sample = f.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
        except csv.Error:
            return ","

    def load(self) -> datasets.Dataset:
        logger.info("Loading CSV from '%s' (encoding=%s)", self.path, self.encoding)
        sep = self._detect_delimiter()
        self._dataset = datasets.load_dataset(
            "csv",
            data_files=str(self.path),
            split="train",
            delimiter=sep,
            encoding=self.encoding,
        )
        self._dataset = normalize_columns(self._dataset)
        return self._dataset

    def get_metadata(self) -> DatasetMetadata:
        ds = self._dataset
        return DatasetMetadata(
            name=self.path.stem,
            source="csv",
            num_examples=len(ds) if ds is not None else 0,
            columns=list(ds.column_names) if ds is not None else [],
        )


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------


class JSONLLoader(BaseDataLoader):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._dataset: datasets.Dataset | None = None

    def load(self) -> datasets.Dataset:
        logger.info("Loading JSONL from '%s'", self.path)
        records: list[dict[str, Any]] = []
        with open(self.path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSON at line %d: %s", lineno, exc)

        if not records:
            raise ValueError(f"No valid records found in {self.path}")

        self._dataset = datasets.Dataset.from_list(records)
        self._dataset = normalize_columns(self._dataset)
        return self._dataset

    def get_metadata(self) -> DatasetMetadata:
        ds = self._dataset
        return DatasetMetadata(
            name=self.path.stem,
            source="jsonl",
            num_examples=len(ds) if ds is not None else 0,
            columns=list(ds.column_names) if ds is not None else [],
        )


# ---------------------------------------------------------------------------
# Database loader (SQLAlchemy)
# ---------------------------------------------------------------------------


class DatabaseLoader(BaseDataLoader):
    def __init__(self, connection_string: str, query: str) -> None:
        self.connection_string = connection_string
        self.query = query
        self._dataset: datasets.Dataset | None = None

    def load(self) -> datasets.Dataset:
        try:
            import sqlalchemy
        except ImportError as exc:
            raise ImportError(
                "SQLAlchemy is required for DatabaseLoader. Install with: uv pip install sqlalchemy"
            ) from exc

        logger.info("Loading from database (query length=%d chars)", len(self.query))
        engine = sqlalchemy.create_engine(self.connection_string)
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(self.query))
            columns = list(result.keys())
            rows = result.fetchall()

        records = [dict(zip(columns, row)) for row in rows]
        if not records:
            raise ValueError("Database query returned no rows")

        self._dataset = datasets.Dataset.from_list(records)
        self._dataset = normalize_columns(self._dataset)
        return self._dataset

    def get_metadata(self) -> DatasetMetadata:
        ds = self._dataset
        return DatasetMetadata(
            name="database_query",
            source="database",
            num_examples=len(ds) if ds is not None else 0,
            columns=list(ds.column_names) if ds is not None else [],
        )


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

_LOADER_TYPE_MAP: dict[str, type[BaseDataLoader]] = {
    "huggingface": HuggingFaceLoader,
    "csv": CSVLoader,
    "jsonl": JSONLLoader,
    "database": DatabaseLoader,
}


class DatasetRegistry:
    """Registry mapping dataset names to loader configurations, populated from YAML."""

    _entries: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, config: dict[str, Any]) -> None:
        cls._entries[name] = config

    @classmethod
    def load_yaml(cls, path: str | Path) -> None:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        registry_data = data if isinstance(data, dict) else {}
        if "datasets" in registry_data:
            registry_data = registry_data["datasets"]

        for name, config in registry_data.items():
            cls.register(name, config)
            logger.info("Registered dataset '%s' (type=%s)", name, config.get("type", "unknown"))

    @classmethod
    def get(cls, name: str) -> BaseDataLoader:
        if name not in cls._entries:
            raise KeyError(f"Dataset '{name}' not found in registry. Available: {list(cls._entries.keys())}")

        config = dict(cls._entries[name])
        loader_type = config.pop("type", None)
        if loader_type not in _LOADER_TYPE_MAP:
            raise ValueError(
                f"Unknown loader type '{loader_type}' for dataset '{name}'. "
                f"Supported: {list(_LOADER_TYPE_MAP.keys())}"
            )

        loader_cls = _LOADER_TYPE_MAP[loader_type]
        return loader_cls(**config)

    @classmethod
    def list_datasets(cls) -> list[str]:
        return list(cls._entries.keys())

    @classmethod
    def clear(cls) -> None:
        cls._entries.clear()
