"""Golden datasets for agent evaluation and optimization.

A :class:`GoldenDataset` is an ordered collection of :class:`Example` records,
each pairing an agent input with an (optional) expected output and arbitrary
metadata. Optimizers evaluate candidate configurations against the dataset and
keep the configuration that scores best.

Loading is dependency-free: lists of dicts, JSON, JSON Lines, and CSV are all
supported using only the standard library. Splitting and sampling are
deterministic when a seed is supplied so optimization runs are reproducible.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Keys recognised as the *input* field when loading from records/files.
_INPUT_KEYS = ("inputs", "input", "question", "prompt", "query", "request")
#: Keys recognised as the *expected output* field when loading from records.
_EXPECTED_KEYS = ("expected", "expected_output", "answer", "output", "label", "target", "gold")


@dataclass
class Example:
    """A single golden record: an input, its expected output, and metadata.

    Args:
        inputs: The input handed to the agent. May be a string, a messages list,
            or any payload the wrapped agent accepts.
        expected: The reference / gold output, if known. ``None`` for unlabeled
            examples (still usable with reference-free metrics such as an
            LLM-as-judge rubric).
        metadata: Free-form annotations (difficulty, tags, ids, judge criteria).
    """

    inputs: Any
    expected: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Example:
        """Build an :class:`Example` from a loosely-structured dict.

        The input and expected fields are located by trying the well-known key
        names in :data:`_INPUT_KEYS` / :data:`_EXPECTED_KEYS`. Any remaining keys
        become metadata, so extra columns survive the round-trip.
        """
        record = dict(record)
        inputs = _pop_first(record, _INPUT_KEYS)
        expected = _pop_first(record, _EXPECTED_KEYS)
        # An explicit metadata mapping is merged with leftover columns.
        meta = record.pop("metadata", None)
        metadata: dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}
        metadata.update(record)
        return cls(inputs=inputs, expected=expected, metadata=metadata)


def _pop_first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record:
            return record.pop(key)
    return None


class GoldenDataset:
    """An ordered collection of :class:`Example` records.

    Supports iteration, indexing, slicing, deterministic shuffling/splitting,
    and sampling. Construct directly from examples or via the ``from_*`` loaders.
    """

    def __init__(self, examples: Iterable[Example] = ()):
        self._examples: list[Example] = list(examples)
        for i, ex in enumerate(self._examples):
            if not isinstance(ex, Example):
                raise TypeError(f"GoldenDataset item {i} is not an Example: {type(ex)!r}")

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_list(cls, records: Iterable[Example | dict[str, Any]]) -> GoldenDataset:
        """Build a dataset from a list of :class:`Example` or plain dicts."""
        examples: list[Example] = []
        for record in records:
            if isinstance(record, Example):
                examples.append(record)
            elif isinstance(record, dict):
                examples.append(Example.from_record(record))
            else:
                raise TypeError(f"Cannot build Example from {type(record)!r}")
        return cls(examples)

    @classmethod
    def from_json(cls, path: str | Path) -> GoldenDataset:
        """Load from a JSON file containing a list of records (or ``{"examples": [...]}``)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "examples" in data:
            data = data["examples"]
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list or an object with an 'examples' list")
        return cls.from_list(data)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> GoldenDataset:
        """Load from a JSON Lines file (one JSON object per line)."""
        examples: list[Example] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            examples.append(Example.from_record(json.loads(line)))
        return cls(examples)

    @classmethod
    def from_csv(cls, path: str | Path) -> GoldenDataset:
        """Load from a CSV file with a header row."""
        with Path(path).open(newline="", encoding="utf-8") as fh:
            return cls.from_list(list(csv.DictReader(fh)))

    # -- sequence protocol -----------------------------------------------------

    def __iter__(self) -> Iterator[Example]:
        return iter(self._examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int | slice) -> Example | GoldenDataset:
        if isinstance(index, slice):
            return GoldenDataset(self._examples[index])
        return self._examples[index]

    def __bool__(self) -> bool:
        return bool(self._examples)

    @property
    def examples(self) -> list[Example]:
        """A shallow copy of the underlying example list."""
        return list(self._examples)

    # -- transforms ------------------------------------------------------------

    def shuffled(self, seed: int | None = None) -> GoldenDataset:
        """Return a new, deterministically shuffled dataset."""
        items = list(self._examples)
        random.Random(seed).shuffle(items)
        return GoldenDataset(items)

    def split(
        self,
        train: float = 0.8,
        *,
        seed: int | None = None,
        shuffle: bool = True,
    ) -> tuple[GoldenDataset, GoldenDataset]:
        """Split into ``(train, holdout)`` datasets by fraction.

        Args:
            train: Fraction (0..1) of examples assigned to the training split.
            seed: Seed for the optional shuffle (reproducible splits).
            shuffle: Shuffle before splitting (recommended for ordered data).
        """
        if not 0.0 <= train <= 1.0:
            raise ValueError("train fraction must be between 0 and 1")
        source = self.shuffled(seed) if shuffle else self
        cut = int(round(len(source) * train))
        return source[:cut], source[cut:]  # type: ignore[return-value]

    def sample(self, n: int, *, seed: int | None = None) -> GoldenDataset:
        """Return a random subset of up to ``n`` examples (without replacement)."""
        if n >= len(self._examples):
            return GoldenDataset(self._examples)
        chosen = random.Random(seed).sample(self._examples, n)
        return GoldenDataset(chosen)

    def filter(self, predicate) -> GoldenDataset:
        """Return a new dataset keeping examples for which ``predicate`` is true."""
        return GoldenDataset(ex for ex in self._examples if predicate(ex))

    def to_records(self) -> list[dict[str, Any]]:
        """Serialise to a list of plain dicts (round-trips through ``from_list``)."""
        records: list[dict[str, Any]] = []
        for ex in self._examples:
            record: dict[str, Any] = {"inputs": ex.inputs, "expected": ex.expected}
            if ex.metadata:
                record["metadata"] = dict(ex.metadata)
            records.append(record)
        return records


__all__ = ["Example", "GoldenDataset"]
