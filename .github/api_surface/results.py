# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured operation results with edge case tracking.

Every compound operation returns an OperationResult containing:
- data: deterministic, confident results (auto-processable)
- edge_cases: situations needing LLM/human judgment
- errors: API failures, timeouts, etc.
- metadata: timing, cache status, counts
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EdgeCase:
    """A situation the code identified but couldn't resolve deterministically."""

    kind: str  # e.g. "ambiguous_match", "missing_link", "status_conflict"
    description: str  # Human-readable explanation
    context: dict = field(default_factory=dict)  # All relevant data for decision-making
    suggested_actions: list[str] = field(
        default_factory=list
    )  # What the code thinks should happen
    confidence: float = 0.0  # How confident the suggestion is (0-1)


@dataclass
class OpError:
    """An error encountered during an operation."""

    operation: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class OperationResult:
    """Standard return type for all compound operations."""

    data: dict = field(default_factory=dict)
    edge_cases: list[EdgeCase] = field(default_factory=list)
    errors: list[OpError] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    preview: bool = False

    @property
    def needs_judgment(self) -> bool:
        """Whether this result has edge cases requiring human/LLM judgment."""
        return bool(self.edge_cases)

    @property
    def has_errors(self) -> bool:
        """Whether any errors occurred during the operation."""
        return bool(self.errors)

    @property
    def ok(self) -> bool:
        """Whether the operation completed without errors."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary, including computed properties."""
        return {
            "data": self.data,
            "edge_cases": [asdict(ec) for ec in self.edge_cases],
            "errors": [asdict(e) for e in self.errors],
            "metadata": self.metadata,
            "preview": self.preview,
            "ok": self.ok,
            "has_errors": self.has_errors,
            "needs_judgment": self.needs_judgment,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path) -> None:
        """Save result to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: dict) -> "OperationResult":
        """Deserialize from a dictionary."""
        return cls(
            data=data.get("data", {}),
            edge_cases=[EdgeCase(**ec) for ec in data.get("edge_cases", [])],
            errors=[OpError(**e) for e in data.get("errors", [])],
            metadata=data.get("metadata", {}),
            preview=data.get("preview", False),
        )

    @classmethod
    def load(cls, path: str | Path) -> "OperationResult":
        """Load result from a JSON file."""
        path = Path(path)
        raw = json.loads(path.read_text())
        return cls.from_dict(raw)

    def add_timing(self, start_time: float) -> None:
        """Add duration_ms to metadata from a start timestamp."""
        self.metadata["duration_ms"] = int((time.time() - start_time) * 1000)

    def diff(self, previous: "OperationResult", id_key: str = "id") -> dict[str, Any]:
        """Compute delta between this result and a previous one.

        Compares items in data by stable ID to find added, removed, and changed items.

        Args:
            previous: The earlier result to compare against
            id_key: Key used to identify items for stable diffing

        Returns:
            Dict with 'added', 'removed', 'changed' lists
        """

        def _index_items(data: dict, key: str) -> dict[str, dict]:
            """Build ID-keyed index from data dict's list values."""
            index: dict[str, dict] = {}
            for items in data.values():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and key in item:
                            index[str(item[key])] = item
            return index

        curr = _index_items(self.data, id_key)
        prev = _index_items(previous.data, id_key)

        added = [curr[k] for k in curr if k not in prev]
        removed = [prev[k] for k in prev if k not in curr]
        changed = []
        for k in curr:
            if k in prev and curr[k] != prev[k]:
                changed.append({"id": k, "before": prev[k], "after": curr[k]})

        return {"added": added, "removed": removed, "changed": changed}
