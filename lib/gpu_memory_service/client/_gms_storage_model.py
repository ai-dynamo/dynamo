# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

CURRENT_VERSION = "1.0"


@dataclass(frozen=True)
class AllocationEntry:
    """Immutable record of one dumped allocation."""

    allocation_id: str
    size: int
    aligned_size: int
    tag: str
    tensor_file: str
    tensor_offset: int = 0


@dataclass
class SaveManifest:
    """Manifest for a GMS dump directory."""

    version: str
    timestamp: float
    layout_hash: str
    device: int
    allocations: List[AllocationEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "layout_hash": self.layout_hash,
            "device": self.device,
            "allocations": [asdict(a) for a in self.allocations],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SaveManifest":
        version = payload["version"]
        if version != CURRENT_VERSION:
            raise ValueError(
                f"Unsupported manifest version {version!r} "
                f"(expected {CURRENT_VERSION!r})"
            )
        allocations = [
            AllocationEntry(
                allocation_id=entry["allocation_id"],
                size=entry["size"],
                aligned_size=entry["aligned_size"],
                tag=entry["tag"],
                tensor_file=entry["tensor_file"],
                tensor_offset=entry.get("tensor_offset", 0),
            )
            for entry in payload.get("allocations", [])
        ]
        return cls(
            version=payload["version"],
            timestamp=payload["timestamp"],
            layout_hash=payload["layout_hash"],
            device=payload["device"],
            allocations=allocations,
        )
