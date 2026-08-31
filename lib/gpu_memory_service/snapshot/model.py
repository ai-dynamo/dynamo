# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


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

    timestamp: float
    layout_hash: str
    device: int
    allocations: List[AllocationEntry] = field(default_factory=list)
