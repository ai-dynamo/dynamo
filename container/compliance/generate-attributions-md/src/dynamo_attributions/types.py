# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for the attributions workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Ecosystem(str, Enum):
    """Language ecosystem for transitive dependency tracking."""

    RUST = "rust"
    PYTHON = "python"
    GO = "go"


@dataclass
class ResolvedPackage:
    """A fully resolved dependency from a lock file or package manager output."""

    name: str
    version: str
    ecosystem: Ecosystem
    is_direct: bool = False
    dependencies: list[str] = field(default_factory=list)
    source: str = ""
