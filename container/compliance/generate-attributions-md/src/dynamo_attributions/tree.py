# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency tree builder from resolved packages."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .types import ResolvedPackage


class DependencyTree:
    """Directed dependency graph built from resolved packages."""

    def __init__(self, packages: list[ResolvedPackage]) -> None:
        self._packages: dict[str, ResolvedPackage] = {}
        self._forward: dict[str, list[str]] = defaultdict(list)
        self._reverse: dict[str, list[str]] = defaultdict(list)
        self._direct: set[str] = set()

        for pkg in packages:
            self._packages[pkg.name] = pkg
            self._forward[pkg.name] = list(pkg.dependencies)
            if pkg.is_direct:
                self._direct.add(pkg.name)
            for dep in pkg.dependencies:
                self._reverse[dep].append(pkg.name)

    def all_packages(self) -> list[ResolvedPackage]:
        """Return all packages sorted by name."""
        return sorted(self._packages.values(), key=lambda p: p.name)

    def summary(self) -> dict[str, Any]:
        """Summary statistics for the dependency tree."""
        return {
            "total": len(self._packages),
            "direct": len(self._direct),
            "transitive": len(self._packages) - len(self._direct),
        }
