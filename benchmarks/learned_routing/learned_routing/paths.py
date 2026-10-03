# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Campaign-root layout (CONTRACT "Paths").

The root defaults to ``~/learned-routing`` and is normally set with ``LR_ROOT`` or a CLI ``--root``
(remote bundles run with the bundle directory as their root).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path.home() / "learned-routing"
NUM_SLOTS = 20


@dataclass(frozen=True)
class Layout:
    root: Path

    @classmethod
    def resolve(cls, root: str | os.PathLike | None = None) -> "Layout":
        chosen = root or os.environ.get("LR_ROOT") or DEFAULT_ROOT
        return cls(Path(chosen).resolve())

    @property
    def slots_dir(self) -> Path:
        return self.root / "slots"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def cache_dir(self) -> Path:
        return self.runs_dir / "cache"

    @property
    def policies_dir(self) -> Path:
        return self.runs_dir / "policies"

    @property
    def replicates_dir(self) -> Path:
        return self.runs_dir / "replicates"

    @property
    def e0_dir(self) -> Path:
        return self.cache_dir / "e0"

    @property
    def engine_json(self) -> Path:
        return self.root / "config" / "engine.json"

    def resolve_path(self, value: str | os.PathLike) -> Path:
        """Resolve a cell path: ``CR/...`` and relative paths are relative to the root."""
        text = os.fspath(value)
        if text.startswith("CR/"):
            return self.root / text[3:]
        path = Path(text)
        return path if path.is_absolute() else self.root / path
