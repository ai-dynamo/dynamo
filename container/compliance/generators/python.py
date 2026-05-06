# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOTICES-Python.txt generator.

Runs `pip-licenses --format=json --with-license-file --no-license-path`
against the runtime venv. Normalizes PyPI classifier strings to SPDX with
the AIPerf normalization map. Pre/post `uv pip freeze` snapshots so the
licensing tool's own ephemeral deps (pip-licenses + transitively-installed
prettytable, etc.) don't pollute the output.

First-party packages (`ai-dynamo`, `ai-dynamo-runtime`, `kvbm`, `nixl_*`,
`nvidia-*`, `dynamo-*`) are KEPT in the output — same principle as Rust.

TODO: implement. Scope:
  - shell out to `pip-licenses --format=json --with-license-file
    --no-license-path --python <venv>/bin/python` and parse the JSON
  - normalize legacy PyPI classifier strings to SPDX (port AIPerf's mapping)
  - handle compound expressions
  - exclude only the licensing tool's own ephemeral deps via pre/post
    `uv pip freeze` diff (NOT first-party packages — those stay)
"""

from __future__ import annotations

from pathlib import Path

from .common import Component

ECOSYSTEM = "python"


def collect_components(search_paths: list[Path]) -> list[Component]:
    raise NotImplementedError("Python generator not yet implemented")


def generate(search_paths: list[Path], output_dir: Path) -> list[Component]:
    raise NotImplementedError("Python generator not yet implemented")
