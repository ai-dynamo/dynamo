# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOTICES-Native.txt generator.

Reads container/compliance/native_packages.yaml — the hand-curated overlay
for from-source components (CRIU, ucx, libfabric, ffmpeg, gdrcopy, NIXL
when treated as native, etc.) and emits per-component sections.

The YAML is filtered by the `dockerfile` field on each entry, so each
runtime image only sees the native components it actually contains.

TODO: implement. Scope:
  - load native_packages.yaml (PyYAML dependency)
  - filter by --image / --dockerfile-path argument
  - emit one section per artifact (binaries/libraries listed under each
    package's `artifacts:` array)
"""

from __future__ import annotations

from pathlib import Path

from .common import Component

ECOSYSTEM = "native"


def collect_components(yaml_path: Path, image_filter: str | None = None) -> list[Component]:
    raise NotImplementedError("Native generator not yet implemented")


def generate(
    yaml_path: Path, output_dir: Path, image_filter: str | None = None
) -> list[Component]:
    raise NotImplementedError("Native generator not yet implemented")
