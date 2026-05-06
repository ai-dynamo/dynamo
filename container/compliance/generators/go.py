# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOTICES-Go.txt generator.

Reads CycloneDX SBOMs produced by `cyclonedx-gomod app -licenses -json`
in the Go builder stage of operator/snapshot-agent/EPP Dockerfiles. The
SBOM is COPYed into the licenses stage by the Dockerfile.

TODO: implement. Scope:
  - read /tmp/sbom-go.cdx.json (path conventionally where the builder
    drops it) — same CycloneDX shape as the Rust SBOMs
  - filter components whose purl starts with `pkg:golang/`
  - normalize compound license expressions
  - render NOTICES-Go.txt with one section per module
"""

from __future__ import annotations

from pathlib import Path

from .common import Component

ECOSYSTEM = "go"


def collect_components(sbom_path: Path) -> list[Component]:
    raise NotImplementedError("Go generator not yet implemented")


def generate(sbom_path: Path, output_dir: Path) -> list[Component]:
    raise NotImplementedError("Go generator not yet implemented")
