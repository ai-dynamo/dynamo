# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOTICES-Native.txt generator.

Reads container/compliance/native_packages.yaml — the hand-curated
overlay for from-source components we build inside our Dockerfiles
(CRIU, cuda-checkpoint, ucx, libfabric, gdrcopy, ffmpeg, ...). These
don't show up via dpkg-query (we build them from upstream source, not
apt-installed packages) and they're not Python wheels / Rust crates /
Go modules, so the per-ecosystem generators miss them. This file makes
them visible in /legal/native/NOTICES-Native.txt.

YAML schema:

  packages:
    - name: CRIU
      version: criu-dev
      license: GPL-2.0-only           # SPDX expression
      source: https://github.com/checkpoint-restore/criu
      license_text_path: /legal/CRIU/COPYING   # optional; inline the
                                               # text in NOTICES
      images:                          # which final images contain
        - snapshot-agent               # this component. Used to
                                       # filter via --native-image at
                                       # generator time so each
                                       # runtime only sees its own
                                       # native deps.

The `subtract` parameter from the orchestrator API is accepted but
ignored — native components by definition come from our build process,
not from any upstream baseline image, so they can never overlap with
a baseline_sbom by (name, version).
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from . import common
from .common import UNKNOWN, Component, spdx_license_text

logger = logging.getLogger(__name__)

ECOSYSTEM = "native"


def collect_components(
    yaml_path: Path, image_filter: str | None = None
) -> list[Component]:
    """Read native_packages.yaml, filter by image_filter, return Components.

    image_filter selects entries whose `images` list contains it. None
    means "return everything" (useful for inspection / tests).
    """
    if not yaml_path.is_file():
        logger.warning("native_packages.yaml not found at %s", yaml_path)
        return []

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    packages = data.get("packages") or []

    out: list[Component] = []
    for pkg in packages:
        name = pkg.get("name")
        version = pkg.get("version")
        if not name or not version:
            logger.warning("skipping malformed native entry: %r", pkg)
            continue

        if image_filter is not None:
            images = pkg.get("images") or []
            if image_filter not in images:
                continue

        spdx_value = pkg.get("license") or UNKNOWN
        license_text: str | None = None
        is_canonical = False
        text_path_str = pkg.get("license_text_path")
        if text_path_str:
            text_path = Path(text_path_str)
            if text_path.is_file():
                try:
                    license_text = text_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError as exc:
                    logger.warning(
                        "could not read license text at %s: %s", text_path, exc
                    )
            else:
                logger.warning(
                    "native package %s declares license_text_path=%s but the "
                    "file is not present in this stage's filesystem; falling "
                    "back to canonical SPDX text",
                    name,
                    text_path,
                )

        if license_text is None:
            # No bundled license file for this component — use the canonical
            # SPDX text (render_notices adds the no-license-file disclaimer).
            license_text = spdx_license_text(spdx_value)
            is_canonical = license_text is not None

        out.append(
            Component(
                ecosystem=ECOSYSTEM,
                name=name,
                version=str(version),
                spdx=spdx_value,
                source_url=pkg.get("source"),
                license_text=license_text,
                license_text_is_canonical=is_canonical,
            )
        )

    logger.info(
        "Native generator: %d components match image_filter=%r " "(out of %d in YAML)",
        len(out),
        image_filter,
        len(packages),
    )
    return out


def generate(
    yaml_path: Path,
    output_dir: Path,
    image_filter: str | None = None,
    subtract: set[tuple[str, str]] | None = None,
) -> list[Component]:
    """Write NOTICES-Native.txt + native-deps.csv. `subtract` is a no-op."""
    del subtract  # native components are first-party from-source builds —
    # never present in upstream baselines, so subtraction is meaningless.
    components = collect_components(yaml_path, image_filter)
    common.write_notices(ECOSYSTEM, components, output_dir)
    common.write_deps_csv(ECOSYSTEM, components, output_dir)
    return components
