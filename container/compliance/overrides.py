# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hand-curated license override lookup.

Reads container/compliance/license_overrides.yaml — the version-agnostic
(ecosystem, name) → SPDX map used by every NOTICES generator to correct
upstream metadata that is empty, malformed, or wrong (NVIDIA-proprietary
dpkgs misclassified as GPL-3, openssl's unparseable DEP-5, PyPI wheels
with empty License field, etc.).

This is the only license-resolution side-table the pipeline carries. The
old license-db.json snapshot tier was removed: every generator already
reads from a deterministic local source (CycloneDX SBOMs embedded in
wheels, cyclonedx-gomod's local detection, dpkg-query + DEP-5,
importlib.metadata), so there was nothing for a snapshot to mediate.

Within-run dedup uses functools.lru_cache.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

Ecosystem = str  # "rust" | "go" | "python" | "dpkg" | "native"

_DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent / "license_overrides.yaml"


def _load_overrides(path: Path) -> dict[tuple[Ecosystem, str], str]:
    """Load license_overrides.yaml.

    Returns {} if the file is absent. PyYAML is a hard dependency of every
    licenses stage (without it a generator would misclassify NVIDIA dpkgs
    etc.), so it's imported at module scope rather than guarded here.
    """
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[tuple[Ecosystem, str], str] = {}
    for e in data.get("overrides", []) or []:
        out[(e["ecosystem"], e["name"])] = e["license"]
    return out


@functools.lru_cache(maxsize=1)
def _overrides() -> dict[tuple[Ecosystem, str], str]:
    return _load_overrides(_DEFAULT_OVERRIDES_PATH)


@functools.lru_cache(maxsize=None)
def lookup(ecosystem: Ecosystem, name: str, version: str = "") -> str | None:
    """Return the override SPDX expression for (ecosystem, name), or None.

    `version` is accepted but unused — overrides are version-agnostic. It
    stays in the signature so call sites that pass a version (because they
    have one handy) remain readable.
    """
    del version  # version-agnostic by design
    return _overrides().get((ecosystem, name))


def reset_caches() -> None:
    """Clear memoization. Tests use this; production code shouldn't need it."""
    _overrides.cache_clear()
    lookup.cache_clear()
