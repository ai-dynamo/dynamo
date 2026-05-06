# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""License-database lookup helper.

Resolution order, used by every NOTICES generator:

  1. The committed JSON snapshot at license-db.json (deterministic, no network).
  2. License-overrides YAML (hand-curated entries for proprietary/no-registry packages).
  3. Network query against the appropriate registry (crates.io, deps.dev, PyPI).

Production CI (container builds) is expected to find every shipped package in
the JSON or overrides — the weekly refresh-bot keeps the snapshot current.
A network fall-through is a soft signal that the snapshot needs a refresh.

Within-run dedup uses functools.lru_cache. There is no persistent on-disk
cache; the JSON snapshot is the persistent layer.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

Ecosystem = str  # "rust" | "go" | "python" | "dpkg" | "native"

_DEFAULT_DB_PATH = Path(__file__).resolve().parent / "license-db.json"
_DEFAULT_OVERRIDES_PATH = (
    Path(__file__).resolve().parent.parent / "license_overrides.yaml"
)


def _load_json_db(path: Path) -> dict[tuple[Ecosystem, str, str], str]:
    """Load license-db.json into a {(ecosystem, name, version): spdx} dict."""
    if not path.is_file():
        logger.warning("license-db.json not found at %s", path)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[Ecosystem, str, str], str] = {}
    for e in data.get("entries", []) or []:
        key = (e["ecosystem"], e["name"], e["version"])
        out[key] = e["spdx"]
    return out


def _load_overrides(path: Path) -> dict[tuple[Ecosystem, str], str]:
    """Load license_overrides.yaml. Match is by (ecosystem, name) — version-agnostic.

    Returns {} if the file is absent or PyYAML is not installed (overrides are optional).
    """
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed; skipping license_overrides.yaml")
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[tuple[Ecosystem, str], str] = {}
    for e in data.get("overrides", []) or []:
        out[(e["ecosystem"], e["name"])] = e["license"]
    return out


@functools.lru_cache(maxsize=1)
def _db() -> dict[tuple[Ecosystem, str, str], str]:
    return _load_json_db(_DEFAULT_DB_PATH)


@functools.lru_cache(maxsize=1)
def _overrides() -> dict[tuple[Ecosystem, str], str]:
    return _load_overrides(_DEFAULT_OVERRIDES_PATH)


@functools.lru_cache(maxsize=None)
def lookup(ecosystem: Ecosystem, name: str, version: str) -> str | None:
    """Resolve an SPDX expression for (ecosystem, name, version).

    Returns the SPDX string on hit. Returns None on miss — the caller decides
    whether to fall through to a network query (refresh.py does; runtime
    generators report UNKNOWN and let the policy gate fail the build).
    """
    key = (ecosystem, name, version)
    if key in _db():
        return _db()[key]

    override_key = (ecosystem, name)
    if override_key in _overrides():
        return _overrides()[override_key]

    return None


def reset_caches() -> None:
    """Clear memoization. Tests use this; production code shouldn't need it."""
    _db.cache_clear()
    _overrides.cache_clear()
    lookup.cache_clear()
