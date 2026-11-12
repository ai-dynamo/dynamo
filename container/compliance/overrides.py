# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated license corrections, exact-version matching, and actual-use evidence.

Versionless entries remain compatible legacy review debt. An override corrects
metadata; it does not itself grant a license-policy exception.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .policy.validate import parse_spdx

_DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent / "license_overrides.yaml"
_EXACT_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+:~_-]*$")


def exact_version(value) -> bool:
    return (
        isinstance(value, str)
        and bool(_EXACT_VERSION.fullmatch(value))
        and value.lower() not in {"unknown", "latest", "any", "all"}
    )


_ECOSYSTEMS = frozenset({"rust", "go", "python", "dpkg", "native"})


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject repeated YAML mapping keys instead of silently taking the last."""


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(
                f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


@dataclass(frozen=True)
class Override:
    ecosystem: str
    name: str
    license: str
    source: str
    version: str | None = None
    usage: str | None = None
    evidence: str | None = None
    reason: str | None = None

    @property
    def key(self) -> tuple[str, str, str | None]:
        return self.ecosystem, self.name, self.version


def load_records(path: Path) -> tuple[Override, ...]:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if (
        not isinstance(data, dict)
        or set(data) != {"overrides"}
        or not isinstance(data["overrides"], list)
    ):
        raise ValueError(f"Expected an overrides list in {path}")
    records = []
    seen = set()
    for row in data["overrides"]:
        if not isinstance(row, dict) or set(row) - set(Override.__dataclass_fields__):
            raise ValueError(f"Invalid override fields: {row!r}")
        for key in ("ecosystem", "name", "license", "source"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Override requires nonempty {key}: {row!r}")
        if row["ecosystem"] not in _ECOSYSTEMS:
            raise ValueError(f"Unknown override ecosystem: {row['ecosystem']}")
        for key in ("version", "usage", "evidence", "reason"):
            if key in row and (not isinstance(row[key], str) or not row[key].strip()):
                raise ValueError(f"Invalid override {key}: {row!r}")
        if "version" in row and not exact_version(row["version"]):
            raise ValueError(f"Override version must be exact: {row['version']!r}")
        parse_spdx(row["license"])
        record = Override(**row)
        if record.key in seen:
            raise ValueError(f"Duplicate override identity: {record.key}")
        seen.add(record.key)
        records.append(record)
    return tuple(records)


@functools.lru_cache(maxsize=1)
def _overrides() -> dict[tuple[str, str, str | None], Override]:
    return {row.key: row for row in load_records(_DEFAULT_OVERRIDES_PATH)}


def _match(ecosystem: str, name: str, version: str) -> Override | None:
    records = _overrides()
    return records.get((ecosystem, name, version)) or records.get(
        (ecosystem, name, None)
    )


_observations: dict[tuple[str, str, str], dict] = {}


def observe(
    ecosystem: str,
    name: str,
    version: str,
    raw_license: str | None,
    resolved_license: str,
    *,
    applied: bool = False,
) -> None:
    """Record generator observations before base subtraction; lookup is the only application signal."""
    record = _match(ecosystem, name, version)
    key = (ecosystem, name, version)
    previous = _observations.get(key, {})
    applied = applied or previous.get("applied", False)
    _observations[key] = {
        "ecosystem": ecosystem,
        "name": name,
        "version": version,
        "raw_license": raw_license,
        "resolved_license": resolved_license,
        "override_version": record.version if record else None,
        "override_license": record.license if record else None,
        "legacy_unversioned": bool(record and record.version is None),
        "applied": applied,
        "status": "applied"
        if applied
        else "matched_unused"
        if record
        else "no_override",
    }


def lookup(
    ecosystem: str, name: str, version: str = "", *, raw_license: str | None = None
) -> str | None:
    """Return an exact-version correction first, then a legacy versionless fallback."""
    record = _match(ecosystem, name, version)
    if record is None:
        return None
    observe(ecosystem, name, version, raw_license, record.license, applied=True)
    return record.license


def ensure_observed(components) -> None:
    """Cover supplemental/final inventory rows without erasing generator evidence."""
    for component in components:
        key = (component.ecosystem, component.name, component.version)
        if key not in _observations:
            observe(*key, raw_license=None, resolved_license=component.spdx)


def write_usage(
    output: Path, ecosystems: list[str], baseline: Path | None = None
) -> None:
    """Write actual generator evidence and the exact base inventory used by this build."""
    base = None
    if baseline is not None:
        raw = baseline.read_bytes()
        json.loads(raw)  # Validate before publishing evidence.
        bundled = output.parent / "override-base.cdx.json"
        bundled.write_bytes(raw)
        base = {
            "file": bundled.name,
            "source_file": baseline.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ecosystems": ecosystems,
                "override_catalog_sha256": hashlib.sha256(
                    _DEFAULT_OVERRIDES_PATH.read_bytes()
                ).hexdigest(),
                "baseline": base,
                "observations": sorted(
                    _observations.values(),
                    key=lambda r: (r["ecosystem"], r["name"], r["version"]),
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def reset_caches() -> None:
    _overrides.cache_clear()
    _observations.clear()
