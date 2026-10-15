# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict manifests for independently configured Planner Gym cases."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from autoscaling_arena.match_config import (
    MatchConfig,
    MatchConfigError,
    _load_yaml,
    parse_match_config,
)


class SuiteConfigError(ValueError):
    """A suite manifest or one of its referenced configurations is invalid."""


@dataclass(frozen=True)
class SuiteCase:
    case_id: str
    config: MatchConfig
    config_file_sha256: str
    labels: dict[str, str]
    run_ids: tuple[str, ...] | None = None

    @property
    def expected_runs(self) -> int:
        return (
            len(self.run_ids) if self.run_ids is not None else self.config.expected_runs
        )


@dataclass(frozen=True)
class SuiteConfig:
    schema_version: int
    name: str
    cases: tuple[SuiteCase, ...]
    source_path: Path

    def selected_cases(self, case_ids: set[str] | None = None) -> tuple[SuiteCase, ...]:
        """Select exact IDs in manifest order; reject empty or unknown selections."""

        if case_ids is None:
            return self.cases
        unknown = case_ids - {case.case_id for case in self.cases}
        if unknown:
            raise SuiteConfigError("unknown case id(s): " + ", ".join(sorted(unknown)))
        if not case_ids:
            raise SuiteConfigError("case_ids must select at least one case")
        return tuple(case for case in self.cases if case.case_id in case_ids)


def load_suite_config(path: str | Path) -> SuiteConfig:
    """Validate a manifest and all referenced Match Configs without a runtime."""

    source = Path(path).expanduser().resolve()
    try:
        data = _load_yaml(source.read_text(), source=str(source))
        return parse_suite_config(data, source_path=source)
    except (OSError, UnicodeError, MatchConfigError) as exc:
        raise SuiteConfigError(str(exc)) from exc


def parse_suite_config(data: Any, *, source_path: str | Path) -> SuiteConfig:
    """Resolve Match Config references relative to the manifest's directory."""

    source = Path(source_path).expanduser().resolve()
    root = _mapping(data, "suite", {"schema_version", "name", "cases"})
    version = root.get("schema_version")
    if type(version) is not int or version != 1:
        raise SuiteConfigError("schema_version: expected 1")
    name = _string(root.get("name"), "name")
    entries = root.get("cases")
    if not isinstance(entries, list) or not entries:
        raise SuiteConfigError("cases: expected a non-empty list")
    cases: list[SuiteCase] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        field = f"cases[{index}]"
        item = _mapping(entry, field, {"id", "config", "labels", "run_ids"})
        case_id = _string(item.get("id"), f"{field}.id")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", case_id):
            raise SuiteConfigError(
                f"{field}.id: use 1-128 letters, digits, dots, underscores or hyphens; "
                "start with a letter or digit"
            )
        if case_id in seen_ids:
            raise SuiteConfigError(f"{field}.id: duplicate case id {case_id!r}")
        seen_ids.add(case_id)
        config_path = Path(_string(item.get("config"), f"{field}.config")).expanduser()
        if not config_path.is_absolute():
            config_path = source.parent / config_path
        config_path = config_path.resolve()
        try:
            content = config_path.read_bytes()
            config = parse_match_config(
                _load_yaml(content.decode("utf-8"), source=str(config_path)),
                source_path=config_path,
            )
        except (OSError, UnicodeError, MatchConfigError) as exc:
            raise SuiteConfigError(f"{field}.config: {exc}") from exc
        labels_raw = item.get("labels", {})
        if not isinstance(labels_raw, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in labels_raw.items()
        ):
            raise SuiteConfigError(f"{field}.labels: expected string keys and values")
        run_ids = None
        if "run_ids" in item:
            values = item["run_ids"]
            if not isinstance(values, list) or not values:
                raise SuiteConfigError(f"{field}.run_ids: expected a non-empty list")
            run_ids = tuple(_string(value, f"{field}.run_ids") for value in values)
            if len(set(run_ids)) != len(run_ids):
                raise SuiteConfigError(f"{field}.run_ids: duplicate run id")
            unknown = set(run_ids) - {run.run_id for run in config.iter_runs()}
            if unknown:
                raise SuiteConfigError(
                    f"{field}.run_ids: unknown run id(s): " + ", ".join(sorted(unknown))
                )
        cases.append(
            SuiteCase(
                case_id=case_id,
                config=config,
                config_file_sha256=hashlib.sha256(content).hexdigest(),
                labels=dict(labels_raw),
                run_ids=run_ids,
            )
        )
    return SuiteConfig(version, name, tuple(cases), source)


def suite_config_sha256(config: SuiteConfig) -> str:
    """Fingerprint the manifest's contents and referenced configuration bytes."""

    payload = {
        "schema_version": config.schema_version,
        "name": config.name,
        "cases": [
            {
                "id": case.case_id,
                "config_file_sha256": case.config_file_sha256,
                "labels": case.labels,
                "run_ids": case.run_ids,
            }
            for case in config.cases
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _mapping(value: Any, field: str, allowed: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SuiteConfigError(f"{field}: expected a mapping with string keys")
    unknown = set(value) - allowed
    if unknown:
        raise SuiteConfigError(
            f"{field}: unknown key(s): " + ", ".join(sorted(unknown))
        )
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SuiteConfigError(f"{field}: expected a non-empty string")
    return value
