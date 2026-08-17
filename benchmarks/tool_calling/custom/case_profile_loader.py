#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic loader for declarative tool-calling case profiles."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Callable

PROFILE_ROOT = Path(__file__).resolve().parent / "configs" / "case_profiles"
_PROFILE_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_TUPLE_FIELDS = (
    "messages",
    "tools",
    "expected_finish_reasons",
    "expected_tool_names",
    "expected_tool_calls",
    "forbidden_output_fragments",
    "regression_prs",
)


def available_case_profiles() -> tuple[str, ...]:
    """List declarative profiles available in this checkout."""

    if not PROFILE_ROOT.is_dir():
        return ()
    return tuple(sorted(path.stem for path in PROFILE_ROOT.glob("*.json")))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_case_profile(
    profile: str, case_factory: Callable[..., Any]
) -> tuple[Any, ...] | None:
    """Load one profile, returning ``None`` when no declarative file exists."""

    if not _PROFILE_NAME.fullmatch(profile):
        return None
    path = PROFILE_ROOT / f"{profile}.json"
    if not path.is_file():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("profile") != profile:
        raise ValueError(f"profile metadata mismatch in {path}")
    defaults = payload.get("defaults") or {}
    presets = payload.get("request_presets") or {}
    tools = payload.get("tools") or {}
    raw_cases = payload.get("cases")
    if not all(isinstance(value, dict) for value in (defaults, presets, tools)):
        raise ValueError(
            f"profile defaults, presets, and tools must be objects in {path}"
        )
    if not isinstance(raw_cases, list):
        raise ValueError(f"profile cases must be a list in {path}")

    cases: list[Any] = []
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError(f"profile case must be an object in {path}")
        values = _deep_merge(defaults, raw)
        preset_name = values.pop("request_preset", None)
        preset: dict[str, Any] = {}
        if preset_name is not None:
            preset = presets.get(preset_name)
            if not isinstance(preset, dict):
                raise ValueError(
                    f"case {values.get('case_id')} uses unknown preset "
                    f"{preset_name!r} in {path}"
                )
        values["request_overrides"] = _deep_merge(
            preset, values.get("request_overrides") or {}
        )

        resolved_tools: list[dict[str, Any]] = []
        for reference in values.get("tools") or ():
            if isinstance(reference, dict):
                resolved_tools.append(copy.deepcopy(reference))
                continue
            definition = tools.get(reference)
            if not isinstance(definition, dict):
                raise ValueError(
                    f"case {values.get('case_id')} uses unknown tool "
                    f"{reference!r} in {path}"
                )
            resolved_tools.append(copy.deepcopy(definition))
        values["tools"] = resolved_tools

        for field in _TUPLE_FIELDS:
            values[field] = tuple(values.get(field) or ())
        cases.append(case_factory(**values))

    expected = int(payload.get("logical_cases") or 0)
    if len(cases) != expected:
        raise ValueError(
            f"profile {profile} declares {expected} cases but loaded {len(cases)}"
        )
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"profile {profile} contains duplicate case IDs")
    return tuple(cases)
