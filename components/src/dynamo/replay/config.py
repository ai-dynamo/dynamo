# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical engine configuration at the Dynamo replay boundary."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from aisimulate.runner import canonical_performance_config

from dynamo.mocker.args import resolve_planner_profile_data as _resolve_profile
from dynamo.mocker.config import normalize_mocker_config


class PlannerProfileDataResult(Protocol):
    npz_path: Path | None


def canonical_upstream_config(
    config: Mapping[str, Any], *, worker_type: str
) -> dict[str, Any]:
    return canonical_performance_config(config, worker_type=worker_type)


def resolve_planner_profile_data(
    planner_profile_data: Path | None,
) -> PlannerProfileDataResult:
    if planner_profile_data is None or planner_profile_data.suffix == ".npz":
        return SimpleNamespace(npz_path=planner_profile_data)
    return _resolve_profile(planner_profile_data)


def load_engine_args(raw_args: str | Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Load a canonical AISimulate launch config with optional Dynamo runtime options."""
    return None if raw_args is None else normalize_mocker_config(raw_args)
