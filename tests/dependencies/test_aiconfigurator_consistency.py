# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify Dynamo consumes AIC through the consolidated AISimulate release."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.aiconfigurator,
]

ROOT = Path(__file__).resolve().parents[2]
LEGACY_DISTRIBUTIONS = {"aiconfigurator", "aiconfigurator-core"}


def _requirement_names(requirements: list[str]) -> set[str]:
    return {canonicalize_name(Requirement(item).name) for item in requirements}


def _requirements_file_names(path: Path) -> set[str]:
    requirements = [
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]
    return _requirement_names(requirements)


def test_no_manifest_installs_retired_aic_distributions() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        root_project = tomllib.load(handle)["project"]
    with (ROOT / "benchmarks/pyproject.toml").open("rb") as handle:
        benchmark_project = tomllib.load(handle)["project"]
    with (ROOT / "lib/bindings/python/Cargo.toml").open("rb") as handle:
        bindings_cargo = tomllib.load(handle)
    with (ROOT / "lib/bindings/python/Cargo.lock").open("rb") as handle:
        bindings_lock = tomllib.load(handle)

    requirement_sets = [
        _requirement_names(root_project["dependencies"]),
        *(
            _requirement_names(requirements)
            for requirements in root_project["optional-dependencies"].values()
        ),
        _requirement_names(benchmark_project["dependencies"]),
        _requirements_file_names(ROOT / "container/deps/requirements.frontend.txt"),
        _requirements_file_names(ROOT / "container/deps/requirements.planner.txt"),
    ]
    assert all(not (names & LEGACY_DISTRIBUTIONS) for names in requirement_sets)

    features = bindings_cargo["features"]
    dependencies = bindings_cargo["dependencies"]
    assert "aiconfigurator-core" not in dependencies
    assert all(
        package["name"] != "aiconfigurator-core" for package in bindings_lock["package"]
    )
    assert features["aic-forward-pass"] == ["dep:aisimulate-core"]
    assert dependencies["aisimulate-core"] == {
        "version": "=0.12.0",
        "optional": True,
        "features": ["python"],
    }


def test_aisimulate_wheel_preserves_aic_import_namespaces() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        pytest.skip("AISimulate supports Python 3.11 through 3.13")

    import aiconfigurator
    import aiconfigurator_core
    from aiconfigurator_core.sdk import RustForwardPassPerfModel
    from aisimulate_core.sdk import RustForwardPassPerfModel as PublicPerfModel

    assert aiconfigurator is not None
    assert aiconfigurator_core is not None
    assert RustForwardPassPerfModel is PublicPerfModel
