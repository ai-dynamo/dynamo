# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify Dynamo consumes AIC through the consolidated AISimulate release."""

from __future__ import annotations

import sys
from importlib import metadata
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
CARGO_LOCKFILES = (
    ROOT / "Cargo.lock",
    ROOT / "lib/bindings/python/Cargo.lock",
    ROOT / "lib/bindings/kvbm/Cargo.lock",
)


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
    requirement_sets = [
        (
            "pyproject.toml project.dependencies",
            _requirement_names(root_project["dependencies"]),
        ),
        *(
            (
                f"pyproject.toml project.optional-dependencies.{name}",
                _requirement_names(requirements),
            )
            for name, requirements in root_project["optional-dependencies"].items()
        ),
        (
            "benchmarks/pyproject.toml project.dependencies",
            _requirement_names(benchmark_project["dependencies"]),
        ),
        (
            "container/deps/requirements.frontend.txt",
            _requirements_file_names(ROOT / "container/deps/requirements.frontend.txt"),
        ),
        (
            "container/deps/requirements.planner.txt",
            _requirements_file_names(ROOT / "container/deps/requirements.planner.txt"),
        ),
    ]
    for label, names in requirement_sets:
        retired = names & LEGACY_DISTRIBUTIONS
        assert not retired, f"{label} installs retired distributions: {sorted(retired)}"

    features = bindings_cargo["features"]
    dependencies = bindings_cargo["dependencies"]
    assert "aiconfigurator-core" not in dependencies
    for lockfile in CARGO_LOCKFILES:
        with lockfile.open("rb") as handle:
            packages = tomllib.load(handle)["package"]
        assert all(package["name"] != "aiconfigurator-core" for package in packages)
    assert features["aic-forward-pass"] == ["dep:aisimulate-core"]
    assert dependencies["aisimulate-core"] == {
        "version": "=0.1.0-dev.2",
        "optional": True,
        "features": ["python"],
    }


def test_aisimulate_wheel_preserves_aic_import_namespaces() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        pytest.skip("AISimulate supports Python 3.11 through 3.13")

    release = metadata.distribution("aisimulate")
    release_requirements = _requirement_names(release.requires or [])
    release_files = {str(path) for path in release.files or []}

    assert not (release_requirements & LEGACY_DISTRIBUTIONS)
    assert "aiconfigurator/__init__.py" in release_files
    assert "aiconfigurator_core/__init__.py" in release_files

    import aiconfigurator
    import aiconfigurator_core
    from aiconfigurator_core.sdk import RustForwardPassPerfModel
    from aisimulate_core.sdk import RustForwardPassPerfModel as PublicPerfModel

    assert aiconfigurator is not None
    assert aiconfigurator_core is not None
    assert RustForwardPassPerfModel is PublicPerfModel
