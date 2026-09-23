# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify Dynamo consumes AIC through the consolidated AISimulate source pin."""

from __future__ import annotations

import re
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
        re.split(r"\s+#", line, maxsplit=1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "--"))
    ]
    return _requirement_names(requirements)


def test_no_manifest_installs_retired_aic_distributions() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        root_project = tomllib.load(handle)["project"]
    with (ROOT / "benchmarks/pyproject.toml").open("rb") as handle:
        benchmark_project = tomllib.load(handle)["project"]
    with (ROOT / "lib/bindings/python/Cargo.toml").open("rb") as handle:
        bindings_cargo = tomllib.load(handle)
    with (ROOT / "Cargo.toml").open("rb") as handle:
        workspace_cargo = tomllib.load(handle)
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
        (
            "container/deps/requirements.aisimulate.txt",
            _requirements_file_names(
                ROOT / "container/deps/requirements.aisimulate.txt"
            ),
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
    native_dependency = workspace_cargo["workspace"]["dependencies"]["aisimulate-core"]
    assert native_dependency["git"] == "https://github.com/ai-dynamo/aisimulate.git"
    assert re.fullmatch(r"[0-9a-f]{40}", native_dependency["rev"])
    assert dependencies["aisimulate-core"] == {
        **native_dependency,
        "optional": True,
        "features": ["python"],
    }


def test_aisimulate_wheel_exposes_the_canonical_estimator_api() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        pytest.skip("AISimulate supports Python 3.11 through 3.13")

    release = metadata.distribution("aisimulate")
    release_requirements = _requirement_names(release.requires or [])
    release_files = {str(path) for path in release.files or []}

    assert not (release_requirements & LEGACY_DISTRIBUTIONS)
    assert "aisimulate/__init__.py" in release_files
    assert "aisimulate_core/__init__.py" in release_files
    assert any(
        path.startswith("aisimulate/_runtime.") and path.endswith((".so", ".pyd"))
        for path in release_files
    )

    import aisimulate_core
    from aisimulate import _runtime
    from aisimulate_core.sdk import RustForwardPassPerfModel

    assert aisimulate_core.RustForwardPassPerfModel is _runtime.RustForwardPassPerfModel
    assert callable(RustForwardPassPerfModel)
