# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep every Dynamo AIC dependency on one published dev release."""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

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
AIC_PACKAGES = {"aiconfigurator", "aiconfigurator-core"}


def _python_exact_version(requirement: str, *, package: str) -> Version:
    parsed = Requirement(requirement)
    assert canonicalize_name(parsed.name) == canonicalize_name(package), parsed
    assert parsed.marker is None, "AIC release dependency must not be conditional"
    assert (
        parsed.url is None
    ), f"AIC release dependency must be index-resolvable: {parsed}"

    specifiers = list(parsed.specifier)
    assert (
        len(specifiers) == 1 and specifiers[0].operator == "=="
    ), f"AIC release dependency must use one exact version: {parsed}"
    assert (
        "*" not in specifiers[0].version
    ), f"AIC release dependency must not use a wildcard: {parsed}"
    return Version(specifiers[0].version)


def _cargo_exact_version(dependency: object) -> Version:
    assert isinstance(dependency, dict), dependency
    forbidden = {"git", "rev", "branch", "tag", "path"} & dependency.keys()
    assert not forbidden, f"AIC Cargo dependency must use crates.io: {dependency}"

    version = str(dependency.get("version", ""))
    assert version.startswith("="), (
        "AIC Cargo dependency must use one exact crates.io version: " f"{dependency}"
    )
    return Version(version.removeprefix("="))


def _cargo_lock_version(path: Path) -> Version:
    with path.open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    matches = [
        package for package in packages if package["name"] == "aiconfigurator-core"
    ]
    assert len(matches) == 1, f"expected one aiconfigurator-core package in {path}"

    package = matches[0]
    source = str(package.get("source", ""))
    assert (
        source == "registry+https://github.com/rust-lang/crates.io-index"
    ), f"aiconfigurator-core must resolve from crates.io in {path}: {source}"
    checksum = str(package.get("checksum", ""))
    assert re.fullmatch(
        r"[0-9a-f]{64}", checksum
    ), f"aiconfigurator-core must have a registry checksum in {path}: {checksum}"
    return Version(str(package["version"]))


def _project_requirements(path: Path, *, extra: str | None = None) -> dict[str, str]:
    with path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    requirements = (
        project["optional-dependencies"][extra] if extra else project["dependencies"]
    )
    return _aic_requirements(requirements, source=str(path))


def _installed_requirements(package: str) -> dict[str, str]:
    requirements = metadata.requires(package)
    assert requirements is not None, f"{package} has no installed requirements"
    return _aic_requirements(requirements, source=f"installed {package} distribution")


def _requirements_file(path: Path) -> dict[str, str]:
    requirements = [
        line.partition(" #")[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return _aic_requirements(requirements, source=str(path))


def _aic_requirements(requirements: list[str], *, source: str) -> dict[str, str]:
    matches: dict[str, str] = {}
    for requirement in requirements:
        parsed = Requirement(requirement)
        package = canonicalize_name(parsed.name)
        if package not in AIC_PACKAGES:
            continue
        assert package not in matches, f"duplicate {package} requirement in {source}"
        matches[package] = requirement
    return matches


def test_all_aiconfigurator_dependencies_use_one_release() -> None:
    with (ROOT / "lib/bindings/python/Cargo.toml").open("rb") as handle:
        bindings_cargo = tomllib.load(handle)

    root_requirements = _project_requirements(ROOT / "pyproject.toml", extra="mocker")
    aisimulate_requirements = (
        _installed_requirements("aisimulate") if sys.version_info < (3, 13) else {}
    )
    benchmark_requirements = _project_requirements(ROOT / "benchmarks/pyproject.toml")
    frontend_requirements = _requirements_file(
        ROOT / "container/deps/requirements.frontend.txt"
    )
    planner_requirements = _requirements_file(
        ROOT / "container/deps/requirements.planner.txt"
    )

    assert set(root_requirements) == {"aiconfigurator-core"}
    if aisimulate_requirements:
        assert set(aisimulate_requirements) == AIC_PACKAGES
    assert set(benchmark_requirements) == {"aiconfigurator-core"}
    assert set(frontend_requirements) == {"aiconfigurator-core"}
    assert set(planner_requirements) == AIC_PACKAGES

    versions = {
        "Python bindings Cargo": _cargo_exact_version(
            bindings_cargo["dependencies"]["aiconfigurator-core"]
        ),
        "Python bindings Cargo.lock": _cargo_lock_version(
            ROOT / "lib/bindings/python/Cargo.lock"
        ),
        "ai-dynamo[mocker]": _python_exact_version(
            root_requirements["aiconfigurator-core"],
            package="aiconfigurator-core",
        ),
        "benchmarks core": _python_exact_version(
            benchmark_requirements["aiconfigurator-core"],
            package="aiconfigurator-core",
        ),
        "frontend core": _python_exact_version(
            frontend_requirements["aiconfigurator-core"],
            package="aiconfigurator-core",
        ),
        "planner upper": _python_exact_version(
            planner_requirements["aiconfigurator"],
            package="aiconfigurator",
        ),
        "planner core": _python_exact_version(
            planner_requirements["aiconfigurator-core"],
            package="aiconfigurator-core",
        ),
    }
    if aisimulate_requirements:
        versions.update(
            {
                "AI Simulate upper": _python_exact_version(
                    aisimulate_requirements["aiconfigurator"],
                    package="aiconfigurator",
                ),
                "AI Simulate core": _python_exact_version(
                    aisimulate_requirements["aiconfigurator-core"],
                    package="aiconfigurator-core",
                ),
            }
        )
    expected_version = versions["Python bindings Cargo"]
    mismatches = {
        consumer: version
        for consumer, version in versions.items()
        if version != expected_version
    }
    assert (
        not mismatches
    ), f"AIC release versions differ; expected {expected_version}: {mismatches}"


def test_installed_aiconfigurator_packages_match_declared_release() -> None:
    expected_version = _cargo_lock_version(ROOT / "lib/bindings/python/Cargo.lock")
    installed = {
        "aiconfigurator": Version(metadata.version("aiconfigurator")),
        "aiconfigurator-core": Version(metadata.version("aiconfigurator-core")),
    }
    mismatches = {
        package: version
        for package, version in installed.items()
        if version != expected_version
    }
    assert (
        not mismatches
    ), f"installed AIC versions differ; expected {expected_version}: {mismatches}"
