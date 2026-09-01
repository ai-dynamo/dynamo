# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python AIC comes from the aisimulate wheel, not standalone AIC pins.

The optional Rust ``aic-forward-pass`` feature still pins ``aiconfigurator-core``
on crates.io. That crate pin is checked here so Cargo.toml and Cargo.lock stay
on one published version.
"""

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
PYTHON_AIC_SOURCES = (
    "pyproject.toml",
    "benchmarks/pyproject.toml",
    "container/deps/requirements.frontend.txt",
    "container/deps/requirements.planner.txt",
    "container/deps/requirements.aisimulate.txt",
)
requires_aisimulate = pytest.mark.skipif(
    not ((3, 11) <= sys.version_info[:2] < (3, 14)),
    reason="aisimulate publishes wheels for Python 3.11-3.13 only",
)


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


def _requirement_lines(path: Path) -> list[str]:
    if path.suffix == ".toml":
        with path.open("rb") as handle:
            document = tomllib.load(handle)
        project = document.get("project", {})
        lines = list(project.get("dependencies", []))
        for extra_reqs in project.get("optional-dependencies", {}).values():
            lines.extend(extra_reqs)
        for group_reqs in document.get("dependency-groups", {}).values():
            # PEP 735 groups may hold {include-group = ...} tables alongside strings.
            lines.extend(req for req in group_reqs if isinstance(req, str))
        return lines
    return [
        line.partition(" #")[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _aic_python_pins(path: Path) -> dict[str, str]:
    matches: dict[str, str] = {}
    for requirement in _requirement_lines(path):
        parsed = Requirement(requirement)
        package = canonicalize_name(parsed.name)
        if package not in AIC_PACKAGES:
            continue
        assert package not in matches, f"duplicate {package} requirement in {path}"
        matches[package] = requirement
    return matches


@pytest.mark.parametrize("source", PYTHON_AIC_SOURCES)
def test_python_source_does_not_pin_standalone_aic(source: str) -> None:
    pinned = _aic_python_pins(ROOT / source)
    assert pinned == {}, f"standalone AIC pins must be removed from {source}: {pinned}"


@requires_aisimulate
def test_installed_aisimulate_does_not_depend_on_aic_wheels() -> None:
    requirements = metadata.requires("aisimulate")
    assert requirements is not None, "aisimulate has no installed requirements"
    aic_requires = [
        requirement
        for requirement in requirements
        if canonicalize_name(Requirement(requirement).name) in AIC_PACKAGES
    ]
    assert aic_requires == [], (
        "aisimulate 0.1.0.dev2 vendors AIC; it must not Requires-Dist AIC wheels: "
        f"{aic_requires}"
    )


def test_rust_aiconfigurator_core_crate_uses_one_release() -> None:
    with (ROOT / "lib/bindings/python/Cargo.toml").open("rb") as handle:
        bindings_cargo = tomllib.load(handle)
    expected = _cargo_exact_version(
        bindings_cargo["dependencies"]["aiconfigurator-core"]
    )
    locked = _cargo_lock_version(ROOT / "lib/bindings/python/Cargo.lock")
    assert locked == expected, (
        f"Python bindings Cargo.toml ({expected}) and Cargo.lock ({locked}) "
        "disagree on aiconfigurator-core"
    )


@requires_aisimulate
def test_python_aic_modules_come_from_aisimulate() -> None:
    for dist in AIC_PACKAGES:
        with pytest.raises(metadata.PackageNotFoundError):
            metadata.version(dist)

    import aiconfigurator
    import aiconfigurator_core

    assert aiconfigurator
    assert aiconfigurator_core
