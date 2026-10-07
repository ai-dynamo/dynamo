# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep AISimulate release requirements and optional source patches consistent."""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from tests.wheels.smoke_install import AISIMULATE_FIND_LINKS

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

ROOT = Path(__file__).resolve().parents[2]
AISIMULATE_REQUIREMENTS = ROOT / "container/deps/requirements.aisimulate.txt"
LOCKFILES = (
    ROOT / "Cargo.lock",
    ROOT / "lib/bindings/python/Cargo.lock",
    ROOT / "lib/bindings/kvbm/Cargo.lock",
)


def _root_configs() -> tuple[dict, dict]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    with (ROOT / "Cargo.toml").open("rb") as handle:
        cargo = tomllib.load(handle)
    return pyproject, cargo


def _python_requirement(pyproject: dict) -> Requirement:
    matches = [
        Requirement(requirement)
        for requirement in pyproject["project"]["dependencies"]
        if canonicalize_name(Requirement(requirement).name) == "aisimulate"
    ]
    assert len(matches) == 1, "ai-dynamo must declare one AISimulate dependency"
    return matches[0]


def _requirements_file_aisimulate_requirement(path: Path) -> Requirement:
    matches: list[Requirement] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if not requirement or requirement.startswith("--"):
            continue
        parsed = Requirement(requirement)
        if canonicalize_name(parsed.name) == "aisimulate":
            matches.append(parsed)
    assert len(matches) == 1, f"{path} must declare one AISimulate dependency"
    return matches[0]


def _python_version_range(requirement: Requirement) -> SpecifierSet:
    assert requirement.url is None, "AISimulate must resolve from PyPI"
    bounds = {
        specifier.operator: Version(specifier.version)
        for specifier in requirement.specifier
    }
    assert len(requirement.specifier) == 2 and set(bounds) == {">", "<"}
    assert bounds[">"] < bounds["<"]
    assert not requirement.specifier.contains(bounds[">"])
    assert not requirement.specifier.contains(bounds["<"])
    return requirement.specifier


def _locked_cargo_version(path: Path, patch: dict | None = None) -> Version:
    with path.open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    matches = [package for package in packages if package["name"] == "aisimulate-core"]
    assert len(matches) == 1, f"expected one aisimulate-core package in {path}"

    package = matches[0]
    if patch is None:
        assert package.get("source") == (
            "registry+https://github.com/rust-lang/crates.io-index"
        ), f"aisimulate-core must resolve from crates.io in {path}"
        assert re.fullmatch(
            r"[0-9a-f]{64}", str(package.get("checksum", ""))
        ), f"aisimulate-core must have a registry checksum in {path}"
    else:
        assert package.get("source") == (
            f"git+{patch['git']}?rev={patch['rev']}#{patch['rev']}"
        ), f"aisimulate-core must resolve from the exact source patch in {path}"
        assert "checksum" not in package
    return Version(str(package["version"]))


def test_dynamo_declares_matching_aisimulate_requirements() -> None:
    pyproject, cargo = _root_configs()
    python_requirement = _python_requirement(pyproject)
    python_versions = _python_version_range(python_requirement)
    container_requirement = _requirements_file_aisimulate_requirement(
        AISIMULATE_REQUIREMENTS
    )

    assert python_requirement.marker is not None
    environment = default_environment()
    environment["python_version"] = "3.10"
    assert not python_requirement.marker.evaluate(environment)
    environment["python_version"] = "3.11"
    assert python_requirement.marker.evaluate(environment)
    environment["python_version"] = "3.12"
    assert python_requirement.marker.evaluate(environment)
    environment["python_version"] = "3.13"
    assert python_requirement.marker.evaluate(environment)
    environment["python_version"] = "3.14"
    assert not python_requirement.marker.evaluate(environment)
    assert container_requirement.marker is None
    assert _python_version_range(container_requirement) == python_versions
    with (ROOT / "benchmarks/pyproject.toml").open("rb") as handle:
        benchmarks = tomllib.load(handle)
    assert _python_version_range(_python_requirement(benchmarks)) == python_versions
    assert "aisimulate" not in pyproject.get("tool", {}).get("uv", {}).get(
        "sources", {}
    )

    cargo_dependency = cargo["workspace"]["dependencies"]["aisimulate-core"]
    assert not {"path", "git", "rev", "branch", "tag"} & cargo_dependency.keys()
    cargo_requirement = str(cargo_dependency["version"])
    assert cargo_requirement.startswith(
        "="
    ), "aisimulate-core must use one exact crates.io version"
    cargo_version = Version(cargo_requirement.removeprefix("="))

    patch = cargo.get("patch", {}).get("crates-io", {}).get("aisimulate-core")
    assert python_versions.contains(cargo_version)
    assert python_versions.contains("0.13.0")
    assert not python_versions.contains("0.14.0")
    if patch is not None:
        # Release freezes pin Rust source independently of the Python wheel.
        assert set(patch) == {"git", "rev"}
        assert patch["git"] == "https://github.com/ai-dynamo/aisimulate.git"
        assert re.fullmatch(r"[0-9a-f]{40}", patch["rev"])
        for path in LOCKFILES[1:]:
            with path.with_name("Cargo.toml").open("rb") as handle:
                manifest = tomllib.load(handle)
            assert manifest["patch"]["crates-io"]["aisimulate-core"] == patch
    assert all(
        _locked_cargo_version(path, patch) == cargo_version for path in LOCKFILES
    )


def test_container_stages_the_published_aisimulate_wheel() -> None:
    pyproject, _ = _root_configs()
    python_versions = _python_version_range(_python_requirement(pyproject))
    container_versions = _python_version_range(
        _requirements_file_aisimulate_requirement(AISIMULATE_REQUIREMENTS)
    )
    wheel_builder = (ROOT / "container/templates/wheel_builder.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert container_versions == python_versions
    assert "requirements.aisimulate.txt" in wheel_builder
    assert (
        "--requirement /opt/dynamo/container/deps/requirements.aisimulate.txt"
        in wheel_builder
    )
    assert "--only-binary=:all:" in wheel_builder
    assert "--no-deps" in wheel_builder
    assert "--no-index" in wheel_builder
    assert AISIMULATE_FIND_LINKS == "https://pypi.nvidia.com/aisimulate/"
    assert f"--find-links {AISIMULATE_FIND_LINKS}" in wheel_builder
    assert "COPY aisimulate" not in wheel_builder
    assert "/opt/dynamo/aisimulate" not in wheel_builder
    assert not (ROOT / "aisimulate").exists()


def test_planner_ci_image_collects_unified_cli_e2e_tests() -> None:
    planner_dockerfile = ROOT / "container/templates/planner.Dockerfile"
    if not planner_dockerfile.is_file():
        pytest.skip("planner Dockerfile is not staged in this component image")
    planner_template = planner_dockerfile.read_text(encoding="utf-8")

    assert "components/src/dynamo/replay/tests/e2e" in planner_template
    assert "components/src/dynamo/replay/tests/test_main.py" not in planner_template


def test_installed_aisimulate_satisfies_the_declared_range() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        pytest.skip("AISimulate supports Python 3.11 through 3.13")
    pyproject, _ = _root_configs()
    requirement = _python_requirement(pyproject)

    assert requirement.specifier.contains(Version(metadata.version("aisimulate")))


def test_ai_dynamo_registers_only_its_aisimulate_providers() -> None:
    pyproject, _ = _root_configs()
    project = pyproject["project"]

    extras = set(project.get("optional-dependencies", {}))
    assert {"sweeper", "simulate", "simulation"}.isdisjoint(extras)
    assert project["entry-points"]["aisimulate.sweep_config_providers"] == {
        "dynamo.planner": "dynamo.planner.simulation:create_provider",
        "dynamo.router": "dynamo.router.simulation:create_provider",
    }
    assert project["entry-points"]["aisimulate.runner_factories"] == {
        "dynamo": "dynamo.replay.simulation:DynamoReplayRunnerFactory"
    }
    assert project["entry-points"]["aisimulate.config_adapters"] == {
        "dynamo.planner": "dynamo.planner.simulation:create_provider",
        "dynamo.router": "dynamo.router.simulation:create_provider",
    }
