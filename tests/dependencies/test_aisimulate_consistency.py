# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Rust, Python, and containers on one exact AISimulate source/version."""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
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


def _exact_version(requirement: Requirement) -> Version:
    assert requirement.url is None, "AISimulate wheel requirements must pin a version"
    specifiers = list(requirement.specifier)
    assert (
        len(specifiers) == 1 and specifiers[0].operator == "=="
    ), "AISimulate must use one exact wheel version"
    return Version(specifiers[0].version)


def _locked_cargo_version(path: Path, dependency: dict) -> Version:
    with path.open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    matches = [package for package in packages if package["name"] == "aisimulate-core"]
    assert len(matches) == 1, f"expected one aisimulate-core package in {path}"

    package = matches[0]
    if "git" in dependency:
        revision = dependency["rev"]
        assert package.get("source") == (
            f"git+{dependency['git']}?rev={revision}#{revision}"
        ), f"aisimulate-core must resolve to the exact source revision in {path}"
        assert "checksum" not in package
    else:
        assert package.get("source") == (
            "registry+https://github.com/rust-lang/crates.io-index"
        ), f"aisimulate-core must resolve from crates.io in {path}"
        assert re.fullmatch(
            r"[0-9a-f]{64}", str(package.get("checksum", ""))
        ), f"aisimulate-core must have a registry checksum in {path}"
    return Version(str(package["version"]))


def test_dynamo_pins_matching_aisimulate_sources_and_versions() -> None:
    pyproject, cargo = _root_configs()
    python_requirement = _python_requirement(pyproject)
    python_version = _exact_version(python_requirement)
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
    assert _exact_version(container_requirement) == python_version

    cargo_dependency = cargo["workspace"]["dependencies"]["aisimulate-core"]
    assert not {"path", "branch", "tag"} & cargo_dependency.keys()
    if "git" in cargo_dependency:
        assert cargo_dependency["git"] == "https://github.com/ai-dynamo/aisimulate.git"
        assert re.fullmatch(r"[0-9a-f]{40}", cargo_dependency.get("rev", ""))
    else:
        assert "rev" not in cargo_dependency
    cargo_requirement = str(cargo_dependency["version"])
    assert cargo_requirement.startswith(
        "="
    ), "aisimulate-core must use one exact version even when built from source"
    cargo_version = Version(cargo_requirement.removeprefix("="))

    assert cargo_version == python_version
    assert all(
        _locked_cargo_version(path, cargo_dependency) == cargo_version
        for path in LOCKFILES
    )

    with (ROOT / "lib/bindings/python/Cargo.toml").open("rb") as handle:
        binding_cargo = tomllib.load(handle)
    binding_dependency = binding_cargo["dependencies"]["aisimulate-core"]
    assert {
        key: value
        for key, value in binding_dependency.items()
        if key not in {"optional", "features"}
    } == cargo_dependency
    assert binding_dependency["optional"] is True
    assert binding_dependency["features"] == ["python"]

    with (ROOT / "benchmarks/pyproject.toml").open("rb") as handle:
        benchmarks = tomllib.load(handle)
    assert _exact_version(_python_requirement(benchmarks)) == python_version
    for manifest in (cargo, binding_cargo):
        assert not manifest.get(
            "patch"
        ), "local patches must not replace the source pin"
        assert not manifest.get(
            "replace"
        ), "replacement dependencies hide the source pin"
    for config_path in (ROOT / ".cargo/config", ROOT / ".cargo/config.toml"):
        if config_path.is_file():
            with config_path.open("rb") as handle:
                assert not tomllib.load(handle).get("paths")


def test_container_stages_the_matching_aisimulate_wheel() -> None:
    pyproject, cargo = _root_configs()
    python_version = _exact_version(_python_requirement(pyproject))
    container_version = _exact_version(
        _requirements_file_aisimulate_requirement(AISIMULATE_REQUIREMENTS)
    )
    wheel_builder = (ROOT / "container/templates/wheel_builder.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert container_version == python_version
    assert "requirements.aisimulate.txt" in wheel_builder
    assert "--no-deps" in wheel_builder
    assert AISIMULATE_FIND_LINKS == "https://pypi.nvidia.com/aisimulate/"
    dependency = cargo["workspace"]["dependencies"]["aisimulate-core"]
    if "git" in dependency:
        assert 'pathlib.Path("/opt/dynamo/Cargo.toml")' in wheel_builder
        assert (
            '["workspace"]["dependencies"]["aisimulate-core"]["rev"]' in wheel_builder
        )
        assert "python -m pip wheel" in wheel_builder
        assert "--config-settings=build-args=--locked" in wheel_builder
        assert (
            "--constraint /opt/dynamo/container/deps/requirements.aisimulate.txt"
            in wheel_builder
        )
        assert (
            "aisimulate @ git+https://github.com/ai-dynamo/aisimulate.git@"
            "${AISIMULATE_REV}#subdirectory=python/aisimulate"
        ) in wheel_builder
    else:
        assert (
            "--requirement /opt/dynamo/container/deps/requirements.aisimulate.txt"
            in wheel_builder
        )
        assert "--only-binary=:all:" in wheel_builder
        assert "--no-index" in wheel_builder
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


def test_installed_aisimulate_matches_the_declared_release() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        pytest.skip("AISimulate supports Python 3.11 through 3.13")
    pyproject, _ = _root_configs()
    expected = _exact_version(_python_requirement(pyproject))

    assert Version(metadata.version("aisimulate")) == expected


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
