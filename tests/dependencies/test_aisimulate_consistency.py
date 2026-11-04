# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Dynamo on one source-pinned AISim release."""

from __future__ import annotations

import re
import runpy
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from tests.wheels import smoke_install

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
        if canonicalize_name(Requirement(requirement).name) == "aisim"
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
        if canonicalize_name(parsed.name) == "aisim":
            matches.append(parsed)
    assert len(matches) == 1, f"{path} must declare one AISimulate dependency"
    return matches[0]


def _exact_version(requirement: Requirement) -> Version:
    assert requirement.url is None, "AISim must declare a version requirement"
    specifiers = list(requirement.specifier)
    assert (
        len(specifiers) == 1 and specifiers[0].operator == "=="
    ), "AISim must use one exact version"
    return Version(specifiers[0].version)


def _locked_cargo_version(path: Path, patch: dict) -> Version:
    with path.open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    matches = [package for package in packages if package["name"] == "aisim-core"]
    assert len(matches) == 1, f"expected one aisim-core package in {path}"

    package = matches[0]
    assert package.get("source") == (
        f"git+{patch['git']}?rev={patch['rev']}#{patch['rev']}"
    ), f"aisim-core must resolve from the exact RC source in {path}"
    assert "checksum" not in package
    return Version(str(package["version"]))


def test_dynamo_pins_matching_aisim_rc_sources() -> None:
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
    with (ROOT / "benchmarks/pyproject.toml").open("rb") as handle:
        benchmarks = tomllib.load(handle)
    assert _exact_version(_python_requirement(benchmarks)) == python_version
    assert "aisim" not in pyproject.get("tool", {}).get("uv", {}).get("sources", {})

    cargo_dependency = cargo["workspace"]["dependencies"]["aisimulate-core"]
    assert not {"path", "git", "rev", "branch", "tag"} & cargo_dependency.keys()
    cargo_requirement = str(cargo_dependency["version"])
    assert cargo_requirement.startswith("="), "aisim-core must use one exact version"
    cargo_version = Version(cargo_requirement.removeprefix("="))

    with (ROOT / "lib/bindings/python/Cargo.toml").open("rb") as handle:
        bindings = tomllib.load(handle)
    bindings_dependency = bindings["dependencies"]["aisimulate-core"]
    assert not {"path", "git", "rev", "branch", "tag"} & bindings_dependency.keys()
    assert bindings_dependency["version"] == cargo_requirement
    assert cargo_dependency["package"] == bindings_dependency["package"] == "aisim-core"

    patch = cargo["patch"]["crates-io"]["aisim-core"]
    assert set(patch) == {"git", "rev"}
    assert patch["git"] == "https://github.com/ai-dynamo/aisimulate.git"
    assert re.fullmatch(r"[0-9a-f]{40}", patch["rev"])
    for path in LOCKFILES[1:]:
        with path.with_name("Cargo.toml").open("rb") as handle:
            manifest = tomllib.load(handle)
        assert manifest["patch"]["crates-io"]["aisim-core"] == patch
    assert cargo_version == python_version
    assert all(
        _locked_cargo_version(path, patch) == cargo_version for path in LOCKFILES
    )


def test_container_builds_the_matching_aisim_source_wheel() -> None:
    wheel_builder = (ROOT / "container/templates/wheel_builder.Dockerfile").read_text()
    assert "container/deps/build_aisimulate.py" in wheel_builder
    assert "--root /opt/dynamo" in wheel_builder
    assert "--output /opt/dynamo/dist" in wheel_builder
    assert "pip download" not in wheel_builder
    assert "pypi.nvidia.com/aisimulate" not in wheel_builder
    for name in (
        "planner",
        "sglang_runtime",
        "trtllm_runtime",
        "vllm_runtime",
        "frontend",
        "dynamo_runtime",
        "triton_runtime",
    ):
        template = (ROOT / f"container/templates/{name}.Dockerfile").read_text()
        assert "/opt/dynamo/wheelhouse/aisim-*.whl" in template
        assert "/opt/dynamo/wheelhouse/aisimulate*.whl" not in template


@pytest.mark.parametrize(
    "name,version",
    [
        ("aisim", "0.14.0"),
        ("aisim", "0.14.0.dev1"),
        ("aisim", "0.15.0"),
        ("aisimulate", "0.14.0"),
    ],
)
def test_source_wheel_build_validates_identity_before_compilation(
    tmp_path, monkeypatch, name, version
):
    _, cargo = _root_configs()
    patch = cargo["patch"]["crates-io"]["aisim-core"]
    source, output = tmp_path / "source", tmp_path / "wheels"
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "checkout" in command:
            project = source / "python/aisimulate"
            project.mkdir(parents=True)
            (project / "pyproject.toml").write_text(
                f'[project]\nname = "{name}"\nversion = "{version}"\n'
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    build = runpy.run_path(str(ROOT / "container/deps/build_aisimulate.py"))["build"]
    if (name, version) == ("aisim", "0.14.0"):
        build(ROOT, source, output)
        assert commands[1] == [
            "git",
            "-C",
            str(source),
            "fetch",
            "--depth=1",
            patch["git"],
            patch["rev"],
        ]
        assert commands[-1][0] == "maturin"
        assert "--locked" in commands[-1]
    else:
        with pytest.raises(ValueError):
            build(ROOT, source, output)
        assert all(command[0] != "maturin" for command in commands)


def test_wheel_smoke_requires_the_source_built_aisim_wheel(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(smoke_install, "run", lambda command: commands.append(command))
    with pytest.raises(AssertionError, match="missing required 'aisim'"):
        smoke_install.pip_install(Path("python"), tmp_path, ["ai-dynamo"])
    wheel = tmp_path / "aisim-0.14.0-cp311-abi3-manylinux_2_28_x86_64.whl"
    wheel.touch()
    smoke_install.pip_install(Path("python"), tmp_path, ["ai-dynamo"])
    assert str(wheel) in commands[0]
    assert "aisim" in commands[0]
    assert not any("pypi.nvidia.com" in arg for arg in commands[0])


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

    assert Version(metadata.version("aisim")) == expected


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
