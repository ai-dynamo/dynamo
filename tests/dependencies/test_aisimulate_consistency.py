# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Dynamo's Python and Rust engines on one immutable AISimulate commit."""

from __future__ import annotations

import json
import re
import sys
from importlib import metadata
from pathlib import Path

import pytest
from packaging.markers import default_environment
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
    pytest.mark.planner,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

ROOT = Path(__file__).resolve().parents[2]
AISIMULATE_REQUIREMENTS = ROOT / "container/deps/requirements.aisimulate.txt"
AISIMULATE_REPOSITORY = "https://github.com/ai-dynamo/aisimulate.git"
AISIMULATE_SUBDIRECTORY = "python/aisimulate"
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
        requirement = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        if not requirement or requirement.startswith(("#", "--")):
            continue
        parsed = Requirement(requirement)
        if canonicalize_name(parsed.name) == "aisimulate":
            matches.append(parsed)
    assert len(matches) == 1, f"{path} must declare one AISimulate dependency"
    return matches[0]


def _git_commit(requirement: Requirement) -> str:
    assert not requirement.specifier, "AISimulate must use its immutable source pin"
    match = re.fullmatch(
        rf"git\+{re.escape(AISIMULATE_REPOSITORY)}@([0-9a-f]{{40}})"
        rf"#subdirectory={re.escape(AISIMULATE_SUBDIRECTORY)}",
        requirement.url or "",
    )
    assert match, "AISimulate must use a full Git SHA and its Python subdirectory"
    return match.group(1)


def _locked_cargo_version(path: Path, commit: str) -> Version:
    with path.open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    matches = [package for package in packages if package["name"] == "aisimulate-core"]
    assert len(matches) == 1, f"expected one aisimulate-core package in {path}"

    package = matches[0]
    assert package.get("source") == (
        f"git+{AISIMULATE_REPOSITORY}?rev={commit}#{commit}"
    ), f"aisimulate-core must resolve to the Python source commit in {path}"
    assert "checksum" not in package, f"unexpected registry package in {path}"
    return Version(str(package["version"]))


def test_dynamo_pins_one_immutable_aisimulate_commit() -> None:
    pyproject, cargo = _root_configs()
    python_requirement = _python_requirement(pyproject)
    commit = _git_commit(python_requirement)
    with (ROOT / "benchmarks/pyproject.toml").open("rb") as handle:
        benchmark_requirement = _python_requirement(tomllib.load(handle))
    container_requirement = _requirements_file_aisimulate_requirement(
        AISIMULATE_REQUIREMENTS
    )

    assert python_requirement.marker is not None
    assert benchmark_requirement.marker == python_requirement.marker
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
    assert _git_commit(benchmark_requirement) == commit
    assert _git_commit(container_requirement) == commit

    cargo_dependency = cargo["workspace"]["dependencies"]["aisimulate-core"]
    assert cargo_dependency == {"git": AISIMULATE_REPOSITORY, "rev": commit}
    with (ROOT / "lib/bindings/python/Cargo.toml").open("rb") as handle:
        binding_dependency = tomllib.load(handle)["dependencies"]["aisimulate-core"]
    assert binding_dependency == {
        **cargo_dependency,
        "optional": True,
        "features": ["python"],
    }
    assert len({_locked_cargo_version(path, commit) for path in LOCKFILES}) == 1


def test_container_builds_the_pinned_aisimulate_source_wheel() -> None:
    pyproject, _ = _root_configs()
    python_commit = _git_commit(_python_requirement(pyproject))
    container_commit = _git_commit(
        _requirements_file_aisimulate_requirement(AISIMULATE_REQUIREMENTS)
    )
    wheel_builder = (ROOT / "container/templates/wheel_builder.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert container_commit == python_commit
    build_step = next(
        step
        for step in wheel_builder.split("\nRUN ")
        if "--requirement /opt/dynamo/container/deps/requirements.aisimulate.txt"
        in step
    )
    assert "python -m pip wheel" in build_step
    assert "--wheel-dir /opt/dynamo/dist" in build_step
    assert "--no-deps" in build_step
    assert "--only-binary" not in build_step
    assert "--no-index" not in build_step
    assert "--find-links https://pypi.nvidia.com/aisimulate/" not in wheel_builder
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


def test_installed_aisimulate_matches_the_native_version() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        pytest.skip("AISimulate supports Python 3.11 through 3.13")
    pyproject, _ = _root_configs()
    commit = _git_commit(_python_requirement(pyproject))
    expected = _locked_cargo_version(LOCKFILES[0], commit)

    assert Version(metadata.version("aisimulate")) == expected


def test_installed_aisimulate_vcs_provenance_matches_the_declared_commit() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        pytest.skip("AISimulate supports Python 3.11 through 3.13")
    pyproject, _ = _root_configs()
    expected = _git_commit(_python_requirement(pyproject))
    direct_url = metadata.distribution("aisimulate").read_text("direct_url.json")
    if direct_url is None:
        pytest.skip("installed AISimulate wheel has no direct_url.json VCS provenance")
    provenance = json.loads(direct_url)
    if "vcs_info" not in provenance:
        pytest.skip(
            "installed AISimulate local wheel does not retain its source commit"
        )
    assert provenance["url"] == AISIMULATE_REPOSITORY
    assert provenance["subdirectory"] == AISIMULATE_SUBDIRECTORY
    assert provenance["vcs_info"]["vcs"] == "git"
    assert provenance["vcs_info"]["commit_id"] == expected
    assert provenance["vcs_info"]["requested_revision"] == expected


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
