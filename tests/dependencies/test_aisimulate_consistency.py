# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Dynamo on one published AISim release."""

from __future__ import annotations

import re
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

pytestmark = [
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]
ROOT = Path(__file__).resolve().parents[2]
LOCKFILES = (
    ROOT / "Cargo.lock",
    ROOT / "lib/bindings/python/Cargo.lock",
    ROOT / "lib/bindings/kvbm/Cargo.lock",
)


def _source_requirement(project: dict) -> Requirement:
    pins = [
        Requirement(value)
        for value in project["project"]["dependencies"]
        if Requirement(value).name == "aisim"
    ]
    assert len(pins) == 1
    return pins[0]


def _source_pin() -> tuple[Requirement, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pin = _source_requirement(project)
    match = re.fullmatch(
        r"git\+https://github\.com/ai-dynamo/aisimulate\.git@([0-9a-f]{40})#subdirectory=python/aisim",
        pin.url or "",
    )
    assert match is not None, "AISim must use one immutable reviewed source commit"
    return pin, match[1]


def test_dynamo_pins_one_aisim_source_across_python_rust_and_images() -> None:
    pin, revision = _source_pin()
    lines = (ROOT / "container/deps/requirements.aisim.txt").read_text().splitlines()
    container = [
        Requirement(line)
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(container) == 1 and container[0].url == pin.url
    benchmarks = tomllib.loads((ROOT / "benchmarks/pyproject.toml").read_text())
    assert _source_requirement(benchmarks).url == pin.url
    assert pin.marker is not None
    for version, expected in [
        ("3.10", False),
        ("3.11", True),
        ("3.12", True),
        ("3.13", True),
        ("3.14", False),
    ]:
        environment = default_environment()
        environment["python_version"] = version
        assert pin.marker.evaluate(environment) is expected
    root = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"][
        "dependencies"
    ]["aisim-core"]
    bindings = tomllib.loads((ROOT / "lib/bindings/python/Cargo.toml").read_text())[
        "dependencies"
    ]["aisim-core"]
    assert root == {"git": "https://github.com/ai-dynamo/aisimulate", "rev": revision}
    assert bindings == {**root, "optional": True, "features": ["python"]}
    expected = f"git+{root['git']}?rev={revision}#{revision}"
    for lock in LOCKFILES:
        packages = tomllib.loads(lock.read_text())["package"]
        matching = [package for package in packages if package["name"] == "aisim-core"]
        assert len(matching) == 1 and matching[0]["source"] == expected
        assert all(package["name"] != "aisimulate-core" for package in packages)


def test_container_builds_aisim_from_the_source_requirement() -> None:
    template = (ROOT / "container/templates/wheel_builder.Dockerfile").read_text()
    assert "python -m pip wheel" in template
    assert "--wheel-dir /opt/dynamo/dist" in template
    assert "--requirement /opt/dynamo/container/deps/requirements.aisim.txt" in template
    assert "--no-deps" in template
    assert "https://pypi.nvidia.com/aisimulate/" not in template


def test_planner_ci_image_collects_unified_cli_e2e_tests() -> None:
    planner_dockerfile = ROOT / "container/templates/planner.Dockerfile"
    if not planner_dockerfile.is_file():
        pytest.skip("planner Dockerfile is not staged in this component image")
    planner_template = planner_dockerfile.read_text(encoding="utf-8")

    assert "components/src/dynamo/replay/tests/e2e" in planner_template
    assert "components/src/dynamo/replay/tests/test_main.py" not in planner_template


def test_installed_aisim_comes_from_the_reviewed_source_commit() -> None:
    if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
        pytest.skip("AISim supports Python 3.11 through 3.13")
    _, revision = _source_pin()
    distribution = metadata.distribution("aisim")
    provenance = __import__("json").loads(
        distribution.read_text("direct_url.json") or "{}"
    )
    assert provenance["vcs_info"]["commit_id"] == revision
    assert provenance["subdirectory"] == "python/aisim"


def test_ai_dynamo_registers_only_its_aisimulate_providers() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]

    extras = set(project.get("optional-dependencies", {}))
    assert {"sweeper", "simulate", "simulation"}.isdisjoint(extras)
    assert project["entry-points"]["aisim.sweep_config_providers"] == {
        "dynamo.planner": "dynamo.planner.simulation:create_provider",
        "dynamo.router": "dynamo.router.simulation:create_provider",
    }
    assert project["entry-points"]["aisim.runner_factories"] == {
        "dynamo": "dynamo.replay.simulation:DynamoReplayRunnerFactory"
    }
    assert project["entry-points"]["aisim.config_adapters"] == {
        "dynamo.planner": "dynamo.planner.simulation:create_provider",
        "dynamo.router": "dynamo.router.simulation:create_provider",
    }
