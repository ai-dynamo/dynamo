# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep AIPerf install requirements compatible across the source tree."""

import ast
import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]

ROOT = Path(__file__).resolve().parents[2]


def test_epd_aiperf_version_gate_matches_release() -> None:
    """Keep the EPD version check aligned with the supported AIPerf release."""
    path = ROOT / "benchmarks/multimodal/sweep/experiments/epd/run_experiment.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    versions = [
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        if any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_AIPERF_VERSION"
            for target in node.targets
        )
    ]
    assert versions == ["0.13.0"]


def test_aiperf_install_pins_match() -> None:
    """Require benchmark packages and container images to pin the same AIPerf."""
    with (ROOT / "benchmarks/pyproject.toml").open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]
    benchmark_pin = next(
        Requirement(item) for item in dependencies if Requirement(item).name == "aiperf"
    )
    container_pin = next(
        Requirement(line)
        for line in (ROOT / "container/deps/requirements.benchmark.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("aiperf==")
    )
    assert (
        benchmark_pin.specifier == container_pin.specifier == SpecifierSet("==0.13.0")
    )


@pytest.mark.parametrize("version", ["3.10", "3.11", "3.12", "3.13", "3.14"])
def test_benchmark_python_range_matches_aiperf(version: str) -> None:
    with (ROOT / "benchmarks/pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    expected = version in SpecifierSet(">=3.11,<3.14")
    assert (version in SpecifierSet(project["requires-python"])) == expected
    assert "Programming Language :: Python :: 3.10" not in project["classifiers"]


@pytest.mark.parametrize("component", ["common", "test", "vllm"])
def test_zstandard_pin_accepts_aiperf_requirement(component: str) -> None:
    requirements = ROOT / f"container/deps/requirements.{component}.txt"
    pin = next(
        Requirement(line)
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.startswith("zstandard==")
    )
    assert pin.specifier == SpecifierSet("==0.25.0")


def test_pillow_floor_accepts_aiperf_requirement() -> None:
    """Reject Pillow floor bumps that AIPerf would undo in the images."""
    requirements = ROOT / "container/deps/requirements.common.txt"
    pin = next(
        Requirement(line)
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.startswith("pillow")
    )
    floors = [spec.version for spec in pin.specifier if spec.operator == ">="]
    assert len(floors) == 1, "Expected one inclusive Pillow minimum version"
    # AIPerf 0.13.0 limits Pillow to the 12.3 patch series, so a floor outside
    # that range needs an override in every image that resolves AIPerf.
    if floors[0] in SpecifierSet("~=12.3.0"):
        return
    for image in ("frontend", "planner"):
        overrides = ROOT / f"container/deps/overrides.{image}.txt"
        assert overrides.is_file(), (
            f"Pillow floor {floors[0]} is outside AIPerf 0.13.0's ~=12.3.0 range; "
            f"restore {overrides.name} or update AIPerf"
        )
        override = next(
            Requirement(line)
            for line in overrides.read_text(encoding="utf-8").splitlines()
            if line.startswith("pillow")
        )
        assert floors[0] in override.specifier, (
            f"{overrides.name} pins {override.specifier}, which excludes the "
            f"Pillow floor {floors[0]}"
        )


def _pillow_requirements() -> dict[str, SpecifierSet]:
    specs = {}
    deps = ROOT / "container/deps"
    for path in sorted(
        [*deps.glob("requirements.*.txt"), *deps.glob("overrides.*.txt")]
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line.lower().startswith("pillow") and Requirement(line).name == "pillow":
                specs[path.name] = Requirement(line).specifier
    return specs


def _template(name: str) -> str:
    path = ROOT / "container/templates" / name
    if not path.is_file():
        pytest.skip(f"{name} is not staged in this component image (.dockerignore)")
    return path.read_text(encoding="utf-8")


def test_pillow_specifiers_match_everywhere() -> None:
    """Require every Pillow pin and override to agree."""
    specs = _pillow_requirements()
    assert {"requirements.common.txt", "overrides.frontend.txt"} <= specs.keys()
    expected = specs["requirements.common.txt"]
    mismatched = {name: str(spec) for name, spec in specs.items() if spec != expected}
    assert not mismatched, f"Pillow specifiers differ from {expected}: {mismatched}"


def test_trtllm_system_pillow_matches_requirements() -> None:
    """Require the TRT-LLM system-site upgrade and its checks to match the pins."""
    expected = _pillow_requirements()["requirements.common.txt"]
    dockerfile = _template("trtllm_runtime.Dockerfile")
    installs = re.findall(r'--upgrade "(pillow[^"]*)"', dockerfile)
    assert [Requirement(item).specifier for item in installs] == [expected]
    bounds = {
        match
        for line in dockerfile.splitlines()
        if "pillow dist-info" in line
        for match in re.findall(r"\(([\d, ]+)\) <= tv\(vs\[0\]\) < \(([\d, ]+)\)", line)
    }
    assert bounds, "No TRT-LLM Pillow version checks found"
    assert len(bounds) == 1, f"TRT-LLM Pillow version checks disagree: {bounds}"
    floor, ceiling = (
        ".".join(part.strip() for part in b.split(",")) for b in bounds.pop()
    )
    assert SpecifierSet(f">={floor},<{ceiling}") == expected


@pytest.mark.parametrize(("image", "installs"), [("frontend", 3), ("planner", 1)])
def test_pillow_overrides_reach_image_installs(image: str, installs: int) -> None:
    """Fail when any image install stops passing its overrides file to uv."""
    dockerfile = _template(f"{image}.Dockerfile")
    target = f"/tmp/overrides.{image}.txt"
    assert (
        f"source=./container/deps/overrides.{image}.txt,target={target}" in dockerfile
    )
    assert dockerfile.count(f"--overrides {target}") == installs
