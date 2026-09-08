# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep AIPerf install requirements compatible across the source tree."""

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


def test_aiperf_install_pins_match() -> None:
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
        benchmark_pin.specifier == container_pin.specifier == SpecifierSet("==0.12.0")
    )


@pytest.mark.parametrize("version", ["3.10", "3.11", "3.12", "3.13", "3.14"])
def test_benchmark_python_range_matches_aiperf(version: str) -> None:
    with (ROOT / "benchmarks/pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    expected = version in SpecifierSet(">=3.11,<3.14")
    assert (version in SpecifierSet(project["requires-python"])) == expected
    assert "Programming Language :: Python :: 3.10" not in project["classifiers"]


@pytest.mark.parametrize("component", ["common", "test"])
def test_zstandard_pin_accepts_aiperf_requirement(component: str) -> None:
    requirements = ROOT / f"container/deps/requirements.{component}.txt"
    pin = next(
        Requirement(line)
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.startswith("zstandard==")
    )
    assert "0.25.0" in pin.specifier
