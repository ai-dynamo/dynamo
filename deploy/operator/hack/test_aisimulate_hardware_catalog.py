# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep DGDR's hardware hints synchronized with the pinned AISimulate wheel."""

from __future__ import annotations

import json
import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from deploy.operator.hack.generate_aisimulate_schema import generated_files

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "deploy/operator/internal/aisimulate/hardware_skus.txt"
API_PATH = (
    REPO_ROOT / "deploy/operator/api/v1beta2/dynamographdeploymentrequest_types.go"
)
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _pinned_version() -> Version:
    with PYPROJECT_PATH.open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["optional-dependencies"]["ais"]
    requirements = [
        Requirement(dependency)
        for dependency in dependencies
        if canonicalize_name(Requirement(dependency).name) == "aisimulate"
    ]
    assert (
        len(requirements) == 1
    ), "pyproject.toml's ais extra must contain one AISimulate pin"

    requirement = requirements[0]
    specifiers = list(requirement.specifier)
    assert requirement.url is None
    assert len(specifiers) == 1 and specifiers[0].operator == "=="
    return Version(specifiers[0].version)


def _packaged_hardware_skus() -> list[str]:
    distribution = metadata.distribution("aisimulate")
    system_files = [
        path
        for path in distribution.files or []
        if path.parent.as_posix() == "aisimulate_core/systems"
        and path.suffix == ".yaml"
    ]
    result = []
    for packaged_path in system_files:
        with distribution.locate_file(packaged_path).open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if isinstance(document, dict) and "gpu" in document and "node" in document:
            result.append(packaged_path.stem)
    return sorted(result)


def _documented_hardware_skus() -> list[str]:
    source = API_PATH.read_text()
    match = re.search(
        r"Known values in the bundled AISimulate release are (.*?)\n\s*// The field remains open",
        source,
        re.DOTALL,
    )
    assert match is not None, "GPUHardwareSpec.SKU must document the bundled catalog"
    return sorted(re.findall(r"`([^`]+)`", match.group(1)))


def _operator_hardware_skus() -> list[str]:
    return sorted(
        line.strip()
        for line in CATALOG_PATH.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def test_dgdr_hardware_catalog_matches_pinned_aisimulate() -> None:
    assert Version(metadata.version("aisimulate")) == _pinned_version()

    expected = _packaged_hardware_skus()
    assert _operator_hardware_skus() == expected
    assert _documented_hardware_skus() == expected


@pytest.mark.parametrize(
    "declarations",
    [[], ["aisimulate==1.2.3", "aisimulate==1.2.3"], ["aisimulate>=1.2.3"]],
    ids=["missing", "duplicate", "unbounded"],
)
def test_hardware_catalog_requires_one_exact_ais_extra_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declarations: list[str]
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\ndependencies = []\n[project.optional-dependencies]\n"
        f"ais = {json.dumps(declarations)}\n"
    )
    monkeypatch.setattr(f"{__name__}.PYPROJECT_PATH", pyproject)
    with pytest.raises(AssertionError):
        _pinned_version()


@pytest.mark.parametrize(
    "declarations",
    [
        [],
        ["aisimulate==1.2.3", "aisimulate==1.2.3"],
        ["aisimulate>=1.2.3"],
        ["aisimulate==1.2.4"],
    ],
    ids=["missing", "duplicate", "unbounded", "container-mismatch"],
)
def test_schema_export_rejects_invalid_ais_extra_before_runtime_import(
    tmp_path: Path, declarations: list[str]
) -> None:
    requirements = tmp_path / "container/deps/requirements.aisimulate.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("aisimulate==1.2.3\n")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n[project.optional-dependencies]\n"
        f"ais = {json.dumps(declarations)}\n"
    )
    with pytest.raises(ValueError, match="ais extra.*must pin the same"):
        generated_files(tmp_path)
