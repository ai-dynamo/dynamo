# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Kubernetes authentication support consistent across install paths."""

from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

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


def _requirements_file_kubernetes_requirement(path: Path) -> Requirement:
    matches = [
        Requirement(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("kubernetes[")
    ]
    assert len(matches) == 1, f"{path} must declare one Kubernetes dependency"
    return matches[0]


def test_kubernetes_requirements_preserve_google_auth_extra() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]
    project = next(
        Requirement(item)
        for item in dependencies
        if Requirement(item).name == "kubernetes"
    )
    requirements = [
        project,
        _requirements_file_kubernetes_requirement(
            ROOT / "container/deps/requirements.dev.txt"
        ),
        _requirements_file_kubernetes_requirement(
            ROOT / "container/deps/requirements.planner.txt"
        ),
    ]

    assert all(requirement.extras == {"google-auth"} for requirement in requirements)
    assert project.specifier == SpecifierSet(">=37.0.1,<38.0.0")
    assert all(
        requirement.specifier == SpecifierSet("==37.0.1")
        for requirement in requirements[1:3]
    )


def test_nemotron_recipe_uses_selected_wheel_kubernetes_requirement() -> None:
    dockerfile = (ROOT / "recipes/nemotron-3-nano-omni/Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "from importlib.metadata import requires" in dockerfile
    assert '"${KUBERNETES_REQUIREMENT}"' in dockerfile
    assert 'requires("kubernetes")' in dockerfile
    assert '"${DYNAMO_PYDANTIC_REQUIREMENT}"' in dockerfile
    assert '"${KUBERNETES_PYDANTIC_REQUIREMENT}"' in dockerfile
    assert '"pydantic<2.13"' not in dockerfile
