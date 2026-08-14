# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test nightly version stamping across separately versioned Python wheels."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/apply_dev_version.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("apply_dev_version", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_project(path: Path, version: str, *, root: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dependencies = ""
    if root:
        dependencies = """
dependencies = [
    "ai-dynamo-runtime==1.4.0",
    "aisimulate==0.1.0; python_version < '3.13'",
]
"""
    path.write_text(
        f'[project]\nname = "fixture"\nversion = "{version}"\n{dependencies}',
        encoding="utf-8",
    )


def _create_repo(module: ModuleType, root: Path) -> None:
    for relative_path in module.PYPROJECT_TARGETS:
        version = "0.1.0" if relative_path == "aisimulate/pyproject.toml" else "1.4.0"
        _write_project(
            root / relative_path,
            version,
            root=(relative_path == "pyproject.toml"),
        )

    (root / "Cargo.toml").write_text(
        """
[workspace.package]
version = "1.4.0"

[workspace.dependencies]
dynamo-runtime = { path = "lib/runtime", version = "1.4.0" }
anyhow = { version = "1" }
""".lstrip(),
        encoding="utf-8",
    )
    for relative_path in module.SUBCRATE_CARGO_TARGETS:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '[package]\nname = "fixture"\nversion = "1.4.0"\n',
            encoding="utf-8",
        )


def _run_stamper(root: Path, suffix: str) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), suffix, str(root)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_ai_dynamo_pins_the_aisimulate_project_version() -> None:
    root_project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    aisimulate_project = tomllib.loads((ROOT / "aisimulate/pyproject.toml").read_text())

    requirement = next(
        Requirement(value)
        for value in root_project["project"]["dependencies"]
        if Requirement(value).name == "aisimulate"
    )
    assert str(requirement.specifier) == f"=={aisimulate_project['project']['version']}"
    assert str(requirement.marker) == 'python_version < "3.13"'


def test_nightly_stamps_aisimulate_and_root_dependency_pins(tmp_path: Path) -> None:
    module = _load_script()
    assert "aisimulate/pyproject.toml" in module.PYPROJECT_TARGETS
    _create_repo(module, tmp_path)

    _run_stamper(tmp_path, ".dev20260813")

    root_project = tomllib.loads((tmp_path / "pyproject.toml").read_text())
    aisimulate_project = tomllib.loads(
        (tmp_path / "aisimulate/pyproject.toml").read_text()
    )
    assert root_project["project"]["version"] == "1.4.0.dev20260813"
    assert root_project["project"]["dependencies"][:2] == [
        "ai-dynamo-runtime==1.4.0.dev20260813",
        "aisimulate==0.1.0.dev20260813; python_version < '3.13'",
    ]
    assert aisimulate_project["project"]["version"] == "0.1.0.dev20260813"


def test_nightly_stamping_is_idempotent(tmp_path: Path) -> None:
    module = _load_script()
    _create_repo(module, tmp_path)

    _run_stamper(tmp_path, ".dev20260813")
    first = {
        path.relative_to(tmp_path): path.read_text()
        for path in tmp_path.rglob("*.toml")
    }
    _run_stamper(tmp_path, ".dev20260813")
    second = {
        path.relative_to(tmp_path): path.read_text()
        for path in tmp_path.rglob("*.toml")
    }

    assert second == first
