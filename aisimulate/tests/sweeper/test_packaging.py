# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed-package contracts for the experimental Sweeper feature."""

import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

pytestmark = pytest.mark.timeout(30)


def test_aisimulate_distribution_publishes_aisimulate_sweeper_package():
    distribution = importlib.metadata.distribution("aisimulate")
    packaged_files = {str(path) for path in distribution.files or ()}

    assert distribution.metadata["Name"] == "aisimulate"
    assert importlib.util.find_spec("aisimulate.replay") is not None
    assert importlib.util.find_spec("aisimulate.sweeper") is not None
    # Editable installs expose only their .pth/dist-info records. In wheel-based
    # Planner CI, assert the artifact contains the canonical package and no alias.
    if any(path.startswith("aisimulate/") for path in packaged_files):
        assert any(path.startswith("aisimulate/sweeper/") for path in packaged_files)
        assert any(path.startswith("aisimulate/replay/") for path in packaged_files)
        assert not any(path.startswith("aisimulate/spica/") for path in packaged_files)
        assert not any(path.startswith("sweeper/") for path in packaged_files)


def test_aisimulate_has_no_console_script():
    distribution = importlib.metadata.distribution("aisimulate")

    assert all(entry.group != "console_scripts" for entry in distribution.entry_points)


def test_ai_dynamo_has_no_aisimulate_extra():
    distribution = importlib.metadata.distribution("ai-dynamo")

    extras = set(distribution.metadata.get_all("Provides-Extra", []))
    assert {"sweeper", "simulate", "simulation"}.isdisjoint(extras)


def test_aisimulate_has_no_dynamo_or_component_adapter_dependencies():
    distribution = importlib.metadata.distribution("aisimulate")

    requirements = distribution.requires or []
    names = {Requirement(requirement).name.lower() for requirement in requirements}
    assert "ai-dynamo" not in names
    assert "prometheus-api-client" not in names
    assert "filterpy" not in names
    assert "pmdarima" not in names
    assert "prophet" not in names


def test_importing_sweeper_does_not_import_dynamo():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import aisimulate.sweeper; "
                "assert not any(name == 'dynamo' or name.startswith('dynamo.') "
                "for name in sys.modules)"
            ),
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_ai_dynamo_registers_optional_sweeper_providers():
    distribution = importlib.metadata.distribution("ai-dynamo")
    entry_points = {
        entry_point.name: entry_point.value
        for entry_point in distribution.entry_points
        if entry_point.group == "aisimulate.sweep_config_providers"
    }

    assert entry_points == {
        "dynamo.planner": "dynamo.planner.simulation:create_provider",
        "dynamo.router": "dynamo.router.simulation:create_provider",
    }


def test_aisimulate_builds_a_planner_local_native_runtime_wheel():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    wheel_builder = (
        root.parent / "container/templates/wheel_builder.Dockerfile"
    ).read_text()
    release_workflow = (root.parent / ".github/workflows/release.yml").read_text()

    assert project["build-system"]["build-backend"] == "maturin"
    assert project["tool"]["maturin"]["module-name"] == "aisimulate._runtime"
    assert project["tool"]["maturin"]["profile"] == "release"
    assert '{% if target == "planner" %}' in wheel_builder
    assert "uv build --wheel --out-dir /opt/dynamo/dist /opt/dynamo/aisimulate" in (
        wheel_builder
    )
    assert "aisimulate-*" not in release_workflow


def test_profiler_does_not_publish_or_reexport_sweeper():
    assert importlib.util.find_spec("dynamo.profiler.sweeper") is None
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import dynamo.profiler; assert not hasattr(dynamo.profiler, 'sweeper')",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
