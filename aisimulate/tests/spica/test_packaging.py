# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed-package contracts for the experimental Spica feature."""

import importlib.metadata
import importlib.util
import re
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


def test_aisimulate_distribution_publishes_aisimulate_spica_package():
    distribution = importlib.metadata.distribution("aisimulate")
    packaged_files = {str(path) for path in distribution.files or ()}

    assert distribution.metadata["Name"] == "aisimulate"
    assert importlib.util.find_spec("aisimulate.spica") is not None
    # Editable installs expose only their .pth/dist-info records. In wheel-based
    # Planner CI, assert the artifact contains the canonical package and no alias.
    if any(path.startswith("aisimulate/") for path in packaged_files):
        assert any(path.startswith("aisimulate/spica/") for path in packaged_files)
        assert not any(path.startswith("spica/") for path in packaged_files)


def test_aisimulate_has_no_console_script():
    distribution = importlib.metadata.distribution("aisimulate")

    assert all(entry.group != "console_scripts" for entry in distribution.entry_points)


def test_ai_dynamo_has_no_spica_extra():
    distribution = importlib.metadata.distribution("ai-dynamo")

    assert "spica" not in distribution.metadata.get_all("Provides-Extra", [])


def test_aisimulate_has_no_dynamo_or_component_adapter_dependencies():
    distribution = importlib.metadata.distribution("aisimulate")

    requirements = distribution.requires or []
    names = {Requirement(requirement).name.lower() for requirement in requirements}
    assert "ai-dynamo" not in names
    assert "prometheus-api-client" not in names
    assert "filterpy" not in names
    assert "pmdarima" not in names
    assert "prophet" not in names


def test_importing_spica_does_not_import_dynamo():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import aisimulate.spica; "
                "assert not any(name == 'dynamo' or name.startswith('dynamo.') "
                "for name in sys.modules)"
            ),
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_ai_dynamo_publishes_optional_spica_adapters():
    root_pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    match = re.search(
        r'^\[project\.entry-points\."aisimulate\.adapters"\]\n'
        r"(?P<body>.*?)(?=^\[|\Z)",
        root_pyproject.read_text(),
        flags=re.MULTILINE | re.DOTALL,
    )

    assert match is not None
    body = match.group("body")
    assert '"dynamo.planner" = "dynamo.planner.simulation:create_adapter"' in body
    assert '"dynamo.router" = "dynamo.router.simulation:create_adapter"' in body


def test_ai_dynamo_simulation_extra_installs_adapter_runtime_dependencies():
    root_pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    project = tomllib.loads(root_pyproject.read_text())["project"]
    requirements = {
        Requirement(requirement).name.lower()
        for requirement in project["optional-dependencies"]["simulation"]
    }

    assert requirements.issuperset(
        {
            "aisimulate",
            "filterpy",
            "pmdarima",
            "prometheus-api-client",
            "prophet",
            "scikit-learn",
            "scipy",
        }
    )


def test_profiler_does_not_publish_or_reexport_spica():
    assert importlib.util.find_spec("dynamo.profiler.spica") is None
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import dynamo.profiler; assert not hasattr(dynamo.profiler, 'spica')",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
