# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed-package contracts for the experimental Spica feature."""

import importlib.metadata
import importlib.util
import subprocess
import sys

import pytest

pytestmark = pytest.mark.timeout(30)


def test_aisimulate_distribution_publishes_canonical_spica_package():
    distribution = importlib.metadata.distribution("aisimulate")

    assert distribution.metadata["Name"] == "aisimulate"
    assert importlib.util.find_spec("spica") is not None


def test_aisimulate_has_no_console_script():
    distribution = importlib.metadata.distribution("aisimulate")

    assert all(entry.group != "console_scripts" for entry in distribution.entry_points)


def test_ai_dynamo_has_no_spica_extra():
    distribution = importlib.metadata.distribution("ai-dynamo")

    assert "spica" not in distribution.metadata.get_all("Provides-Extra", [])


def test_aisimulate_declares_planner_monitoring_dependency():
    distribution = importlib.metadata.distribution("aisimulate")

    requirements = distribution.requires or []
    assert any(
        requirement.startswith("prometheus-api-client==0.6.0")
        for requirement in requirements
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
