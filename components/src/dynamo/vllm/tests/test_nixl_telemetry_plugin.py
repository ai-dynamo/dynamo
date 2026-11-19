# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the installed vLLM plugin in independently inherited worker processes."""

import json
import os
import subprocess
import sys
from importlib.metadata import entry_points

import pytest

from dynamo.vllm.nixl_telemetry import allow_nixl_telemetry_capture

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
]

_PLUGIN_NAME = "dynamo_nixl_telemetry"


def test_installed_general_plugin_resolves_to_telemetry_helper():
    """The installed wheel exposes the helper through vLLM's plugin group."""
    plugins = entry_points(group="vllm.general_plugins", name=_PLUGIN_NAME)
    assert len(plugins) == 1
    (plugin,) = plugins
    assert plugin.load() is allow_nixl_telemetry_capture


@pytest.mark.parametrize(
    "enable, expected_enable, expected_exporter",
    [("n", None, "NOP"), ("y", "y", "prometheus"), (None, None, "prometheus")],
)
def test_plugin_normalizes_inherited_worker_environment(
    monkeypatch, enable, expected_enable, expected_exporter
):
    """Workers normalize inherited defaults while preserving explicit opt-in."""
    for name in (
        "NIXL_TELEMETRY_ENABLE",
        "NIXL_TELEMETRY_EXPORTER",
        "NIXL_TELEMETRY_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    if enable is not None:
        monkeypatch.setenv("NIXL_TELEMETRY_ENABLE", enable)
    monkeypatch.setenv("NIXL_TELEMETRY_EXPORTER", "prometheus")

    # Ray starts before the driver normalizes its own environment. Its actors
    # inherit the original settings, even after the driver's helper runs.
    worker_env = os.environ.copy()
    allow_nixl_telemetry_capture()
    driver_env = {
        name: value for name, value in os.environ.items() if name.startswith("NIXL_")
    }
    for name, value in driver_env.items():
        worker_env.setdefault(name, value)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from importlib.metadata import entry_points; import json, os; "
            "plugin, = entry_points(group='vllm.general_plugins', "
            "name='dynamo_nixl_telemetry'); "
            "plugin.load()(); "
            "print(json.dumps({'enable': os.environ.get('NIXL_TELEMETRY_ENABLE'), "
            "'exporter': os.environ.get('NIXL_TELEMETRY_EXPORTER')}))",
        ],
        env=worker_env,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert json.loads(result.stdout) == {
        "enable": expected_enable,
        "exporter": expected_exporter,
    }
