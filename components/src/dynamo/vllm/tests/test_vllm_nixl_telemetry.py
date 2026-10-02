# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from dynamo.vllm.nixl_telemetry import allow_nixl_telemetry_capture

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_NIXL_ENV_VARS = (
    "NIXL_TELEMETRY_ENABLE",
    "NIXL_TELEMETRY_EXPORTER",
    "NIXL_TELEMETRY_DIR",
    "NIXL_TELEMETRY_PROMETHEUS_PORT",
)


@pytest.fixture(autouse=True)
def _clean_nixl_env(monkeypatch):
    for var in _NIXL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    "value", ["n", "N", "no", " off ", "false", "disable", "0"]
)
def test_false_enable_switches_to_collect_only(monkeypatch, value):
    monkeypatch.setenv("NIXL_TELEMETRY_ENABLE", value)
    monkeypatch.setenv("NIXL_TELEMETRY_EXPORTER", "prometheus")
    monkeypatch.setenv("NIXL_TELEMETRY_DIR", "/tmp/x")
    monkeypatch.setenv("NIXL_TELEMETRY_PROMETHEUS_PORT", "19090")

    assert allow_nixl_telemetry_capture() is True

    assert "NIXL_TELEMETRY_ENABLE" not in os.environ
    assert "NIXL_TELEMETRY_EXPORTER" not in os.environ
    assert "NIXL_TELEMETRY_DIR" not in os.environ
    assert os.environ["NIXL_TELEMETRY_PROMETHEUS_PORT"] == "19090"


@pytest.mark.parametrize("value", ["y", "1", "true", "bogus"])
def test_truthy_or_garbage_enable_untouched(monkeypatch, value):
    monkeypatch.setenv("NIXL_TELEMETRY_ENABLE", value)
    monkeypatch.setenv("NIXL_TELEMETRY_EXPORTER", "prometheus")
    monkeypatch.setenv("NIXL_TELEMETRY_DIR", "/tmp/x")

    assert allow_nixl_telemetry_capture() is False

    assert os.environ["NIXL_TELEMETRY_ENABLE"] == value
    assert os.environ["NIXL_TELEMETRY_EXPORTER"] == "prometheus"
    assert os.environ["NIXL_TELEMETRY_DIR"] == "/tmp/x"


def test_unset_enable_leaves_sink_envs(monkeypatch):
    monkeypatch.setenv("NIXL_TELEMETRY_EXPORTER", "prometheus")

    assert allow_nixl_telemetry_capture() is False

    assert "NIXL_TELEMETRY_ENABLE" not in os.environ
    assert os.environ["NIXL_TELEMETRY_EXPORTER"] == "prometheus"
