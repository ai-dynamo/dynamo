# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import types
from types import SimpleNamespace

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


@pytest.mark.parametrize("value", ["n", "N", "no", "false", "disable", "0"])
def test_false_enable_switches_to_collect_only(monkeypatch, value):
    monkeypatch.setenv("NIXL_TELEMETRY_ENABLE", value)
    monkeypatch.setenv("NIXL_TELEMETRY_EXPORTER", "prometheus")
    monkeypatch.setenv("NIXL_TELEMETRY_DIR", "/tmp/x")
    monkeypatch.setenv("NIXL_TELEMETRY_PROMETHEUS_PORT", "19090")

    assert allow_nixl_telemetry_capture() is True

    assert os.environ["NIXL_TELEMETRY_ENABLE"] == "y"
    assert os.environ["NIXL_TELEMETRY_EXPORTER"] == "NOP"
    assert "NIXL_TELEMETRY_DIR" not in os.environ
    assert os.environ["NIXL_TELEMETRY_PROMETHEUS_PORT"] == "19090"


@pytest.mark.parametrize("value", ["y", "1", "true", "bogus", " off ", "n "])
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


def test_headless_clears_false_enable_before_workers(monkeypatch):
    monkeypatch.setenv("NIXL_TELEMETRY_ENABLE", "n")
    monkeypatch.setenv("NIXL_TELEMETRY_EXPORTER", "prometheus")
    monkeypatch.setenv("NIXL_TELEMETRY_DIR", "/tmp/x")
    monkeypatch.setenv("NIXL_TELEMETRY_PROMETHEUS_PORT", "19090")

    seen = {}

    def run_headless(_args):
        seen["enable"] = os.environ.get("NIXL_TELEMETRY_ENABLE")
        seen["exporter"] = os.environ.get("NIXL_TELEMETRY_EXPORTER")
        seen["directory"] = os.environ.get("NIXL_TELEMETRY_DIR")
        seen["port"] = os.environ.get("NIXL_TELEMETRY_PROMETHEUS_PORT")

    def package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []
        module.__package__ = name
        return module

    # headless imports vLLM argument types. Stub that module when this test
    # runs without vLLM installed, and drop the stub import afterward.
    loaded_headless = "dynamo.vllm.headless" in sys.modules
    if not loaded_headless and "dynamo.vllm.args" not in sys.modules:
        args_mod = types.ModuleType("dynamo.vllm.args")
        args_mod.Config = object
        monkeypatch.setitem(sys.modules, "dynamo.vllm.args", args_mod)
    for name in ("vllm", "vllm.entrypoints", "vllm.entrypoints.cli"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, package(name))
    serve = types.ModuleType("vllm.entrypoints.cli.serve")
    serve.run_headless = run_headless
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.cli.serve", serve)

    try:
        from dynamo.vllm.headless import run_dynamo_headless

        run_dynamo_headless(
            SimpleNamespace(engine_args=SimpleNamespace(load_format="auto"))
        )
    finally:
        if not loaded_headless:
            sys.modules.pop("dynamo.vllm.headless", None)

    assert seen == {
        "enable": "y",
        "exporter": "NOP",
        "directory": None,
        "port": "19090",
    }
