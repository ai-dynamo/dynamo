# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared Dynamo runtime arguments."""

import argparse
import logging
import os

import pytest

import dynamo.common.configuration.groups.runtime_args as runtime_args
from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _parse_runtime_args(argv: list[str]) -> tuple[DynamoRuntimeConfig, str]:
    parser = argparse.ArgumentParser()
    DynamoRuntimeArgGroup().add_arguments(parser)
    args = parser.parse_args(argv)
    config = DynamoRuntimeConfig.from_cli_args(args)
    config.validate()
    return config, parser.format_help()


def test_fpm_trace_defaults_disabled(monkeypatch):
    monkeypatch.delenv("DYN_FPM_TRACE", raising=False)

    config, _ = _parse_runtime_args([])

    assert config.fpm_trace is False
    assert "DYN_FPM_TRACE" not in os.environ


def test_kv_state_endpoint_supports_cli_and_env(monkeypatch):
    monkeypatch.setenv("DYN_KV_STATE_ENDPOINT", "dynamo/kv/events")
    env_config, help_text = _parse_runtime_args([])
    cli_config, _ = _parse_runtime_args(["--kv-state-endpoint", "other/cache/updates"])

    assert env_config.kv_state_endpoint == "dynamo/kv/events"
    assert cli_config.kv_state_endpoint == "other/cache/updates"
    assert "--kv-state-endpoint" in help_text
    assert "DYN_KV_STATE_ENDPOINT" in help_text


def test_response_plane_defaults_to_tcp_and_accepts_quic(monkeypatch):
    monkeypatch.delenv("DYN_RESPONSE_PLANE", raising=False)

    default_config, help_text = _parse_runtime_args([])
    quic_config, _ = _parse_runtime_args(["--response-plane", "quic"])
    monkeypatch.setenv("DYN_RESPONSE_PLANE", "quic")
    env_config, _ = _parse_runtime_args([])

    assert default_config.response_plane == "tcp"
    assert quic_config.response_plane == "quic"
    assert env_config.response_plane == "quic"
    assert os.environ["DYN_RESPONSE_PLANE"] == "quic"
    assert "--response-plane" in help_text


def test_response_plane_rejects_invalid_value():
    with pytest.raises(SystemExit):
        _parse_runtime_args(["--response-plane", "invalid"])


def test_fpm_trace_env_enables_and_is_canonicalized(monkeypatch):
    monkeypatch.setenv("DYN_FPM_TRACE", "on")

    config, _ = _parse_runtime_args([])

    assert config.fpm_trace is True
    assert os.environ["DYN_FPM_TRACE"] == "1"


def test_fpm_trace_env_is_trimmed(monkeypatch):
    monkeypatch.setenv("DYN_FPM_TRACE", " true ")

    config, _ = _parse_runtime_args([])

    assert config.fpm_trace is True
    assert os.environ["DYN_FPM_TRACE"] == "1"


def test_invalid_fpm_trace_warns_once_and_is_disabled(monkeypatch, caplog):
    monkeypatch.setenv("DYN_FPM_TRACE", "sometimes")
    monkeypatch.setattr(runtime_args, "_fpm_trace_invalid_warning_emitted", False)

    with caplog.at_level(logging.WARNING, logger=runtime_args.__name__):
        config, _ = _parse_runtime_args([])
        monkeypatch.setenv("DYN_FPM_TRACE", "still-invalid")
        _parse_runtime_args([])

    assert config.fpm_trace is False
    assert os.environ["DYN_FPM_TRACE"] == "0"
    assert caplog.text.count("Invalid DYN_FPM_TRACE value") == 1


def test_explicit_fpm_port_preserves_precedence_over_invalid_trace(monkeypatch, caplog):
    monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "23456")
    monkeypatch.setenv("DYN_FPM_TRACE", "sometimes")
    monkeypatch.setattr(runtime_args, "_fpm_trace_invalid_warning_emitted", False)

    with caplog.at_level(logging.WARNING, logger=runtime_args.__name__):
        config, _ = _parse_runtime_args([])

    assert config.fpm_trace is False
    assert os.environ["DYN_FPM_TRACE"] == "0"
    assert "Invalid DYN_FPM_TRACE value" not in caplog.text


def test_fpm_trace_cli_enables_and_is_exported(monkeypatch):
    monkeypatch.delenv("DYN_FPM_TRACE", raising=False)

    config, _ = _parse_runtime_args(["--fpm-trace"])

    assert config.fpm_trace is True
    assert os.environ["DYN_FPM_TRACE"] == "1"


def test_no_fpm_trace_cli_overrides_enabled_env(monkeypatch):
    monkeypatch.setenv("DYN_FPM_TRACE", "true")

    config, _ = _parse_runtime_args(["--no-fpm-trace"])

    assert config.fpm_trace is False
    assert os.environ["DYN_FPM_TRACE"] == "0"


def test_fpm_trace_help_lists_flag_and_env(monkeypatch):
    monkeypatch.delenv("DYN_FPM_TRACE", raising=False)

    _, help_text = _parse_runtime_args([])

    assert "--fpm-trace" in help_text
    assert "--no-fpm-trace" in help_text
    assert "DYN_FPM_TRACE" in help_text


@pytest.mark.parametrize("mode", ["enabled", "disabled"])
def test_default_thinking_mode_cli(mode, monkeypatch):
    monkeypatch.delenv("DYN_DEFAULT_THINKING_MODE", raising=False)

    config, help_text = _parse_runtime_args(["--dyn-default-thinking-mode", mode])

    assert config.dyn_default_thinking_mode == mode
    assert "DYN_DEFAULT_THINKING_MODE" in help_text


def test_default_thinking_mode_env(monkeypatch):
    monkeypatch.setenv("DYN_DEFAULT_THINKING_MODE", "disabled")

    config, _ = _parse_runtime_args([])

    assert config.dyn_default_thinking_mode == "disabled"


def test_default_thinking_mode_rejects_invalid_value(monkeypatch):
    monkeypatch.delenv("DYN_DEFAULT_THINKING_MODE", raising=False)

    with pytest.raises(SystemExit):
        _parse_runtime_args(["--dyn-default-thinking-mode", "adaptive"])


_ENGINE_LIMIT_ENV = (
    "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT",
    "DYN_ENGINE_REQUEST_LIMIT",
    "DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT",
)


@pytest.fixture
def engine_limit_env(monkeypatch):
    """Clear the gate's engine-limit variables, so nothing ambient decides the
    outcome and whatever validate() writes is restored afterwards."""
    for name in (*_ENGINE_LIMIT_ENV, "DYN_RESPONSE_PLANE"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_engine_request_limit_is_left_to_the_gate_by_default(engine_limit_env):
    config, _ = _parse_runtime_args([])

    assert config.engine_request_limit is None
    assert not any(name in os.environ for name in _ENGINE_LIMIT_ENV)


def test_engine_request_limit_flag_outranks_both_environment_names(engine_limit_env):
    engine_limit_env.setenv("DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT", "5")
    engine_limit_env.setenv("DYN_ENGINE_REQUEST_LIMIT", "3")

    config, help_text = _parse_runtime_args(["--engine-request-limit", "7"])

    assert config.engine_request_limit == 7
    # Exported under the name the gate reads first, and only as the
    # full-request limit.
    assert os.environ["DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT"] == "7"
    assert os.environ["DYN_ENGINE_REQUEST_LIMIT"] == "3"
    assert "DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT" not in os.environ
    assert "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT" in help_text
    assert "DYN_ENGINE_REQUEST_LIMIT" in help_text


@pytest.mark.parametrize(
    "environment",
    [
        {"DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT": "5"},
        {"DYN_ENGINE_REQUEST_LIMIT": "3"},
        # An invalid alias shadowed by a valid canonical value is never read.
        {
            "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT": "5",
            "DYN_ENGINE_REQUEST_LIMIT": "invalid",
        },
    ],
)
def test_engine_request_limit_environment_is_resolved_by_the_gate(
    engine_limit_env, environment
):
    for name, value in environment.items():
        engine_limit_env.setenv(name, value)

    config, _ = _parse_runtime_args([])

    assert config.engine_request_limit is None
    assert {name: os.environ.get(name) for name in _ENGINE_LIMIT_ENV} == {
        name: environment.get(name) for name in _ENGINE_LIMIT_ENV
    }


@pytest.mark.parametrize("value", ["0", "-1"])
def test_engine_request_limit_rejects_non_positive_values(engine_limit_env, value):
    with pytest.raises(ValueError, match="must be a positive integer"):
        _parse_runtime_args(["--engine-request-limit", value])

    assert "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT" not in os.environ
