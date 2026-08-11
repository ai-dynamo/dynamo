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


def test_event_plane_documents_zmq_default_and_supports_cli_and_env(monkeypatch):
    monkeypatch.delenv("DYN_EVENT_PLANE", raising=False)

    default_config, help_text = _parse_runtime_args([])

    assert default_config.event_plane is None
    assert "defaults to 'zmq'" in help_text
    assert "DYN_EVENT_PLANE" in help_text

    monkeypatch.setenv("DYN_EVENT_PLANE", "nats")
    env_config, _ = _parse_runtime_args([])
    cli_config, _ = _parse_runtime_args(["--event-plane", "zmq"])

    assert env_config.event_plane == "nats"
    assert cli_config.event_plane == "zmq"


def test_event_plane_rejects_unknown_value(monkeypatch):
    monkeypatch.delenv("DYN_EVENT_PLANE", raising=False)

    with pytest.raises(SystemExit):
        _parse_runtime_args(["--event-plane", "invalid"])


def test_engine_request_limit_supports_environment_and_cli_precedence(monkeypatch):
    monkeypatch.setenv("DYN_ENGINE_REQUEST_LIMIT", "8")

    env_config, help_text = _parse_runtime_args([])
    assert env_config.engine_request_limit == 8
    assert os.environ["DYN_ENGINE_REQUEST_LIMIT"] == "8"

    cli_config, _ = _parse_runtime_args(["--engine-request-limit", "16"])
    assert cli_config.engine_request_limit == 16
    assert os.environ["DYN_ENGINE_REQUEST_LIMIT"] == "16"
    assert "DYN_ENGINE_REQUEST_LIMIT" in help_text


@pytest.mark.parametrize("limit", [0, -1])
def test_engine_request_limit_rejects_non_positive_values(monkeypatch, limit):
    monkeypatch.delenv("DYN_ENGINE_REQUEST_LIMIT", raising=False)

    with pytest.raises(ValueError, match="must be a positive integer"):
        _parse_runtime_args(["--engine-request-limit", str(limit)])

    assert "DYN_ENGINE_REQUEST_LIMIT" not in os.environ


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
