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


# --- Per-model frontend admission override (DIS-2186) ---


@pytest.fixture()
def _clear_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(
        "DYN_REJECTION_FRONTEND_REQUEST_CONCURRENCY_LIMIT", raising=False
    )


def test_worker_concurrency_override_disabled_by_default(_clear_override_env):
    config, _ = _parse_runtime_args([])

    assert config.rejection_frontend_request_concurrency_limit is None


def test_worker_concurrency_override_parses(_clear_override_env):
    config, _ = _parse_runtime_args(
        ["--rejection-frontend-request-concurrency-limit", "64"]
    )

    assert config.rejection_frontend_request_concurrency_limit == 64


@pytest.mark.parametrize("value", ["0", "-1"])
def test_worker_concurrency_override_rejects_non_positive(
    _clear_override_env, value: str
):
    with pytest.raises(ValueError, match="must be a positive integer"):
        _parse_runtime_args(["--rejection-frontend-request-concurrency-limit", value])


def test_worker_concurrency_override_env_var(monkeypatch):
    monkeypatch.setenv("DYN_REJECTION_FRONTEND_REQUEST_CONCURRENCY_LIMIT", "32")

    config, _ = _parse_runtime_args([])

    assert config.rejection_frontend_request_concurrency_limit == 32
