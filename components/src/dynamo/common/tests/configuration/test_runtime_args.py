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


def _clear_structural_tag_env(monkeypatch):
    for name in (
        "DYN_ENABLE_STRUCTURAL_TAG",
        "DYN_STRUCTURAL_TAG_SCOPE",
        "DYN_STRUCTURAL_TAG_SCHEMA",
        "DYN_STRUCTURAL_TAG",
    ):
        monkeypatch.delenv(name, raising=False)


def test_structural_tags_are_disabled_by_default(monkeypatch):
    _clear_structural_tag_env(monkeypatch)

    config, _ = _parse_runtime_args([])

    assert config.structural_tag is None


def test_structural_tag_flag_uses_default_config(monkeypatch):
    _clear_structural_tag_env(monkeypatch)

    config, help_text = _parse_runtime_args(["--dyn-structural-tag"])

    assert config.structural_tag == {
        "scope": "auto",
        "schema": "auto",
        "allow_tool_calls_with_structured_output": False,
        "exclude_special_tokens": None,
        "reasoning_boundary": "structural_tag",
        "tool_arguments_any_order": False,
    }
    assert "DYN_STRUCTURAL_TAG" in help_text
    assert "--dyn-enable-structural-tag" not in help_text


def test_structural_tag_json_config_enables_and_fills_defaults(monkeypatch):
    _clear_structural_tag_env(monkeypatch)

    config, _ = _parse_runtime_args(
        [
            "--dyn-structural-tag",
            '{"scope":"always","schema":"strict",'
            '"allow_tool_calls_with_structured_output":true,'
            '"exclude_special_tokens":false,'
            '"reasoning_boundary":"backend",'
            '"tool_arguments_any_order":true}',
        ]
    )

    assert config.structural_tag == {
        "scope": "always",
        "schema": "strict",
        "allow_tool_calls_with_structured_output": True,
        "exclude_special_tokens": False,
        "reasoning_boundary": "backend",
        "tool_arguments_any_order": True,
    }


def test_structural_tag_environment_is_supported(monkeypatch):
    _clear_structural_tag_env(monkeypatch)
    monkeypatch.setenv(
        "DYN_STRUCTURAL_TAG",
        '{"scope":"always","schema":"strict"}',
    )

    config, _ = _parse_runtime_args([])

    assert config.structural_tag is not None
    assert config.structural_tag["scope"] == "always"
    assert config.structural_tag["schema"] == "strict"


def test_legacy_structural_tag_options_resolve_to_canonical_config(monkeypatch, caplog):
    _clear_structural_tag_env(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=runtime_args.__name__):
        config, _ = _parse_runtime_args(
            [
                "--dyn-enable-structural-tag",
                "--dyn-structural-tag-scope",
                "always",
                "--dyn-structural-tag-schema",
                "strict",
            ]
        )

    assert config.structural_tag is not None
    assert config.structural_tag["scope"] == "always"
    assert config.structural_tag["schema"] == "strict"
    assert "deprecated" in caplog.text


def test_structural_tag_json_rejects_legacy_tuning(monkeypatch):
    _clear_structural_tag_env(monkeypatch)

    with pytest.raises(ValueError, match="cannot be combined"):
        _parse_runtime_args(
            [
                "--dyn-structural-tag",
                "{}",
                "--dyn-structural-tag-scope",
                "always",
            ]
        )


@pytest.mark.parametrize(
    "raw_config",
    [
        "{",
        "[]",
        '{"unknown":true}',
        '{"scope":"sometimes"}',
        '{"allow_tool_calls_with_structured_output":"yes"}',
    ],
)
def test_structural_tag_json_config_is_strict(monkeypatch, raw_config):
    _clear_structural_tag_env(monkeypatch)

    with pytest.raises(SystemExit):
        _parse_runtime_args(["--dyn-structural-tag", raw_config])


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
