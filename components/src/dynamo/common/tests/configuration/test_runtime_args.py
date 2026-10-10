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


def test_structural_tags_default_on_for_supported_parsers(monkeypatch):
    _clear_structural_tag_env(monkeypatch)

    config, _ = _parse_runtime_args([])

    assert config.structural_tag is not None
    assert config.structural_tag["scope"] == "always"
    assert config.structural_tag["schema"] == "auto"


def test_structural_tag_schema_help_describes_omitted_strict(monkeypatch):
    _clear_structural_tag_env(monkeypatch)

    _, help_text = _parse_runtime_args([])

    help_text = " ".join(help_text.split())
    assert "strict omitted or true" in help_text
    assert "overriding strict=false" in help_text


def test_structural_tag_global_opt_out(monkeypatch):
    _clear_structural_tag_env(monkeypatch)
    monkeypatch.setenv("DYN_ENABLE_STRUCTURAL_TAG", "false")

    config, _ = _parse_runtime_args([])

    assert config.dyn_enable_structural_tag is False
    assert config.structural_tag is None


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


def _clear_structural_tag_env(monkeypatch):
    for name in (
        "DYN_ENABLE_STRUCTURAL_TAG",
        "DYN_STRUCTURAL_TAG_SCOPE",
        "DYN_STRUCTURAL_TAG_SCHEMA",
        "DYN_STRUCTURAL_TAG",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_structural_tag_policy_does_not_warn_about_legacy_options(
    monkeypatch, caplog
):
    _clear_structural_tag_env(monkeypatch)

    config, _ = _parse_runtime_args([])

    assert config.structural_tag is not None
    assert config.structural_tag["scope"] == "always"
    assert "deprecated" not in caplog.text
    assert config.dyn_enable_structural_tag is None


def test_structural_tag_flag_uses_default_config(monkeypatch):
    _clear_structural_tag_env(monkeypatch)

    config, help_text = _parse_runtime_args(["--dyn-structural-tag"])

    assert config.structural_tag == {
        "scope": "always",
        "schema": "auto",
        "allow_tool_calls_with_structured_output": False,
        "exclude_special_tokens": None,
        "reasoning_boundary": "auto",
        "tool_arguments_any_order": False,
    }
    assert "DYN_STRUCTURAL_TAG" in help_text
    assert "--dyn-enable-structural-tag" not in help_text


@pytest.mark.parametrize(
    "argv", [["--dyn-structural-tag", "false"], ["--no-dyn-enable-structural-tag"]]
)
def test_structural_tag_cli_opt_out(monkeypatch, argv):
    _clear_structural_tag_env(monkeypatch)
    config, _ = _parse_runtime_args(argv)
    assert config.structural_tag is None


@pytest.mark.parametrize(
    "argv, scope, schema",
    [
        (["--dyn-structural-tag-scope", "auto"], "auto", "auto"),
        (["--dyn-structural-tag-schema", "strict"], "always", "strict"),
    ],
)
def test_legacy_tuning_alone_keeps_default_guidance_enabled(
    monkeypatch, argv, scope, schema
):
    _clear_structural_tag_env(monkeypatch)
    config, _ = _parse_runtime_args(argv)
    assert config.structural_tag is not None
    assert config.structural_tag["scope"] == scope
    assert config.structural_tag["schema"] == schema


def test_structural_tag_json_config_enables_and_fills_defaults(monkeypatch):
    _clear_structural_tag_env(monkeypatch)
    monkeypatch.setattr(runtime_args, "distribution_version", lambda _: "0.2.3")

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


@pytest.mark.parametrize(
    ("xgrammar_version", "supported"), [("0.2.1", False), ("0.2.3", True)]
)
def test_structural_tag_any_order_requires_supported_xgrammar(
    monkeypatch, xgrammar_version, supported
):
    _clear_structural_tag_env(monkeypatch)
    monkeypatch.setattr(
        runtime_args, "distribution_version", lambda _: xgrammar_version
    )
    argv = ["--dyn-structural-tag", '{"tool_arguments_any_order":true}']

    if supported:
        config, _ = _parse_runtime_args(argv)
        assert config.structural_tag["tool_arguments_any_order"] is True
    else:
        with pytest.raises(ValueError, match="requires XGrammar >= 0.2.3"):
            _parse_runtime_args(argv)


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
