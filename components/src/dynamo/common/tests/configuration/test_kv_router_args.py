# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import warnings
from pathlib import Path

import pytest

from dynamo.common.configuration.groups import kv_router_args
from dynamo.common.configuration.groups.ais_perf_args import (
    AisPerfArgGroup,
    AisPerfConfigBase,
)
from dynamo.common.configuration.groups.kv_router_args import (
    REMOVED_KV_ROUTER_ENV_VARS,
    REMOVED_KV_ROUTER_FLAGS,
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.common.configuration.groups.router_args import (
    REMOVED_ROUTER_ENV_VARS,
    REMOVED_ROUTER_FLAGS,
)
from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _clear_rejection_threshold_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",
        "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
        "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
        "DYN_ADMISSION_CONTROL",
        "DYN_ROUTER_QUEUE_THRESHOLD",
        "DYN_ROUTER_SESSION_AFFINITY_MODE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_ais_perf_moe_cli_flows_to_binding_kwargs() -> None:
    parser = argparse.ArgumentParser()
    AisPerfArgGroup().add_arguments(parser)

    args = parser.parse_args(
        [
            "--aic-backend",
            "vllm",
            "--aic-system",
            "h200_sxm",
            "--aic-model-path",
            "moonshotai/Kimi-K2-Instruct",
            "--aic-tp-size",
            "2",
            "--aic-moe-tp-size",
            "2",
            "--aic-moe-ep-size",
            "1",
            "--aic-attention-dp-size",
            "1",
        ]
    )

    config = AisPerfConfigBase.from_cli_args(args)

    payload = config.ais_perf_kwargs()["config"]
    assert payload["backend"] == "vllm"
    assert payload["model"] == "moonshotai/Kimi-K2-Instruct"
    assert payload["tp"] == 2
    assert payload["moe_tp_size"] == 2
    assert payload["moe_ep_size"] == 1
    assert payload["attention_dp"] == 1


def test_ais_perf_moe_env_flows_to_binding_kwargs(monkeypatch) -> None:
    monkeypatch.setenv("DYN_AIS_BACKEND", "vllm")
    monkeypatch.setenv("DYN_AIS_SYSTEM", "h200_sxm")
    monkeypatch.setenv("DYN_AIS_MODEL_PATH", "moonshotai/Kimi-K2-Instruct")
    monkeypatch.setenv("DYN_AIS_TP_SIZE", "2")
    monkeypatch.setenv("DYN_AIS_MOE_TP_SIZE", "2")
    monkeypatch.setenv("DYN_AIS_MOE_EP_SIZE", "1")
    monkeypatch.setenv("DYN_AIS_ATTENTION_DP_SIZE", "1")

    parser = argparse.ArgumentParser()
    AisPerfArgGroup().add_arguments(parser)
    args = parser.parse_args([])

    config = AisPerfConfigBase.from_cli_args(args)

    assert config.ais_perf_kwargs()["config"]["moe_tp_size"] == 2
    assert config.ais_perf_kwargs()["config"]["moe_ep_size"] == 1
    assert config.ais_perf_kwargs()["config"]["attention_dp"] == 1


def test_ais_mtp_cli_documents_conditional_rates_and_seed() -> None:
    parser = argparse.ArgumentParser()
    AisPerfArgGroup().add_arguments(parser)
    args = parser.parse_args(
        [
            "--aic-nextn",
            "3",
            "--aic-nextn-accept-rates",
            "1,0.5",
            "--aic-mtp-seed",
            "99",
        ]
    )

    config = AisPerfConfigBase.from_cli_args(args)
    assert config.ais_nextn == 3
    assert config.ais_nextn_accept_rates == "1,0.5"
    assert config.ais_mtp_seed == 99
    assert "all earlier drafts were accepted" in " ".join(parser.format_help().split())


def test_session_prefix_index_is_opt_in() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    default_kwargs = KvRouterConfigBase.from_cli_args(
        parser.parse_args([])
    ).kv_router_kwargs()
    assert default_kwargs["enable_session_prefix_index"] is False

    enabled_kwargs = KvRouterConfigBase.from_cli_args(
        parser.parse_args(["--enable-session-prefix-index"])
    ).kv_router_kwargs()
    assert enabled_kwargs["enable_session_prefix_index"] is True

    disabled_kwargs = KvRouterConfigBase.from_cli_args(
        parser.parse_args(["--no-enable-session-prefix-index"])
    ).kv_router_kwargs()
    assert disabled_kwargs["enable_session_prefix_index"] is False


def test_session_prefix_index_environment_flows_to_binding_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DYN_ENABLE_SESSION_PREFIX_INDEX", "true")
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    kwargs = KvRouterConfigBase.from_cli_args(parser.parse_args([])).kv_router_kwargs()
    assert kwargs["enable_session_prefix_index"] is True

    overridden = KvRouterConfigBase.from_cli_args(
        parser.parse_args(["--no-enable-session-prefix-index"])
    ).kv_router_kwargs()
    assert overridden["enable_session_prefix_index"] is False


def test_tracking_hash_cli_flows_to_binding_kwargs(tmp_path: Path) -> None:
    key_file = tmp_path / "tracking-key"
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(
        [
            "--router-tracking-hash",
            "keyed-xxh3-v1",
            "--router-tracking-key-file",
            str(key_file),
            "--router-tracking-key-id",
            "2026-01",
        ]
    )
    kwargs = KvRouterConfigBase.from_cli_args(args).kv_router_kwargs()

    assert kwargs["router_tracking_hash"] == "keyed-xxh3-v1"
    assert kwargs["router_tracking_key_file"] == str(key_file)
    assert kwargs["router_tracking_key_id"] == "2026-01"


def test_tracking_hash_environment_flows_to_binding_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key_file = tmp_path / "tracking-key"
    monkeypatch.setenv("DYN_ROUTER_TRACKING_HASH", "keyed-xxh3-v1")
    monkeypatch.setenv("DYN_ROUTER_TRACKING_KEY_FILE", str(key_file))
    monkeypatch.setenv("DYN_ROUTER_TRACKING_KEY_ID", "2026-01")
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args([])
    kwargs = KvRouterConfigBase.from_cli_args(args).kv_router_kwargs()

    assert kwargs["router_tracking_hash"] == "keyed-xxh3-v1"
    assert kwargs["router_tracking_key_file"] == str(key_file)
    assert kwargs["router_tracking_key_id"] == "2026-01"


def test_policy_config_cli_overrides_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_policy_path = str(tmp_path / "env-policy.yaml")
    explicit_policy_path = str(tmp_path / "explicit-policy.yaml")
    Path(explicit_policy_path).write_text("router: {}\n")
    monkeypatch.setenv("DYN_ROUTER_POLICY_CONFIG", env_policy_path)
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--router-policy-config", explicit_policy_path])
    config = KvRouterConfigBase.from_cli_args(args)

    assert config.kv_router_kwargs()["router_policy_config"] == explicit_policy_path


def _frontend_config(argv: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(argv))
    config.validate()
    return config


@pytest.mark.parametrize(
    ("argv", "policy_file", "expected_mode"),
    [
        pytest.param([], False, "round-robin", id="neither"),
        pytest.param([], True, "kv", id="file-alone"),
        pytest.param(["--router-mode", "kv"], False, "kv", id="kv"),
        pytest.param(["--router-mode", "kv"], True, "kv", id="kv-with-file"),
        pytest.param(
            ["--router-mode", "least-loaded"], False, "least-loaded", id="other-mode"
        ),
    ],
)
def test_frontend_routes_by_a_router_mode_or_a_policy_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
    policy_file: bool,
    expected_mode: str,
) -> None:
    monkeypatch.delenv("DYN_ROUTER_MODE", raising=False)
    monkeypatch.delenv("DYN_ROUTER_POLICY_CONFIG", raising=False)
    if policy_file:
        path = tmp_path / "policy.yaml"
        path.write_text("worker_selection:\n  decode: round-robin\n")
        argv = [*argv, "--router-policy-config", str(path)]

    config = _frontend_config(argv)

    assert config.router_mode == expected_mode
    kwargs = config.kv_router_kwargs()
    assert (kwargs["router_policy_config"] is not None) == policy_file


@pytest.mark.parametrize("router_mode", ["round-robin", "direct"])
def test_frontend_rejects_a_policy_file_with_a_router_mode_other_than_kv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, router_mode: str
) -> None:
    monkeypatch.delenv("DYN_ROUTER_MODE", raising=False)
    path = tmp_path / "policy.yaml"
    path.write_text("worker_selection:\n  decode: round-robin\n")

    with pytest.raises(ValueError, match="mutually exclusive"):
        _frontend_config(
            ["--router-mode", router_mode, "--router-policy-config", str(path)]
        )


def test_frontend_reasoning_field_name_cli_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DYN_REASONING_FIELD_NAME", raising=False)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    assert config.reasoning_field_name == "reasoning_content"

    monkeypatch.setenv("DYN_REASONING_FIELD_NAME", "reasoning")
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    assert config.reasoning_field_name == "reasoning"

    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(
        parser.parse_args(["--reasoning-field-name", "reasoning_content"])
    )
    assert config.reasoning_field_name == "reasoning_content"


def test_frontend_reasoning_field_name_rejects_invalid_choice() -> None:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--reasoning-field-name", "invalid"])


def test_frontend_response_plane_defaults_to_unset_and_accepts_quic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DYN_RESPONSE_PLANE", raising=False)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    default_config = FrontendConfig.from_cli_args(parser.parse_args([]))
    quic_config = FrontendConfig.from_cli_args(
        parser.parse_args(["--response-plane", "quic"])
    )
    monkeypatch.setenv("DYN_RESPONSE_PLANE", "quic")
    env_parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(env_parser)
    env_config = FrontendConfig.from_cli_args(env_parser.parse_args([]))

    assert default_config.response_plane is None
    assert quic_config.response_plane == "quic"
    assert parser.parse_args(["--response-plane", "velo"]).response_plane == "velo"
    assert env_config.response_plane == "quic"

    with pytest.raises(SystemExit):
        parser.parse_args(["--response-plane", "invalid"])


def test_conditional_disagg_config_cli_lowers_to_router_kwargs() -> None:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    config = FrontendConfig.from_cli_args(
        parser.parse_args(
            [
                "--router-mode",
                "kv",
                "--router-conditional-disagg",
                "--router-conditional-disagg-config",
                '{"policy":"isl_or_load","eff_isl_threshold":4096,'
                '"eff_isl_ratio_threshold":0.8,"prefill_busy_threshold":16,'
                '"decode_busy_threshold":0.9}',
            ]
        )
    )
    config.validate()
    kwargs = config.kv_router_kwargs()

    assert kwargs["conditional_disagg_enabled"] is True
    assert kwargs["conditional_disagg_policy"] == "isl_or_load"
    assert kwargs["conditional_disagg_eff_isl_threshold"] == 4096
    assert kwargs["conditional_disagg_eff_isl_ratio_threshold"] == 0.8
    assert kwargs["conditional_disagg_prefill_busy_threshold"] == 16
    assert kwargs["conditional_disagg_decode_busy_threshold"] == 0.9


def test_conditional_disagg_requires_router_kv_events() -> None:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    config = FrontendConfig.from_cli_args(
        parser.parse_args(
            [
                "--router-mode",
                "kv",
                "--router-conditional-disagg",
                "--no-router-kv-events",
            ]
        )
    )

    with pytest.raises(ValueError, match="requires --router-kv-events"):
        config.validate()


def test_conditional_disagg_prefill_load_errors_without_busy_threshold() -> None:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    config = FrontendConfig.from_cli_args(
        parser.parse_args(
            [
                "--router-mode",
                "kv",
                "--router-conditional-disagg",
                "--router-conditional-disagg-config",
                '{"policy":"prefill_load"}',
            ]
        )
    )
    with pytest.raises(
        ValueError,
        match="needs prefill_busy_threshold; set it in --router-conditional-disagg-config",
    ):
        config.validate()


@pytest.mark.parametrize(
    ("config_json", "message"),
    [
        ("not-json", "must be a JSON object"),
        ("[]", "must be a JSON object"),
        ('{"unknown":1}', "unknown field"),
        ('{"policy":1}', "policy must be a string"),
        ('{"eff_isl_threshold":"4096"}', "eff_isl_threshold must be an integer"),
        (
            '{"eff_isl_ratio_threshold":"0.8"}',
            "eff_isl_ratio_threshold must be a number",
        ),
        (
            '{"eff_isl_ratio_threshold":null}',
            "eff_isl_ratio_threshold must be a number",
        ),
        ('{"prefill_busy_threshold":true}', "prefill_busy_threshold must be a number"),
    ],
)
def test_conditional_disagg_config_rejects_invalid_json(
    config_json: str, message: str
) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        kv_router_args._conditional_disagg_config_arg(config_json)


def test_frontend_rejection_thresholds_default_to_none(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _clear_rejection_threshold_env(monkeypatch)
    caplog.set_level("INFO")
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    config.validate()

    assert not hasattr(config, "admission_control")
    assert config.active_decode_blocks_threshold is None
    assert config.active_prefill_tokens_threshold_frac is None
    assert config.router_kwargs() == {
        "active_decode_blocks_threshold": None,
        "active_prefill_tokens_threshold_frac": None,
        "session_affinity_ttl_secs": None,
        "session_affinity_mode": "hard",
    }
    assert "busy-worker rejection disabled" in caplog.text


@pytest.mark.parametrize(
    ("flag", "value", "field", "expected"),
    [
        (
            "--active-decode-blocks-threshold",
            "0.5",
            "active_decode_blocks_threshold",
            0.5,
        ),
        (
            "--active-prefill-tokens-threshold-frac",
            "2.0",
            "active_prefill_tokens_threshold_frac",
            2.0,
        ),
    ],
)
def test_each_cli_rejection_threshold_is_independently_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    flag: str,
    value: str,
    field: str,
    expected: float | int,
) -> None:
    _clear_rejection_threshold_env(monkeypatch)
    caplog.set_level("INFO")
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    config = FrontendConfig.from_cli_args(parser.parse_args([flag, value]))
    config.validate()

    thresholds = {
        "active_decode_blocks_threshold": config.active_decode_blocks_threshold,
        "active_prefill_tokens_threshold_frac": config.active_prefill_tokens_threshold_frac,
    }
    assert thresholds[field] == expected
    assert all(value is None for name, value in thresholds.items() if name != field)
    assert f"{flag}={expected}" in caplog.text


@pytest.mark.parametrize(
    ("env_var", "field", "expected"),
    [
        (
            "DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",
            "active_decode_blocks_threshold",
            0.6,
        ),
        (
            "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
            "active_prefill_tokens_threshold_frac",
            3.0,
        ),
    ],
)
def test_each_environment_rejection_threshold_is_independently_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    field: str,
    expected: float | int,
) -> None:
    _clear_rejection_threshold_env(monkeypatch)
    monkeypatch.setenv(env_var, str(expected))
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    config.validate()

    thresholds = {
        "active_decode_blocks_threshold": config.active_decode_blocks_threshold,
        "active_prefill_tokens_threshold_frac": config.active_prefill_tokens_threshold_frac,
    }
    assert thresholds[field] == expected
    assert all(value is None for name, value in thresholds.items() if name != field)


def test_all_rejection_thresholds_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_rejection_threshold_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    config = FrontendConfig.from_cli_args(
        parser.parse_args(
            [
                "--active-decode-blocks-threshold",
                "0.5",
                "--active-prefill-tokens-threshold-frac",
                "2.0",
            ]
        )
    )
    config.validate()

    assert config.router_kwargs() == {
        "active_decode_blocks_threshold": 0.5,
        "active_prefill_tokens_threshold_frac": 2.0,
        "session_affinity_ttl_secs": None,
        "session_affinity_mode": "hard",
    }


@pytest.mark.parametrize(
    ("flag", "expected_value"),
    [("--enforce-disagg", True), ("--no-enforce-disagg", False)],
)
def test_enforce_disagg_cli_is_deprecated_and_not_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    flag: str,
    expected_value: bool,
) -> None:
    monkeypatch.delenv("DYN_ENFORCE_DISAGG", raising=False)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    config = FrontendConfig.from_cli_args(parser.parse_args([flag]))
    config.validate()
    kwargs = config.router_kwargs()

    assert config.enforce_disagg is expected_value
    assert "enforce_disagg" not in kwargs
    warning = f"{flag} is deprecated and ignored"
    assert caplog.text.count(warning) == 1


@pytest.mark.parametrize("value", ["true", "false"])
def test_enforce_disagg_environment_is_deprecated_and_not_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    value: str,
) -> None:
    monkeypatch.setenv("DYN_ENFORCE_DISAGG", value)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    config.validate()
    kwargs = config.router_kwargs()

    assert config.enforce_disagg is (value == "true")
    assert "enforce_disagg" not in kwargs
    warning = "DYN_ENFORCE_DISAGG is deprecated and ignored"
    assert caplog.text.count(warning) == 1


def test_admission_control_cli_flag_warns_and_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _clear_rejection_threshold_env(monkeypatch)
    caplog.set_level("WARNING")
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    assert "--admission-control" not in parser.format_help()

    config = FrontendConfig.from_cli_args(
        parser.parse_args(["--admission-control", "none"])
    )
    config.validate()

    assert not hasattr(config, "admission_control")
    assert config.active_decode_blocks_threshold is None
    assert config.active_prefill_tokens_threshold_frac is None
    assert "--admission-control is no longer supported and is ignored" in caplog.text

    with pytest.raises(SystemExit):
        parser.parse_args(["--admission-control", "bogus"])


def test_removed_admission_control_environment_warns_and_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _clear_rejection_threshold_env(monkeypatch)
    monkeypatch.setenv("DYN_ADMISSION_CONTROL", "token-capacity")
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    config.validate()

    assert not hasattr(config, "admission_control")
    assert config.active_decode_blocks_threshold is None
    assert config.active_prefill_tokens_threshold_frac is None
    assert "DYN_ADMISSION_CONTROL is no longer supported and is ignored" in caplog.text


@pytest.mark.parametrize(
    "flag",
    [
        "--active-decode-blocks-threshold",
        "--active-prefill-tokens-threshold-frac",
    ],
)
def test_explicit_none_keeps_rejection_threshold_disabled(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
) -> None:
    _clear_rejection_threshold_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    config = FrontendConfig.from_cli_args(parser.parse_args([flag, "None"]))
    config.validate()

    assert config.active_decode_blocks_threshold is None
    assert config.active_prefill_tokens_threshold_frac is None


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--active-decode-blocks-threshold", "-0.1"),
        ("--active-decode-blocks-threshold", "1.1"),
        ("--active-decode-blocks-threshold", "nan"),
        ("--active-prefill-tokens-threshold-frac", "-0.1"),
        ("--active-prefill-tokens-threshold-frac", "inf"),
    ],
)
def test_rejection_threshold_validation_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
) -> None:
    _clear_rejection_threshold_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    config = FrontendConfig.from_cli_args(parser.parse_args([flag, value]))
    with pytest.raises(ValueError, match=flag):
        config.validate()


def test_session_affinity_ttl_cli_and_environment(monkeypatch) -> None:
    monkeypatch.delenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", raising=False)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    config.validate()
    assert config.session_affinity_ttl_secs is None
    assert config.router_kwargs()["session_affinity_ttl_secs"] is None

    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "600")
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    config.validate()
    assert config.session_affinity_ttl_secs == 600
    assert config.router_kwargs()["session_affinity_ttl_secs"] == 600

    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(
        parser.parse_args(["--router-session-affinity-ttl-secs", "900"])
    )
    config.validate()
    assert config.session_affinity_ttl_secs == 900


def test_session_affinity_mode_cli_and_environment(monkeypatch) -> None:
    monkeypatch.delenv("DYN_ROUTER_SESSION_AFFINITY_MODE", raising=False)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    assert config.session_affinity_mode == "hard"
    assert config.router_kwargs()["session_affinity_mode"] == "hard"

    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_MODE", "soft")
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    assert config.session_affinity_mode == "soft"

    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(
        parser.parse_args(["--router-session-affinity-mode", "hard"])
    )
    assert config.session_affinity_mode == "hard"


def test_session_affinity_max_entries_cli_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def parse(argv: list[str]) -> FrontendConfig:
        parser = argparse.ArgumentParser()
        FrontendArgGroup().add_arguments(parser)
        config = FrontendConfig.from_cli_args(parser.parse_args(argv))
        config.validate()
        return config

    monkeypatch.delenv("DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES", raising=False)
    assert parse([]).session_affinity_max_entries == 65_536

    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES", "1000000")
    assert parse([]).session_affinity_max_entries == 1_000_000

    config = parse(["--router-session-affinity-max-entries", "2000000"])
    assert config.session_affinity_max_entries == 2_000_000
    assert "session_affinity_max_entries" not in config.router_kwargs()

    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(
        parser.parse_args(["--router-session-affinity-max-entries", "0"])
    )
    with pytest.raises(ValueError, match="router-session-affinity-max-entries"):
        config.validate()


@pytest.mark.parametrize("ttl", [0, 31_536_001])
def test_session_affinity_ttl_rejects_out_of_range(ttl: int) -> None:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(
        parser.parse_args(["--router-session-affinity-ttl-secs", str(ttl)])
    )
    with pytest.raises(ValueError, match="router-session-affinity-ttl-secs"):
        config.validate()


@pytest.mark.parametrize(
    ("flag", "env", "parameter"),
    [
        (
            "--router-kv-overlap-score-credit",
            "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT",
            "overlap_score_credit",
        ),
        (
            "--router-prefill-load-scale",
            "DYN_ROUTER_PREFILL_LOAD_SCALE",
            "prefill_load_scale",
        ),
        ("--router-temperature", "DYN_ROUTER_TEMPERATURE", "router_temperature"),
    ],
)
def test_deprecated_policy_flags_preserve_values_and_cli_precedence(
    monkeypatch, flag, env, parameter
):
    monkeypatch.setenv(env, "2.5")
    parser = argparse.ArgumentParser()
    with pytest.warns(
        FutureWarning,
        match=rf"{env} is deprecated.*{parameter}.*--router-policy-config",
    ):
        KvRouterArgGroup().add_arguments(parser)
    config = KvRouterConfigBase.from_cli_args(parser.parse_args([]))
    assert config.kv_router_kwargs()[parameter] == 2.5

    # An explicit zero must override the environment, not fall back to it.
    with pytest.warns(
        FutureWarning,
        match=rf"{flag} is deprecated.*{parameter}.*--router-policy-config",
    ):
        args = parser.parse_args([flag, "0"])
    assert KvRouterConfigBase.from_cli_args(args).kv_router_kwargs()[parameter] == 0.0


@pytest.mark.parametrize("source", ["default", "cli", "env", "yaml"])
def test_shared_cache_weight_default_is_left_to_the_policy(
    monkeypatch, tmp_path, source
):
    monkeypatch.delenv("DYN_SHARED_CACHE_MULTIPLIER", raising=False)
    monkeypatch.delenv("DYN_SHARED_CACHE_TYPE", raising=False)
    args = []
    if source == "cli":
        args = ["--shared-cache-type", "hicache"]
    elif source == "env":
        monkeypatch.setenv("DYN_SHARED_CACHE_TYPE", "hicache")
    elif source == "yaml":
        path = tmp_path / "policy.yaml"
        path.write_text("router:\n  shared_cache_type: hicache\n")
        args = ["--router-policy-config", str(path)]
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    kwargs = KvRouterConfigBase.from_cli_args(
        parser.parse_args(args)
    ).kv_router_kwargs()
    assert kwargs["shared_cache_type"] == ("none" if source == "default" else "hicache")
    # Not forwarded, so the binding keeps None and the policy picks its default.
    assert "shared_cache_multiplier" not in kwargs


# The removed spellings, written out and not read from the tables: a
# spelling dropped from a table would drop out of these tests with it.
_REMOVED_FLAGS = (
    "--active-prefill-tokens-threshold",
    "--kv-overlap-score-weight",
    "--load-aware",
    "--no-load-aware",
    "--router-decode-active-request-weight",
    "--router-decode-policy",
    "--router-kv-overlap-score-credit-decay",
    "--router-kv-overlap-score-weight",
    "--router-prefill-policy",
    "--router-queue-policy",
    "--router-queue-threshold",
    "--shared-cache-multiplier",
)


def test_every_removed_flag_is_listed() -> None:
    assert set(_REMOVED_FLAGS) == {*REMOVED_KV_ROUTER_FLAGS, *REMOVED_ROUTER_FLAGS}


_REMOVED_ENV_VARS = {**REMOVED_KV_ROUTER_ENV_VARS, **REMOVED_ROUTER_ENV_VARS}
_REMOVED_ENV_VAR_NAMES = (
    "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
    "DYN_OVERLAP_SCORE_WEIGHT",
    "DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT",
    "DYN_ROUTER_DECODE_POLICY",
    "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY",
    "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT",
    "DYN_ROUTER_LOAD_AWARE",
    "DYN_ROUTER_PREFILL_POLICY",
    "DYN_ROUTER_QUEUE_POLICY",
    "DYN_ROUTER_QUEUE_THRESHOLD",
    "DYN_SHARED_CACHE_MULTIPLIER",
)


def test_every_removed_environment_variable_is_listed() -> None:
    assert set(_REMOVED_ENV_VAR_NAMES) == set(_REMOVED_ENV_VARS)


@pytest.mark.parametrize("flag", _REMOVED_FLAGS)
def test_removed_flag_fails_and_stays_out_of_help(
    capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    # A value too: --active-prefill-tokens-threshold must not be read as an
    # abbreviation of --active-prefill-tokens-threshold-frac.
    with pytest.raises(SystemExit):
        parser.parse_args([flag, "1"])

    assert f"{flag} was removed in v1.8; use " in capsys.readouterr().err
    assert flag not in parser.format_help().split()


@pytest.mark.parametrize(
    ("flag", "replacement"),
    [
        (
            "--router-queue-threshold",
            "policy_classes[].prefill_busy_threshold_frac in --router-policy-config, "
            "and set queue_policy there: the removed queue defaulted to fcfs, a "
            "class defaults to edf",
        ),
        (
            "--router-queue-policy",
            "policy_classes[].queue_policy in --router-policy-config, which takes "
            "edf, fcfs or wspt and not lcfs",
        ),
        (
            "--active-prefill-tokens-threshold",
            "--active-prefill-tokens-threshold-frac",
        ),
    ],
)
def test_removed_flag_names_its_replacement(
    capsys: pytest.CaptureFixture[str], flag: str, replacement: str
) -> None:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([flag, "1"])

    error = capsys.readouterr().err
    assert error.endswith(f"{flag} was removed in v1.8; use {replacement}\n")


@pytest.mark.parametrize("name", _REMOVED_ENV_VAR_NAMES)
def test_removed_environment_variable_fails_a_kv_frontend(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    for other in _REMOVED_ENV_VAR_NAMES:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(name, "1")
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(["--router-mode", "kv"]))

    with pytest.raises(ValueError) as error:
        config.validate()

    assert f"removed in v1.8 are still set: {name} (use " in str(error.value)


@pytest.mark.parametrize(
    ("name", "fails"),
    [
        # Forced KV routing, so its absence changes this frontend's routing.
        ("DYN_ROUTER_LOAD_AWARE", True),
        # Applied to every mode but direct.
        ("DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD", True),
        # Never read outside KV routing; the EPP beside this frontend may read it.
        ("DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY", False),
        ("DYN_ROUTER_QUEUE_THRESHOLD", False),
    ],
)
def test_round_robin_frontend_fails_only_on_removed_variables_it_read(
    monkeypatch: pytest.MonkeyPatch, name: str, fails: bool
) -> None:
    for other in _REMOVED_ENV_VARS:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.delenv("DYN_ROUTER_MODE", raising=False)
    monkeypatch.setenv(name, "1")
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args([]))

    if fails:
        with pytest.raises(ValueError, match=f"removed in v1.8 are still set: {name}"):
            config.validate()
    else:
        config.validate()
        assert config.router_mode == "round-robin"


def test_removed_environment_variable_fails_a_standalone_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DYN_ROUTER_QUEUE_THRESHOLD", "0.5")
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    config = KvRouterConfigBase.from_cli_args(parser.parse_args([]))

    with pytest.raises(ValueError, match="policy_classes.*prefill_busy_threshold_frac"):
        config.kv_router_kwargs()


def test_default_policy_flags_do_not_warn(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("DYN_ROUTER_", "DYN_SHARED_CACHE_", "DYN_OVERLAP_")):
            monkeypatch.delenv(name)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        parser = argparse.ArgumentParser()
        KvRouterArgGroup().add_arguments(parser)
        config = KvRouterConfigBase.from_cli_args(parser.parse_args([]))
        assert config.kv_router_kwargs()["overlap_score_credit"] == 1.0
    assert not recorded


def test_router_yaml_overrides_flags_and_reads_once(tmp_path):
    path = tmp_path / "router.yaml"
    path.write_text(
        "router:\n  use_kv_events: true\n  router_predicted_ttl_secs: null\n  router_track_output_blocks: true\n"
    )
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    args = parser.parse_args(
        ["--no-router-kv-events", "--router-policy-config", str(path)]
    )
    config = FrontendConfig.from_cli_args(args)
    config.validate()
    assert config.router_mode == "kv"
    assert config.use_kv_events is True
    assert config.router_track_output_blocks is True
    path.unlink()
    kwargs = config.kv_router_kwargs()
    assert kwargs["use_kv_events"] is True
    assert kwargs["router_predicted_ttl_secs"] is None


def test_router_yaml_applies_before_frontend_validation(tmp_path):
    path = tmp_path / "router.yaml"
    path.write_text("router:\n  router_prefill_load_model: ais\n")
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(
        parser.parse_args(
            [
                "--router-mode",
                "kv",
                "--router-policy-config",
                str(path),
            ]
        )
    )
    with pytest.raises(
        ValueError,
        match=r"--router-prefill-load-model=ais requires an AIS perf model",
    ):
        config.validate()


@pytest.mark.parametrize(
    "settings",
    [
        "overlap_score_credit: 1",  # policy parameters do not belong under router
        "router_event_threads: false",
        'use_kv_events: "false"',
        "router_event_threads: null",
    ],
)
def test_router_yaml_rejects_invalid_settings(tmp_path, settings):
    path = tmp_path / "router.yaml"
    path.write_text(f"router:\n  {settings}\n")
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    config = KvRouterConfigBase.from_cli_args(
        parser.parse_args(["--router-policy-config", str(path)])
    )
    with pytest.raises(ValueError):
        config.kv_router_kwargs()


def test_shared_router_settings_fixture_flows_to_binding_kwargs(pytestconfig):
    path = (
        pytestconfig.rootpath / "lib/router/kv-router/tests/data/router-settings.yaml"
    )
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    config = KvRouterConfigBase.from_cli_args(
        parser.parse_args(
            [
                "--router-policy-config",
                str(path),
                "--router-predicted-ttl-secs",
                "5",
            ]
        )
    )
    kwargs = config.kv_router_kwargs()
    expected = kv_router_args.yaml.safe_load(path.read_text())["router"]
    assert set(expected) == set(kv_router_args._ROUTER_SETTINGS)
    for name, value in expected.items():
        assert kwargs[name] == value, name
