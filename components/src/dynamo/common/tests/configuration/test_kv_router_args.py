# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import pytest

from dynamo.common.configuration.groups import kv_router_args
from dynamo.common.configuration.groups.aic_perf_args import (
    AicPerfArgGroup,
    AicPerfConfigBase,
)
from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

# `dynamo._core` is the compiled pyo3 extension. It is present in a normal dev
# build but absent from a pure-Python checkout, so guard it at collection rather
# than letting the import error out mid-test.
KvRouterConfig = pytest.importorskip(
    "dynamo._core",
    reason="the compiled dynamo._core binding is not built in this environment",
).KvRouterConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _clear_rejection_threshold_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",
        "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
        "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
        "DYN_ADMISSION_CONTROL",
        "DYN_ROUTER_QUEUE_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)


def test_aic_perf_moe_cli_flows_to_binding_kwargs() -> None:
    parser = argparse.ArgumentParser()
    AicPerfArgGroup().add_arguments(parser)

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

    config = AicPerfConfigBase.from_cli_args(args)

    assert config.aic_perf_kwargs() == {
        "aic_backend": "vllm",
        "aic_system": "h200_sxm",
        "aic_backend_version": None,
        "aic_tp_size": 2,
        "aic_model_path": "moonshotai/Kimi-K2-Instruct",
        "aic_moe_tp_size": 2,
        "aic_moe_ep_size": 1,
        "aic_attention_dp_size": 1,
        "aic_nextn": None,
        "aic_nextn_accept_rates": None,
    }


def test_aic_perf_moe_env_flows_to_binding_kwargs(monkeypatch) -> None:
    monkeypatch.setenv("DYN_AIC_BACKEND", "vllm")
    monkeypatch.setenv("DYN_AIC_SYSTEM", "h200_sxm")
    monkeypatch.setenv("DYN_AIC_MODEL_PATH", "moonshotai/Kimi-K2-Instruct")
    monkeypatch.setenv("DYN_AIC_TP_SIZE", "2")
    monkeypatch.setenv("DYN_AIC_MOE_TP_SIZE", "2")
    monkeypatch.setenv("DYN_AIC_MOE_EP_SIZE", "1")
    monkeypatch.setenv("DYN_AIC_ATTENTION_DP_SIZE", "1")

    parser = argparse.ArgumentParser()
    AicPerfArgGroup().add_arguments(parser)
    args = parser.parse_args([])

    config = AicPerfConfigBase.from_cli_args(args)

    assert config.aic_perf_kwargs()["aic_moe_tp_size"] == 2
    assert config.aic_perf_kwargs()["aic_moe_ep_size"] == 1
    assert config.aic_perf_kwargs()["aic_attention_dp_size"] == 1


def test_aic_mtp_cli_documents_conditional_rates_and_seed() -> None:
    parser = argparse.ArgumentParser()
    AicPerfArgGroup().add_arguments(parser)
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

    config = AicPerfConfigBase.from_cli_args(args)
    assert config.aic_nextn == 3
    assert config.aic_nextn_accept_rates == "1,0.5"
    assert config.aic_mtp_seed == 99
    assert "all earlier drafts were accepted" in parser.format_help()


def test_deprecated_overlap_score_weight_cli_flows_to_binding_kwargs() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    with pytest.warns(FutureWarning, match="overlap score weight is deprecated"):
        args = parser.parse_args(["--router-kv-overlap-score-weight", "2.5"])

    assert args.overlap_score_credit == 1.0
    assert args.prefill_load_scale == 1.0
    assert args.overlap_score_weight == 2.5

    config = KvRouterConfigBase.from_cli_args(args)
    assert config.kv_router_kwargs()["overlap_score_weight"] == 2.5


def test_deprecated_overlap_score_weight_env_flows_to_binding_kwargs(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "2.5")

    with pytest.warns(FutureWarning, match="DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT"):
        parser = argparse.ArgumentParser()
        KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    assert args.overlap_score_credit == 1.0
    assert args.prefill_load_scale == 1.0
    assert args.overlap_score_weight == 2.5

    config = KvRouterConfigBase.from_cli_args(args)
    assert config.kv_router_kwargs()["overlap_score_weight"] == 2.5


@pytest.mark.parametrize(
    ("canonical_env", "cli_args", "expected_credit", "expected_scale"),
    [
        (("DYN_ROUTER_PREFILL_LOAD_SCALE", "2.5"), [], 1.0, 2.5),
        (("DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT", "0.5"), [], 0.5, 1.0),
        (None, ["--router-prefill-load-scale", "3"], 1.0, 3.0),
        (None, ["--router-kv-overlap-score-credit", "0.5"], 0.5, 1.0),
    ],
    ids=["scale-env", "credit-env", "scale-cli", "credit-cli"],
)
def test_deprecated_overlap_score_weight_env_coexists_with_canonical_settings(
    monkeypatch,
    canonical_env,
    cli_args,
    expected_credit,
    expected_scale,
) -> None:
    monkeypatch.setenv("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "0")
    if canonical_env is not None:
        monkeypatch.setenv(*canonical_env)

    with pytest.warns(FutureWarning, match="deprecated"):
        parser = argparse.ArgumentParser()
        KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(cli_args)

    assert args.overlap_score_credit == expected_credit
    assert args.prefill_load_scale == expected_scale
    assert args.overlap_score_weight == 0.0

    config = KvRouterConfigBase.from_cli_args(args)
    assert config.kv_router_kwargs()["overlap_score_weight"] == 0.0


def test_decode_active_request_weight_flows_to_binding_kwargs() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--router-decode-active-request-weight", "64"])
    kwargs = KvRouterConfigBase.from_cli_args(args).kv_router_kwargs()

    assert kwargs["decode_active_request_weight"] == 64.0


def test_session_prefix_index_is_opt_in_and_reaches_the_binding() -> None:
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

    # The kwargs dict is unpacked straight into the Rust binding, so a name that
    # does not match the pyo3 signature fails here rather than at serve time.
    KvRouterConfig(**enabled_kwargs)


def test_session_prefix_index_environment_flows_to_binding_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The env var is consumed when the flag is registered, so it has to be set
    # before the group builds its arguments.
    monkeypatch.setenv("DYN_ENABLE_SESSION_PREFIX_INDEX", "true")
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    kwargs = KvRouterConfigBase.from_cli_args(parser.parse_args([])).kv_router_kwargs()
    assert kwargs["enable_session_prefix_index"] is True

    # An explicit flag still wins over the environment.
    overridden = KvRouterConfigBase.from_cli_args(
        parser.parse_args(["--no-enable-session-prefix-index"])
    ).kv_router_kwargs()
    assert overridden["enable_session_prefix_index"] is False


def test_load_aware_cli_applies_no_cache_load_balancing_preset() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--load-aware"])

    assert args.load_aware is True
    config = KvRouterConfigBase.from_cli_args(args)
    kwargs = config.kv_router_kwargs()

    assert kwargs["overlap_score_credit"] == 0.0
    assert kwargs["use_kv_events"] is False
    assert kwargs["router_track_active_blocks"] is True
    assert kwargs["router_assume_kv_reuse"] is False
    assert kwargs["router_track_prefill_tokens"] is True
    assert kwargs["use_remote_indexer"] is False
    assert kwargs["serve_indexer"] is False
    assert kwargs["shared_cache_multiplier"] == 0.0
    assert kwargs["shared_cache_type"] == "none"
    assert "load_aware" not in kwargs
    assert kwargs["overlap_score_weight"] is None


def test_load_aware_env_applies_no_cache_load_balancing_preset(monkeypatch) -> None:
    monkeypatch.setenv("DYN_ROUTER_LOAD_AWARE", "true")
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    assert args.load_aware is True
    config = KvRouterConfigBase.from_cli_args(args)
    kwargs = config.kv_router_kwargs()

    assert kwargs["overlap_score_credit"] == 0.0
    assert kwargs["use_kv_events"] is False
    assert kwargs["router_assume_kv_reuse"] is False
    assert kwargs["router_track_prefill_tokens"] is True


def test_load_aware_preserves_prefill_load_scale() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--load-aware", "--router-prefill-load-scale", "2.5"])

    config = KvRouterConfigBase.from_cli_args(args)
    kwargs = config.kv_router_kwargs()

    assert kwargs["overlap_score_credit"] == 0.0
    assert kwargs["prefill_load_scale"] == 2.5


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


def test_load_aware_preserves_cache_hit_weights() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(
        [
            "--load-aware",
            "--router-host-cache-hit-weight",
            "0.9",
            "--router-disk-cache-hit-weight",
            "0.1",
        ]
    )

    config = KvRouterConfigBase.from_cli_args(args)
    kwargs = config.kv_router_kwargs()

    assert kwargs["overlap_score_credit"] == 0.0
    assert kwargs["host_cache_hit_weight"] == 0.9
    assert kwargs["disk_cache_hit_weight"] == 0.1


def test_policy_config_cli_overrides_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_policy_path = str(tmp_path / "env-policy.yaml")
    explicit_policy_path = str(tmp_path / "explicit-policy.yaml")
    monkeypatch.setenv("DYN_ROUTER_POLICY_CONFIG", env_policy_path)
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--router-policy-config", explicit_policy_path])
    config = KvRouterConfigBase.from_cli_args(args)

    assert config.kv_router_kwargs()["router_policy_config"] == explicit_policy_path


def test_stage_policy_cli_and_environment_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DYN_ROUTER_PREFILL_POLICY", "env-prefill")
    monkeypatch.setenv("DYN_ROUTER_DECODE_POLICY", "env-decode")
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--router-prefill-policy", "cli-prefill"])
    kwargs = KvRouterConfigBase.from_cli_args(args).kv_router_kwargs()

    assert kwargs["router_prefill_policy"] == "cli-prefill"
    assert kwargs["router_decode_policy"] == "env-decode"


def test_stage_policy_help_describes_router_process_pool_roles() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    help_by_dest = {action.dest: action.help for action in parser._actions}

    assert "Process-local" in help_by_dest["router_prefill_policy"]
    assert "disaggregated prefill workers" in help_by_dest["router_prefill_policy"]
    assert "Process-local" in help_by_dest["router_decode_policy"]
    assert "this router process" in help_by_dest["router_decode_policy"]


def test_load_aware_clears_predicted_ttl() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--load-aware", "--router-predicted-ttl-secs", "5"])

    config = KvRouterConfigBase.from_cli_args(args)
    kwargs = config.kv_router_kwargs()

    assert kwargs["use_kv_events"] is False
    assert kwargs["router_predicted_ttl_secs"] is None


def test_load_aware_preserves_deprecated_overlap_score_weight_env(monkeypatch) -> None:
    monkeypatch.setenv("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "2.5")

    with pytest.warns(FutureWarning, match="deprecated"):
        parser = argparse.ArgumentParser()
        KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--load-aware"])

    config = KvRouterConfigBase.from_cli_args(args)
    kwargs = config.kv_router_kwargs()

    assert kwargs["overlap_score_credit"] == 0.0
    assert kwargs["prefill_load_scale"] == 1.0
    assert kwargs["overlap_score_weight"] == 2.5


def test_load_aware_frontend_implies_kv_router_mode() -> None:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    args = parser.parse_args(["--load-aware"])

    config = FrontendConfig.from_cli_args(args)
    config.validate()

    assert config.router_mode == "kv"
    assert config.overlap_score_credit == 0.0
    assert config.use_kv_events is False
    assert config.router_assume_kv_reuse is False


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


def test_conditional_disagg_prefill_busy_threshold_defaults_to_queue_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
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
                "--router-queue-threshold",
                "16",
            ]
        )
    )
    config.validate()

    assert config.conditional_disagg_prefill_busy_threshold == 16.0
    assert (
        "conditional_disagg prefill_busy_threshold defaults to "
        "--router-queue-threshold=16.0"
    ) in caplog.text


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
    with pytest.raises(ValueError, match="needs prefill_busy_threshold"):
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
    assert config.active_prefill_tokens_threshold is None
    assert config.active_prefill_tokens_threshold_frac is None
    assert config.router_queue_threshold is None
    assert config.router_kwargs() == {
        "active_decode_blocks_threshold": None,
        "active_prefill_tokens_threshold": None,
        "active_prefill_tokens_threshold_frac": None,
        "session_affinity_ttl_secs": None,
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
            "--active-prefill-tokens-threshold",
            "1000",
            "active_prefill_tokens_threshold",
            1000,
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
        "active_prefill_tokens_threshold": config.active_prefill_tokens_threshold,
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
            "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
            "active_prefill_tokens_threshold",
            2000,
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
        "active_prefill_tokens_threshold": config.active_prefill_tokens_threshold,
        "active_prefill_tokens_threshold_frac": config.active_prefill_tokens_threshold_frac,
    }
    assert thresholds[field] == expected
    assert all(value is None for name, value in thresholds.items() if name != field)


def test_all_rejection_thresholds_and_queue_override_are_forwarded(
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
                "--active-prefill-tokens-threshold",
                "1000",
                "--active-prefill-tokens-threshold-frac",
                "2.0",
                "--router-queue-threshold",
                "32.0",
            ]
        )
    )
    config.validate()

    assert config.active_decode_blocks_threshold == 0.5
    assert config.active_prefill_tokens_threshold == 1000
    assert config.active_prefill_tokens_threshold_frac == 2.0
    assert config.router_queue_threshold == 32.0
    assert config.router_kwargs() == {
        "active_decode_blocks_threshold": 0.5,
        "active_prefill_tokens_threshold": 1000,
        "active_prefill_tokens_threshold_frac": 2.0,
        "session_affinity_ttl_secs": None,
    }
    assert config.kv_router_kwargs()["router_queue_threshold"] == 32.0


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
    assert config.active_prefill_tokens_threshold is None
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
    assert config.active_prefill_tokens_threshold is None
    assert config.active_prefill_tokens_threshold_frac is None
    assert "DYN_ADMISSION_CONTROL is no longer supported and is ignored" in caplog.text


@pytest.mark.parametrize(
    "flag",
    [
        "--active-decode-blocks-threshold",
        "--active-prefill-tokens-threshold",
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
    assert config.active_prefill_tokens_threshold is None
    assert config.active_prefill_tokens_threshold_frac is None


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--active-decode-blocks-threshold", "-0.1"),
        ("--active-decode-blocks-threshold", "1.1"),
        ("--active-decode-blocks-threshold", "nan"),
        ("--active-prefill-tokens-threshold", "-1"),
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


@pytest.mark.parametrize("ttl", [0, 31_536_001])
def test_session_affinity_ttl_rejects_out_of_range(ttl: int) -> None:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(
        parser.parse_args(["--router-session-affinity-ttl-secs", str(ttl)])
    )
    with pytest.raises(ValueError, match="router-session-affinity-ttl-secs"):
        config.validate()
