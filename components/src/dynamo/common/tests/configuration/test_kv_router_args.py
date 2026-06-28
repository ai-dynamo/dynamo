# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import pytest

from dynamo.common.configuration.groups.aic_perf_args import (
    AicPerfArgGroup,
    AicPerfConfigBase,
)
from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _clear_admission_control_env(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_overlap_score_credit_cli_uses_kv_router_config_field() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--router-kv-overlap-score-credit", "0.5"])

    assert args.overlap_score_credit == 0.5
    assert args.overlap_score_weight is None


def test_overlap_score_credit_decay_cli_uses_kv_router_config_field() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--router-kv-overlap-score-credit-decay", "0.5"])

    assert args.overlap_score_credit_decay == 0.5
    config = KvRouterConfigBase.from_cli_args(args)
    assert config.kv_router_kwargs()["overlap_score_credit_decay"] == 0.5


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


def test_prefill_load_scale_cli_uses_kv_router_config_field() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--router-prefill-load-scale", "2.5"])

    assert args.prefill_load_scale == 2.5
    assert not hasattr(args, "router_prefill_load_scale")


def test_prefill_load_scale_env_uses_kv_router_config_field(monkeypatch) -> None:
    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_SCALE", "3.5")
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    assert args.prefill_load_scale == 3.5
    assert not hasattr(args, "router_prefill_load_scale")


def test_router_selection_policy_cli_flows_to_binding_kwargs() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--router-selection-policy", "lmetric"])

    config = KvRouterConfigBase.from_cli_args(args)
    assert config.kv_router_kwargs()["router_selection_policy"] == "lmetric"


def test_router_selection_policy_env_flows_to_binding_kwargs(monkeypatch) -> None:
    monkeypatch.setenv("DYN_ROUTER_SELECTION_POLICY", "lmetric")
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    config = KvRouterConfigBase.from_cli_args(args)
    assert config.kv_router_kwargs()["router_selection_policy"] == "lmetric"


def test_load_aware_cli_applies_no_cache_load_balancing_preset() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--load-aware"])

    assert args.load_aware is True
    config = KvRouterConfigBase.from_cli_args(args)
    kwargs = config.kv_router_kwargs()

    assert kwargs["overlap_score_credit"] == 0.0
    assert kwargs["use_kv_events"] is False
    assert kwargs["durable_kv_events"] is False
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


def test_frontend_admission_control_defaults_to_none(monkeypatch) -> None:
    """Default --admission-control is 'none': busy thresholds are cleared
    even though the underlying threshold flags have non-None defaults."""
    _clear_admission_control_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    config = FrontendConfig.from_cli_args(args)
    config.validate()

    assert config.admission_control == "none"
    assert config.active_decode_blocks_threshold is None
    assert config.active_prefill_tokens_threshold is None
    assert config.active_prefill_tokens_threshold_frac is None
    assert config.router_queue_threshold == 16.0


def test_admission_control_token_capacity_preserves_busy_thresholds(
    monkeypatch,
) -> None:
    """With --admission-control token-capacity, the configured busy thresholds
    flow through to router_kwargs unchanged."""
    _clear_admission_control_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    args = parser.parse_args(["--admission-control", "token-capacity"])

    config = FrontendConfig.from_cli_args(args)
    config.validate()

    assert config.admission_control == "token-capacity"
    assert config.active_decode_blocks_threshold == 1.0
    assert config.active_prefill_tokens_threshold == 10_000_000
    assert config.active_prefill_tokens_threshold_frac == 64.0
    assert config.router_queue_threshold == 16.0
    assert config.router_kwargs() == {
        "active_decode_blocks_threshold": 1.0,
        "active_prefill_tokens_threshold": 10_000_000,
        "active_prefill_tokens_threshold_frac": 64.0,
        "enforce_disagg": False,
        "session_affinity_ttl_secs": None,
    }


def test_admission_control_token_capacity_with_custom_thresholds(
    monkeypatch,
) -> None:
    _clear_admission_control_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    args = parser.parse_args(
        [
            "--admission-control",
            "token-capacity",
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

    config = FrontendConfig.from_cli_args(args)
    config.validate()

    assert config.admission_control == "token-capacity"
    assert config.active_decode_blocks_threshold == 0.5
    assert config.active_prefill_tokens_threshold == 1000
    assert config.active_prefill_tokens_threshold_frac == 2.0
    assert config.router_queue_threshold == 32.0
    assert config.router_kwargs() == {
        "active_decode_blocks_threshold": 0.5,
        "active_prefill_tokens_threshold": 1000,
        "active_prefill_tokens_threshold_frac": 2.0,
        "enforce_disagg": False,
        "session_affinity_ttl_secs": None,
    }
    assert config.kv_router_kwargs()["router_queue_threshold"] == 32.0


def test_admission_control_explicit_none_with_threshold_raises(
    monkeypatch,
) -> None:
    """Explicit --admission-control none combined with an explicit
    threshold flag is a contradiction and must raise. The implicit-default
    case (no --admission-control flag passed) auto-promotes instead — see
    test_admission_control_default_none_with_explicit_threshold_auto_switches.
    """
    _clear_admission_control_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    args = parser.parse_args(
        [
            "--admission-control",
            "none",
            "--active-decode-blocks-threshold",
            "0.5",
        ]
    )

    config = FrontendConfig.from_cli_args(args)
    with pytest.raises(ValueError, match="cannot be combined with explicit"):
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


def test_admission_control_explicit_none_without_thresholds_resolves_to_none(
    monkeypatch,
) -> None:
    """Explicit --admission-control none with no threshold flags is a
    legal config: admission disabled, queue threshold preserved."""
    _clear_admission_control_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    args = parser.parse_args(
        [
            "--admission-control",
            "none",
            "--router-queue-threshold",
            "32.0",
        ]
    )

    config = FrontendConfig.from_cli_args(args)
    config.validate()

    assert config.admission_control == "none"
    assert config.active_decode_blocks_threshold is None
    assert config.active_prefill_tokens_threshold is None
    assert config.active_prefill_tokens_threshold_frac is None
    assert config.router_queue_threshold == 32.0


def test_admission_control_default_none_with_explicit_threshold_auto_switches(
    monkeypatch,
) -> None:
    """Pre-v1.1.2 launch-config compatibility: passing a threshold flag
    without --admission-control auto-promotes mode from the new 'none'
    default to 'token-capacity' so the threshold actually fires.

    User-set thresholds keep their values; unset thresholds receive
    production defaults — same as passing --admission-control token-capacity
    explicitly. This matches the v1.0.x/v1.1.x contract where setting any
    threshold flag implicitly activated admission control with defaults
    filling in the rest.
    """
    _clear_admission_control_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    args = parser.parse_args(
        [
            "--active-decode-blocks-threshold",
            "0.85",
            "--active-prefill-tokens-threshold",
            "10000",
        ]
    )

    config = FrontendConfig.from_cli_args(args)
    config.validate()

    assert config.admission_control == "token-capacity"
    assert config.active_decode_blocks_threshold == 0.85
    assert config.active_prefill_tokens_threshold == 10000
    # _frac was not passed → filled in with production default 64.0
    # (auto-switch matches --admission-control token-capacity).
    assert config.active_prefill_tokens_threshold_frac == 64.0


def test_admission_control_apply_is_idempotent(monkeypatch) -> None:
    """``apply_admission_control()`` runs once in ``validate()`` and again
    when ``router_kwargs()`` builds the worker config. The second call
    must not raise the explicit-none contradiction against the ``None``
    threshold values its first call normalized them to."""
    _clear_admission_control_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    config = FrontendConfig.from_cli_args(args)
    config.validate()  # first apply
    # router_kwargs() runs apply_admission_control() again on the
    # already-normalized state. Must not raise.
    kwargs = config.router_kwargs()

    assert config.admission_control == "none"
    assert kwargs["active_decode_blocks_threshold"] is None
    assert kwargs["active_prefill_tokens_threshold"] is None
    assert kwargs["active_prefill_tokens_threshold_frac"] is None


def test_admission_control_explicit_none_threshold_with_none_mode_ok(
    monkeypatch,
) -> None:
    """``--admission-control none --active-decode-blocks-threshold None``
    is consistent (both say disabled) — must not raise. Only a *numeric*
    threshold value alongside explicit ``none`` is a contradiction."""
    _clear_admission_control_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    args = parser.parse_args(
        [
            "--admission-control",
            "none",
            "--active-decode-blocks-threshold",
            "None",
        ]
    )

    config = FrontendConfig.from_cli_args(args)
    config.validate()

    assert config.admission_control == "none"
    assert config.active_decode_blocks_threshold is None


def test_admission_control_token_capacity_with_explicit_none_threshold_keeps_disabled(
    monkeypatch,
) -> None:
    """Explicit ``--<threshold> None`` keeps that specific check disabled
    even in ``--admission-control token-capacity`` mode, matching the
    "Pass 'None' on the CLI to disable this check" help text. The other
    thresholds still receive production defaults."""
    _clear_admission_control_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    args = parser.parse_args(
        [
            "--admission-control",
            "token-capacity",
            "--active-decode-blocks-threshold",
            "None",
        ]
    )

    config = FrontendConfig.from_cli_args(args)
    config.validate()

    assert config.admission_control == "token-capacity"
    # Explicit `None` is preserved — the documented disable-this-check semantic.
    assert config.active_decode_blocks_threshold is None
    # Un-passed thresholds still receive their production defaults.
    assert config.active_prefill_tokens_threshold == 10_000_000
    assert config.active_prefill_tokens_threshold_frac == 64.0


def test_admission_control_env_var(monkeypatch) -> None:
    _clear_admission_control_env(monkeypatch)
    monkeypatch.setenv("DYN_ADMISSION_CONTROL", "token-capacity")
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    config = FrontendConfig.from_cli_args(args)
    config.validate()

    assert config.admission_control == "token-capacity"
    assert config.active_decode_blocks_threshold == 1.0
