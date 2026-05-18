# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

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
        "DYN_NO_ADMISSION_CONTROL",
        "DYN_ROUTER_QUEUE_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)


def test_overlap_score_credit_cli_uses_kv_router_config_field() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--router-kv-overlap-score-credit", "0.5"])

    assert args.overlap_score_credit == 0.5
    assert args.overlap_score_weight is None


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


def test_deprecated_overlap_score_weight_zero_cli_flows_to_binding_kwargs() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    with pytest.warns(FutureWarning, match="overlap score weight is deprecated"):
        args = parser.parse_args(["--router-kv-overlap-score-weight", "0"])

    assert args.overlap_score_credit == 1.0
    assert args.prefill_load_scale == 1.0
    assert args.overlap_score_weight == 0.0

    config = KvRouterConfigBase.from_cli_args(args)
    assert config.kv_router_kwargs()["overlap_score_weight"] == 0.0


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


def test_deprecated_overlap_score_weight_zero_env_flows_to_binding_kwargs(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "0")

    with pytest.warns(FutureWarning, match="deprecated"):
        parser = argparse.ArgumentParser()
        KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    assert args.overlap_score_credit == 1.0
    assert args.prefill_load_scale == 1.0
    assert args.overlap_score_weight == 0.0

    config = KvRouterConfigBase.from_cli_args(args)
    assert config.kv_router_kwargs()["overlap_score_weight"] == 0.0


def test_deprecated_overlap_score_weight_env_overrides_new_scale_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "0")
    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_SCALE", "2.5")

    with pytest.warns(FutureWarning, match="deprecated"):
        parser = argparse.ArgumentParser()
        KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    assert args.overlap_score_credit == 1.0
    assert args.prefill_load_scale == 2.5
    assert args.overlap_score_weight == 0.0

    config = KvRouterConfigBase.from_cli_args(args)
    assert config.kv_router_kwargs()["overlap_score_weight"] == 0.0


def test_deprecated_overlap_score_weight_env_overrides_new_credit_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "0")
    monkeypatch.setenv("DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT", "0.5")

    with pytest.warns(FutureWarning, match="deprecated"):
        parser = argparse.ArgumentParser()
        KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    assert args.overlap_score_credit == 0.5
    assert args.prefill_load_scale == 1.0
    assert args.overlap_score_weight == 0.0

    config = KvRouterConfigBase.from_cli_args(args)
    assert config.kv_router_kwargs()["overlap_score_weight"] == 0.0


def test_deprecated_overlap_score_weight_env_overrides_new_scale_cli(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "0")

    with pytest.warns(FutureWarning, match="deprecated"):
        parser = argparse.ArgumentParser()
        KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--router-prefill-load-scale", "3"])

    assert args.overlap_score_credit == 1.0
    assert args.prefill_load_scale == 3.0
    assert args.overlap_score_weight == 0.0

    config = KvRouterConfigBase.from_cli_args(args)
    assert config.kv_router_kwargs()["overlap_score_weight"] == 0.0


def test_deprecated_overlap_score_weight_env_overrides_new_credit_cli(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "0")

    with pytest.warns(FutureWarning, match="deprecated"):
        parser = argparse.ArgumentParser()
        KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--router-kv-overlap-score-credit", "0.5"])

    assert args.overlap_score_credit == 0.5
    assert args.prefill_load_scale == 1.0
    assert args.overlap_score_weight == 0.0

    config = KvRouterConfigBase.from_cli_args(args)
    assert config.kv_router_kwargs()["overlap_score_weight"] == 0.0


def test_deprecated_overlap_score_weight_cli_order_does_not_change_presence() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    with pytest.warns(FutureWarning, match="overlap score weight is deprecated"):
        old_then_new = parser.parse_args(
            [
                "--router-kv-overlap-score-weight",
                "0",
                "--router-prefill-load-scale",
                "3",
            ]
        )
    with pytest.warns(FutureWarning, match="overlap score weight is deprecated"):
        new_then_old = parser.parse_args(
            [
                "--router-prefill-load-scale",
                "3",
                "--router-kv-overlap-score-weight",
                "0",
            ]
        )

    assert old_then_new.overlap_score_weight == 0.0
    assert old_then_new.prefill_load_scale == 3.0
    assert new_then_old.overlap_score_weight == 0.0
    assert new_then_old.prefill_load_scale == 3.0


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


def test_frontend_uses_admission_control_defaults(monkeypatch) -> None:
    _clear_admission_control_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    config = FrontendConfig.from_cli_args(args)
    config.validate()

    assert config.active_decode_blocks_threshold == 1.0
    assert config.active_prefill_tokens_threshold == 10_000_000
    assert config.active_prefill_tokens_threshold_frac == 64.0
    assert config.router_queue_threshold == 16.0


def test_no_admission_control_clears_busy_thresholds_but_keeps_queue(
    monkeypatch,
) -> None:
    _clear_admission_control_env(monkeypatch)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    args = parser.parse_args(
        [
            "--no-admission-control",
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

    assert config.no_admission_control is True
    assert config.active_decode_blocks_threshold is None
    assert config.active_prefill_tokens_threshold is None
    assert config.active_prefill_tokens_threshold_frac is None
    assert config.router_queue_threshold == 32.0
    assert config.router_kwargs() == {
        "active_decode_blocks_threshold": None,
        "active_prefill_tokens_threshold": None,
        "active_prefill_tokens_threshold_frac": None,
        "enforce_disagg": False,
    }
    assert config.kv_router_kwargs()["router_queue_threshold"] == 32.0
