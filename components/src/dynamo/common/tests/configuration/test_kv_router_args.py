# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_overlap_score_credit_cli_uses_kv_router_config_field() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--router-kv-overlap-score-credit", "0.5"])

    assert args.overlap_score_credit == 0.5
    assert not hasattr(args, "overlap_score_weight")


def test_deprecated_overlap_score_weight_cli_maps_to_prefill_load_scale() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    with pytest.warns(FutureWarning, match="overlap score weight is deprecated"):
        args = parser.parse_args(["--router-kv-overlap-score-weight", "2.5"])

    assert args.overlap_score_credit == 1.0
    assert args.prefill_load_scale == 2.5
    assert not hasattr(args, "overlap_score_weight")


def test_deprecated_overlap_score_weight_zero_cli_disables_overlap_credit() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    with pytest.warns(FutureWarning, match="overlap score weight is deprecated"):
        args = parser.parse_args(["--router-kv-overlap-score-weight", "0"])

    assert args.overlap_score_credit == 0.0
    assert args.prefill_load_scale == 0.0
    assert not hasattr(args, "overlap_score_weight")


def test_deprecated_overlap_score_weight_env_maps_to_prefill_load_scale(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "2.5")

    with pytest.warns(FutureWarning, match="DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT"):
        parser = argparse.ArgumentParser()
        KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    assert args.overlap_score_credit == 1.0
    assert args.prefill_load_scale == 2.5
    assert not hasattr(args, "overlap_score_weight")


def test_deprecated_overlap_score_weight_zero_env_disables_overlap_credit(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "0")

    with pytest.warns(FutureWarning, match="deprecated"):
        parser = argparse.ArgumentParser()
        KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    assert args.overlap_score_credit == 0.0
    assert args.prefill_load_scale == 0.0
    assert not hasattr(args, "overlap_score_weight")


def test_deprecated_overlap_score_weight_zero_env_disables_credit_with_new_scale_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "0")
    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_SCALE", "2.5")

    with pytest.warns(FutureWarning, match="deprecated"):
        parser = argparse.ArgumentParser()
        KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args([])

    assert args.overlap_score_credit == 0.0
    assert args.prefill_load_scale == 2.5
    assert not hasattr(args, "overlap_score_weight")


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


def test_kv_router_config_rejects_negative_prefill_load_scale() -> None:
    config = KvRouterConfigBase.__new__(KvRouterConfigBase)
    config.overlap_score_credit = 1.0
    config.prefill_load_scale = -0.1

    with pytest.raises(ValueError, match="--router-prefill-load-scale must be >= 0.0"):
        config.validate_kv_router_config()
