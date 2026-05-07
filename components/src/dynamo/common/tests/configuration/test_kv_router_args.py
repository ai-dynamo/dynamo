# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.common.configuration.groups.router_args import RouterArgGroup


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


def test_router_mode_accepts_token_dp_balance() -> None:
    parser = argparse.ArgumentParser()
    RouterArgGroup().add_arguments(parser)

    args = parser.parse_args(["--router-mode", "token-dp-balance"])

    assert args.router_mode == "token-dp-balance"


def test_kv_router_config_rejects_negative_prefill_load_scale() -> None:
    config = KvRouterConfigBase.__new__(KvRouterConfigBase)
    config.prefill_load_scale = -0.1

    with pytest.raises(ValueError, match="--router-prefill-load-scale must be >= 0.0"):
        config.validate_kv_router_config()
