# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.unit,
]


def parse_frontend_config(*args: str) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    namespace = parser.parse_args(list(args))
    config = FrontendConfig.from_cli_args(namespace)
    config.validate()
    return config


def test_conditional_prefill_defaults_to_disabled() -> None:
    config = parse_frontend_config()

    assert config.conditional_prefill_enabled is False
    assert config.conditional_prefill_max_new_tokens == 5000
    assert config.kv_router_kwargs()["conditional_prefill_enabled"] is False
    assert config.kv_router_kwargs()["conditional_prefill_max_new_tokens"] == 5000


def test_conditional_prefill_requires_kv_router_mode() -> None:
    with pytest.raises(ValueError, match="requires --router-mode=kv"):
        parse_frontend_config("--router-conditional-prefill")


def test_conditional_prefill_accepts_token_cap_in_kv_mode() -> None:
    config = parse_frontend_config(
        "--router-mode",
        "kv",
        "--router-conditional-prefill",
        "--router-conditional-prefill-max-new-tokens",
        "1234",
    )

    assert config.conditional_prefill_enabled is True
    assert config.conditional_prefill_max_new_tokens == 1234
    assert config.kv_router_kwargs()["conditional_prefill_enabled"] is True
    assert config.kv_router_kwargs()["conditional_prefill_max_new_tokens"] == 1234
