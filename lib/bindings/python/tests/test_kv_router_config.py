# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import KvRouterConfig


def test_removed_router_options_cannot_shift_positional_arguments() -> None:
    with pytest.raises(TypeError):
        KvRouterConfig(None, 0.75, 0.25, 0.0, True, False)


def test_decode_active_request_weight_defaults_to_zero_and_validates() -> None:
    config = KvRouterConfig()
    assert config.decode_active_request_weight == 0.0

    assert (
        config.with_overrides(
            decode_active_request_weight=64.0
        ).decode_active_request_weight
        == 64.0
    )

    for invalid in [-1.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="decode_active_request_weight"):
            KvRouterConfig(decode_active_request_weight=invalid)


def test_decode_affinity_high_watermark_defaults_to_none_and_validates() -> None:
    assert KvRouterConfig().router_decode_affinity_high_watermark is None
    assert (
        KvRouterConfig(
            router_decode_affinity_high_watermark=0.7
        ).router_decode_affinity_high_watermark
        == 0.7
    )

    for invalid in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError, match="router_decode_affinity_high_watermark"):
            KvRouterConfig(router_decode_affinity_high_watermark=invalid)
