# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import LoadThresholdConfig, RouterConfig, RouterMode

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_load_threshold_config_defaults_are_disabled():
    config = LoadThresholdConfig()

    assert config.active_decode_blocks_threshold is None
    assert config.active_prefill_tokens_threshold_frac is None


def test_load_threshold_config_preserves_valid_values():
    config = LoadThresholdConfig(
        active_decode_blocks_threshold=0.75,
        active_prefill_tokens_threshold_frac=0.5,
    )

    assert config.active_decode_blocks_threshold == 0.75
    assert config.active_prefill_tokens_threshold_frac == 0.5


def test_load_threshold_config_rejects_removed_absolute_threshold():
    with pytest.raises(TypeError, match="active_prefill_tokens_threshold"):
        LoadThresholdConfig(active_prefill_tokens_threshold=512)
    assert not hasattr(LoadThresholdConfig(), "active_prefill_tokens_threshold")


@pytest.mark.parametrize(
    "legacy_positional",
    [
        # The removed absolute threshold, then the fraction, enforce_disagg, and TTL
        # in their former positions.
        (RouterMode.KV, None, None, 1000),
        (RouterMode.KV, None, None, None, 0.8),
        (RouterMode.KV, None, None, None, None, True),
        (RouterMode.KV, None, None, None, None, False, 60),
    ],
)
def test_router_config_rejects_shifted_legacy_positional_arguments(legacy_positional):
    with pytest.raises(TypeError, match="positional"):
        RouterConfig(*legacy_positional)


def test_router_config_rejects_removed_absolute_threshold_keyword():
    with pytest.raises(TypeError, match="active_prefill_tokens_threshold"):
        RouterConfig(RouterMode.KV, active_prefill_tokens_threshold=1000)


def test_router_config_accepts_fractional_thresholds():
    RouterConfig(RouterMode.KV, None, 0.9, active_prefill_tokens_threshold_frac=0.5)
    RouterConfig(
        RouterMode.KV,
        active_decode_blocks_threshold=0.9,
        active_prefill_tokens_threshold_frac=2.0,
        enforce_disagg=False,
        session_affinity_ttl_secs=60,
        session_affinity_mode="soft",
    )

    # Each value reaches the threshold it names.
    with pytest.raises(ValueError, match="active_decode_blocks_threshold"):
        RouterConfig(RouterMode.KV, None, 1.5)
    with pytest.raises(ValueError, match="active_prefill_tokens_threshold_frac"):
        RouterConfig(RouterMode.KV, active_prefill_tokens_threshold_frac=-0.1)


def test_load_threshold_config_rejects_invalid_values():
    with pytest.raises(
        ValueError,
        match=(
            "invalid load threshold config: active_decode_blocks_threshold "
            "must be between 0.0 and 1.0"
        ),
    ):
        LoadThresholdConfig(active_decode_blocks_threshold=1.1)
