# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import LoadThresholdConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_load_threshold_config_defaults_are_disabled():
    config = LoadThresholdConfig()

    assert config.active_decode_blocks_threshold is None
    assert config.active_prefill_tokens_threshold is None
    assert config.active_prefill_tokens_threshold_frac is None


def test_load_threshold_config_preserves_valid_values():
    config = LoadThresholdConfig(
        active_decode_blocks_threshold=0.75,
        active_prefill_tokens_threshold=512,
        active_prefill_tokens_threshold_frac=0.5,
    )

    assert config.active_decode_blocks_threshold == 0.75
    assert config.active_prefill_tokens_threshold == 512
    assert config.active_prefill_tokens_threshold_frac == 0.5


def test_load_threshold_config_rejects_invalid_values():
    with pytest.raises(
        ValueError,
        match=(
            "invalid load threshold config: active_decode_blocks_threshold "
            "must be between 0.0 and 1.0"
        ),
    ):
        LoadThresholdConfig(active_decode_blocks_threshold=1.1)
