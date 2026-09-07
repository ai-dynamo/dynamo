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
