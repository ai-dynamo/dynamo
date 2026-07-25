# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import KvRouterConfig, RouterConfig, RouterMode


def test_removed_router_options_cannot_shift_positional_arguments() -> None:
    with pytest.raises(TypeError):
        KvRouterConfig(None, 0.75, 0.25, 0.0, True, False)


def test_router_config_rejects_invalid_non_cpu_to_cpu_ratio_assignment() -> None:
    config = RouterConfig(RouterMode.DeviceAwareWeighted, non_cpu_to_cpu_ratio=2)

    with pytest.raises(ValueError, match="non_cpu_to_cpu_ratio must be >= 1"):
        config.non_cpu_to_cpu_ratio = 0

    assert config.non_cpu_to_cpu_ratio == 2
    config.non_cpu_to_cpu_ratio = None
    assert config.non_cpu_to_cpu_ratio is None
