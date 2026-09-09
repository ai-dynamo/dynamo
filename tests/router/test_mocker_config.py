# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.router.mocker_config import MockerConfig

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.router,
]


def test_mocker_config_renders_scalar_and_boolean_flags() -> None:
    config = MockerConfig(
        speedup_ratio=10.0,
        block_size=16,
        enable_prefix_caching=False,
        aic_perf_model=True,
    )

    assert config.to_cli_args() == [
        "--speedup-ratio",
        "10.0",
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
        "--aic-perf-model",
    ]


def test_mocker_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown mocker config field.*speedup_rato"):
        MockerConfig.from_value({"speedup_rato": 10.0})


@pytest.mark.parametrize("dp_size", [0, -1])
def test_mocker_config_requires_positive_dp_size(dp_size: int) -> None:
    with pytest.raises(ValueError, match="dp_size must be positive"):
        MockerConfig(dp_size=dp_size)
