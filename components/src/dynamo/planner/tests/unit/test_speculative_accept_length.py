# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.state_machine import normalize_speculative_accept_length
from dynamo.planner.core.types import EngineCapabilities, WorkerCapabilities

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]


def _config(speculative_nextn: int) -> PlannerConfig:
    """Build a decode planner config with the requested speculation setting."""
    return PlannerConfig(
        mode="decode",
        optimization_target="sla",
        enable_load_scaling=False,
        enable_throughput_scaling=True,
        min_endpoint=1,
        max_gpu_budget=16,
        speculative_nextn=speculative_nextn,
    )


def _caps(decode_nextn=None, decode=True) -> WorkerCapabilities:
    """Build worker capabilities with an optional decode speculation setting."""
    if not decode:
        return WorkerCapabilities()
    return WorkerCapabilities(
        decode=EngineCapabilities(num_gpu=1, speculative_nextn=decode_nextn)
    )


@pytest.mark.parametrize(
    "accept_length, caps, config_nextn, expected",
    [
        # The decode capability wins over the config.
        (9.0, _caps(decode_nextn=4), 1, 5.0),
        # Config fallback when the capability is missing, zero, or absent.
        (9.0, _caps(decode_nextn=None), 2, 3.0),
        (9.0, _caps(decode_nextn=0), 2, 3.0),
        (9.0, _caps(decode=False), 2, 3.0),
        # Speculation off gives 1.0.
        (2.5, _caps(), 0, 1.0),
        # Missing or non-finite observations give 1.0.
        (None, _caps(), 3, 1.0),
        (math.nan, _caps(), 3, 1.0),
        (math.inf, _caps(), 3, 1.0),
        # Bounded to [1.0, nextn + 1].
        (0.5, _caps(), 3, 1.0),
        (2.5, _caps(), 3, 2.5),
    ],
)
def test_normalize_speculative_accept_length(
    accept_length, caps, config_nextn, expected
):
    """Check capability precedence, fallbacks, and accept-length bounds."""
    config = _config(config_nextn)
    assert normalize_speculative_accept_length(accept_length, caps, config) == expected
