# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.state_machine import (
    PlannerScalingState,
    normalize_speculative_accept_length,
)
from dynamo.planner.core.types import EngineCapabilities, WorkerCapabilities
from dynamo.planner.plugins.builtins.local_planner import BuiltinLoadPredict
from dynamo.planner.plugins.types import (
    ObservationData,
    PipelineContext,
    PredictStageRequest,
    TrafficMetrics,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]


def _config(speculative_nextn: int) -> PlannerConfig:
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
    if not decode:
        return WorkerCapabilities()
    return WorkerCapabilities(
        decode=EngineCapabilities(num_gpu=1, speculative_nextn=decode_nextn)
    )


@pytest.mark.parametrize(
    "accept_length, caps, config_nextn, expected",
    [
        # The decode capability wins over the config.
        (4.5, _caps(decode_nextn=4), 1, 4.5),
        (9.0, _caps(decode_nextn=4), 1, 5.0),
        # Config fallback when the capability is missing, zero, or absent.
        (9.0, _caps(decode_nextn=None), 2, 3.0),
        (9.0, _caps(decode_nextn=0), 2, 3.0),
        (9.0, _caps(decode=False), 2, 3.0),
        # Speculation off (including a negative config) gives 1.0.
        (2.5, _caps(), 0, 1.0),
        (2.5, _caps(), -3, 1.0),
        # Missing or non-finite observations give 1.0.
        (None, _caps(), 3, 1.0),
        (math.nan, _caps(), 3, 1.0),
        (math.inf, _caps(), 3, 1.0),
        # Bounded to [1.0, nextn + 1].
        (0.5, _caps(), 3, 1.0),
        (2.5, _caps(), 3, 2.5),
        (7.0, _caps(), 3, 4.0),
    ],
)
def test_normalize_speculative_accept_length(
    accept_length, caps, config_nextn, expected
):
    config = _config(0)
    # Assigned after construction because field validation rejects negatives.
    config.speculative_nextn = config_nextn
    assert normalize_speculative_accept_length(accept_length, caps, config) == expected


def _state_observe(state: PlannerScalingState, accept_length: float) -> float:
    state.observe_runtime_metadata(accept_length=accept_length)
    return state._current_decode_accept_length()


def _plugin_observe(plugin: BuiltinLoadPredict, accept_length: float) -> float:
    request = PredictStageRequest(
        context=PipelineContext(
            observations=ObservationData(
                traffic=TrafficMetrics(
                    duration_s=60.0,
                    num_req=10.0,
                    isl=128.0,
                    osl=64.0,
                    accept_length=accept_length,
                )
            )
        )
    )
    response = asyncio.run(plugin.Predict(request))
    assert response.predictions is not None
    return response.predictions.predicted_accept_length


@pytest.mark.parametrize(
    "make, observe",
    [
        (lambda c, caps: PlannerScalingState(c, caps), _state_observe),
        (lambda c, caps: BuiltinLoadPredict(c, caps), _plugin_observe),
    ],
    ids=["scaling_state", "load_predict"],
)
def test_callers_clamp_on_observe_update_and_read(make, observe):
    caller = make(_config(1), _caps())
    assert observe(caller, 3.5) == 2.0

    # A capability update re-clamps the stored value; it does not restore the
    # raw observation.
    caller.update_capabilities(_caps(decode_nextn=4))
    assert observe(caller, math.nan) == 2.0

    assert observe(caller, 3.5) == 3.5
