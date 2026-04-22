# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.health_check import TrtllmHealthCheckPayload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "mode,expect_canary",
    [
        (DisaggregationMode.AGGREGATED, False),
        (DisaggregationMode.PREFILL, False),
        (DisaggregationMode.DECODE, True),
    ],
)
def test_payload_marks_only_decode(mode, expect_canary):
    payload = TrtllmHealthCheckPayload(disaggregation_mode=mode).to_dict()
    assert bool(payload.get("_HEALTH_CHECK")) is expect_canary
    if expect_canary:
        assert payload["disaggregated_params"] == {
            "request_type": "context_and_generation"
        }
    else:
        assert "disaggregated_params" not in payload
