# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.health_check import HEALTH_CHECK_KEY, TrtllmHealthCheckPayload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "mode,expect_disagg",
    [
        (DisaggregationMode.AGGREGATED, False),
        (DisaggregationMode.PREFILL, False),
        (DisaggregationMode.DECODE, True),
    ],
)
def test_payload_shape(mode, expect_disagg):
    payload = TrtllmHealthCheckPayload(disaggregation_mode=mode).to_dict()
    assert payload[HEALTH_CHECK_KEY] is True
    if expect_disagg:
        assert payload["disaggregated_params"] == {
            "request_type": "context_and_generation"
        }
    else:
        assert "disaggregated_params" not in payload
