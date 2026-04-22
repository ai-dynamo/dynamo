# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.health_check import build_worker_health_check_payload

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
def test_builder_marks_only_decode(mode, expect_canary):
    payload = build_worker_health_check_payload(disaggregation_mode=mode)
    assert bool(payload.get("_CANARY_HEALTH_CHECK")) is expect_canary
    if expect_canary:
        assert payload["disaggregated_params"] == {
            "request_type": "context_and_generation"
        }
    else:
        assert "disaggregated_params" not in payload
