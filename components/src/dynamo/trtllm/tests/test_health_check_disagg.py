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
    "mode,expect_disagg",
    [
        (DisaggregationMode.AGGREGATED, False),
        (DisaggregationMode.PREFILL, False),
        (DisaggregationMode.DECODE, True),
    ],
)
def test_builder_adds_disagg_params_only_for_decode(mode, expect_disagg):
    payload = build_worker_health_check_payload(disaggregation_mode=mode)
    if expect_disagg:
        assert payload["disaggregated_params"] == {
            "request_type": "context_and_generation"
        }
    else:
        assert "disaggregated_params" not in payload
