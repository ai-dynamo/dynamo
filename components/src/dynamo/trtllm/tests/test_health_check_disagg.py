# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.health_check import (
    TrtllmHealthCheckPayload,
    build_worker_health_check_payload,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_1,  # needs trtllm packages installed but does not use GPU
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def test_trtllm_health_check_payload_has_no_disagg_params():
    """Standard TrtllmHealthCheckPayload should NOT include disaggregated_params."""
    payload = TrtllmHealthCheckPayload().to_dict()
    assert "disaggregated_params" not in payload
    assert "prefill_result" not in payload
    assert "token_ids" in payload


@pytest.mark.parametrize(
    "mode",
    [DisaggregationMode.AGGREGATED, DisaggregationMode.PREFILL],
)
def test_non_decode_modes_register_canary_payload(mode):
    """Aggregated and prefill workers register the standard canary payload."""
    payload = build_worker_health_check_payload(disaggregation_mode=mode)
    assert payload is not None
    assert "token_ids" in payload
    assert "sampling_options" in payload


def test_decode_mode_opts_out_of_canary():
    """Decode workers return None so no canary target is registered.

    The trtllm decode handler strictly rejects requests without
    `disaggregated_params` (handler_base.py: "Disaggregated params are required
    for decode mode"), so a generic canary probe cannot satisfy that contract.
    The worker opts out; readiness is signalled by successful endpoint
    registration (see lib/runtime/src/system_health.rs::set_endpoint_registered).
    """
    payload = build_worker_health_check_payload(
        disaggregation_mode=DisaggregationMode.DECODE
    )
    assert payload is None
