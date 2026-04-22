# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.health_check import (
    CANARY_PROBE_KEY,
    TrtllmDisaggDecodeHealthCheckPayload,
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


def test_decode_mode_registers_probe_payload():
    """Decode workers register a probe payload carrying CANARY_PROBE_KEY.

    The handler detects the marker in `_setup_disaggregated_params_for_mode`
    and routes the probe through `request_type="context_and_generation"`
    so the engine runs it as a local agg request (no cache transceiver).
    Pattern mirrors SGLang's FAKE_BOOTSTRAP_HOST.
    """
    payload = build_worker_health_check_payload(
        disaggregation_mode=DisaggregationMode.DECODE
    )
    assert payload is not None
    assert payload.get(CANARY_PROBE_KEY) is True
    # Standard fields must still be present so the handler's downstream logic
    # (sampling, stop conditions, token_ids) runs normally.
    assert "token_ids" in payload
    assert "sampling_options" in payload
    assert "stop_conditions" in payload


def test_disagg_decode_payload_class_sets_probe_marker():
    """Direct construction of TrtllmDisaggDecodeHealthCheckPayload carries the marker."""
    payload = TrtllmDisaggDecodeHealthCheckPayload().to_dict()
    assert payload.get(CANARY_PROBE_KEY) is True
    assert "disaggregated_params" not in payload  # real params built by handler
