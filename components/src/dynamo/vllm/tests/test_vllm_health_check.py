#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression guard: vLLM workers must always register a canary payload.

DIS-1737 originally routed disagg decode through a `payload = None` branch in
`worker_factory.py`. That silently opted decode workers out of canary, which
after DIS-1185 (canary = sole readiness authority) left them stuck NotReady.
These tests ensure the payload constructor works for both agg and decode paths
so no one re-introduces a DECODE-specific None branch.
"""

import pytest

from dynamo.vllm.health_check import VllmHealthCheckPayload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize("use_text_input", [False, True])
def test_vllm_health_check_payload_is_non_none(use_text_input):
    """VllmHealthCheckPayload.to_dict() returns a non-None dict regardless of
    tokenizer mode. Worker code must always register a canary target; decode
    workers rely on the vLLM handler's natural agg-style fallback when
    `prefill_result` is absent (handlers.py::_generate_token_mode)."""
    payload = VllmHealthCheckPayload(
        engine_client=None, use_text_input=use_text_input
    ).to_dict()

    assert payload is not None
    assert isinstance(payload, dict)
    # Payload must contain some form of input the engine can decode.
    assert "token_ids" in payload or "prompt" in payload
