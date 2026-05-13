# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python tests for the TRT-LLM unified prefill→decode handoff codec.

Asymmetry between `_encode_prefill_handoff` and `_decode_prefill_handoff`
is silent corruption — a wrongly-imported KV cache surfaces as a TRT-LLM
error on the decode peer, or worse, garbage tokens.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.skipif(
        importlib.util.find_spec("tensorrt_llm") is None,
        reason="tensorrt_llm not installed in this container",
    ),
]


class _StubGenOutput:
    # Minimal stand-in for a TRT-LLM GenerationResult.
    def __init__(self, disaggregated_params) -> None:
        self.disaggregated_params = disaggregated_params


def test_prefill_handoff_codec_round_trip_preserves_opaque_state():
    from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams

    from dynamo.trtllm.llm_engine import TrtllmLLMEngine

    input_params = LlmDisaggregatedParams(
        request_type="context_only",
        disagg_request_id=0xDEADBEEF,
        opaque_state=b"\x00\x01\x02opaque-kv-handle\xff",
    )
    wire_dict = TrtllmLLMEngine._encode_prefill_handoff(
        _StubGenOutput(input_params), input_params
    )
    assert wire_dict is not None
    # opaque_state survives JSON via base64.
    assert isinstance(wire_dict["opaque_state"], str)

    decoded = TrtllmLLMEngine._decode_prefill_handoff(
        {"disaggregated_params": wire_dict}
    )
    assert decoded.opaque_state == b"\x00\x01\x02opaque-kv-handle\xff"
    assert decoded.disagg_request_id == 0xDEADBEEF
    # decode flips request_type so TRT-LLM skips the context phase.
    assert decoded.request_type == "generation_only"


def test_decode_handoff_strips_router_worker_id():
    # The Rust router stamps `worker_id` onto the wire dict for routing;
    # it's not a DisaggregatedParams constructor arg, so decode must drop
    # it before the dataclass call or TRT-LLM raises TypeError.
    from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams

    from dynamo.trtllm.llm_engine import TrtllmLLMEngine

    input_params = LlmDisaggregatedParams(
        request_type="context_only",
        disagg_request_id=1,
        opaque_state=b"x",
    )
    wire_dict = TrtllmLLMEngine._encode_prefill_handoff(
        _StubGenOutput(input_params), input_params
    )
    wire_dict["worker_id"] = {"prefill_worker_id": 7, "prefill_dp_rank": 0}

    decoded = TrtllmLLMEngine._decode_prefill_handoff(
        {"disaggregated_params": wire_dict}
    )
    assert decoded.opaque_state == b"x"


def test_decode_handoff_rejects_empty_payload():
    from dynamo.trtllm.llm_engine import TrtllmLLMEngine

    with pytest.raises(ValueError, match="prefill_result"):
        TrtllmLLMEngine._decode_prefill_handoff({"disaggregated_params": {}})
