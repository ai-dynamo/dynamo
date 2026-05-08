# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python tests for the TRT-LLM unified prefill→decode handoff codec.

The unified path's ``_encode_prefill_handoff`` runs on the prefill side
to produce a JSON-safe wire payload; ``_decode_prefill_handoff`` runs on
the decode side to reconstruct an ``LlmDisaggregatedParams`` ready for a
``generation_only`` call. The two halves MUST be exactly symmetric — any
field that round-trips through one but not the other is silent
corruption that surfaces as a TRT-LLM error on the decode peer or, worse,
a wrongly-imported KV cache.

These tests exercise the codec without a GPU; they're gated on
``tensorrt_llm`` only because the codec uses TRT-LLM's
``DisaggregatedParams`` dataclass.
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
    """Minimal stand-in for a TRT-LLM ``GenerationResult`` carrying just
    the field ``_encode_prefill_handoff`` reads."""

    def __init__(self, disaggregated_params) -> None:
        self.disaggregated_params = disaggregated_params


def test_prefill_handoff_codec_round_trip_preserves_opaque_state():
    """The round trip must preserve ``opaque_state`` bytes through base64
    transport and ``disagg_request_id`` through the JSON-safe encoding."""
    from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams

    from dynamo.trtllm.llm_engine import TrtllmLLMEngine

    input_params = LlmDisaggregatedParams(
        request_type="context_only",
        disagg_request_id=0xDEADBEEF,
        opaque_state=b"\x00\x01\x02opaque-kv-handle\xff",
    )
    stub_output = _StubGenOutput(input_params)

    wire_dict = TrtllmLLMEngine._encode_prefill_handoff(stub_output, input_params)
    assert wire_dict is not None
    # opaque_state must be a string after encode (base64) so it survives
    # JSON serialisation across the Rust transport.
    assert isinstance(wire_dict["opaque_state"], str)

    decoded = TrtllmLLMEngine._decode_prefill_handoff(
        {"disaggregated_params": wire_dict}
    )
    assert decoded.opaque_state == b"\x00\x01\x02opaque-kv-handle\xff"
    assert decoded.disagg_request_id == 0xDEADBEEF
    # The decode side flips request_type so TRT-LLM skips the context phase.
    assert decoded.request_type == "generation_only"


def test_prefill_handoff_codec_round_trip_falls_back_to_input_params():
    """``_encode_prefill_handoff`` falls back to the input params when the
    engine returns ``None`` for ``output.disaggregated_params``. The round
    trip on this fallback path must still recover the original opaque
    state — otherwise prefill silently drops the handoff."""
    from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams

    from dynamo.trtllm.llm_engine import TrtllmLLMEngine

    input_params = LlmDisaggregatedParams(
        request_type="context_only",
        disagg_request_id=42,
        opaque_state=b"input-only",
    )
    stub_output_no_params = _StubGenOutput(disaggregated_params=None)

    wire_dict = TrtllmLLMEngine._encode_prefill_handoff(
        stub_output_no_params, input_params
    )
    assert wire_dict is not None

    decoded = TrtllmLLMEngine._decode_prefill_handoff(
        {"disaggregated_params": wire_dict}
    )
    assert decoded.opaque_state == b"input-only"
    assert decoded.request_type == "generation_only"


def test_decode_handoff_drops_router_only_worker_id():
    """The Rust router can stamp a ``worker_id`` field onto the wire dict
    for routing decisions. That field is NOT a ``DisaggregatedParams``
    constructor arg — ``_decode_prefill_handoff`` must drop it before
    handing the dict off, otherwise TRT-LLM raises ``TypeError``."""
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

    # Must not raise — worker_id is dropped before the dataclass constructor.
    decoded = TrtllmLLMEngine._decode_prefill_handoff(
        {"disaggregated_params": wire_dict}
    )
    assert decoded.opaque_state == b"x"


def test_decode_handoff_rejects_request_without_disaggregated_params():
    """Empty payload is unrecoverable on the decode side — fail fast
    instead of letting a downstream ``DisaggregatedParams()`` call raise
    a confusing ``TypeError``."""
    from dynamo.trtllm.llm_engine import TrtllmLLMEngine

    with pytest.raises(ValueError, match="prefill_result"):
        TrtllmLLMEngine._decode_prefill_handoff({"disaggregated_params": {}})
