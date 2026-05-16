# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for routed_experts wire-format pass-through (DYN-3046).

sglang >= 0.5.11 emits ``meta_info["routed_experts"]`` as a base64-encoded
UTF-8 string (see upstream PR sgl-project/sglang#21634 and the conditional
encode in ``python/sglang/srt/managers/tokenizer_manager.py``). Earlier
Dynamo code called ``.numpy().tobytes()`` on the value, which crashed with
``AttributeError: 'str' object has no attribute 'numpy'`` whenever
``--enable-return-routed-experts`` was on against any non-DSv4 sglang build.

These tests pin the contract: when meta_info carries a pre-encoded string,
both stream paths must pass it through unchanged.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, List

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


# A representative base64-encoded payload matching sglang's wire format:
# pybase64.b64encode(<int32 tensor>.numpy().tobytes()).decode("utf-8")
ROUTED_EXPERTS_B64 = "AQAAAAIAAAADAAAABAAAAA=="  # 4x int32: [1, 2, 3, 4]


class _Context:
    id_value: str = "dyn-3046-test"
    trace_id: str = "dyn-3046-trace"

    def id(self) -> str:
        return self.id_value

    def is_stopped(self) -> bool:
        return False


def _new_decode_handler() -> DecodeWorkerHandler:
    """Build a DecodeWorkerHandler without invoking sgl.Engine.

    Mirrors the pattern in test_sglang_frontend_decoding.py.
    """
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.use_sglang_tokenizer = False
    handler.enable_trace = False
    handler.serving_mode = DisaggregationMode.AGGREGATED
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(served_model_name="moe-test-model")
    )
    handler._routed_experts_kwargs = {"return_routed_experts": True}
    handler._enable_frontend_decoding = False
    handler._image_loader = None

    @asynccontextmanager
    async def no_cancellation_monitor(*args, **kwargs):
        yield None

    handler._cancellation_monitor = no_cancellation_monitor
    return handler


async def _moe_token_stream(b64_payload: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Mimic sglang >= 0.5.11 token-mode output for an MoE model.

    One mid-stream token chunk with routed_experts, then a finishing chunk.
    """
    yield {
        "output_ids": [42],
        "meta_info": {
            "id": "sgl-req-1",
            "finish_reason": None,
            "routed_experts": b64_payload,
        },
    }
    yield {
        "output_ids": [],
        "meta_info": {
            "id": "sgl-req-1",
            "finish_reason": {"type": "stop", "matched": None},
            "prompt_tokens": 19,
            "completion_tokens": 1,
            "cached_tokens": 0,
        },
    }


async def _moe_text_stream(b64_payload: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Mimic sglang >= 0.5.11 text-mode (BatchStrOutput) chunks for MoE."""
    yield {
        "text": "Hi",
        "meta_info": {
            "id": "sgl-req-1",
            "finish_reason": None,
            "routed_experts": b64_payload,
        },
    }
    yield {
        "text": "Hi",
        "meta_info": {
            "id": "sgl-req-1",
            "finish_reason": {"type": "stop", "matched": None},
            "routed_experts": b64_payload,
        },
    }


@pytest.mark.asyncio
async def test_token_stream_passes_routed_experts_string_through():
    """Token-mode path must accept sglang's pre-encoded base64 string and
    forward it verbatim in ``disaggregated_params.routed_experts``."""
    handler = _new_decode_handler()

    outputs: List[Dict[str, Any]] = []
    async for out in handler._process_token_stream(
        _moe_token_stream(ROUTED_EXPERTS_B64),
        _Context(),
    ):
        outputs.append(out)

    routed_chunks = [o for o in outputs if "disaggregated_params" in o]
    assert routed_chunks, "expected at least one chunk carrying routed_experts"
    assert routed_chunks[0]["disaggregated_params"] == {
        "routed_experts": ROUTED_EXPERTS_B64
    }


@pytest.mark.asyncio
async def test_text_stream_passes_routed_experts_string_through():
    """Text-mode (sglang tokenizer) path must forward the pre-encoded
    base64 string into ``nvext.routed_experts`` verbatim."""
    handler = _new_decode_handler()
    handler.use_sglang_tokenizer = True

    outputs: List[Dict[str, Any]] = []
    async for resp in handler._process_text_stream(
        _moe_text_stream(ROUTED_EXPERTS_B64),
        _Context(),
        request={},
    ):
        outputs.append(resp)

    routed_chunks = [
        o for o in outputs if "nvext" in o and "routed_experts" in o["nvext"]
    ]
    assert routed_chunks, "expected at least one chunk carrying nvext.routed_experts"
    assert routed_chunks[0]["nvext"]["routed_experts"] == ROUTED_EXPERTS_B64


@pytest.mark.asyncio
async def test_token_stream_omits_routed_experts_when_absent():
    """Sanity check: when meta_info has no routed_experts, no
    disaggregated_params field appears."""
    handler = _new_decode_handler()

    async def plain_stream() -> AsyncGenerator[Dict[str, Any], None]:
        yield {
            "output_ids": [42],
            "meta_info": {
                "id": "sgl-req-1",
                "finish_reason": None,
            },
        }
        yield {
            "output_ids": [],
            "meta_info": {
                "id": "sgl-req-1",
                "finish_reason": {"type": "stop", "matched": None},
                "prompt_tokens": 19,
                "completion_tokens": 1,
                "cached_tokens": 0,
            },
        }

    async for out in handler._process_token_stream(plain_stream(), _Context()):
        assert "disaggregated_params" not in out
