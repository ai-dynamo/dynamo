# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""routed_experts wire-format pass-through.

sglang >= 0.5.11 emits meta_info["routed_experts"] as a base64 UTF-8
string (sgl-project/sglang#21634). Both stream paths must forward it
unchanged.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


B64 = "AQAAAAIAAAADAAAABAAAAA=="


class _Context:
    trace_id = "trace"

    def id(self) -> str:
        return "req"

    def is_stopped(self) -> bool:
        return False


def _handler() -> DecodeWorkerHandler:
    h = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    h.use_sglang_tokenizer = False
    h.enable_trace = False
    h.serving_mode = DisaggregationMode.AGGREGATED
    h.config = SimpleNamespace(server_args=SimpleNamespace(served_model_name="m"))
    h._routed_experts_kwargs = {"return_routed_experts": True}
    h._enable_frontend_decoding = False
    h._image_loader = None

    @asynccontextmanager
    async def noop(*a, **k):
        yield None

    h._cancellation_monitor = noop
    return h


async def _token_stream() -> AsyncGenerator[Dict[str, Any], None]:
    yield {
        "output_ids": [42],
        "meta_info": {"id": "r", "finish_reason": None, "routed_experts": B64},
    }


async def _text_stream() -> AsyncGenerator[Dict[str, Any], None]:
    yield {
        "text": "Hi",
        "meta_info": {"id": "r", "finish_reason": None, "routed_experts": B64},
    }


@pytest.mark.asyncio
async def test_token_stream_passes_routed_experts_through():
    h = _handler()
    out = [o async for o in h._process_token_stream(_token_stream(), _Context())]
    assert out[0]["disaggregated_params"] == {"routed_experts": B64}


@pytest.mark.asyncio
async def test_text_stream_passes_routed_experts_through():
    h = _handler()
    h.use_sglang_tokenizer = True
    out = [
        o async for o in h._process_text_stream(_text_stream(), _Context(), request={})
    ]
    assert out[0]["nvext"]["routed_experts"] == B64
