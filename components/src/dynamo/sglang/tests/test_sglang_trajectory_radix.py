# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def _handler(*, enabled: bool = True):
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(enable_session_radix_cache=enabled)
    )
    return handler


def test_trajectory_id_becomes_sglang_session_tag():
    handler = _handler()
    request = {"agent_context": {"trajectory_id": "trajectory-1"}}

    assert handler._session_kwargs(request) == {
        "session_params": {"id": "trajectory-1"}
    }
    assert _handler(enabled=False)._session_kwargs(request) == {}


@pytest.mark.asyncio
async def test_only_final_trajectory_closes_after_stream():
    closed = []

    async def close_session(request, _context):
        closed.append(request.session_id)

    async def source():
        yield "first"
        yield "last"

    handler = _handler()
    handler.engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(close_session=close_session)
    )

    nonfinal = handler._wrap_trajectory_stream(
        source(), {"agent_context": {"trajectory_id": "still-running"}}
    )
    assert [item async for item in nonfinal] == ["first", "last"]
    assert closed == []

    final = handler._wrap_trajectory_stream(
        source(),
        {
            "agent_context": {
                "trajectory_id": "trajectory-final",
                "trajectory_final": True,
            }
        },
    )
    assert [item async for item in final] == ["first", "last"]
    assert closed == ["trajectory-final"]
