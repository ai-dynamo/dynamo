# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SGLang unified engine's disagg dispatch.

Targets the pure-Python helpers that decide which (host, port, room)
triple a request uses, so the tests don't need a running engine. They
do require ``sglang`` to be importable because the module imports it
at top level — gated with ``importorskip``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "sglang",
    reason="sglang not installed in this container",
)

from dynamo.common.constants import DisaggregationMode  # noqa: E402
from dynamo.sglang.llm_engine import (  # noqa: E402
    SglangDeferredAbort,
    SglangLLMEngine,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _engine_with_cached_bootstrap() -> SglangLLMEngine:
    """Bypass ``__init__`` so we can exercise the bootstrap helpers
    without spinning up a real ``sgl.Engine`` (which needs a GPU)."""
    engine = SglangLLMEngine.__new__(SglangLLMEngine)
    engine._bootstrap_host = "10.0.0.5"
    engine._bootstrap_port = 12345
    engine.serving_mode = DisaggregationMode.PREFILL
    return engine


def test_prefill_bootstrap_uses_router_provided_info_when_present():
    # Bootstrap path: the Rust PrefillRouter resolved bootstrap upfront
    # and writes (host, port, room) onto request.bootstrap_info before
    # calling prefill. The engine MUST honour that exact triple so the
    # decode peer (which the router writes the same triple onto) finds
    # the matching room.
    engine = _engine_with_cached_bootstrap()
    request = {
        "token_ids": [1, 2, 3],
        "bootstrap_info": {
            "bootstrap_host": "router.host",
            "bootstrap_port": 9999,
            "bootstrap_room": 42,
        },
    }
    out = engine._resolve_prefill_bootstrap(request)
    assert out == {
        "bootstrap_host": "router.host",
        "bootstrap_port": 9999,
        "bootstrap_room": 42,
    }


def test_prefill_bootstrap_falls_back_to_engine_address_when_request_has_none():
    # Completed path: the router didn't resolve bootstrap upfront, so
    # the engine generates its own room and uses its own address. The
    # router will pull this triple off the prefill response and forward
    # it to decode.
    engine = _engine_with_cached_bootstrap()
    out = engine._resolve_prefill_bootstrap({"token_ids": [1, 2, 3]})
    assert out["bootstrap_host"] == "10.0.0.5"
    assert out["bootstrap_port"] == 12345
    assert isinstance(out["bootstrap_room"], int)


def test_decode_bootstrap_reads_bootstrap_info_first():
    # Bootstrap path: router puts bootstrap info directly on the decode
    # request. Decode workers must read it from there.
    out = SglangLLMEngine._resolve_decode_bootstrap(
        {
            "token_ids": [1, 2, 3],
            "bootstrap_info": {
                "bootstrap_host": "h",
                "bootstrap_port": 1,
                "bootstrap_room": 2,
            },
        }
    )
    assert out == {
        "bootstrap_host": "h",
        "bootstrap_port": 1,
        "bootstrap_room": 2,
    }


def test_decode_bootstrap_falls_back_to_prefill_result():
    # Completed path: router waits for prefill, packs the prefill
    # response's disaggregated_params into prefill_result. Decode
    # workers must accept that fallback shape.
    out = SglangLLMEngine._resolve_decode_bootstrap(
        {
            "token_ids": [1, 2, 3],
            "prefill_result": {
                "disaggregated_params": {
                    "bootstrap_host": "h",
                    "bootstrap_port": 1,
                    "bootstrap_room": 2,
                },
            },
        }
    )
    assert out == {
        "bootstrap_host": "h",
        "bootstrap_port": 1,
        "bootstrap_room": 2,
    }


def test_decode_bootstrap_raises_when_neither_path_populated():
    # Both router paths missing → loud failure. Silent zero-fill would
    # leave the decode worker waiting on a phantom KV transfer.
    with pytest.raises(ValueError, match="bootstrap"):
        SglangLLMEngine._resolve_decode_bootstrap({"token_ids": [1, 2, 3]})


def test_decode_bootstrap_raises_when_required_field_missing():
    # Partial bootstrap info is a configuration bug — surface the
    # specific missing field so operators don't have to debug from a
    # generic NIXL connection error later.
    with pytest.raises(ValueError, match="bootstrap_room"):
        SglangLLMEngine._resolve_decode_bootstrap(
            {
                "token_ids": [1, 2, 3],
                "bootstrap_info": {"bootstrap_host": "h", "bootstrap_port": 1},
            }
        )


@pytest.mark.asyncio
async def test_sglang_deferred_abort_invokes_tokenizer_abort_request():
    """`SglangDeferredAbort._do_abort_now` forwards to
    `tokenizer_manager.abort_request(rid=..., abort_all=False)`."""
    tokenizer_manager = MagicMock()
    guard = SglangDeferredAbort(tokenizer_manager, "req-xyz")
    guard.signal_first_token()
    guard.abort()
    await asyncio.sleep(0)
    tokenizer_manager.abort_request.assert_called_once_with(
        rid="req-xyz", abort_all=False
    )
