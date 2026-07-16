# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wiring tests for engine-owned conversation-affinity ADP routing on the unified
``TrtllmLLMEngine`` path.

Exercises the four branches of ``_generate_started`` around
``_conversation_affinity``:

* off + router-forced rank → ``SchedulingParams(attention_dp_rank=rank)`` and no
  ``conversation_params`` kwarg.
* off + no rank → ``scheduling_params=None`` and no ``conversation_params`` kwarg.
* on + ``agent_context.session_id`` present → ``scheduling_params=None`` and
  ``conversation_params.conversation_id == session_id``.
* on + no session id → ``scheduling_params=None`` and ``conversation_params=None``
  (the kwarg is still passed so the engine's no-id balancing fallback runs).

Plus the manual ``--conversation-affinity`` / ``DYN_ENGINE_CONV_AFFINITY`` override
(``_engine_conversation_affinity_override``), which forces the affinity branch even when
engine detection is off and raises when the ConversationParams API is missing — matching
the legacy ``HandlerBase`` override tests.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from dynamo.trtllm.constants import DisaggregationMode

# Guard the engine import: package metadata can be present without submodules,
# and native-lib loads (e.g., libcuda.so.1) raise ImportError, not
# ModuleNotFoundError. Catch both via ImportError.
try:
    from dynamo.trtllm.llm_engine import TrtllmLLMEngine
except ImportError:
    pytest.skip("tensorrt_llm backend not available", allow_module_level=True)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


class _FakeContext:
    def id(self) -> str:
        return "conv-affinity-test-req"

    def trace_headers(self) -> dict[str, str] | None:
        return None


async def _empty_async_iter():
    for _ in ():
        yield


class _FakeConversationParams:
    """Stand-in for tensorrt_llm.llmapi.ConversationParams. The unified path
    threads it through opaquely; tests only need to read back ``conversation_id``."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id


def _make_engine(
    generate_async, *, conversation_affinity: bool, attention_dp_size: int = 1
) -> TrtllmLLMEngine:
    # Couples to TrtllmLLMEngine.__init__ attribute names — keep in sync.
    engine = TrtllmLLMEngine.__new__(TrtllmLLMEngine)
    engine._engine = SimpleNamespace(llm=SimpleNamespace(generate_async=generate_async))
    engine._logits_processor_spec = None
    engine._default_sampling_params = SimpleNamespace()
    engine.disaggregation_mode = DisaggregationMode.AGGREGATED
    engine.max_seq_len = 1024
    engine._active_requests = {}
    engine._additional_metrics = None
    engine._inflight_lock = asyncio.Lock()
    engine._inflight_requests = 0
    engine._no_inflight_requests = asyncio.Event()
    engine._no_inflight_requests.set()
    engine._reject_new_requests = False
    engine._conversation_affinity = conversation_affinity
    # Manual override (--conversation-affinity / DYN_ENGINE_CONV_AFFINITY); off by
    # default, individual tests flip it to exercise the override path.
    engine._engine_conversation_affinity_override = False
    engine._attention_dp_size = attention_dp_size
    engine._override_sampling_params = lambda default, request: SimpleNamespace(
        max_tokens=None
    )
    return engine


async def _drain(engine: TrtllmLLMEngine, request: dict) -> None:
    async for _ in engine.generate(request, _FakeContext()):
        pass


async def test_affinity_off_forwards_router_dp_rank():
    """affinity=False + router dp_rank → SchedulingParams with that rank."""
    captured: dict = {}

    def fake_generate_async(**kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    engine = _make_engine(
        fake_generate_async, conversation_affinity=False, attention_dp_size=8
    )
    await _drain(engine, {"token_ids": [1, 2, 3], "routing": {"dp_rank": 3}})

    scheduling_params = captured["scheduling_params"]
    assert scheduling_params is not None
    assert scheduling_params.attention_dp_rank == 3
    assert scheduling_params.attention_dp_relax is False
    assert "conversation_params" not in captured


async def test_affinity_off_no_rank_leaves_scheduling_params_none():
    """affinity=False + no router rank → scheduling_params=None, no conv kwarg."""
    captured: dict = {}

    def fake_generate_async(**kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    engine = _make_engine(
        fake_generate_async, conversation_affinity=False, attention_dp_size=8
    )
    await _drain(engine, {"token_ids": [1, 2, 3]})

    assert captured["scheduling_params"] is None
    assert "conversation_params" not in captured


async def test_affinity_on_with_session_id_suppresses_rank_and_forwards_conv_params(
    monkeypatch,
):
    """affinity=True + session id → suppress rank, forward ConversationParams."""
    monkeypatch.setattr(
        "dynamo.trtllm.conversation_affinity.ConversationParams",
        _FakeConversationParams,
    )

    captured: dict = {}

    def fake_generate_async(**kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    engine = _make_engine(
        fake_generate_async, conversation_affinity=True, attention_dp_size=8
    )
    # Router still stamps a rank; affinity mode must ignore it (an explicit rank
    # would bypass the engine's ConversationAwareADPRouter).
    await _drain(
        engine,
        {
            "token_ids": [1, 2, 3],
            "routing": {"dp_rank": 3},
            "agent_context": {"session_id": "run-42:agent-0"},
        },
    )

    assert captured["scheduling_params"] is None
    conv_params = captured["conversation_params"]
    assert conv_params is not None
    assert conv_params.conversation_id == "run-42:agent-0"


async def test_affinity_on_without_session_id_passes_none_conversation_params(
    monkeypatch,
):
    """affinity=True + no session id → scheduling_params=None,
    conversation_params kwarg present with value None (no-id balancing)."""
    monkeypatch.setattr(
        "dynamo.trtllm.conversation_affinity.ConversationParams",
        _FakeConversationParams,
    )

    captured: dict = {}

    def fake_generate_async(**kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    engine = _make_engine(
        fake_generate_async, conversation_affinity=True, attention_dp_size=8
    )
    await _drain(engine, {"token_ids": [1, 2, 3]})

    assert captured["scheduling_params"] is None
    assert "conversation_params" in captured
    assert captured["conversation_params"] is None


async def test_override_suppresses_dp_rank_and_forwards_conversation_params(
    monkeypatch,
):
    """DYN_ENGINE_CONV_AFFINITY override=True + engine detection disabled →
    dp_rank suppressed, conversation_params forwarded. Mirrors the legacy
    HandlerBase override test so both entry points stay in lockstep."""
    monkeypatch.setattr(
        "dynamo.trtllm.llm_engine.CONVERSATION_PARAMS_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        "dynamo.trtllm.conversation_affinity.ConversationParams",
        _FakeConversationParams,
    )

    captured: dict = {}

    def fake_generate_async(**kwargs):
        captured.update(kwargs)
        return _empty_async_iter()

    # Engine detection returns False (no engine-side affinity config)...
    engine = _make_engine(
        fake_generate_async, conversation_affinity=False, attention_dp_size=8
    )
    # ...but the operator forced it via DYN_ENGINE_CONV_AFFINITY.
    engine._engine_conversation_affinity_override = True
    # Router still stamps a rank; the override must ignore it just like
    # auto-detection does.
    await _drain(
        engine,
        {
            "token_ids": [1, 2, 3],
            "routing": {"dp_rank": 3},
            "agent_context": {"session_id": "run-99:agent-0"},
        },
    )

    assert captured["scheduling_params"] is None
    conv_params = captured["conversation_params"]
    assert conv_params is not None
    assert conv_params.conversation_id == "run-99:agent-0"


async def test_override_raises_when_conversation_params_api_missing(monkeypatch):
    """DYN_ENGINE_CONV_AFFINITY=true on a build without ConversationParams →
    RuntimeError in _generate_started, before generate_async is called."""
    monkeypatch.setattr(
        "dynamo.trtllm.llm_engine.CONVERSATION_PARAMS_AVAILABLE",
        False,
    )

    called = False

    def fake_generate_async(**kwargs):
        nonlocal called
        called = True
        return _empty_async_iter()

    engine = _make_engine(
        fake_generate_async, conversation_affinity=False, attention_dp_size=8
    )
    engine._engine_conversation_affinity_override = True

    with pytest.raises(RuntimeError, match="DYN_ENGINE_CONV_AFFINITY"):
        await _drain(engine, {"token_ids": [1, 2, 3]})
    assert called is False
