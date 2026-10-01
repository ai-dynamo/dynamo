# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for forwarding agent_context session ids to SGLang.

Engine-independent: every kwarg is filtered against the engine's declared
``async_generate`` signature, so the stubs below stand in for a session-aware
build, a build predating ``parent_session_id``, and one predating both.
"""

from types import SimpleNamespace
from typing import Any, Optional

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.sglang.agent_session import agent_session_kwargs, session_ids_from_request
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler
from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

_AGENT_CONTEXT = {"session_id": "child-1", "parent_session_id": "root-0"}

# Default for the stub's session kwargs, so a recorded call distinguishes a kwarg
# the handler omitted from one it passed explicitly (even as None).
_UNSET = object()


class _SessionAwareEngine:
    """Engine whose async_generate declares both session kwargs.

    Each recorded call holds only the session kwargs the handler actually passed.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_generate(
        self,
        session_id: Any = _UNSET,
        parent_session_id: Any = _UNSET,
        **kwargs: Any,
    ):
        # **kwargs absorbs the unrelated per-request arguments the handlers pass.
        # A **kwargs signature makes the compat filter pass every kwarg through, so
        # the record below is exactly what the handler sent.
        self.calls.append(
            {
                name: value
                for name, value in (
                    ("session_id", session_id),
                    ("parent_session_id", parent_session_id),
                )
                if value is not _UNSET
            }
        )

        async def stream():
            yield {"text": "", "output_ids": [], "meta_info": {"id": "sglang-rid"}}

        return stream()


class _LegacyEngine:
    """Engine predating session forwarding: neither kwarg exists."""

    async def async_generate(self, prompt=None):
        raise NotImplementedError


class _ParentUnawareEngine:
    """Engine with the long-standing session_id but no parent_session_id."""

    async def async_generate(self, prompt=None, session_id: Optional[str] = None):
        raise NotImplementedError


# --------------------------------------------------------------------------
# session_ids_from_request: normalization of the agent_context payload
# --------------------------------------------------------------------------


def test_session_ids_from_request_returns_both_for_a_subagent_request():
    assert session_ids_from_request({"agent_context": _AGENT_CONTEXT}) == {
        "session_id": "child-1",
        "parent_session_id": "root-0",
    }


def test_session_ids_from_request_root_request_has_no_parent():
    assert session_ids_from_request({"agent_context": {"session_id": "root-0"}}) == {
        "session_id": "root-0"
    }


def test_session_ids_from_request_trims_surrounding_whitespace():
    assert session_ids_from_request(
        {"agent_context": {"session_id": " child-1 ", "parent_session_id": " root-0 "}}
    ) == {"session_id": "child-1", "parent_session_id": "root-0"}


@pytest.mark.parametrize(
    "request_payload",
    [
        {},
        {"agent_context": None},
        {"agent_context": "not-a-dict"},
        {"agent_context": {}},
        {"agent_context": {"session_id": ""}},
        {"agent_context": {"session_id": "   "}},
        {"agent_context": {"session_id": 123}},
    ],
)
def test_session_ids_from_request_absent_or_malformed(request_payload):
    assert session_ids_from_request(request_payload) == {}


def test_session_ids_from_request_normalizes_each_field_independently():
    """A malformed session_id must not suppress the parent keepalive hint."""
    assert session_ids_from_request(
        {"agent_context": {"session_id": None, "parent_session_id": "root-0"}}
    ) == {"parent_session_id": "root-0"}


# --------------------------------------------------------------------------
# agent_session_kwargs: filtering against the engine signature
# --------------------------------------------------------------------------


def test_session_aware_engine_receives_both_ids():
    assert agent_session_kwargs(
        _SessionAwareEngine(), {"agent_context": _AGENT_CONTEXT}
    ) == {"session_id": "child-1", "parent_session_id": "root-0"}


def test_engine_without_the_kwargs_receives_nothing():
    assert (
        agent_session_kwargs(_LegacyEngine(), {"agent_context": _AGENT_CONTEXT}) == {}
    )


def test_engine_receives_only_the_kwargs_it_declares():
    assert agent_session_kwargs(
        _ParentUnawareEngine(), {"agent_context": _AGENT_CONTEXT}
    ) == {"session_id": "child-1"}


def test_request_without_agent_context_sends_nothing_to_a_capable_engine():
    assert agent_session_kwargs(_SessionAwareEngine(), {}) == {}


# --------------------------------------------------------------------------
# Handler wiring: the ids reach engine.async_generate on every LLM path
# --------------------------------------------------------------------------


def _stub_generate_dependencies(handler: Any) -> None:
    """Neutralize the per-request helpers unrelated to session forwarding."""
    handler._get_input_param = lambda request: {"input_ids": [1, 2, 3]}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}
    handler.enable_trace = False


def _new_decode_handler(engine: Any, serving_mode: DisaggregationMode):
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.engine = engine
    handler.serving_mode = serving_mode
    handler.use_sglang_tokenizer = False
    handler._first_token_source = None
    handler._enable_frontend_decoding = False
    handler._mm_hashes_supported = False
    handler._routed_experts_kwargs = {}
    _stub_generate_dependencies(handler)
    handler._build_sampling_params = lambda request: {"max_new_tokens": 1}
    handler._build_logprob_kwargs = lambda request: {}
    handler._metadata_uploader_from_request = lambda request: None

    async def passthrough(stream, context, *args, **kwargs):
        async for chunk in stream:
            yield chunk

    handler._process_token_stream = passthrough
    return handler


def _new_prefill_handler(engine: Any):
    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.engine = engine
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 8998
    handler._generate_bootstrap_room = lambda: 42
    _stub_generate_dependencies(handler)
    return handler


def _context():
    return SimpleNamespace(
        id=lambda: "request-id",
        trace_id="trace-id",
        trace_headers=lambda: None,
        is_stopped=lambda: False,
        notify_first_token=lambda: None,
    )


@pytest.mark.asyncio
async def test_aggregated_generate_forwards_both_session_ids():
    engine = _SessionAwareEngine()
    handler = _new_decode_handler(engine, DisaggregationMode.AGGREGATED)

    async for _ in handler.generate(
        {"agent_context": _AGENT_CONTEXT}, _context()
    ):  # noqa: B007
        pass

    assert engine.calls == [{"session_id": "child-1", "parent_session_id": "root-0"}]


@pytest.mark.asyncio
async def test_disaggregated_decode_generate_forwards_both_session_ids():
    engine = _SessionAwareEngine()
    handler = _new_decode_handler(engine, DisaggregationMode.DECODE)
    request = {
        "agent_context": _AGENT_CONTEXT,
        "bootstrap_info": {
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": 8998,
            "bootstrap_room": 42,
        },
    }

    async for _ in handler.generate(request, _context()):  # noqa: B007
        pass

    assert engine.calls == [{"session_id": "child-1", "parent_session_id": "root-0"}]


@pytest.mark.asyncio
async def test_prefill_generate_reads_the_inner_disagg_request():
    """agent_context rides the inner PreprocessedRequest, not the disagg envelope."""
    engine = _SessionAwareEngine()
    handler = _new_prefill_handler(engine)
    request = {
        "request": {"agent_context": _AGENT_CONTEXT},
        "sampling_params": {"max_new_tokens": 16},
    }

    async for _ in handler.generate(request, _context()):  # noqa: B007
        break

    assert engine.calls == [{"session_id": "child-1", "parent_session_id": "root-0"}]


@pytest.mark.asyncio
async def test_generate_without_agent_context_sends_no_session_kwargs():
    engine = _SessionAwareEngine()
    handler = _new_decode_handler(engine, DisaggregationMode.AGGREGATED)

    async for _ in handler.generate({}, _context()):  # noqa: B007
        pass

    assert engine.calls == [{}]
