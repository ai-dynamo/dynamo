# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the realtime orchestrator's turn handling and ASR->LLM->TTS chaining.

The downstream ASR/LLM/TTS clients are faked so the test exercises the
orchestration state machine without a runtime or real workers.
"""

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest
from riva_nim import orchestrator

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class _FakeStream:
    """Async-iterable matching the Dynamo client response stream (items expose .data())."""

    def __init__(self, items):
        self._items = [_FakeResponse(d) for d in items]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _client(streams):
    """A mock endpoint client whose round_robin() yields one fresh stream per call."""
    client = MagicMock()
    client.round_robin = AsyncMock(
        side_effect=[_FakeStream(items) for items in streams]
    )
    return client


async def _events(events):
    for event in events:
        yield event


def _llm_chunk(content):
    return {"choices": [{"delta": {"role": "assistant", "content": content}}]}


@pytest.fixture
def context():
    ctx = MagicMock()
    ctx.is_stopped.return_value = False
    return ctx


async def test_accumulates_audio_and_runs_chain_on_commit(context):
    asr_client = _client([[{"transcript": "what is the capital of france"}]])
    llm_client = _client([[_llm_chunk("The capital "), _llm_chunk("is Paris.")]])
    tts_audio = base64.b64encode(b"reply-pcm").decode()
    tts_client = _client([[{"audio_base64": tts_audio, "sample_rate_hz": 22050}]])

    pipeline = orchestrator.CascadedPipeline(
        asr_client, llm_client, tts_client, llm_model="my-llm"
    )

    chunk_a = base64.b64encode(b"AAAA").decode()
    chunk_b = base64.b64encode(b"BBBB").decode()
    events = [
        {"type": "session.update", "session": {"voice": "aria"}},
        {"type": "input_audio_buffer.append", "audio": chunk_a},
        {"type": "input_audio_buffer.append", "audio": chunk_b},
        {"type": "input_audio_buffer.commit"},
    ]
    out = [e async for e in pipeline.handle(_events(events), context)]

    # Appends emit nothing; the response sequence appears only after commit.
    assert [e["type"] for e in out] == [
        "session.updated",
        "response.created",
        "response.output_audio.delta",
        "response.output_audio.done",
        "response.done",
    ]

    # ASR receives the decoded, concatenated audio of both append chunks.
    asr_req = asr_client.round_robin.call_args.args[0]
    assert base64.b64decode(asr_req["audio_base64"]) == b"AAAABBBB"

    # The transcript is sent to the LLM as a chat message with the configured model.
    llm_req = llm_client.round_robin.call_args.args[0]
    assert llm_req["model"] == "my-llm"
    assert llm_req["messages"] == [
        {"role": "user", "content": "what is the capital of france"}
    ]

    # The accumulated LLM delta text is what gets synthesized.
    tts_req = tts_client.round_robin.call_args.args[0]
    assert tts_req["text"] == "The capital is Paris."

    # The synthesized audio is carried in the output_audio delta.
    delta = next(e for e in out if e["type"] == "response.output_audio.delta")
    assert delta["delta"] == tts_audio


async def test_cancelled_mid_turn_aborts_without_responding(context):
    # is_stopped() is False for the two event-loop checks (append, commit), then
    # True once the chain checks it after ASR — so the turn aborts post-ASR.
    calls = {"n": 0}

    def stopped():
        calls["n"] += 1
        return calls["n"] > 2

    context.is_stopped.side_effect = stopped

    asr_client = _client([[{"transcript": "hello"}]])
    llm_client = _client([])
    tts_client = _client([])
    pipeline = orchestrator.CascadedPipeline(
        asr_client, llm_client, tts_client, llm_model="my-llm"
    )

    events = [
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"X").decode()},
        {"type": "input_audio_buffer.commit"},
    ]
    out = [e async for e in pipeline.handle(_events(events), context)]

    assert out == []  # cancelled mid-turn: no response.* events emitted
    asr_client.round_robin.assert_called_once()
    llm_client.round_robin.assert_not_called()
    tts_client.round_robin.assert_not_called()


async def test_downstream_failure_emits_error_event(context):
    asr_client = MagicMock()
    asr_client.round_robin = AsyncMock(side_effect=RuntimeError("asr unavailable"))
    llm_client = _client([])
    tts_client = _client([])
    pipeline = orchestrator.CascadedPipeline(
        asr_client, llm_client, tts_client, llm_model="my-llm"
    )

    events = [
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"X").decode()},
        {"type": "input_audio_buffer.commit"},
    ]
    out = [e async for e in pipeline.handle(_events(events), context)]

    # The failure surfaces as a realtime error event, not an unhandled exception.
    assert [e["type"] for e in out] == ["error"]
    llm_client.round_robin.assert_not_called()
    tts_client.round_robin.assert_not_called()


async def test_empty_commit_emits_nothing(context):
    asr_client, llm_client, tts_client = _client([]), _client([]), _client([])
    pipeline = orchestrator.CascadedPipeline(
        asr_client, llm_client, tts_client, llm_model="my-llm"
    )

    events = [{"type": "input_audio_buffer.commit"}]
    out = [e async for e in pipeline.handle(_events(events), context)]

    assert out == []  # nothing buffered → no chain run, no events
    asr_client.round_robin.assert_not_called()


async def test_malformed_audio_emits_invalid_request_error(context):
    asr_client, llm_client, tts_client = _client([]), _client([]), _client([])
    pipeline = orchestrator.CascadedPipeline(
        asr_client, llm_client, tts_client, llm_model="my-llm"
    )

    events = [
        {"type": "input_audio_buffer.append", "audio": "!!!not-base64!!!"},
        {"type": "input_audio_buffer.commit"},
    ]
    out = [e async for e in pipeline.handle(_events(events), context)]

    assert [e["type"] for e in out] == ["error"]
    assert out[0]["error"]["type"] == "invalid_request_error"
    asr_client.round_robin.assert_not_called()  # never reaches the chain


async def test_buffer_resets_between_turns(context):
    asr_client = _client([[{"transcript": "first"}], [{"transcript": "second"}]])
    llm_client = _client([[_llm_chunk("one")], [_llm_chunk("two")]])
    tts_client = _client(
        [
            [
                {
                    "audio_base64": base64.b64encode(b"a").decode(),
                    "sample_rate_hz": 22050,
                }
            ],
            [
                {
                    "audio_base64": base64.b64encode(b"b").decode(),
                    "sample_rate_hz": 22050,
                }
            ],
        ]
    )
    pipeline = orchestrator.CascadedPipeline(
        asr_client, llm_client, tts_client, llm_model="my-llm"
    )

    turn1 = base64.b64encode(b"TURN1").decode()
    turn2 = base64.b64encode(b"TURN2").decode()
    events = [
        {"type": "input_audio_buffer.append", "audio": turn1},
        {"type": "input_audio_buffer.commit"},
        {"type": "input_audio_buffer.append", "audio": turn2},
        {"type": "input_audio_buffer.commit"},
    ]
    _ = [e async for e in pipeline.handle(_events(events), context)]

    # Each turn's ASR call sees only that turn's audio (buffer cleared after commit).
    first_audio = asr_client.round_robin.call_args_list[0].args[0]["audio_base64"]
    second_audio = asr_client.round_robin.call_args_list[1].args[0]["audio_base64"]
    assert base64.b64decode(first_audio) == b"TURN1"
    assert base64.b64decode(second_audio) == b"TURN2"
