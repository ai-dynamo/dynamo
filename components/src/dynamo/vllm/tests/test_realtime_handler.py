# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import numpy as np
import pytest

from dynamo.vllm.realtime import (
    RealtimeHandler,
    RealtimeTextHandler,
    RealtimeTranscriptionHandler,
)
from dynamo.vllm.realtime.handler import _TextPrefill, _TextTurn

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

MODEL = "test/realtime-asr"


class _Context:
    def __init__(self) -> None:
        self.stopped = False

    def is_stopped(self) -> bool:
        return self.stopped


class _RecordingHandler:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def generate(self, request_stream, context):
        del context
        async for event in request_stream:
            self.events.append(event)
        yield {"type": "session.updated"}


class _FakeEngine:
    def __init__(self) -> None:
        self.audio: list[np.ndarray] = []
        self.request_ids: list[str] = []

    async def generate(self, *, prompt, sampling_params, request_id):
        del sampling_params
        self.request_ids.append(request_id)
        async for chunk in prompt:
            self.audio.append(chunk)
        yield SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            outputs=[SimpleNamespace(text="hello ", token_ids=[10, 11])],
        )
        yield SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            outputs=[SimpleNamespace(text="world", token_ids=[12])],
        )


async def _stream_audio(audio_stream, input_stream):
    del input_stream
    async for chunk in audio_stream:
        yield chunk


def _handler(engine: _FakeEngine) -> RealtimeTranscriptionHandler:
    return RealtimeTranscriptionHandler(
        engine_client=engine,
        model_name=MODEL,
        model_sample_rate=16_000.0,
        streaming_input_factory=_stream_audio,
        sampling_params_factory=lambda: object(),
    )


def _session(*, turn_detection=None) -> dict:
    return {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24_000},
                "transcription": {"model": MODEL, "language": "en"},
                "noise_reduction": None,
                "turn_detection": turn_detection,
            }
        },
    }


async def _drive(handler, events, *, before_commit: asyncio.Event | None = None):
    async def request_stream():
        for event in events:
            if event["type"] == "input_text.commit" and before_commit is not None:
                await asyncio.wait_for(before_commit.wait(), timeout=5)
            yield event

    return [event async for event in handler.generate(request_stream(), _Context())]


def test_dispatches_session_and_replays_initial_update():
    transcription = _RecordingHandler()
    handler = RealtimeHandler({"transcription": transcription})
    events = [
        {
            "type": "session.update",
            "event_id": "event_1",
            "session": {"type": "transcription"},
        },
        {"type": "input_audio_buffer.commit"},
    ]

    result = asyncio.run(_drive(handler, events))

    assert result == [{"type": "session.updated"}]
    assert transcription.events == events


def test_rejects_unsupported_session_type():
    transcription = _RecordingHandler()
    handler = RealtimeHandler({"transcription": transcription})

    result = asyncio.run(
        _drive(
            handler,
            [
                {
                    "type": "session.update",
                    "event_id": "event_1",
                    "session": {"type": "realtime"},
                }
            ],
        )
    )

    assert result[0]["type"] == "error"
    assert result[0]["error"]["code"] == "unsupported_session"
    assert result[0]["error"]["event_id"] == "event_1"
    assert transcription.events == []


TEXT_MODEL = "test/realtime-llm"


def _text_session(**updates) -> dict:
    session = {
        "type": "realtime",
        "model": TEXT_MODEL,
        "instructions": "Answer clearly.",
        "max_output_tokens": 32,
        "output_modalities": ["text"],
    }
    session.update(updates)
    return session


def _text_item(text: str, *, item_id: str = "user_1") -> dict:
    return {
        "type": "conversation.item.create",
        "item": {
            "id": item_id,
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }


def test_text_turn_finalization_is_idempotent():
    items = []
    messages = []
    turn = _TextTurn(
        messages=[],
        max_output_tokens=32,
        wire_max_output_tokens=32,
        add_to_conversation=True,
        items=items,
        conversation_messages=messages,
    )

    first = turn.final_events(status="completed")
    second = turn.final_events(status="cancelled")

    assert first[-1]["response"]["status"] == "completed"
    assert second == []
    assert len(items) == 1
    assert messages == [{"role": "assistant", "content": ""}]


def test_text_session_streams_canonical_response_and_preserves_usage():
    calls = []

    async def chat_completion(messages, max_output_tokens):
        calls.append((messages, max_output_tokens))

        async def frames():
            yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            yield 'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
            yield 'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}\n\n'
            yield (
                'data: {"choices":[],"usage":{"prompt_tokens":5,'
                '"completion_tokens":2,"total_tokens":7}}\n\n'
            )
            yield "data: [DONE]\n\n"

        return frames()

    handler = RealtimeTextHandler(
        model_name=TEXT_MODEL,
        chat_completion_factory=chat_completion,
    )
    result = asyncio.run(
        _drive(
            handler,
            [
                {"type": "session.update", "session": _text_session()},
                _text_item("Hello"),
                {"type": "response.create"},
            ],
        )
    )

    assert calls == [
        (
            [
                {"role": "system", "content": "Answer clearly."},
                {"role": "user", "content": "Hello"},
            ],
            32,
        )
    ]
    assert [event["type"] for event in result] == [
        "session.updated",
        "conversation.item.added",
        "conversation.item.done",
        "response.created",
        "response.output_item.added",
        "conversation.item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "conversation.item.done",
        "response.done",
    ]
    assert (
        "".join(
            event["delta"]
            for event in result
            if event["type"] == "response.output_text.delta"
        )
        == "hello world"
    )
    response = result[-1]["response"]
    assert response["status"] == "completed"
    assert response["output"][0]["content"] == [
        {"type": "output_text", "text": "hello world"}
    ]
    assert response["usage"] == {
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }


@pytest.mark.parametrize("overflow", [False, True])
def test_text_commit_cancels_warming_before_final_generation(monkeypatch, overflow):
    monkeypatch.setattr(
        "dynamo.vllm.realtime.handler.MAX_TEXT_BUFFER_BYTES", len("Hello world")
    )
    updates_seen = []
    prefill_messages = []
    prefill_done = asyncio.Event()
    prefill_seen = asyncio.Event()

    async def prefill(messages, updates):
        prefill_messages.extend(messages)
        try:
            async for update in updates:
                updates_seen.append(update)
                if update == "Hello world":
                    prefill_seen.set()
        except asyncio.CancelledError:
            prefill_done.set()
            raise

    async def chat_completion(messages, max_output_tokens):
        assert prefill_done.is_set()
        assert messages[-1] == {"role": "user", "content": "Hello world"}
        assert max_output_tokens == 32

        async def frames():
            yield 'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        return frames()

    handler = RealtimeTextHandler(
        model_name=TEXT_MODEL,
        chat_completion_factory=chat_completion,
        text_prefill_factory=prefill,
    )
    result = asyncio.run(
        _drive(
            handler,
            [
                {"type": "session.update", "session": _text_session()},
                {"type": "input_text.append", "text": "Hello"},
                {"type": "input_text.append", "text": " world"},
                *([{"type": "input_text.append", "text": "!"}] if overflow else []),
                {"type": "input_text.commit"},
                {"type": "response.create"},
            ],
            before_commit=prefill_seen,
        )
    )

    assert prefill_messages == [{"role": "system", "content": "Answer clearly."}]
    assert updates_seen == ["Hello world"]
    errors = [event for event in result if event["type"] == "error"]
    assert len(errors) == int(overflow)
    if overflow:
        assert errors[0]["error"]["code"] == "invalid_text"
    user_items = [
        event["item"]
        for event in result
        if event["type"] == "conversation.item.done" and event["item"]["role"] == "user"
    ]
    assert len(user_items) == 1
    assert user_items[0]["content"] == [{"type": "input_text", "text": "Hello world"}]
    done = next(event for event in result if event["type"] == "response.done")
    assert done["response"]["status"] == "completed"


def test_text_buffer_clear_discards_input_and_allows_replay():
    prefill_texts = []
    prefill_seen = asyncio.Event()

    async def prefill(messages, updates):
        del messages
        async for text in updates:
            prefill_texts.append(text)
            if text == "correct":
                prefill_seen.set()

    async def chat_completion(messages, max_output_tokens):
        del max_output_tokens
        assert messages[-1] == {"role": "user", "content": "correct"}

        async def frames():
            yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'

        return frames()

    result = asyncio.run(
        _drive(
            RealtimeTextHandler(
                model_name=TEXT_MODEL,
                chat_completion_factory=chat_completion,
                text_prefill_factory=prefill,
            ),
            [
                {"type": "session.update", "session": _text_session()},
                {"type": "input_text.append", "text": "incorrect"},
                {"type": "input_text.clear"},
                {"type": "input_text.append", "text": "correct"},
                {"type": "input_text.commit"},
                {"type": "response.create"},
            ],
            before_commit=prefill_seen,
        )
    )

    user_items = [
        event["item"]
        for event in result
        if event["type"] == "conversation.item.done" and event["item"]["role"] == "user"
    ]
    assert len(user_items) == 1
    assert user_items[0]["content"][0]["text"] == "correct"
    assert "correct" in prefill_texts
    done = next(event for event in result if event["type"] == "response.done")
    assert done["response"]["status"] == "completed"


def test_text_prefill_failure_does_not_fail_final_generation():
    prefill_failed = asyncio.Event()

    async def failed_prefill(messages, updates):
        del messages
        async for _ in updates:
            prefill_failed.set()
            raise RuntimeError("prefill failed")

    async def chat_completion(messages, max_output_tokens):
        del messages, max_output_tokens

        async def frames():
            yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'

        return frames()

    result = asyncio.run(
        _drive(
            RealtimeTextHandler(
                model_name=TEXT_MODEL,
                chat_completion_factory=chat_completion,
                text_prefill_factory=failed_prefill,
            ),
            [
                {"type": "session.update", "session": _text_session()},
                {"type": "input_text.append", "text": "Hello"},
                {"type": "input_text.commit"},
                {"type": "response.create"},
            ],
            before_commit=prefill_failed,
        )
    )

    done = next(event for event in result if event["type"] == "response.done")
    assert done["response"]["status"] == "completed"


@pytest.mark.parametrize("prefill_fails", [False, True])
def test_text_buffer_coalesces_updates_and_limits_bytes(monkeypatch, prefill_fails):
    monkeypatch.setattr("dynamo.vllm.realtime.handler.MAX_TEXT_BUFFER_BYTES", 8)

    async def scenario():
        seen = asyncio.Event()
        updates_seen = []

        async def factory(messages, updates):
            async for text in updates:
                updates_seen.append(text)
                seen.set()
                if prefill_fails:
                    raise RuntimeError("prefill failed")

        prefill = _TextPrefill(messages=[], factory=factory)
        try:
            prefill.append("ab")
            prefill.append("\u20ac")
            await seen.wait()
            if prefill_fails:
                await asyncio.gather(prefill.task, return_exceptions=True)
            assert updates_seen == ["ab\u20ac"]
            prefill.append("123")
            with pytest.raises(ValueError, match="input text exceeds"):
                prefill.append("4")
            assert prefill.text == "ab\u20ac123"
        finally:
            await prefill.cancel()

    asyncio.run(asyncio.wait_for(scenario(), timeout=1))


def test_text_buffer_must_be_committed_before_response():
    async def unused_chat_completion(messages, max_output_tokens):
        raise AssertionError((messages, max_output_tokens))

    async def prefill(messages, updates):
        del messages
        async for _ in updates:
            pass

    result = asyncio.run(
        _drive(
            RealtimeTextHandler(
                model_name=TEXT_MODEL,
                chat_completion_factory=unused_chat_completion,
                text_prefill_factory=prefill,
            ),
            [
                {"type": "session.update", "session": _text_session()},
                {"type": "input_text.append", "text": "Hello"},
                {"type": "response.create"},
            ],
        )
    )

    error = next(event for event in result if event["type"] == "error")
    assert error["error"]["code"] == "input_not_committed"


@pytest.mark.parametrize(
    "event,message",
    [
        ({"type": "input_text.append", "text": ""}, "non-empty"),
        ({"type": "input_text.append", "text": 123}, "non-empty"),
        ({"type": "input_text.append", "text": "123456789"}, "exceeds"),
        ({"type": "input_text.commit"}, "buffer is empty"),
    ],
)
def test_invalid_text_buffer_events_are_recoverable(monkeypatch, event, message):
    monkeypatch.setattr("dynamo.vllm.realtime.handler.MAX_TEXT_BUFFER_BYTES", 8)

    async def chat_completion(messages, max_output_tokens):
        assert messages == [
            {"role": "system", "content": "Recovered"},
            {"role": "user", "content": "Hi"},
        ]

        async def frames():
            yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'

        return frames()

    async def unused_prefill(messages, updates):
        raise AssertionError((messages, updates))

    result = asyncio.run(
        _drive(
            RealtimeTextHandler(
                model_name=TEXT_MODEL,
                chat_completion_factory=chat_completion,
                text_prefill_factory=unused_prefill,
            ),
            [
                {"type": "session.update", "session": _text_session()},
                event,
                {
                    "type": "session.update",
                    "session": _text_session(instructions="Recovered"),
                },
                _text_item("Hi"),
                {"type": "response.create"},
            ],
        )
    )

    errors = [item for item in result if item["type"] == "error"]
    assert len(errors) == 1
    assert message in errors[0]["error"]["message"]
    assert result[-1]["response"]["status"] == "completed"


@pytest.mark.parametrize(
    "option",
    [
        {"prompt": {"id": "pmpt_required", "variables": {"context": "required"}}},
        {"reasoning": {"effort": "high"}},
    ],
)
def test_response_rejects_unsupported_prompt_and_reasoning(option):
    async def unused_chat_completion(messages, max_output_tokens):
        raise AssertionError((messages, max_output_tokens))

    # Supply a complete response object accepted by the frontend's typed decoder.
    response = {
        "audio": {
            "output": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "voice": "alloy",
            }
        },
        "conversation": "none",
        "input": [],
        "instructions": "Fallback instruction",
        "max_output_tokens": 32,
        "metadata": {},
        "output_modalities": ["text"],
        **option,
    }
    result = asyncio.run(
        _drive(
            RealtimeTextHandler(
                model_name=TEXT_MODEL,
                chat_completion_factory=unused_chat_completion,
            ),
            [
                {"type": "session.update", "session": _text_session()},
                {"type": "response.create", "response": response},
            ],
        )
    )
    assert [event["type"] for event in result] == ["session.updated", "error"]
    assert result[-1]["error"]["code"] == "invalid_response"


@pytest.mark.parametrize(
    "session_update, item, code",
    [
        ({"output_modalities": ["audio"]}, _text_item("Hello"), "invalid_session"),
        ({"truncation": "auto"}, _text_item("Hello"), "invalid_session"),
        (
            {
                "truncation": {
                    "type": "retention_ratio",
                    "retention_ratio": 0.5,
                    "token_limits": {"post_instructions": 1},
                }
            },
            _text_item("Hello"),
            "invalid_session",
        ),
        (
            {},
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "audio": "..."}],
                },
            },
            "invalid_item",
        ),
    ],
)
def test_text_session_rejects_unsupported_options(session_update, item, code):
    async def unused_chat_completion(messages, max_output_tokens):
        raise AssertionError((messages, max_output_tokens))

    result = asyncio.run(
        _drive(
            RealtimeTextHandler(
                model_name=TEXT_MODEL,
                chat_completion_factory=unused_chat_completion,
            ),
            [
                {
                    "type": "session.update",
                    "session": _text_session(**session_update),
                },
                item,
            ],
        )
    )

    errors = [event for event in result if event["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == code
    if code == "invalid_session":
        assert not any(event["type"] == "session.updated" for event in result)


@pytest.mark.parametrize("after_first_delta", [False, True])
def test_response_cancel_aborts_generation(after_first_delta):
    async def scenario():
        started = asyncio.Event()

        async def chat_completion(messages, max_output_tokens):
            del messages, max_output_tokens

            async def frames():
                started.set()
                yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                await asyncio.Event().wait()
                yield "data: [DONE]\n\n"

            return frames()

        handler = RealtimeTextHandler(
            model_name=TEXT_MODEL,
            chat_completion_factory=chat_completion,
        )

        async def request_stream():
            yield {"type": "session.update", "session": _text_session()}
            yield _text_item("Wait")
            yield {"type": "response.create"}
            if after_first_delta:
                await started.wait()
            yield {"type": "response.cancel"}

        return [event async for event in handler.generate(request_stream(), _Context())]

    result = asyncio.run(asyncio.wait_for(scenario(), timeout=1))

    done = [event for event in result if event["type"] == "response.done"]
    assert len(done) == 1
    assert done[0]["response"]["status"] == "cancelled"
    assert done[0]["response"]["status_details"] == {
        "type": "cancelled",
        "reason": "client_cancelled",
    }
    response_events = [
        event["type"] for event in result if event["type"].startswith("response.")
    ]
    assert response_events == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        *(["response.output_text.delta"] if after_first_delta else []),
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.done",
    ]
    assert done[0]["response"]["output"][0]["status"] == "incomplete"
    assert done[0]["response"]["output"][0]["content"] == (
        [{"type": "output_text", "text": "partial"}] if after_first_delta else []
    )


def test_cancellation_under_backpressure_closes_chat_stream():
    async def scenario():
        started = asyncio.Event()
        closed = asyncio.Event()

        async def frames():
            try:
                started.set()
                yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            finally:
                await asyncio.sleep(0)
                closed.set()

        stream = frames()

        async def chat_completion(messages, max_tokens):
            return stream

        handler = RealtimeTextHandler(
            model_name=TEXT_MODEL, chat_completion_factory=chat_completion
        )
        turn = _TextTurn(
            messages=[],
            max_output_tokens=32,
            wire_max_output_tokens=32,
            add_to_conversation=False,
            items=[],
            conversation_messages=[],
        )
        while not turn.events.full():
            turn.events.put_nowait({})
        task = asyncio.create_task(handler._run_turn(turn, _Context()))
        await started.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert closed.is_set()

    asyncio.run(asyncio.wait_for(scenario(), timeout=1))


@pytest.mark.parametrize("close_early", [False, True])
def test_text_serving_closes_nested_engine_stream(monkeypatch, close_early):
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    from dynamo.vllm.realtime.serving import build_realtime_text_factories

    monkeypatch.setattr(OpenAIServingChat, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(
        "vllm.renderers.online_renderer.OnlineRenderer", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "dynamo.vllm.realtime.serving._build_models", lambda **kwargs: None
    )

    async def scenario():
        closed = asyncio.Event()

        async def engine_output():
            try:
                yield "data: [DONE]\n\n"
            finally:
                await asyncio.sleep(0)
                closed.set()

        engine_stream = engine_output()

        async def create(self, request):
            return self.chat_completion_stream_generator(request, engine_stream)

        async def frames(self, request, result_generator):
            async for frame in result_generator:
                yield frame

        monkeypatch.setattr(OpenAIServingChat, "create_chat_completion", create)
        monkeypatch.setattr(
            OpenAIServingChat, "chat_completion_stream_generator", frames
        )
        factory, _ = build_realtime_text_factories(
            engine_client=SimpleNamespace(model_config=None, renderer=None),
            model_name=TEXT_MODEL,
            model_path=TEXT_MODEL,
            chat_template_path=None,
        )
        stream = await factory([{"role": "user", "content": "Hi"}], 32)
        await anext(stream)
        if close_early:
            await stream.aclose()
        else:
            assert [frame async for frame in stream] == []
        assert closed.is_set()

    asyncio.run(asyncio.wait_for(scenario(), timeout=1))


def test_generation_failure_closes_announced_response_item():
    async def failed_chat_completion(messages, max_output_tokens):
        del messages, max_output_tokens
        raise RuntimeError("engine unavailable")

    result = asyncio.run(
        _drive(
            RealtimeTextHandler(
                model_name=TEXT_MODEL,
                chat_completion_factory=failed_chat_completion,
            ),
            [
                {"type": "session.update", "session": _text_session()},
                _text_item("Hello"),
                {"type": "response.create"},
            ],
        )
    )

    response_events = [
        event["type"] for event in result if event["type"].startswith("response.")
    ]
    assert response_events == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.done",
    ]
    assert result[-1]["response"]["status"] == "failed"


def test_text_session_starts_next_turn_immediately_after_response_done():
    async def scenario():
        calls = []

        async def chat_completion(messages, max_output_tokens):
            del max_output_tokens
            calls.append(messages)

            async def frames():
                reply = f"reply {len(calls)}"
                yield f'data: {{"choices":[{{"delta":{{"content":"{reply}"}},"finish_reason":"stop"}}]}}\n\n'
                yield "data: [DONE]\n\n"

            return frames()

        requests = asyncio.Queue()

        async def request_stream():
            while (event := await requests.get()) is not None:
                yield event

        responses = RealtimeTextHandler(
            model_name=TEXT_MODEL,
            chat_completion_factory=chat_completion,
        ).generate(request_stream(), _Context())
        await requests.put({"type": "session.update", "session": _text_session()})
        await requests.put(_text_item("First", item_id="user_1"))
        await requests.put({"type": "response.create"})

        first = []
        while not first or first[-1]["type"] != "response.done":
            first.append(await anext(responses))

        await requests.put(_text_item("Second", item_id="user_2"))
        await requests.put({"type": "response.create"})
        await requests.put(None)
        remaining = [event async for event in responses]

        return calls, first + remaining

    calls, events = asyncio.run(asyncio.wait_for(scenario(), timeout=1))

    assert len([event for event in events if event["type"] == "response.done"]) == 2
    assert calls == [
        [
            {"role": "system", "content": "Answer clearly."},
            {"role": "user", "content": "First"},
        ],
        [
            {"role": "system", "content": "Answer clearly."},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "Second"},
        ],
    ]


def test_transcription_session_streams_canonical_events_and_resamples_audio():
    pcm = np.linspace(-12_000, 12_000, 2_400, dtype=np.int16).tobytes()
    engine = _FakeEngine()
    events = [
        {"type": "session.update", "session": _session()},
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode(),
        },
        {"type": "input_audio_buffer.commit"},
    ]

    result = asyncio.run(_drive(_handler(engine), events))
    event_types = [event["type"] for event in result]

    assert event_types == [
        "session.updated",
        "input_audio_buffer.committed",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.completed",
    ]
    committed = result[1]
    assert all(event.get("item_id") == committed["item_id"] for event in result[2:])
    assert "".join(event["delta"] for event in result[2:4]) == "hello world"
    assert result[-1]["transcript"] == "hello world"
    assert result[-1]["usage"] == {
        "type": "tokens",
        "input_tokens": 3,
        "output_tokens": 3,
        "total_tokens": 6,
        "input_token_details": {"audio_tokens": 3, "text_tokens": 0},
    }
    assert len(engine.audio) == 1
    assert abs(engine.audio[0].size - 1_600) <= 1


@pytest.mark.parametrize(
    "event, message",
    [
        ({"type": "input_audio_buffer.append", "audio": "not base64"}, "valid base64"),
        ({"type": "input_audio_buffer.append", "audio": 123}, "base64 string"),
        ({"type": "input_audio_buffer.commit"}, "buffer is empty"),
    ],
)
def test_invalid_audio_events_return_recoverable_errors(event, message):
    result = asyncio.run(
        _drive(
            _handler(_FakeEngine()),
            [{"type": "session.update", "session": _session()}, event],
        )
    )

    errors = [item for item in result if item["type"] == "error"]
    assert len(errors) == 1
    assert message in errors[0]["error"]["message"]


def test_invalid_audio_format_returns_recoverable_session_error():
    session = _session()
    session["audio"]["input"]["format"] = "pcm16"

    result = asyncio.run(
        _drive(
            _handler(_FakeEngine()),
            [{"type": "session.update", "session": session}],
        )
    )

    assert [event["type"] for event in result] == ["error"]
    assert result[0]["error"]["code"] == "invalid_session"
    assert "format must be an object" in result[0]["error"]["message"]


@pytest.mark.parametrize(
    "audio_input_update, transcription_update, message",
    [
        ({"format": {"type": "audio/pcm", "rate": 16_000}}, {}, "24000 Hz"),
        ({}, {"language": "fr"}, "only English"),
        ({}, {"prompt": "Dynamo vocabulary"}, "prompts are not supported"),
        ({"noise_reduction": {"type": "near_field"}}, {}, "not supported"),
    ],
)
def test_unsupported_session_options_are_rejected(
    audio_input_update, transcription_update, message
):
    session = _session()
    audio_input = session["audio"]["input"]
    audio_input.update(audio_input_update)
    audio_input["transcription"].update(transcription_update)

    result = asyncio.run(
        _drive(
            _handler(_FakeEngine()),
            [{"type": "session.update", "session": session}],
        )
    )

    assert [event["type"] for event in result] == ["error"]
    assert message in result[0]["error"]["message"]


def test_server_vad_is_rejected_until_worker_supports_it():
    server_vad = {
        "type": "server_vad",
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
        "threshold": 0.5,
    }
    result = asyncio.run(
        _drive(
            _handler(_FakeEngine()),
            [
                {
                    "type": "session.update",
                    "session": _session(turn_detection=server_vad),
                }
            ],
        )
    )

    assert [event["type"] for event in result] == ["error"]
    assert "server turn detection is not supported" in result[0]["error"]["message"]


def test_clear_cancels_uncommitted_turn_and_allows_next_utterance():
    pcm = base64.b64encode(np.ones(480, dtype=np.int16).tobytes()).decode()
    engine = _FakeEngine()
    result = asyncio.run(
        _drive(
            _handler(engine),
            [
                {"type": "session.update", "session": _session()},
                {"type": "input_audio_buffer.append", "audio": pcm},
                {"type": "input_audio_buffer.clear"},
                {"type": "input_audio_buffer.append", "audio": pcm},
                {"type": "input_audio_buffer.commit"},
            ],
        )
    )

    assert "input_audio_buffer.cleared" in [event["type"] for event in result]
    completed = [
        event
        for event in result
        if event["type"] == "conversation.item.input_audio_transcription.completed"
    ]
    assert len(completed) == 1


def test_next_turn_is_pumped_while_previous_turn_uses_engine_slot():
    class _BlockingFirstEngine:
        def __init__(self, release: asyncio.Event) -> None:
            self.release = release
            self.started = 0

        async def generate(self, *, prompt, sampling_params, request_id):
            del sampling_params, request_id
            async for _ in prompt:
                pass
            self.started += 1
            if self.started == 1:
                await self.release.wait()
            yield SimpleNamespace(
                prompt_token_ids=[1],
                outputs=[SimpleNamespace(text="ok", token_ids=[2])],
            )

    async def scenario():
        release = asyncio.Event()
        engine = _BlockingFirstEngine(release)
        handler = _handler(engine)
        pcm = base64.b64encode(np.ones(480, dtype=np.int16).tobytes()).decode()

        async def request_stream():
            yield {"type": "session.update", "session": _session()}
            for _ in range(2):
                yield {"type": "input_audio_buffer.append", "audio": pcm}
                yield {"type": "input_audio_buffer.commit"}
            release.set()

        result = await asyncio.wait_for(
            _collect(handler.generate(request_stream(), _Context())), timeout=1
        )
        return result, engine

    async def _collect(stream):
        return [event async for event in stream]

    result, engine = asyncio.run(scenario())

    completed = [
        event
        for event in result
        if event["type"] == "conversation.item.input_audio_transcription.completed"
    ]
    assert engine.started == 2
    assert len(completed) == 2
