# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import numpy as np
import pytest

from dynamo.vllm.realtime import RealtimeTranscriptionHandler

try:
    from dynamo.vllm.realtime import qwen3_asr as qwen
except ModuleNotFoundError as exc:
    if exc.name not in ("vllm", "vllm.config.speech_to_text"):
        raise
    pytest.skip("Requires vLLM's speech-to-text API", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

SAMPLE_RATE = 16_000
CHUNK = 2 * SAMPLE_RATE
MODEL = "test/qwen3-asr"


class _Tokenizer:
    """Byte tokens make split Unicode boundaries deterministic in these tests."""

    def encode(self, text, *, add_special_tokens=False):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode(errors="replace")


class _Model:
    @staticmethod
    def get_speech_to_text_config(model_config, task_type):
        return SimpleNamespace(sample_rate=SAMPLE_RATE)

    @staticmethod
    def get_generation_prompt(params):
        assert params.language == "en"
        return {
            "prompt_token_ids": [0],
            "multi_modal_data": {"audio": params.audio},
        }


class _Renderer:
    async def render_cmpl_async(self, prompts):
        return prompts


class _Engine:
    def __init__(self, hypotheses):
        self.hypotheses = iter(hypotheses)
        self.calls = []

    async def generate(self, *, prompt, sampling_params, request_id):
        prefix = _Tokenizer().decode(prompt["prompt_token_ids"][1:])
        self.calls.append(
            (len(prompt["multi_modal_data"]["audio"]), prefix, request_id)
        )
        hypothesis = next(self.hypotheses)
        assert hypothesis.startswith(prefix)
        yield SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            outputs=[
                SimpleNamespace(
                    text=hypothesis[len(prefix) :],
                    token_ids=[10, 11],
                    finish_reason="stop",
                )
            ],
        )


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(qwen, "cached_tokenizer_from_config", lambda _: _Tokenizer())
    monkeypatch.setattr(qwen, "parse_model_prompt", lambda config, prompt: prompt)

    def make(engine):
        return qwen.Qwen3ASRTranscriber(
            engine_client=engine,
            serving=SimpleNamespace(
                model_cls=_Model,
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        thinker_config=SimpleNamespace(audio_token_id=2)
                    )
                ),
                renderer=_Renderer(),
            ),
        )

    return make


async def _audio(*sizes):
    for size in sizes:
        yield np.zeros(size, dtype=np.float32)


async def _collect(stream):
    return [item async for item in stream]


@pytest.mark.parametrize("sizes", [(CHUNK * 3 + 7,), (7, CHUNK - 7, CHUNK, CHUNK, 7)])
def test_audio_chunk_boundaries_do_not_depend_on_client_frames(sizes):
    prefixes = asyncio.run(_collect(qwen._audio_prefixes(_audio(*sizes), CHUNK)))
    assert [(len(audio), final) for audio, final in prefixes] == [
        (CHUNK, False),
        (CHUNK * 2, False),
        (CHUNK * 3, False),
        (CHUNK * 3 + 7, True),
    ]


def test_revises_initial_hypothesis_then_forces_emitted_prefix(adapter):
    engine = _Engine(
        [
            "The wrong hypothesis",
            "The quick brown fox",
            "The quick brown fox jumps",
            "The quick brown fox jumps.",
        ]
    )
    deltas = asyncio.run(_collect(adapter(engine)(_audio(CHUNK * 3 + 7), "turn")))
    assert [item.text for item in deltas] == [
        "",
        "The quick brow",
        "n fox ",
        "jumps.",
    ]
    assert "".join(item.text for item in deltas) == "The quick brown fox jumps."
    assert engine.calls == [
        (CHUNK, "", "turn_0"),
        (CHUNK * 2, "", "turn_1"),
        (CHUNK * 3, "The quick brow", "turn_2"),
        (CHUNK * 3 + 7, "The quick brown fox ", "turn_3"),
    ]
    assert sum(item.input_tokens for item in deltas) == 12
    assert sum(item.output_tokens for item in deltas) == 8


@pytest.mark.parametrize("samples", [13, CHUNK, CHUNK * 2])
def test_commit_flushes_short_audio_and_exact_boundaries_without_redecoding(
    adapter, samples
):
    engine = _Engine(["hello world"] * 2)
    deltas = asyncio.run(_collect(adapter(engine)(_audio(samples), "turn")))
    assert "".join(item.text for item in deltas) == "hello world"
    assert len(engine.calls) == max(1, samples // CHUNK)
    assert deltas[-1].text


def test_short_continuation_does_not_move_prefix_behind_emitted_text(adapter):
    engine = _Engine(["hello world", "hello world", "hello ", "hello again"])
    deltas = asyncio.run(_collect(adapter(engine)(_audio(CHUNK * 3 + 1), "turn")))
    assert [item.text for item in deltas] == ["", "hello ", "", "again"]
    assert engine.calls[-1][1] == "hello "


def test_withheld_tail_does_not_emit_partial_unicode():
    # Five withheld byte tokens would cut between the two bytes of é.
    assert qwen._stable_prefix(_Tokenizer(), "café1234", "") == "caf"


def test_empty_audio_does_not_call_engine(adapter):
    engine = _Engine([])
    assert asyncio.run(_collect(adapter(engine)(_audio(), "turn"))) == []
    assert engine.calls == []


def test_silence_can_finish_without_text(adapter):
    engine = _Engine(["", ""])
    deltas = asyncio.run(_collect(adapter(engine)(_audio(CHUNK * 2), "turn")))
    assert "".join(item.text for item in deltas) == ""
    assert sum(item.input_tokens for item in deltas) == 6


def test_same_adapter_keeps_turn_state_separate(adapter):
    engine = _Engine(["first", "second"])
    transcribe = adapter(engine)
    first = asyncio.run(_collect(transcribe(_audio(13), "one")))
    second = asyncio.run(_collect(transcribe(_audio(13), "two")))
    assert "".join(item.text for item in first) == "first"
    assert "".join(item.text for item in second) == "second"
    assert engine.calls == [(13, "", "one_0"), (13, "", "two_0")]


def test_cancellation_closes_inflight_engine_generator(adapter):
    async def scenario():
        started = asyncio.Event()
        closed = asyncio.Event()

        class _BlockedEngine:
            async def generate(self, **kwargs):
                try:
                    started.set()
                    await asyncio.Event().wait()
                    yield  # pragma: no cover -- waits until cancellation
                finally:
                    closed.set()

        transcribe = adapter(_BlockedEngine())(_audio(CHUNK), "turn")
        task = asyncio.create_task(anext(transcribe))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        await transcribe.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["no_output", "token_limit", "engine_error"])
def test_decode_failure_does_not_return_a_successful_transcript(adapter, failure):
    class _FailingEngine:
        async def generate(self, **kwargs):
            if failure == "engine_error":
                raise RuntimeError("engine failed")
            if failure == "token_limit":
                yield SimpleNamespace(outputs=[SimpleNamespace(finish_reason="length")])

    with pytest.raises(RuntimeError):
        asyncio.run(_collect(adapter(_FailingEngine())(_audio(13), "turn")))


def test_handler_emits_delta_before_commit_and_final_matches_deltas(adapter):
    async def scenario():
        engine = _Engine(["wrong early guess", "hello wonderful world"])
        handler = RealtimeTranscriptionHandler(
            model_name=MODEL,
            model_sample_rate=SAMPLE_RATE,
            transcribe=adapter(engine),
        )
        partial_received = asyncio.Event()
        committed = False
        pcm = base64.b64encode(np.zeros(48_000, dtype=np.int16).tobytes()).decode()

        async def requests():
            nonlocal committed
            yield {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24_000},
                            "transcription": {"model": MODEL, "language": "en"},
                        }
                    },
                },
            }
            for _ in range(2):
                yield {"type": "input_audio_buffer.append", "audio": pcm}
            # A buffered-only implementation deadlocks here and fails the test.
            await asyncio.wait_for(partial_received.wait(), 1)
            committed = True
            yield {"type": "input_audio_buffer.commit"}

        events = []
        context = SimpleNamespace(is_stopped=lambda: False)
        async for event in handler.generate(requests(), context):
            events.append(event)
            if event["type"] == "conversation.item.input_audio_transcription.delta":
                if not partial_received.is_set():
                    assert not committed
                    partial_received.set()
        return events

    events = asyncio.run(scenario())
    delta_type = "conversation.item.input_audio_transcription.delta"
    assert next(event for event in events if event["type"] == delta_type)["delta"]
    final = events[-1]
    assert final["type"] == "conversation.item.input_audio_transcription.completed"
    assert final["transcript"] == "hello wonderful world"
    assert (
        "".join(event["delta"] for event in events if event["type"] == delta_type)
        == final["transcript"]
    )
    assert final["usage"]["input_tokens"] == 6
    assert final["usage"]["output_tokens"] == 4
    assert final["usage"]["input_token_details"] == {
        "audio_tokens": 2,
        "text_tokens": 4,
    }


def test_from_engine_rejects_models_without_a_realtime_adapter(monkeypatch):
    serving = SimpleNamespace(
        model_cls=type("UnsupportedModel", (), {}),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="other")),
    )
    monkeypatch.setattr(
        "dynamo.vllm.realtime.handler.build_realtime_serving", lambda **_: serving
    )
    with pytest.raises(ValueError, match="does not support realtime transcription"):
        RealtimeTranscriptionHandler.from_engine(
            engine_client=object(), model_name="test/unsupported", model_path="unused"
        )
