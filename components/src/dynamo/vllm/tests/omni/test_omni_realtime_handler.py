# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the realtime Omni handler's event translation.

These exercise RealtimeOmniHandler in isolation (no frontend, no vLLM): a fake
engine yields OmniRequestOutput-shaped frames and we assert the OpenAI-spec
server-event sequence the handler emits, including PCM16 round-tripping.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import numpy as np
import pytest

try:
    # Importing the omni package pulls omni_handler -> vllm_omni; the handler
    # logic itself is vllm-free, but the package import is not.
    from dynamo.vllm.omni.realtime_handler import RealtimeOmniHandler
except Exception:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

MODEL_NAME = "omni-realtime-unit"


class _FakeContext:
    """Minimal Context stand-in; the handler only calls is_stopped()."""

    def __init__(self, stopped: bool = False) -> None:
        self._stopped = stopped

    def is_stopped(self) -> bool:
        return self._stopped


def _audio_output(samples: np.ndarray, sample_rate: int = 16000):
    return SimpleNamespace(
        stage_id=1,
        outputs=[],
        multimodal_output={"audio": samples, "sr": sample_rate},
    )


def _text_output(text: str):
    return SimpleNamespace(
        stage_id=0,
        outputs=[SimpleNamespace(text=text, token_ids=[1, 2, 3])],
        prompt_token_ids=[0],
        multimodal_output={},
    )


class _FakeEngine:
    """Echoes appended audio back as two output chunks, preceded by a text delta.

    Consuming the streaming input generator proves the audio-in plumbing; the
    canned text and split audio exercise transcript + multi-delta translation.
    """

    def __init__(self, text: str = "hello") -> None:
        self.text = text
        self.seen_chunks: list = []
        self.seen_output_modalities: list = []

    async def generate(
        self, *, prompt, request_id, sampling_params_list=None, output_modalities=None
    ):
        self.seen_output_modalities.append(output_modalities)
        async for chunk in prompt:
            self.seen_chunks.append(chunk)
        full = (
            np.concatenate(self.seen_chunks)
            if self.seen_chunks
            else np.array([], np.float32)
        )
        yield _text_output(self.text)
        half = len(full) // 2
        yield _audio_output(full[:half])
        yield _audio_output(full[half:])


async def _passthrough_factory(audio_stream, input_stream):
    """Stand-in for transcribe_realtime: yield raw float32 chunks unchanged."""
    async for waveform in audio_stream:
        yield waveform


def _make_handler(engine, **kwargs):
    return RealtimeOmniHandler(
        engine_client=engine,
        model_name=MODEL_NAME,
        streaming_input_factory=_passthrough_factory,
        **kwargs,
    )


async def _drive(handler, events, context):
    async def request_stream():
        for ev in events:
            yield ev

    return [event async for event in handler.generate(request_stream(), context)]


def test_full_turn_event_sequence():
    # Input PCM16 chunk: a short ramp, base64-encoded like the wire format.
    pcm16 = np.linspace(-8000, 8000, 64, dtype=np.int16).tobytes()
    audio_b64 = base64.b64encode(pcm16).decode("utf-8")

    engine = _FakeEngine(text="hi there")
    handler = _make_handler(engine)

    events = [
        {
            "type": "session.update",
            "session": {"type": "realtime", "model": MODEL_NAME},
        },
        {"type": "input_audio_buffer.append", "audio": audio_b64},
        {"type": "input_audio_buffer.commit"},
    ]

    out = asyncio.run(_drive(handler, events, _FakeContext()))
    types = [e["type"] for e in out]

    assert types[0] == "session.updated"
    assert out[0]["session"]["model"] == MODEL_NAME
    assert "response.created" in types
    assert "response.output_audio.delta" in types
    assert "response.output_audio.done" in types
    assert types[-1] == "response.done"

    # created precedes audio precedes done precedes response.done.
    assert types.index("response.created") < types.index("response.output_audio.delta")
    assert types.index("response.output_audio.delta") < types.index(
        "response.output_audio.done"
    )
    assert types.index("response.output_audio.done") < types.index("response.done")

    # response ids are consistent across the turn's frames.
    created = next(e for e in out if e["type"] == "response.created")
    response_id = created["response"]["id"]
    for e in out:
        if e["type"] in ("response.output_audio.delta", "response.output_audio.done"):
            assert e["response_id"] == response_id
        if e["type"] == "response.done":
            assert e["response"]["id"] == response_id
            assert e["response"]["status"] == "completed"

    # Transcript delta carries the thinker text.
    transcripts = [
        e["delta"] for e in out if e["type"] == "response.output_audio_transcript.delta"
    ]
    assert "".join(transcripts) == "hi there"

    # Concatenated audio deltas decode back to the input PCM16 (echo round-trip).
    deltas = b"".join(
        base64.b64decode(e["delta"])
        for e in out
        if e["type"] == "response.output_audio.delta"
    )
    in_f32 = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    out_f32 = np.frombuffer(deltas, dtype=np.int16).astype(np.float32) / 32767.0
    assert out_f32.shape == in_f32.shape
    assert np.allclose(out_f32, in_f32, atol=2e-4)


def test_unknown_client_events_are_ignored():
    engine = _FakeEngine()
    handler = _make_handler(engine, emit_transcript=False)
    events = [
        {"type": "session.update", "session": {"model": MODEL_NAME}},
        {"type": "conversation.item.create", "item": {}},
        {"type": "response.cancel"},
    ]
    out = asyncio.run(_drive(handler, events, _FakeContext()))
    # Only the session.updated echo; no turn started, no error frame.
    assert [e["type"] for e in out] == ["session.updated"]


def test_stopped_context_emits_no_turn():
    engine = _FakeEngine()
    handler = _make_handler(engine)
    events = [
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(b"\x00\x00").decode(),
        },
        {"type": "input_audio_buffer.commit"},
    ]
    out = asyncio.run(_drive(handler, events, _FakeContext(stopped=True)))
    assert out == []


@pytest.mark.parametrize(
    "session, expected",
    [
        ({"model": MODEL_NAME, "output_modalities": ["audio"]}, ["audio"]),
        ({"model": MODEL_NAME}, None),  # unset -> engine's launch default (None)
    ],
)
def test_session_output_modalities_forwarded_to_engine(session, expected):
    audio_b64 = base64.b64encode(
        np.linspace(-8000, 8000, 16, dtype=np.int16).tobytes()
    ).decode()
    engine = _FakeEngine()
    handler = _make_handler(engine)
    events = [
        {"type": "session.update", "session": session},
        {"type": "input_audio_buffer.append", "audio": audio_b64},
        {"type": "input_audio_buffer.commit"},
    ]
    asyncio.run(_drive(handler, events, _FakeContext()))
    assert engine.seen_output_modalities == [expected]


def test_turn_responses_forwarded_in_order():
    # Two commits => two turns. Turns run concurrently, but their responses are
    # forwarded in turn order: distinct response ids, and turn 2's events never
    # interleave with turn 1's (turn 1 fully completes before turn 2 emits).
    audio_b64 = base64.b64encode(
        np.linspace(-8000, 8000, 16, dtype=np.int16).tobytes()
    ).decode()
    handler = _make_handler(_FakeEngine())
    events = [
        {"type": "session.update", "session": {"model": MODEL_NAME}},
        {"type": "input_audio_buffer.append", "audio": audio_b64},
        {"type": "input_audio_buffer.commit"},
        {"type": "input_audio_buffer.append", "audio": audio_b64},
        {"type": "input_audio_buffer.commit"},
    ]
    out = asyncio.run(_drive(handler, events, _FakeContext()))

    created = [e["response"]["id"] for e in out if e["type"] == "response.created"]
    done = [e["response"]["id"] for e in out if e["type"] == "response.done"]
    assert len(created) == 2 and len(set(created)) == 2, created  # two distinct turns
    assert created == done, (created, done)  # same ids, completed in start order

    def rid(e: dict) -> str | None:
        return e.get("response_id") or e.get("response", {}).get("id")

    # Turn 1 fully completes (its response.done) before turn 2 starts, and no
    # turn-2-tagged event appears before that point.
    first_done = next(
        i
        for i, e in enumerate(out)
        if e["type"] == "response.done" and e["response"]["id"] == created[0]
    )
    second_created = next(
        i
        for i, e in enumerate(out)
        if e["type"] == "response.created" and e["response"]["id"] == created[1]
    )
    assert first_done < second_created
    assert all(rid(e) != created[1] for e in out[: first_done + 1])


def test_turns_run_concurrently():
    # Both turns' engine drives must be in flight at once: a barrier engine
    # blocks each generate() until two have started. This only completes if
    # turns are not serialized -- a one-at-a-time model would deadlock the
    # barrier and time out. Responses are still forwarded in turn order.
    class _BarrierEngine:
        def __init__(self, n: int) -> None:
            self.n = n
            self.started = 0
            self.gate = asyncio.Event()

        async def generate(
            self,
            *,
            prompt,
            request_id,
            sampling_params_list=None,
            output_modalities=None,
        ):
            async for _ in prompt:  # drain this turn's audio
                pass
            self.started += 1
            if self.started >= self.n:
                self.gate.set()
            await asyncio.wait_for(self.gate.wait(), timeout=5)
            yield _text_output("ok")
            yield _audio_output(np.zeros(4, dtype=np.float32))

    audio_b64 = base64.b64encode(
        np.linspace(-8000, 8000, 16, dtype=np.int16).tobytes()
    ).decode()
    engine = _BarrierEngine(2)
    handler = _make_handler(engine)
    events = [
        {"type": "session.update", "session": {"model": MODEL_NAME}},
        {"type": "input_audio_buffer.append", "audio": audio_b64},
        {"type": "input_audio_buffer.commit"},
        {"type": "input_audio_buffer.append", "audio": audio_b64},
        {"type": "input_audio_buffer.commit"},
    ]
    out = asyncio.run(_drive(handler, events, _FakeContext()))

    # Both drives passed the barrier => they were concurrently in flight.
    assert engine.started == 2
    assert not any(e["type"] == "error" for e in out)
    created = [e["response"]["id"] for e in out if e["type"] == "response.created"]
    done = [e["response"]["id"] for e in out if e["type"] == "response.done"]
    assert len(set(created)) == 2  # two distinct turns, both completed
    assert created == done  # forwarded in turn order, non-interleaved


def test_engine_failure_emits_error_and_failed_response_done():
    # When generation throws, the turn surfaces a top-level `error` event (human
    # readable, but no response_id) AND a terminal response.done(status=failed)
    # so the in-progress response closes and is correlatable by response id.
    class _FailingEngine:
        async def generate(
            self,
            *,
            prompt,
            request_id,
            sampling_params_list=None,
            output_modalities=None,
        ):
            async for _ in prompt:  # drain this turn's audio
                pass
            yield _text_output("partial")
            raise RuntimeError("boom")

    audio_b64 = base64.b64encode(
        np.linspace(-8000, 8000, 16, dtype=np.int16).tobytes()
    ).decode()
    handler = _make_handler(_FailingEngine())
    events = [
        {"type": "session.update", "session": {"model": MODEL_NAME}},
        {"type": "input_audio_buffer.append", "audio": audio_b64},
        {"type": "input_audio_buffer.commit"},
    ]
    out = asyncio.run(_drive(handler, events, _FakeContext()))
    types = [e["type"] for e in out]

    created = next(e for e in out if e["type"] == "response.created")
    response_id = created["response"]["id"]

    error = next((e for e in out if e["type"] == "error"), None)
    assert error is not None
    assert error["error"]["code"] == "omni_generation_error"

    done = next((e for e in out if e["type"] == "response.done"), None)
    assert done is not None
    assert done["response"]["id"] == response_id  # correlatable, unlike `error`
    assert done["response"]["status"] == "failed"
    assert done["response"]["status_details"]["type"] == "failed"
    assert (
        done["response"]["status_details"]["error"]["code"] == "omni_generation_error"
    )
    assert types.index("error") < types.index("response.done")
