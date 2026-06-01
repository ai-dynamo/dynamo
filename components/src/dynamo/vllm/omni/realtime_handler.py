# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realtime (bidirectional) handler backed by vLLM-Omni's streaming engine.

The Dynamo frontend owns the ``/v1/realtime`` WebSocket and the OpenAI Realtime
envelope: it synthesizes ``session.created``, picks this engine when a
``session.update`` names our model, then forwards every client event (including
that ``session.update``) into ``request_stream`` and streams whatever we yield
straight back to the client as typed ``RealtimeServerEvent`` frames. So this
handler is a pure translation layer between the OpenAI Realtime event contract
and vLLM-Omni's in-process streaming API.

Turn model (matching vLLM-Omni's own realtime client contract):

  * ``session.update``            -> ``session.updated`` echoing the session.
  * ``input_audio_buffer.append`` -> base64 PCM16 chunk decoded to a float32
    waveform and queued for the active turn's audio stream.
  * ``input_audio_buffer.commit`` -> opens a turn on the first event and, when
    ``final`` is set, closes the audio stream so the engine drains.

Each turn emits ``response.created`` -> ``response.output_audio.delta``* (+
optional ``response.output_audio_transcript.delta`` for the thinker text) ->
``response.output_audio.done`` -> ``response.done``. These are the OpenAI-spec
event names the frontend's typed reader requires, which differ from
vLLM-Omni's own ``response.audio.delta`` / ``transcription.delta`` names; the
PCM16 and cumulative-vs-delta waveform handling below is ported from
vLLM-Omni's ``realtime_connection.py`` and only the event tags change.

Engine drive (the real-model path, see ``StreamingInput`` in
``vllm.engine.protocol``): audio is not handed to the engine as raw chunks.
Instead a ``streaming_input_factory`` (``OpenAIServingRealtime.transcribe_realtime``
in production) turns the float32 ``audio_stream`` into an async generator of
``StreamingInput`` prompts via the model's ``buffer_realtime_audio`` cumulative
buffering; ``engine.generate(prompt=streaming_input_gen, ...)`` consumes it, and
the thinker's per-step token ids are fed back into ``input_stream`` for the
talker (autoregressive multi-turn). The factory and engine are injected so the
worker passes the real serving/AsyncOmni while tests pass lightweight fakes.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional, Sequence

import numpy as np

from dynamo._core import Context

logger = logging.getLogger(__name__)

# ``streaming_input_factory(audio_stream, input_stream) -> AsyncGenerator`` —
# mirrors ``OpenAIServingRealtime.transcribe_realtime``: it consumes float32
# audio chunks and an ``asyncio.Queue`` of context token ids, yielding engine
# ``StreamingInput`` prompts.
StreamingInputFactory = Callable[
    [AsyncGenerator[np.ndarray, None], "asyncio.Queue[list[int]]"],
    AsyncGenerator[Any, None],
]


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def _response_payload(response_id: str, status: str) -> dict:
    """Minimal ``RealtimeResponse`` payload accepted by the frontend's reader.

    The typed reader requires ``id``, ``max_output_tokens``, ``object``,
    ``output``, ``output_modalities``, and ``status``; they are minted verbatim
    here, mirroring the realtime echo worker.
    """
    return {
        "id": response_id,
        "max_output_tokens": "inf",
        "object": "realtime.response",
        "output": [],
        "output_modalities": ["audio"],
        "status": status,
    }


class _Turn:
    """State for a single in-flight response turn.

    Audio chunks arrive on ``audio_queue`` as float32 waveforms while the client
    appends them; ``None`` is the end-of-input sentinel pushed on the final
    commit (or when the client stream ends). ``audio_ref`` tracks the last
    emitted waveform so cumulative engine outputs are de-duplicated into true
    deltas.
    """

    def __init__(self) -> None:
        self.response_id = f"resp_{uuid.uuid4().hex}"
        self.item_id = f"item_{uuid.uuid4().hex}"
        self.audio_queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue()
        self.audio_ref: np.ndarray | None = None
        self.task: asyncio.Task | None = None


class RealtimeOmniHandler:
    """Bridge OpenAI Realtime client events to vLLM-Omni streaming generation."""

    def __init__(
        self,
        *,
        engine_client: Any,
        model_name: str,
        streaming_input_factory: StreamingInputFactory,
        default_sampling_params_list: Optional[Sequence[Any]] = None,
        emit_transcript: bool = True,
    ) -> None:
        self.engine_client = engine_client
        self.model_name = model_name
        self._streaming_input_factory = streaming_input_factory
        self._default_sampling_params_list = default_sampling_params_list
        self._emit_transcript = emit_transcript

    async def generate(
        self, request_stream: AsyncGenerator[Any, None], context: Context
    ) -> AsyncGenerator[dict, None]:
        """Serve one realtime connection.

        Inbound client events and the per-turn engine drives both feed server
        events onto a shared queue; this coroutine yields them in arrival order
        until the input stream ends and every in-flight turn has drained.
        """
        out_queue: asyncio.Queue[Optional[dict]] = asyncio.Queue()
        active_turn: _Turn | None = None
        turns: list[_Turn] = []

        async def emit(event: dict) -> None:
            await out_queue.put(event)

        def ensure_turn() -> _Turn:
            nonlocal active_turn
            if active_turn is None:
                active_turn = _Turn()
                turns.append(active_turn)
                active_turn.task = asyncio.create_task(
                    self._run_turn(active_turn, context, emit)
                )
            return active_turn

        async def pump() -> None:
            nonlocal active_turn
            try:
                async for client_event in request_stream:
                    if context.is_stopped():
                        break
                    etype = (
                        client_event.get("type")
                        if isinstance(client_event, dict)
                        else None
                    )

                    if etype == "session.update":
                        await emit(
                            {
                                "type": "session.updated",
                                "event_id": _event_id(),
                                "session": client_event.get("session"),
                            }
                        )
                    elif etype == "input_audio_buffer.append":
                        turn = ensure_turn()
                        waveform = _decode_pcm16(client_event.get("audio", ""))
                        if waveform is not None:
                            turn.audio_queue.put_nowait(waveform)
                    elif etype == "input_audio_buffer.commit":
                        turn = ensure_turn()
                        # `final` absent defaults to True: a bare commit means
                        # "buffer complete, generate". A non-final commit opens
                        # the turn (engine starts) but keeps the input open.
                        if client_event.get("final", True):
                            turn.audio_queue.put_nowait(None)
                            active_turn = None
                    elif etype == "input_audio_buffer.clear":
                        # Drop anything buffered but not yet consumed; best effort.
                        if active_turn is not None:
                            _drain_queue(active_turn.audio_queue)
                    else:
                        # The frontend forwards every client event; ones we don't
                        # drive (conversation.item.*, response.*, etc.) are logged
                        # and ignored so a well-behaved session is not torn down.
                        logger.debug("realtime omni: ignoring client event %s", etype)

                # Input stream ended: close any still-open turn so the engine
                # sees end-of-input rather than hanging.
                if active_turn is not None:
                    active_turn.audio_queue.put_nowait(None)
                    active_turn = None
                for turn in turns:
                    if turn.task is not None:
                        await turn.task
            finally:
                await out_queue.put(None)

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                event = await out_queue.get()
                if event is None:
                    break
                yield event
        finally:
            pump_task.cancel()
            for turn in turns:
                if turn.task is not None:
                    turn.task.cancel()

    async def _run_turn(
        self,
        turn: _Turn,
        context: Context,
        emit: Callable[[dict], Awaitable[None]],
    ) -> None:
        """Drive one engine generation turn and translate its outputs."""
        await emit(
            {
                "type": "response.created",
                "event_id": _event_id(),
                "response": _response_payload(turn.response_id, "in_progress"),
            }
        )

        sent_audio = False
        try:
            async for output in self._drive_engine(turn):
                if context.is_stopped():
                    break

                transcript = self._extract_transcript(output)
                if transcript and self._emit_transcript:
                    await emit(self._transcript_delta_event(turn, transcript))

                for chunk in self._extract_audio_chunks(turn, output):
                    sent_audio = True
                    await emit(self._audio_delta_event(turn, chunk))

            if sent_audio:
                await emit(self._audio_done_event(turn))
            await emit(
                {
                    "type": "response.done",
                    "event_id": _event_id(),
                    "response": _response_payload(turn.response_id, "completed"),
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface engine errors on the wire
            logger.exception("realtime omni turn failed: %s", exc)
            await emit(
                {
                    "type": "error",
                    "event_id": _event_id(),
                    "error": {
                        "type": "server_error",
                        "code": "omni_generation_error",
                        "message": str(exc),
                    },
                }
            )

    async def _drive_engine(self, turn: _Turn) -> AsyncGenerator[Any, None]:
        """Feed buffered audio into the engine and yield its stage outputs.

        The float32 ``audio_stream`` (ending on the ``None`` sentinel) and an
        ``input_stream`` token queue are handed to the streaming-input factory
        (``transcribe_realtime``), whose ``StreamingInput`` generator drives
        ``AsyncOmni.generate``. The thinker's (stage 0) per-step token ids are
        fed back into ``input_stream`` for the talker, matching vLLM-Omni's own
        realtime connection.
        """

        async def audio_stream() -> AsyncGenerator[np.ndarray, None]:
            while True:
                waveform = await turn.audio_queue.get()
                if waveform is None:
                    return
                yield waveform

        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        streaming_input_gen = self._streaming_input_factory(
            audio_stream(), input_stream
        )

        generate_kwargs: dict[str, Any] = {
            "prompt": streaming_input_gen,
            "request_id": turn.response_id,
        }
        if self._default_sampling_params_list is not None:
            generate_kwargs["sampling_params_list"] = list(
                self._default_sampling_params_list
            )

        async for output in self.engine_client.generate(**generate_kwargs):
            token_ids = self._thinker_token_ids(output)
            if token_ids:
                input_stream.put_nowait(token_ids)
            yield output

    # -- output translation (ported from vllm-omni realtime_connection.py) ----

    def _audio_delta_event(self, turn: _Turn, chunk: np.ndarray) -> dict:
        return {
            "type": "response.output_audio.delta",
            "event_id": _event_id(),
            "response_id": turn.response_id,
            "item_id": turn.item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": _pcm16_b64(chunk),
        }

    def _audio_done_event(self, turn: _Turn) -> dict:
        return {
            "type": "response.output_audio.done",
            "event_id": _event_id(),
            "response_id": turn.response_id,
            "item_id": turn.item_id,
            "output_index": 0,
            "content_index": 0,
        }

    def _transcript_delta_event(self, turn: _Turn, delta: str) -> dict:
        return {
            "type": "response.output_audio_transcript.delta",
            "event_id": _event_id(),
            "response_id": turn.response_id,
            "item_id": turn.item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": delta,
        }

    @staticmethod
    def _thinker_token_ids(output: Any) -> list[int]:
        """Stage-0 (thinker) per-step token ids to feed back to the talker."""
        if getattr(output, "stage_id", None) != 0:
            return []
        outputs = getattr(output, "outputs", None)
        if not outputs:
            return []
        token_ids = getattr(outputs[0], "token_ids", None)
        return list(token_ids) if token_ids else []

    @staticmethod
    def _extract_transcript(output: Any) -> str:
        """Pull incremental thinker text from a stage-0 LLM output, if any."""
        if getattr(output, "stage_id", None) != 0:
            return ""
        outputs = getattr(output, "outputs", None)
        if not outputs:
            return ""
        return getattr(outputs[0], "text", "") or ""

    def _extract_audio_chunks(self, turn: _Turn, output: Any) -> list[np.ndarray]:
        """Extract per-step audio deltas from an engine output.

        Audio lives in ``output.multimodal_output['audio'|'model_outputs']`` as a
        float32 waveform (or list of them). Some engine paths emit a growing
        cumulative waveform; ``_raw_waveform_to_deltas`` reconciles both shapes
        against the turn's ``audio_ref`` so the client never hears duplicates.
        """
        mm = getattr(output, "multimodal_output", None)
        if not isinstance(mm, dict):
            return []

        key = (
            "audio"
            if "audio" in mm
            else ("model_outputs" if "model_outputs" in mm else None)
        )
        if key is None:
            return []

        raw_audio = mm.get(key)
        if isinstance(raw_audio, (list, tuple)):
            if not raw_audio:
                return []
            arr = _tensor_to_numpy(raw_audio[-1])
        else:
            arr = _tensor_to_numpy(raw_audio)

        if arr is None or arr.size == 0:
            return []
        return _raw_waveform_to_deltas(turn, arr)


def _decode_pcm16(audio_b64: str) -> np.ndarray | None:
    """Decode a base64 PCM16 chunk to a float32 waveform in [-1, 1].

    Mirrors vLLM's realtime connection decode (int16 / 32768). Empty / blank
    payloads yield ``None`` so they are not queued as audio.
    """
    if not audio_b64:
        return None
    waveform = (
        np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16).astype(np.float32)
        / 32768.0
    )
    return waveform if waveform.size else None


def _drain_queue(queue: asyncio.Queue) -> None:
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break


def _tensor_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value
    elif hasattr(value, "detach"):
        arr = value.detach().float().cpu().numpy()
    else:
        try:
            arr = np.asarray(value)
        except Exception:  # noqa: BLE001 - non-array engine payloads are skipped
            return None
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32, copy=False)


def _numpy_audio_prefix_match(prev: np.ndarray, curr: np.ndarray) -> bool:
    n = prev.shape[0]
    if n == 0:
        return True
    if curr.shape[0] < n:
        return False
    return bool(np.allclose(curr[:n], prev, rtol=1e-3, atol=2e-4))


def _raw_waveform_to_deltas(turn: _Turn, arr: np.ndarray) -> list[np.ndarray]:
    """Convert one streaming PCM f32 chunk into incremental piece(s)."""
    if arr.size == 0:
        return []
    ref = turn.audio_ref
    if ref is None:
        turn.audio_ref = arr.copy()
        return [arr]
    if _numpy_audio_prefix_match(ref, arr):
        delta = arr[ref.shape[0] :]
        turn.audio_ref = arr.copy()
        return [delta] if delta.size > 0 else []
    # True per-step delta (not a prefix extension of what we have seen).
    turn.audio_ref = np.concatenate([ref, arr])
    return [arr]


def _pcm16_b64(audio_f32: np.ndarray) -> str:
    clipped = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode("utf-8")
