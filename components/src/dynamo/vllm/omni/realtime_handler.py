# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realtime (bidirectional) handler backed by vLLM-Omni's streaming engine.

This handler expects ``request_stream`` to yield ``RealtimeServerEvent`` frames.

Turn model (matching vLLM-Omni's single-active-generation contract):

  * ``session.update``            -> ``session.updated`` echoing the session;
    also captures the requested ``output_modalities`` for later turns.
  * ``input_audio_buffer.append`` -> base64 PCM16 chunk decoded to a float32
    waveform and queued for the active turn's audio stream.
  * ``input_audio_buffer.commit`` -> opens a turn on the first event and, when
    ``final`` is set, closes the audio stream so the engine drains.

Turns run concurrently -- a commit's turn drives the engine as soon as it opens
-- but their responses are forwarded to the client in turn order: each turn
buffers its server events, and a later turn's buffer is held until the previous
turn's response completes. Responses never interleave and each is identified by
its ``response_id``.

Each turn emits ``response.created`` -> ``response.output_audio.delta``* (+
optional ``response.output_audio_transcript.delta`` for the thinker text) ->
``response.output_audio.done`` -> ``response.done``. These are the OpenAI-spec
event names the frontend's typed reader requires, which differ from
vLLM-Omni's own ``response.audio.delta`` / ``transcription.delta`` names; the
PCM16 and cumulative-vs-delta waveform handling below is ported from
vLLM-Omni's ``realtime_connection.py`` and only the event tags change.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from typing import Any, AsyncGenerator, Callable, Optional, Sequence

import numpy as np

from dynamo._core import Context

logger = logging.getLogger(__name__)

# ``streaming_input_factory(audio_stream, input_stream) -> AsyncGenerator`` —
# mirrors ``OpenAIServingRealtime.transcribe_realtime``: it consumes float32
# audio chunks and an ``asyncio.Queue`` of context token ids, yielding engine
# ``StreamingInput`` prompts.
# The factory and engine are injected so the worker passes the real serving/AsyncOmni
# while tests pass lightweight fakes.
StreamingInputFactory = Callable[
    [AsyncGenerator[np.ndarray, None], "asyncio.Queue[list[int]]"],
    AsyncGenerator[Any, None],
]


def event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def session_updated_event(session: Any) -> dict:
    """Echo a client ``session.update`` back as the spec ``session.updated``."""
    return {
        "type": "session.updated",
        "event_id": event_id(),
        "session": session,
    }


class Turn:
    """One request->response cycle: a committed span of input audio and the
    single OpenAI-spec ``response`` it produces.

    Owns its engine drive and output extraction (``drive_engine``: it feeds its
    buffered audio into the engine and yields ``(transcript, audio_chunks)`` per
    step); ``RealtimeOmniHandler`` orchestrates turns and translates those into
    OpenAI-spec server events (see ``RealtimeOmniHandler.run_turn``).

    Fields: ``response_id`` / ``item_id`` tag every event of this turn;
    ``audio_queue`` carries the float32 input (``None`` = end of input);
    ``audio_ref`` tracks the last emitted waveform so cumulative engine outputs
    are de-duplicated into true deltas; ``output_modalities`` is snapshotted from
    the latest ``session.update``; ``task`` runs the turn's processing;
    ``events`` buffers its server events for in-order forwarding (``None`` = end
    of the turn's response).
    """

    def __init__(
        self,
        *,
        engine_client: Any,
        streaming_input_factory: StreamingInputFactory,
        default_sampling_params_list: Optional[Sequence[Any]] = None,
        output_modalities: list[str] | None = None,
    ) -> None:
        self.response_id = f"resp_{uuid.uuid4().hex}"
        self.item_id = f"item_{uuid.uuid4().hex}"
        # Unbounded on purpose: filled by non-blocking put_nowait so the inbound
        # demux never stalls control events (commit/clear/session.update) behind
        # audio backpressure; paced by the client's input rate and drained by the
        # engine.
        self.audio_queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue()
        self.audio_ref: np.ndarray | None = None
        self.task: asyncio.Task | None = None
        # This turn's server events, forwarded to the client in turn order by
        # ``RealtimeOmniHandler.generate`` (``None`` = end of response). Bounded
        # so a turn waiting for its forwarding slot exerts backpressure on its
        # own engine drive rather than buffering unbounded.
        self.events: asyncio.Queue[Optional[dict]] = asyncio.Queue(maxsize=256)
        self.output_modalities = output_modalities
        self._engine_client = engine_client
        self._streaming_input_factory = streaming_input_factory
        self._default_sampling_params_list = default_sampling_params_list

    async def drive_engine(
        self,
    ) -> AsyncGenerator[tuple[str | None, list[np.ndarray] | None], None]:
        """Feed this turn's buffered audio into the engine and yield, per engine
        step, the extracted ``(transcript, audio_chunks)`` -- either element is
        ``None`` when that step produced none.

        The float32 ``audio_stream`` (ending on the ``None`` sentinel) and an
        ``input_stream`` token queue are handed to the streaming-input factory
        (``transcribe_realtime``), whose ``StreamingInput`` generator drives
        ``AsyncOmni.generate``.
        """

        # build input stream generator
        async def audio_stream() -> AsyncGenerator[np.ndarray, None]:
            while True:
                waveform = await self.audio_queue.get()
                if waveform is None:
                    return
                yield waveform

        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        streaming_input_gen = self._streaming_input_factory(
            audio_stream(), input_stream
        )

        generate_kwargs: dict[str, Any] = {
            "prompt": streaming_input_gen,
            "request_id": self.response_id,
        }
        if self._default_sampling_params_list is not None:
            generate_kwargs["sampling_params_list"] = list(
                self._default_sampling_params_list
            )
        # Client-requested output modalities (session.update) select the final
        # pipeline stage: include "audio" to drive the talker. Omitted ->
        # AsyncOmni.generate uses the engine's launch-time default.
        if self.output_modalities is not None:
            generate_kwargs["output_modalities"] = self.output_modalities

        # drive the engine
        async for output in self._engine_client.generate(**generate_kwargs):
            token_ids = self.thinker_token_ids(output)
            if token_ids:
                input_stream.put_nowait(token_ids)
            transcript = self.extract_transcript(output) or None
            audio_chunks = self.extract_audio_chunks(output) or None
            yield transcript, audio_chunks

        # Generation done: release the cumulative reference waveform. It only
        # de-dups deltas within this drive; nothing client-bound survives on it
        # (the deltas are already encoded into the queued events), and for long
        # voice sessions a ~10 MB/turn float32 buffer pinned per turn adds up.
        self.audio_ref = None

    @staticmethod
    def thinker_token_ids(output: Any) -> list[int]:
        """Stage-0 (thinker) per-step token ids to feed back to the talker."""
        if getattr(output, "stage_id", None) != 0:
            return []
        outputs = getattr(output, "outputs", None)
        if not outputs:
            return []
        token_ids = getattr(outputs[0], "token_ids", None)
        return list(token_ids) if token_ids else []

    @staticmethod
    def extract_transcript(output: Any) -> str:
        """Pull incremental thinker text from a stage-0 LLM output, if any."""
        if getattr(output, "stage_id", None) != 0:
            return ""
        outputs = getattr(output, "outputs", None)
        if not outputs:
            return ""
        return getattr(outputs[0], "text", "") or ""

    def extract_audio_chunks(self, output: Any) -> list[np.ndarray]:
        """Extract per-step audio deltas from an engine output.

        Audio lives in ``output.multimodal_output['audio'|'model_outputs']`` as a
        float32 waveform (or list of them). Some engine paths emit a growing
        cumulative waveform; ``waveform_to_deltas`` reconciles both shapes
        against ``self.audio_ref`` so the client never hears duplicates.
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
            arr = tensor_to_numpy(raw_audio[-1])
        else:
            arr = tensor_to_numpy(raw_audio)

        if arr is None or arr.size == 0:
            return []
        return self.waveform_to_deltas(arr)

    def waveform_to_deltas(self, arr: np.ndarray) -> list[np.ndarray]:
        """Convert one streaming PCM f32 chunk into incremental piece(s)."""
        if arr.size == 0:
            return []
        ref = self.audio_ref
        if ref is None:
            self.audio_ref = arr.copy()
            return [arr]
        if numpy_audio_prefix_match(ref, arr):
            delta = arr[ref.shape[0] :]
            self.audio_ref = arr.copy()
            return [delta] if delta.size > 0 else []
        # True per-step delta (not a prefix extension of what we have seen). The
        # growing concat makes this O(n^2) over a response, but per-response audio
        # is bounded (seconds); kept to mirror vLLM-Omni's realtime_connection.
        self.audio_ref = np.concatenate([ref, arr])
        return [arr]


class RealtimeOmniHandler:
    """Bridge OpenAI Realtime client events to vLLM-Omni streaming generation.

    Owns the per-connection orchestration (event demux, concurrent turns whose
    responses are forwarded in turn order) and the conversion between the
    realtime API and model output: it drives each ``Turn``'s engine generation
    and translates the engine's stage outputs into OpenAI-spec server events.
    """

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

    def new_turn(self, output_modalities: list[str] | None) -> Turn:
        return Turn(
            engine_client=self.engine_client,
            streaming_input_factory=self._streaming_input_factory,
            default_sampling_params_list=self._default_sampling_params_list,
            output_modalities=output_modalities,
        )

    async def run_turn(self, turn: Turn, context: Context) -> None:
        """Drive one turn's engine generation and buffer its server events.

        Events are appended to ``turn.events`` rather than sent to the client
        directly; ``generate`` forwards each turn's buffer in turn order, so
        turns may run concurrently while their responses never interleave. A
        ``None`` sentinel is always appended last to mark the turn's response
        complete and let the forwarder advance to the next turn.
        """
        events = turn.events
        try:
            await events.put(self.response_created_event(turn))

            sent_audio = False
            async for transcript, audio_chunks in turn.drive_engine():
                if context.is_stopped():
                    break

                if transcript and self._emit_transcript:
                    await events.put(self.transcript_delta_event(turn, transcript))

                for chunk in audio_chunks or ():
                    sent_audio = True
                    await events.put(self.audio_delta_event(turn, chunk))

            if context.is_stopped():
                # Connection torn down mid-turn; don't claim a completed response.
                return

            if sent_audio:
                await events.put(self.audio_done_event(turn))
            await events.put(self.response_done_event(turn))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface engine errors on the wire
            logger.exception("realtime omni turn failed: %s", exc)
            # The top-level ``error`` event carries the human-readable message but
            # no ``response_id``, so also close the dangling in-progress response
            # with a terminal ``response.done(status=failed)`` -- that event
            # carries the id, so the client can correlate and the response reaches
            # a terminal state instead of hanging.
            await events.put(self.error_event(exc))
            await events.put(
                self.response_done_event(
                    turn,
                    status="failed",
                    status_details={
                        "type": "failed",
                        "error": {
                            "code": "omni_generation_error",
                            "type": "server_error",
                        },
                    },
                )
            )
        finally:
            await events.put(None)

    # -- response lifecycle events --------------------------------------------

    def response_created_event(self, turn: Turn) -> dict:
        return {
            "type": "response.created",
            "event_id": event_id(),
            "response": {
                "id": turn.response_id,
                "max_output_tokens": "inf",
                "object": "realtime.response",
                "output": [],
                "output_modalities": ["audio"],
                "status": "in_progress",
            },
        }

    def response_done_event(
        self, turn: Turn, status: str = "completed", status_details: dict | None = None
    ) -> dict:
        response: dict[str, Any] = {
            "id": turn.response_id,
            "max_output_tokens": "inf",
            "object": "realtime.response",
            "output": [],
            "output_modalities": ["audio"],
            "status": status,
        }
        if status_details is not None:
            response["status_details"] = status_details
        return {
            "type": "response.done",
            "event_id": event_id(),
            "response": response,
        }

    def error_event(self, exc: Exception) -> dict:
        return {
            "type": "error",
            "event_id": event_id(),
            "error": {
                "type": "server_error",
                "code": "omni_generation_error",
                "message": str(exc),
            },
        }

    # -- output translation (ported from vllm-omni realtime_connection.py) ----

    def audio_delta_event(self, turn: Turn, chunk: np.ndarray) -> dict:
        return {
            "type": "response.output_audio.delta",
            "event_id": event_id(),
            "response_id": turn.response_id,
            "item_id": turn.item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": pcm16_b64(chunk),
        }

    def audio_done_event(self, turn: Turn) -> dict:
        return {
            "type": "response.output_audio.done",
            "event_id": event_id(),
            "response_id": turn.response_id,
            "item_id": turn.item_id,
            "output_index": 0,
            "content_index": 0,
        }

    def transcript_delta_event(self, turn: Turn, delta: str) -> dict:
        return {
            "type": "response.output_audio_transcript.delta",
            "event_id": event_id(),
            "response_id": turn.response_id,
            "item_id": turn.item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": delta,
        }

    async def generate(
        self, request_stream: AsyncGenerator[Any, None], context: Context
    ) -> AsyncGenerator[dict, None]:
        """Serve one realtime connection.

        Each turn is spawned as a task that drives its engine and buffers its
        server events on the turn's own queue. Standalone events (e.g.
        ``session.updated``) and turns are placed on ``out_stream`` in arrival
        order; this coroutine forwards them in that order, draining a turn's
        buffered events in full before the next turn -- so turns run concurrently
        while their responses never interleave.
        """
        # Ordered hand-off: each item is either a standalone server event (dict)
        # to forward as-is, or a ``Turn`` whose buffered events are drained in
        # order once reached. Low-volume (one item per turn / session.update) so
        # unbounded; per-turn audio backpressure lives on each turn's ``events``.
        out_stream: asyncio.Queue[Any] = asyncio.Queue()
        active_turn: Turn | None = None
        turns: list[Turn] = []
        # Latest output modalities requested by the client via session.update;
        # snapshotted into each turn so the engine emits text/audio accordingly.
        session_output_modalities: list[str] | None = None

        def ensure_turn() -> Turn:
            nonlocal active_turn
            if active_turn is None:
                active_turn = self.new_turn(session_output_modalities)
                turns.append(active_turn)
                out_stream.put_nowait(active_turn)
                active_turn.task = asyncio.create_task(
                    self.run_turn(active_turn, context)
                )
            return active_turn

        async def pump() -> None:
            nonlocal active_turn, session_output_modalities
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
                        session = client_event.get("session")
                        modalities = parse_output_modalities(session)
                        if modalities is not None:
                            session_output_modalities = modalities
                        out_stream.put_nowait(session_updated_event(session))
                    elif etype == "input_audio_buffer.append":
                        turn = ensure_turn()
                        waveform = decode_pcm16(client_event.get("audio", ""))
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
                        # Per the spec, clear only discards the *input* buffer (not
                        # yet committed); it does not cancel an in-flight response
                        # (that's response.cancel). After a final commit active_turn
                        # is None, so a clear correctly no-ops.
                        if active_turn is not None:
                            drain_queue(active_turn.audio_queue)
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
            finally:
                # No more turns/events will be queued; lets the forwarder stop
                # once it has drained every turn already on out_stream.
                out_stream.put_nowait(None)

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                item = await out_stream.get()
                if item is None:
                    break
                if isinstance(item, Turn):
                    # Forward this turn's response in full before the next turn;
                    # run_turn always closes the buffer with a None sentinel.
                    while True:
                        event = await item.events.get()
                        if event is None:
                            break
                        yield event
                    # Fully forwarded (its task is finishing): drop our reference
                    # so the turn's buffers are released instead of pinned in
                    # `turns` for the whole connection.
                    item.audio_ref = None
                    turns.remove(item)
                else:
                    yield item
        finally:
            pump_task.cancel()
            for turn in turns:
                if turn.task is not None:
                    turn.task.cancel()
            # Unblock any turn task parked on a full events queue so its
            # cancellation can propagate, then drive every task to completion --
            # this closes each engine generate() async-gen instead of leaking it.
            for turn in turns:
                drain_queue(turn.events)
            await asyncio.gather(
                pump_task,
                *(turn.task for turn in turns if turn.task is not None),
                return_exceptions=True,
            )


def parse_output_modalities(session: Any) -> list[str] | None:
    """Extract requested output modalities from a session.update `session` block.

    Reads OpenAI Realtime ``output_modalities`` (falling back to the older
    ``modalities``); returns a list of strings, or None when unset/malformed so
    the engine's launch-time default applies.
    """
    if not isinstance(session, dict):
        return None
    modalities = session.get("output_modalities")
    if modalities is None:
        modalities = session.get("modalities")
    if isinstance(modalities, list) and all(isinstance(m, str) for m in modalities):
        return modalities
    return None


def decode_pcm16(audio_b64: str) -> np.ndarray | None:
    """Decode a base64 PCM16 chunk to a float32 waveform in [-1, 1].

    Mirrors vLLM's realtime connection decode (int16 / 32768). Empty / blank
    payloads yield ``None`` so they are not queued as audio.
    """
    if not audio_b64:
        return None
    raw = base64.b64decode(audio_b64)
    if len(raw) % 2:
        # PCM16 is 2-byte aligned; np.frombuffer would raise. Drop the malformed
        # chunk rather than let one bad frame tear down the whole session.
        logger.warning(
            "realtime omni: dropping odd-length (%d-byte) audio chunk", len(raw)
        )
        return None
    waveform = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return waveform if waveform.size else None


def drain_queue(queue: asyncio.Queue) -> None:
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break


def tensor_to_numpy(value: Any) -> np.ndarray | None:
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


def numpy_audio_prefix_match(prev: np.ndarray, curr: np.ndarray) -> bool:
    n = prev.shape[0]
    if n == 0:
        return True
    if curr.shape[0] < n:
        return False
    return bool(np.allclose(curr[:n], prev, rtol=1e-3, atol=2e-4))


def pcm16_b64(audio_f32: np.ndarray) -> str:
    clipped = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode("utf-8")
