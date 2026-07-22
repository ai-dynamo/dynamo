# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Realtime dispatch and transcription for standard vLLM models."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import math
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping
from typing import Any, Protocol

import numpy as np

from dynamo._core import Context

from .realtime_connection import RealtimeConnection, RealtimeTurn
from .realtime_events import (
    input_audio_buffer_cleared_event,
    input_audio_buffer_committed_event,
    input_audio_transcription_completed_event,
    input_audio_transcription_delta_event,
    input_audio_transcription_failed_event,
    invalid_request_error_event,
    session_updated_event,
)

logger = logging.getLogger(__name__)

OPENAI_PCM_SAMPLE_RATE = 24_000
MAX_AUDIO_CHUNK_BYTES = 4 * 1024 * 1024
RESAMPLE_BLOCK_MILLISECONDS = 100
MAX_UTTERANCE_SECONDS = 60

StreamingInputFactory = Callable[
    [AsyncGenerator[np.ndarray, None], "asyncio.Queue[list[int]]"],
    AsyncGenerator[Any, None],
]
SamplingParamsFactory = Callable[[], Any]


class RealtimeSessionHandler(Protocol):
    def generate(
        self,
        request_stream: AsyncGenerator[Any, None],
        context: Context,
    ) -> AsyncGenerator[dict, None]:
        ...


class RealtimeHandler:
    """Select one session handler from the initial ``session.update`` event."""

    def __init__(self, handlers: Mapping[str, RealtimeSessionHandler]) -> None:
        self._handlers = dict(handlers)

    async def generate(
        self,
        request_stream: AsyncGenerator[Any, None],
        context: Context,
    ) -> AsyncGenerator[dict, None]:
        try:
            first_event = await anext(request_stream)
        except StopAsyncIteration:
            return

        if (
            not isinstance(first_event, dict)
            or first_event.get("type") != "session.update"
        ):
            yield invalid_request_error_event(
                "invalid_event",
                "first event must be session.update",
                client_event_id=(
                    first_event.get("event_id")
                    if isinstance(first_event, dict)
                    else None
                ),
            )
            return

        session = first_event.get("session")
        session_type = session.get("type") if isinstance(session, dict) else None
        handler = (
            self._handlers.get(session_type) if isinstance(session_type, str) else None
        )
        if handler is None:
            yield invalid_request_error_event(
                "unsupported_session",
                f"unsupported session type: {session_type!r}",
                client_event_id=first_event.get("event_id"),
            )
            return

        async def replay() -> AsyncGenerator[Any, None]:
            yield first_event
            async for event in request_stream:
                yield event

        async for event in handler.generate(replay(), context):
            yield event


def _default_sampling_params() -> Any:
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    return SamplingParams.from_optional(
        temperature=0.0,
        max_tokens=64,
        output_kind=RequestOutputKind.DELTA,
        skip_clone=True,
    )


def _resample(waveform: np.ndarray, input_rate: int, output_rate: int) -> np.ndarray:
    if input_rate == output_rate:
        return waveform

    # scipy is already a vLLM audio dependency. Polyphase resampling preserves
    # speech bandwidth when adapting OpenAI's fixed 24 kHz PCM stream to the
    # model's native rate (commonly 16 kHz).
    from scipy.signal import resample_poly

    divisor = math.gcd(input_rate, output_rate)
    return resample_poly(
        waveform,
        output_rate // divisor,
        input_rate // divisor,
    ).astype(np.float32, copy=False)


def decode_pcm16(audio_b64: str) -> np.ndarray:
    if not isinstance(audio_b64, str):
        raise ValueError("audio must be a base64 string")
    try:
        raw = base64.b64decode(audio_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("audio must be valid base64") from exc
    if not raw:
        raise ValueError("audio chunk is empty")
    if len(raw) > MAX_AUDIO_CHUNK_BYTES:
        raise ValueError("audio chunk exceeds 4 MiB")
    if len(raw) % 2:
        raise ValueError("PCM16 audio must be 2-byte aligned")

    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class _Turn(RealtimeTurn):
    def __init__(self, *, input_rate: int, model_sample_rate: int) -> None:
        super().__init__()
        self.item_id = f"item_{uuid.uuid4().hex}"
        self.request_id = f"rt_{uuid.uuid4().hex}"
        self.input_rate = input_rate
        self.model_sample_rate = model_sample_rate
        self.pending_audio = np.empty(0, dtype=np.float32)
        self.received_samples = 0
        self.audio: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

    def append_audio(self, waveform: np.ndarray) -> np.ndarray | None:
        received_samples = self.received_samples + len(waveform)
        if received_samples > self.input_rate * MAX_UTTERANCE_SECONDS:
            raise ValueError(f"input audio exceeds {MAX_UTTERANCE_SECONDS} seconds")
        self.received_samples = received_samples
        self.pending_audio = np.concatenate((self.pending_audio, waveform))
        block_size = self.input_rate * RESAMPLE_BLOCK_MILLISECONDS // 1000
        ready_size = len(self.pending_audio) // block_size * block_size
        if not ready_size:
            return None
        ready, self.pending_audio = np.split(self.pending_audio, [ready_size])
        return _resample(ready, self.input_rate, self.model_sample_rate)

    def flush_audio(self) -> np.ndarray | None:
        if not len(self.pending_audio):
            return None
        ready = self.pending_audio
        self.pending_audio = np.empty(0, dtype=np.float32)
        return _resample(ready, self.input_rate, self.model_sample_rate)

    async def audio_stream(self) -> AsyncGenerator[np.ndarray, None]:
        while True:
            chunk = await self.audio.get()
            if chunk is None:
                return
            yield chunk


class RealtimeTranscriptionHandler:
    """Translate OpenAI realtime transcription events to vLLM streaming input."""

    def __init__(
        self,
        *,
        engine_client: Any,
        model_name: str,
        model_sample_rate: int | float,
        streaming_input_factory: StreamingInputFactory,
        sampling_params_factory: SamplingParamsFactory = _default_sampling_params,
    ) -> None:
        self.engine_client = engine_client
        self.model_name = model_name
        self.model_sample_rate = int(model_sample_rate)
        self._streaming_input_factory = streaming_input_factory
        self._sampling_params_factory = sampling_params_factory

    @classmethod
    def from_engine(
        cls,
        *,
        engine_client: Any,
        model_name: str,
        model_path: str,
    ) -> "RealtimeTranscriptionHandler":
        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIServingModels
        from vllm.entrypoints.speech_to_text.realtime.serving import (
            OpenAIServingRealtime,
        )

        models = OpenAIServingModels(
            engine_client=engine_client,
            base_model_paths=[BaseModelPath(name=model_name, model_path=model_path)],
            lora_modules=None,
        )
        serving = OpenAIServingRealtime(
            engine_client=engine_client,
            models=models,
            request_logger=None,
        )
        speech_config = serving.model_cls.get_speech_to_text_config(
            serving.model_config, "transcribe"
        )

        def sampling_params() -> Any:
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            return SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=serving.model_cls.realtime_max_tokens,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )

        return cls(
            engine_client=engine_client,
            model_name=model_name,
            model_sample_rate=speech_config.sample_rate,
            streaming_input_factory=serving.transcribe_realtime,
            sampling_params_factory=sampling_params,
        )

    async def _run_turn(
        self,
        turn: _Turn,
        context: Context,
    ) -> None:
        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        streaming_input = self._streaming_input_factory(
            turn.audio_stream(), input_stream
        )
        transcript = ""
        input_tokens = 0
        output_tokens = 0

        try:
            result_stream = self.engine_client.generate(
                prompt=streaming_input,
                sampling_params=self._sampling_params_factory(),
                request_id=turn.request_id,
            )
            async for result in result_stream:
                if context.is_stopped():
                    return
                outputs = getattr(result, "outputs", None)
                if not outputs:
                    continue
                candidate = outputs[0]
                delta = getattr(candidate, "text", "") or ""
                token_ids = list(getattr(candidate, "token_ids", None) or [])
                if not input_tokens:
                    input_tokens = len(getattr(result, "prompt_token_ids", None) or [])
                output_tokens += len(token_ids)
                if token_ids:
                    input_stream.put_nowait(token_ids)
                if delta:
                    transcript += delta
                    await turn.events.put(
                        input_audio_transcription_delta_event(turn.item_id, delta)
                    )

            if not context.is_stopped():
                await turn.events.put(
                    input_audio_transcription_completed_event(
                        turn.item_id,
                        transcript,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate engine failures per turn
            logger.exception("realtime transcription failed: %s", exc)
            await turn.events.put(
                input_audio_transcription_failed_event(
                    turn.item_id, "Transcription failed"
                )
            )

    def _validate_session(self, session: Any) -> str | None:
        if not isinstance(session, dict) or session.get("type") != "transcription":
            return "session.type must be 'transcription'"
        audio = session.get("audio")
        if not isinstance(audio, dict) or not isinstance(audio.get("input"), dict):
            return "session.audio.input must be an object"
        audio_input = audio["input"]
        transcription = audio_input.get("transcription")
        if (
            not isinstance(transcription, dict)
            or transcription.get("model") != self.model_name
        ):
            return f"session transcription model must be '{self.model_name}'"
        audio_format = audio_input.get("format")
        if not isinstance(audio_format, dict):
            return "session.audio.input.format must be an object"
        if audio_format.get("type") != "audio/pcm":
            return "only audio/pcm input is supported"
        rate = audio_format.get("rate")
        if rate != OPENAI_PCM_SAMPLE_RATE:
            return f"audio/pcm input rate must be {OPENAI_PCM_SAMPLE_RATE} Hz"
        language = transcription.get("language")
        if language not in (None, "en"):
            return "only English realtime transcription is supported"
        if transcription.get("prompt") not in (None, ""):
            return "transcription prompts are not supported"
        if audio_input.get("noise_reduction") is not None:
            return "input audio noise reduction is not supported"
        if audio_input.get("turn_detection") is not None:
            return "server turn detection is not supported; use local VAD and explicit commits"
        return None

    async def generate(
        self,
        request_stream: AsyncGenerator[Any, None],
        context: Context,
    ) -> AsyncGenerator[dict, None]:
        input_rate = OPENAI_PCM_SAMPLE_RATE

        connection = RealtimeConnection[_Turn](
            context=context,
            run_turn=self._run_turn,
            max_concurrent_turns=1,
            max_queued_turns=1,
        )

        def new_turn() -> _Turn:
            return _Turn(
                input_rate=input_rate,
                model_sample_rate=self.model_sample_rate,
            )

        def close_turn(turn: _Turn) -> None:
            remainder = turn.flush_audio()
            if remainder is not None:
                turn.audio.put_nowait(remainder)
            turn.audio.put_nowait(None)

        async def handle_event(
            event: Any,
            connection: RealtimeConnection[_Turn],
        ) -> None:
            nonlocal input_rate
            if not isinstance(event, dict):
                connection.emit(
                    invalid_request_error_event(
                        "invalid_event", "event must be an object"
                    )
                )
                return
            event_type = event.get("type")
            if event_type == "session.update":
                session = event.get("session")
                error = self._validate_session(session)
                if error:
                    connection.emit(
                        invalid_request_error_event(
                            "invalid_session",
                            error,
                            client_event_id=event.get("event_id"),
                        )
                    )
                    return
                input_rate = session["audio"]["input"]["format"]["rate"]
                connection.emit(session_updated_event(session))
            elif event_type == "input_audio_buffer.append":
                try:
                    waveform = decode_pcm16(event.get("audio", ""))
                except ValueError as exc:
                    connection.emit(
                        invalid_request_error_event(
                            "invalid_audio",
                            str(exc),
                            client_event_id=event.get("event_id"),
                        )
                    )
                    return
                turn = await connection.ensure_turn(new_turn)
                try:
                    ready = turn.append_audio(waveform)
                except ValueError as exc:
                    connection.cancel_active_turn()
                    connection.emit(
                        invalid_request_error_event(
                            "invalid_audio",
                            str(exc),
                            client_event_id=event.get("event_id"),
                        )
                    )
                    return
                if ready is not None:
                    turn.audio.put_nowait(ready)
            elif event_type == "input_audio_buffer.commit":
                turn = connection.active_turn
                if turn is None:
                    connection.emit(
                        invalid_request_error_event(
                            "invalid_audio", "input audio buffer is empty"
                        )
                    )
                    return
                remainder = turn.flush_audio()
                if remainder is not None:
                    turn.audio.put_nowait(remainder)
                await connection.emit_for_turn(
                    turn,
                    input_audio_buffer_committed_event(turn.item_id),
                )
                turn.audio.put_nowait(None)
                connection.finish_active_turn()
            elif event_type == "input_audio_buffer.clear":
                connection.cancel_active_turn()
                connection.emit(input_audio_buffer_cleared_event())
            else:
                connection.emit(
                    invalid_request_error_event(
                        "unsupported_event",
                        f"unsupported event type: {event_type}",
                        client_event_id=event.get("event_id"),
                    )
                )

        async for event in connection.generate(
            request_stream,
            handle_event=handle_event,
            close_active_turn=close_turn,
        ):
            yield event
