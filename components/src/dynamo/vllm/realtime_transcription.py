# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI realtime transcription bridge for vLLM streaming ASR models."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import math
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import Any

import numpy as np

from dynamo._core import Context

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


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def _error_event(code: str, message: str, *, event_id: str | None = None) -> dict:
    return {
        "type": "error",
        "event_id": _event_id(),
        "error": {
            "type": "invalid_request_error",
            "code": code,
            "message": message,
            "event_id": event_id,
        },
    }


def _session_updated_event(session: dict) -> dict:
    return {
        "type": "session.updated",
        "event_id": _event_id(),
        "session": session,
    }


def _committed_event(item_id: str) -> dict:
    return {
        "type": "input_audio_buffer.committed",
        "event_id": _event_id(),
        "previous_item_id": None,
        "item_id": item_id,
    }


def _cleared_event() -> dict:
    return {"type": "input_audio_buffer.cleared", "event_id": _event_id()}


def _transcription_delta_event(item_id: str, delta: str) -> dict:
    return {
        "type": "conversation.item.input_audio_transcription.delta",
        "event_id": _event_id(),
        "item_id": item_id,
        "content_index": 0,
        "delta": delta,
        "logprobs": None,
    }


def _transcription_completed_event(
    item_id: str,
    transcript: str,
    *,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": _event_id(),
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
        "logprobs": None,
        "usage": {
            "type": "tokens",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {
                "audio_tokens": input_tokens,
                "text_tokens": 0,
            },
        },
    }


def _transcription_failed_event(item_id: str, message: str) -> dict:
    return {
        "type": "conversation.item.input_audio_transcription.failed",
        "event_id": _event_id(),
        "item_id": item_id,
        "content_index": 0,
        "error": {
            "type": "server_error",
            "code": "transcription_error",
            "message": message,
        },
    }


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


def _is_local_vad_placeholder(turn_detection: Any) -> bool:
    """Recognize the frontend's temporary representation of protocol null."""
    return isinstance(turn_detection, dict) and turn_detection == {
        "type": "server_vad",
        "create_response": False,
        "interrupt_response": False,
        "prefix_padding_ms": 0,
        "silence_duration_ms": 0,
        "threshold": 1.0,
    }


class _Turn:
    def __init__(self, *, input_rate: int, model_sample_rate: int) -> None:
        self.item_id = f"item_{uuid.uuid4().hex}"
        self.request_id = f"rt_{uuid.uuid4().hex}"
        self.input_rate = input_rate
        self.model_sample_rate = model_sample_rate
        self.pending_audio = np.empty(0, dtype=np.float32)
        self.received_samples = 0
        self.audio: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self.task: asyncio.Task | None = None

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
        output: "asyncio.Queue[dict | None]",
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
                    await output.put(_transcription_delta_event(turn.item_id, delta))

            if not context.is_stopped():
                await output.put(
                    _transcription_completed_event(
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
            await output.put(
                _transcription_failed_event(turn.item_id, "Transcription failed")
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
        turn_detection = audio_input.get("turn_detection")
        if turn_detection is not None and not _is_local_vad_placeholder(turn_detection):
            return "server turn detection is not supported; use local VAD and explicit commits"
        return None

    async def generate(
        self,
        request_stream: AsyncGenerator[Any, None],
        context: Context,
    ) -> AsyncGenerator[dict, None]:
        output: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=256)
        active_turn: _Turn | None = None
        tasks: set[asyncio.Task] = set()
        input_rate = OPENAI_PCM_SAMPLE_RATE
        turn_slot = asyncio.Semaphore(1)

        def finish_turn(task: asyncio.Task) -> None:
            tasks.discard(task)
            turn_slot.release()

        async def start_turn() -> _Turn:
            nonlocal active_turn
            if active_turn is None:
                await turn_slot.acquire()
                active_turn = _Turn(
                    input_rate=input_rate,
                    model_sample_rate=self.model_sample_rate,
                )
                task = asyncio.create_task(self._run_turn(active_turn, output, context))
                active_turn.task = task
                tasks.add(task)
                task.add_done_callback(finish_turn)
            return active_turn

        async def pump() -> None:
            nonlocal active_turn, input_rate
            try:
                async for event in request_stream:
                    if context.is_stopped():
                        break
                    if not isinstance(event, dict):
                        await output.put(
                            _error_event("invalid_event", "event must be an object")
                        )
                        continue
                    event_type = event.get("type")
                    if event_type == "session.update":
                        session = event.get("session")
                        error = self._validate_session(session)
                        if error:
                            await output.put(
                                _error_event(
                                    "invalid_session",
                                    error,
                                    event_id=event.get("event_id"),
                                )
                            )
                            continue
                        input_rate = session["audio"]["input"]["format"]["rate"]
                        await output.put(_session_updated_event(session))
                    elif event_type == "input_audio_buffer.append":
                        try:
                            waveform = decode_pcm16(event.get("audio", ""))
                        except ValueError as exc:
                            await output.put(
                                _error_event(
                                    "invalid_audio",
                                    str(exc),
                                    event_id=event.get("event_id"),
                                )
                            )
                            continue
                        turn = await start_turn()
                        try:
                            ready = turn.append_audio(waveform)
                        except ValueError as exc:
                            if turn.task is not None:
                                turn.task.cancel()
                            active_turn = None
                            await output.put(
                                _error_event(
                                    "invalid_audio",
                                    str(exc),
                                    event_id=event.get("event_id"),
                                )
                            )
                            continue
                        if ready is not None:
                            turn.audio.put_nowait(ready)
                    elif event_type == "input_audio_buffer.commit":
                        if active_turn is None:
                            await output.put(
                                _error_event(
                                    "invalid_audio", "input audio buffer is empty"
                                )
                            )
                            continue
                        remainder = active_turn.flush_audio()
                        if remainder is not None:
                            active_turn.audio.put_nowait(remainder)
                        await output.put(_committed_event(active_turn.item_id))
                        active_turn.audio.put_nowait(None)
                        active_turn = None
                    elif event_type == "input_audio_buffer.clear":
                        if active_turn is not None:
                            if active_turn.task is not None:
                                active_turn.task.cancel()
                            active_turn = None
                        await output.put(_cleared_event())
                    else:
                        await output.put(
                            _error_event(
                                "unsupported_event",
                                f"unsupported event type: {event_type}",
                                event_id=event.get("event_id"),
                            )
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the connection diagnosable
                logger.exception("realtime transcription input failed: %s", exc)
                await output.put(
                    _error_event("server_error", "Internal transcription error")
                )
            finally:
                if active_turn is not None:
                    remainder = active_turn.flush_audio()
                    if remainder is not None:
                        active_turn.audio.put_nowait(remainder)
                    active_turn.audio.put_nowait(None)
                    active_turn = None
                await asyncio.gather(*list(tasks), return_exceptions=True)
                await output.put(None)

        pump_task = asyncio.create_task(pump())
        try:
            while (event := await output.get()) is not None:
                yield event
        finally:
            pump_task.cancel()
            pending = list(tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(
                pump_task,
                *pending,
                return_exceptions=True,
            )
