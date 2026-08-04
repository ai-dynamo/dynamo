# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenAI realtime transcription adapter for a Riva streaming ASR service."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import queue
import uuid
from collections.abc import AsyncGenerator, Iterator
from typing import Any

from riva.client import AudioEncoding, RecognitionConfig, StreamingRecognitionConfig

from dynamo._core import Context
from dynamo.vllm.realtime.connection import RealtimeConnection, RealtimeTurn
from dynamo.vllm.realtime.events import (
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
PCM16_BYTES_PER_SAMPLE = 2
MAX_AUDIO_CHUNK_BYTES = 4 * 1024 * 1024
MAX_UTTERANCE_SECONDS = 60


def _append_only_delta(emitted: str, hypothesis: str) -> tuple[str, str]:
    """Return the suffix of an extending hypothesis, or suppress a revision."""
    if not hypothesis.startswith(emitted):
        return emitted, ""
    return hypothesis, hypothesis[len(emitted) :]


class _AudioTurn(RealtimeTurn):
    """Queue-backed PCM input for one committed transcription turn."""

    _END = object()

    def __init__(self) -> None:
        super().__init__()
        self.item_id = f"item_{uuid.uuid4().hex}"
        self.received_bytes = 0
        self.audio: queue.Queue[bytes | object] = queue.Queue()
        self.closed = False
        self.closed_event = asyncio.Event()

    def append(self, audio_base64: Any) -> None:
        if not isinstance(audio_base64, str):
            raise TypeError("audio must be a base64 string")
        try:
            audio = base64.b64decode(audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("audio must be valid base64") from exc
        if not audio:
            raise ValueError("audio chunk is empty")
        if len(audio) > MAX_AUDIO_CHUNK_BYTES:
            raise ValueError("audio chunk exceeds 4 MiB")
        if len(audio) % PCM16_BYTES_PER_SAMPLE:
            raise ValueError("PCM16 audio must be 2-byte aligned")
        self.received_bytes += len(audio)
        max_bytes = (
            OPENAI_PCM_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * MAX_UTTERANCE_SECONDS
        )
        if self.received_bytes > max_bytes:
            raise ValueError(f"input audio exceeds {MAX_UTTERANCE_SECONDS} seconds")
        self.audio.put_nowait(audio)

    def append_silence(self, duration_ms: int) -> None:
        samples = OPENAI_PCM_SAMPLE_RATE * duration_ms // 1000
        if samples:
            self.audio.put_nowait(bytes(samples * PCM16_BYTES_PER_SAMPLE))

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.audio.put_nowait(self._END)
            self.closed_event.set()

    def chunks(self) -> Iterator[bytes]:
        while (chunk := self.audio.get()) is not self._END:
            assert isinstance(chunk, bytes)
            yield chunk


class RivaRealtimeTranscriptionHandler:
    """Translate OpenAI transcription events to Riva's streaming gRPC API."""

    def __init__(
        self,
        *,
        asr_service,
        model_name: str,
        riva_model: str,
        language_code: str,
        commit_padding_ms: int,
        timeout_s: float,
    ) -> None:
        self.asr_service = asr_service
        self.model_name = model_name
        self.riva_model = riva_model
        self.language_code = language_code
        if commit_padding_ms < 0:
            raise ValueError("commit_padding_ms must be non-negative")
        self.commit_padding_ms = commit_padding_ms
        self.timeout_s = timeout_s

    def _streaming_config(self) -> StreamingRecognitionConfig:
        return StreamingRecognitionConfig(
            config=RecognitionConfig(
                encoding=AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=OPENAI_PCM_SAMPLE_RATE,
                language_code=self.language_code,
                model=self.riva_model,
                max_alternatives=1,
                enable_automatic_punctuation=True,
            ),
            interim_results=True,
        )

    def _transcribe(
        self,
        turn: _AudioTurn,
        loop: asyncio.AbstractEventLoop,
    ) -> str:
        final_segments: list[str] = []
        emitted_interim = ""
        responses = self.asr_service.streaming_response_generator(
            audio_chunks=turn.chunks(),
            streaming_config=self._streaming_config(),
        )
        for response in responses:
            for result in response.results:
                if not result.alternatives:
                    continue
                transcript = result.alternatives[0].transcript
                if not transcript:
                    continue
                if result.is_final:
                    final_segments.append(transcript)
                    emitted_interim = ""
                    continue
                emitted_interim, delta = _append_only_delta(emitted_interim, transcript)
                if delta:
                    asyncio.run_coroutine_threadsafe(
                        turn.events.put(
                            input_audio_transcription_delta_event(turn.item_id, delta)
                        ),
                        loop,
                    ).result()
        return " ".join(final_segments) or emitted_interim

    async def _run_turn(self, turn: _AudioTurn, context: Context) -> None:
        transcription_task = asyncio.create_task(
            asyncio.to_thread(
                self._transcribe,
                turn,
                asyncio.get_running_loop(),
            )
        )
        try:
            # The turn task starts with the first audio chunk so Riva can emit
            # interim results. Apply the backend deadline only after commit;
            # otherwise a long but valid utterance would consume the RPC budget.
            await turn.closed_event.wait()
            transcript = await asyncio.wait_for(
                transcription_task,
                timeout=self.timeout_s,
            )
            if not context.is_stopped():
                await turn.events.put(
                    input_audio_transcription_completed_event(
                        turn.item_id,
                        transcript,
                        input_tokens=0,
                        output_tokens=0,
                    )
                )
        except asyncio.CancelledError:
            turn.close()
            transcription_task.cancel()
            raise
        except Exception:
            turn.close()
            transcription_task.cancel()
            logger.exception("Riva realtime transcription failed")
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
        if audio_format.get("rate") != OPENAI_PCM_SAMPLE_RATE:
            return f"audio/pcm input rate must be {OPENAI_PCM_SAMPLE_RATE} Hz"
        language = transcription.get("language")
        supported_language = self.language_code.split("-", 1)[0].lower()
        if language and str(language).split("-", 1)[0].lower() != supported_language:
            return f"only {self.language_code} realtime transcription is supported"
        if transcription.get("prompt") not in (None, ""):
            return "transcription prompts are not supported"
        if audio_input.get("noise_reduction") is not None:
            return "input audio noise reduction is not supported"
        if audio_input.get("turn_detection") is not None:
            return "server turn detection is not supported; use local VAD"
        return None

    async def generate(
        self,
        request_stream: AsyncGenerator[Any, None],
        context: Context,
    ) -> AsyncGenerator[dict, None]:
        connection = RealtimeConnection[_AudioTurn](
            context=context,
            run_turn=self._run_turn,
            max_concurrent_turns=1,
            max_queued_turns=1,
        )

        def close_turn(turn: _AudioTurn) -> None:
            turn.close()

        async def handle_event(
            event: Any,
            connection: RealtimeConnection[_AudioTurn],
        ) -> None:
            if not isinstance(event, dict):
                connection.emit(
                    invalid_request_error_event(
                        "invalid_event", "event must be an object"
                    )
                )
                return
            event_type = event.get("type")
            if event_type == "session.update":
                error = self._validate_session(event.get("session"))
                if error:
                    connection.emit(
                        invalid_request_error_event(
                            "invalid_session",
                            error,
                            client_event_id=event.get("event_id"),
                        )
                    )
                else:
                    connection.emit(session_updated_event(event["session"]))
            elif event_type == "input_audio_buffer.append":
                turn = await connection.ensure_turn(_AudioTurn)
                try:
                    turn.append(event.get("audio"))
                except (TypeError, ValueError) as exc:
                    turn.close()
                    connection.cancel_active_turn()
                    connection.emit(
                        invalid_request_error_event(
                            "invalid_audio",
                            str(exc),
                            client_event_id=event.get("event_id"),
                        )
                    )
            elif event_type == "input_audio_buffer.commit":
                turn = connection.active_turn
                if turn is None:
                    connection.emit(
                        invalid_request_error_event(
                            "invalid_audio", "input audio buffer is empty"
                        )
                    )
                    return
                await connection.emit_for_turn(
                    turn, input_audio_buffer_committed_event(turn.item_id)
                )
                turn.append_silence(self.commit_padding_ms)
                turn.close()
                connection.finish_active_turn()
            elif event_type == "input_audio_buffer.clear":
                turn = connection.active_turn
                if turn is not None:
                    turn.close()
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
