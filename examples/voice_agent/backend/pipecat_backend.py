# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Minimal Pipecat/Dynamo ASR backend for Dynamo's realtime bridge."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

PIPECAT_SRC = os.environ.get("PIPECAT_SRC", "/home/nealv/dynamo/pipecat/src")
if os.path.isdir(PIPECAT_SRC):
    sys.path.insert(0, PIPECAT_SRC)

from asr_protocol import AsrRequest, AsrTranscript
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

from dynamo.runtime import DistributedRuntime

DEFAULT_CONNECT = "127.0.0.1:8081"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_MAX_FRAME_BYTES = 16 * 1024 * 1024
DEFAULT_TRANSCRIPT_DRAIN_TIMEOUT = 0.25
DEFAULT_ASR_ENDPOINT = "voice_agent.asr.transcribe"
SESSION_ID = "voice-agent-pipecat-asr"


@dataclass
class TranscriptResult:
    text: str
    final: bool


@dataclass
class PipelineFailure:
    message: str


def parse_host_port(raw: str) -> tuple[str, int]:
    host, sep, port = raw.rpartition(":")
    if not sep or not host:
        raise argparse.ArgumentTypeError(
            f"expected HOST:PORT for bridge address, got {raw!r}"
        )
    try:
        return host, int(port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid port in {raw!r}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connect",
        default=os.environ.get("DYN_REALTIME_FRONTEND_BACKEND_ADDR", DEFAULT_CONNECT),
        type=parse_host_port,
        metavar="HOST:PORT",
        help=f"Frontend backend-bridge address [default: {DEFAULT_CONNECT}].",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=1.0,
        help="Seconds to wait before reconnecting to the frontend bridge.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Input PCM sample rate [default: {DEFAULT_SAMPLE_RATE}].",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
        help=f"Input PCM channel count [default: {DEFAULT_CHANNELS}].",
    )
    parser.add_argument(
        "--pipeline-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the Pipecat ASR pipeline to emit a transcript.",
    )
    parser.add_argument(
        "--transcript-drain-timeout",
        type=float,
        default=DEFAULT_TRANSCRIPT_DRAIN_TIMEOUT,
        help=(
            "Seconds to wait for additional final transcript frames after the "
            "first final frame."
        ),
    )
    parser.add_argument(
        "--max-frame-bytes",
        type=int,
        default=DEFAULT_MAX_FRAME_BYTES,
        help="Maximum newline-delimited JSON frame size accepted from the frontend.",
    )
    parser.add_argument(
        "--asr-endpoint",
        default=os.environ.get("DYN_ASR_ENDPOINT", DEFAULT_ASR_ENDPOINT),
        help=f"Dynamo ASR endpoint path [default: {DEFAULT_ASR_ENDPOINT}].",
    )
    parser.add_argument(
        "--asr-language",
        default=os.environ.get("DYN_ASR_LANGUAGE", "en-US"),
        help="ASR language code [default: en-US].",
    )
    parser.add_argument(
        "--asr-model",
        default=os.environ.get("DYN_ASR_MODEL"),
        help="Optional ASR model name to pass to the Dynamo ASR worker.",
    )
    args = parser.parse_args()
    if args.sample_rate < 1:
        parser.error("--sample-rate must be >= 1")
    if args.channels < 1:
        parser.error("--channels must be >= 1")
    if args.pipeline_timeout <= 0:
        parser.error("--pipeline-timeout must be > 0")
    if args.transcript_drain_timeout <= 0:
        parser.error("--transcript-drain-timeout must be > 0")
    if args.max_frame_bytes < 1:
        parser.error("--max-frame-bytes must be >= 1")
    return args


def annotated_event(frame: int, event: dict[str, Any]) -> dict[str, Any]:
    return {"id": str(frame), "data": event}


def response_payload(response_id: str, status: str) -> dict[str, Any]:
    return {
        "audio": None,
        "id": response_id,
        "max_output_tokens": "inf",
        "object": "realtime.response",
        "output": [],
        "output_modalities": ["text"],
        "status": status,
        "status_details": None,
        "usage": None,
    }


def error_event(
    frame: int,
    message: str,
    code: str = "pipecat_asr_backend_error",
) -> dict[str, Any]:
    return annotated_event(
        frame,
        {
            "type": "error",
            "event_id": f"event_{SESSION_ID}_{frame}",
            "error": {
                "type": "server_error",
                "code": code,
                "message": message,
                "param": None,
                "event_id": None,
            },
        },
    )


def unsupported_event(frame: int, event: dict[str, Any]) -> dict[str, Any]:
    event_type = event.get("type", "<missing type>")
    return error_event(
        frame,
        f"voice-agent Pipecat ASR backend does not support client event {event_type}",
        code="pipecat_asr_backend_unsupported",
    )


async def write_chunk(writer: asyncio.StreamWriter, chunk: dict[str, Any]) -> None:
    writer.write(json.dumps(chunk, separators=(",", ":")).encode("utf-8"))
    writer.write(b"\n")
    await writer.drain()


class TranscriptCollector(FrameProcessor):
    """Collect transcript frames emitted by the Pipecat STT service."""

    def __init__(
        self,
        output_queue: asyncio.Queue[TranscriptResult | PipelineFailure],
    ):
        super().__init__()
        self._output_queue = output_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self._output_queue.put(TranscriptResult(frame.text, final=True))
        elif isinstance(frame, InterimTranscriptionFrame):
            await self._output_queue.put(TranscriptResult(frame.text, final=False))
        elif isinstance(frame, ErrorFrame):
            await self._output_queue.put(PipelineFailure(frame.error))

        await self.push_frame(frame, direction)


def runtime_from_env() -> DistributedRuntime:
    return DistributedRuntime(
        asyncio.get_running_loop(),
        os.environ.get("DYN_DISCOVERY_BACKEND", "etcd"),
        os.environ.get("DYN_REQUEST_PLANE", "tcp"),
    )


def parse_asr_transcript(chunk: Any) -> AsrTranscript:
    raw = chunk.data() if hasattr(chunk, "data") else chunk
    if isinstance(raw, AsrTranscript):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return AsrTranscript.model_validate_json(raw)
    return AsrTranscript.model_validate(raw)


class DynamoASRService(SegmentedSTTService):
    """Segmented Pipecat STT service backed by a Dynamo ASR endpoint."""

    def __init__(
        self,
        *,
        endpoint_path: str,
        sample_rate: int,
        channels: int,
        language: str,
        model: str | None,
    ):
        super().__init__(
            sample_rate=sample_rate,
            audio_passthrough=False,
            settings=STTSettings(model=model, language=language),
            ttfs_p99_latency=1.0,
        )
        self._endpoint_path = endpoint_path
        self._channels = channels
        self._language = language
        self._model = model
        self._runtime: DistributedRuntime | None = None
        self._client: Any | None = None

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        self._runtime = runtime_from_env()
        endpoint = self._runtime.endpoint(self._endpoint_path)
        self._client = await endpoint.client()
        await self._client.wait_for_instances()
        return self._client

    async def start(self, frame):
        await super().start(frame)
        await self._ensure_client()

    async def run_stt(self, audio: bytes):
        try:
            client = await self._ensure_client()
            request = AsrRequest(
                audio_b64=base64.b64encode(audio).decode("ascii"),
                encoding="wav",
                sample_rate_hz=self._sample_rate,
                channels=self._channels,
                language=self._language,
                model=self._model,
            )
            stream = await client.generate(request.model_dump_json())
            async for chunk in stream:
                transcript = parse_asr_transcript(chunk)
                if not transcript.text:
                    continue

                result = transcript.model_dump()
                if transcript.is_final:
                    yield TranscriptionFrame(
                        transcript.text,
                        self._user_id,
                        time_now_iso8601(),
                        None,
                        result,
                    )
                else:
                    yield InterimTranscriptionFrame(
                        transcript.text,
                        self._user_id,
                        time_now_iso8601(),
                        None,
                        result,
                    )
        except Exception as exc:
            yield ErrorFrame(
                error=f"Dynamo ASR endpoint {self._endpoint_path} failed: {exc}"
            )


class PipecatASRSession:
    def __init__(self, args: argparse.Namespace):
        self._sample_rate = args.sample_rate
        self._channels = args.channels
        self._timeout = args.pipeline_timeout
        self._drain_timeout = args.transcript_drain_timeout
        self._output_queue: asyncio.Queue[
            TranscriptResult | PipelineFailure
        ] = asyncio.Queue()
        self._runner_task: asyncio.Task | None = None

        stt = DynamoASRService(
            endpoint_path=args.asr_endpoint,
            sample_rate=args.sample_rate,
            channels=args.channels,
            language=args.asr_language,
            model=args.asr_model,
        )
        pipeline = Pipeline([stt, TranscriptCollector(self._output_queue)])
        self._task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=args.sample_rate,
                audio_out_sample_rate=args.sample_rate,
            ),
            enable_rtvi=False,
            enable_turn_tracking=False,
            idle_timeout_secs=None,
        )
        self._task.event_handler("on_pipeline_error")(self._on_pipeline_error)
        self._runner = PipelineRunner(handle_sigint=False)

    async def _on_pipeline_error(
        self,
        _task: PipelineTask,
        frame: ErrorFrame,
    ) -> None:
        await self._output_queue.put(PipelineFailure(frame.error))

    async def start(self) -> None:
        if self._runner_task is None:
            self._runner_task = asyncio.create_task(self._runner.run(self._task))
            await asyncio.sleep(0)

    async def transcribe_audio(self, audio: bytes) -> TranscriptResult:
        await self.start()
        while not self._output_queue.empty():
            self._output_queue.get_nowait()

        await self._task.queue_frame(VADUserStartedSpeakingFrame())
        await self._task.queue_frame(
            InputAudioRawFrame(
                audio=audio,
                sample_rate=self._sample_rate,
                num_channels=self._channels,
            )
        )
        await self._task.queue_frame(VADUserStoppedSpeakingFrame())

        text_parts: list[str] = []
        while True:
            timeout = self._drain_timeout if text_parts else self._timeout
            try:
                result = await asyncio.wait_for(
                    self._output_queue.get(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                if text_parts:
                    return TranscriptResult("\n".join(text_parts), final=True)
                raise
            if isinstance(result, PipelineFailure):
                raise RuntimeError(result.message)
            if result.final:
                text_parts.append(result.text)

    def audio_duration_seconds(self, audio: bytes) -> float:
        return len(audio) / (self._sample_rate * self._channels * 2)

    async def close(self) -> None:
        if self._runner_task is None:
            return
        await self._task.cancel(reason="realtime backend connection closed")
        try:
            await asyncio.wait_for(self._runner_task, timeout=self._timeout)
        except asyncio.TimeoutError:
            self._runner_task.cancel()
        self._runner_task = None


async def handle_event(
    writer: asyncio.StreamWriter,
    session: PipecatASRSession,
    event: dict[str, Any],
    frame: int,
) -> int:
    event_type = event.get("type")

    if event_type == "session.update":
        frame += 1
        await write_chunk(
            writer,
            annotated_event(
                frame,
                {
                    "type": "session.updated",
                    "event_id": f"event_{SESSION_ID}_{frame}",
                    "session": event.get("session", {}),
                },
            ),
        )
        return frame

    if event_type == "input_audio_buffer.append":
        response_id = f"resp_{SESSION_ID}_{frame}"
        item_id = f"item_{SESSION_ID}_{frame}"

        frame += 1
        await write_chunk(
            writer,
            annotated_event(
                frame,
                {
                    "type": "response.created",
                    "event_id": f"event_{SESSION_ID}_{frame}",
                    "response": response_payload(response_id, "in_progress"),
                },
            ),
        )

        audio_base64 = event.get("audio", "")
        if not isinstance(audio_base64, str):
            audio_base64 = ""

        try:
            input_audio = base64.b64decode(audio_base64, validate=False)
            transcript = await session.transcribe_audio(input_audio)
        except Exception as exc:
            frame += 1
            await write_chunk(
                writer,
                error_event(frame, f"Pipecat ASR pipeline failed: {exc}"),
            )
            return frame

        audio_seconds = session.audio_duration_seconds(input_audio)

        frame += 1
        await write_chunk(
            writer,
            annotated_event(
                frame,
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "event_id": f"event_{SESSION_ID}_{frame}",
                    "item_id": item_id,
                    "content_index": 0,
                    "transcript": transcript.text,
                    "logprobs": None,
                    "usage": {
                        "type": "duration",
                        "seconds": audio_seconds,
                    },
                },
            ),
        )

        frame += 1
        await write_chunk(
            writer,
            annotated_event(
                frame,
                {
                    "type": "response.output_text.delta",
                    "event_id": f"event_{SESSION_ID}_{frame}",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": transcript.text,
                },
            ),
        )

        frame += 1
        await write_chunk(
            writer,
            annotated_event(
                frame,
                {
                    "type": "response.output_text.done",
                    "event_id": f"event_{SESSION_ID}_{frame}",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": transcript.text,
                },
            ),
        )

        frame += 1
        await write_chunk(
            writer,
            annotated_event(
                frame,
                {
                    "type": "response.done",
                    "event_id": f"event_{SESSION_ID}_{frame}",
                    "response": response_payload(response_id, "completed"),
                },
            ),
        )
        return frame

    frame += 1
    await write_chunk(writer, unsupported_event(frame, event))
    return frame


async def serve_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    args: argparse.Namespace,
) -> None:
    frame = 0
    session = PipecatASRSession(args)
    try:
        while line := await reader.readline():
            event = json.loads(line)
            frame = await handle_event(writer, session, event, frame)
    finally:
        await session.close()


async def run_backend(args: argparse.Namespace) -> None:
    host, port = args.connect
    while True:
        try:
            reader, writer = await asyncio.open_connection(
                host,
                port,
                limit=args.max_frame_bytes,
            )
        except OSError as exc:
            print(
                f"waiting for frontend bridge at tcp://{host}:{port}: {exc}",
                file=sys.stderr,
            )
            await asyncio.sleep(args.reconnect_delay)
            continue

        print(f"Connected Pipecat ASR backend to tcp://{host}:{port}")
        try:
            await serve_connection(reader, writer, args)
        except Exception as exc:
            print(f"backend connection failed: {exc}", file=sys.stderr)
        finally:
            writer.close()
            await writer.wait_closed()
            print("backend connection closed", file=sys.stderr)

        await asyncio.sleep(args.reconnect_delay)


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(run_backend(args))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
