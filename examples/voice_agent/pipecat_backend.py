# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Pipecat-backed echo backend for Dynamo's realtime bridge."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import io
import json
import os
import sys
from typing import Any

os.environ.setdefault("LOGURU_LEVEL", "WARNING")

DEFAULT_CONNECT = "127.0.0.1:8081"
DEFAULT_CHUNK_LEN = 64
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1
SESSION_ID = "voice-agent-pipecat-backend"

_IMPORT_STDERR = io.StringIO()
try:
    with contextlib.redirect_stderr(_IMPORT_STDERR):
        from pipecat.frames.frames import (
            Frame,
            InputAudioRawFrame,
            OutputAudioRawFrame,
        )
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
except ImportError as exc:  # pragma: no cover - exercised by humans.
    import_stderr = _IMPORT_STDERR.getvalue().strip()
    detail = f"\n\nImport stderr:\n{import_stderr}" if import_stderr else ""
    raise SystemExit(
        "Missing Pipecat dependency. Install the local checkout into the requested venv, "
        "for example:\n"
        "  uv pip install --python /mnt/scratch/nealv/venvs/dynamo_realtime/bin/python "
        "--no-deps -e /home/nealv/dynamo/pipecat\n"
        "  uv pip install --python /mnt/scratch/nealv/venvs/dynamo_realtime/bin/python "
        "loguru pydantic pydantic-core annotated-types typing-inspection wait_for2 "
        "docstring_parser aiofiles aiohttp numpy pyloudnorm resampy soxr protobuf "
        "Markdown Pillow openai nltk"
        f"{detail}"
    ) from exc


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
        "--chunk-len",
        type=int,
        default=DEFAULT_CHUNK_LEN,
        help="Maximum base64 characters per response.output_audio.delta event.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="PCM sample rate to use when wrapping input bytes as Pipecat audio.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
        help="PCM channel count to use when wrapping input bytes as Pipecat audio.",
    )
    parser.add_argument(
        "--pipeline-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the Pipecat pipeline to emit output audio.",
    )
    args = parser.parse_args()
    if args.chunk_len < 1:
        parser.error("--chunk-len must be >= 1")
    if args.sample_rate < 1:
        parser.error("--sample-rate must be >= 1")
    if args.channels < 1:
        parser.error("--channels must be >= 1")
    if args.pipeline_timeout <= 0:
        parser.error("--pipeline-timeout must be > 0")
    return args


def annotated_event(frame: int, event: dict[str, Any]) -> dict[str, Any]:
    return {"id": str(frame), "data": event}


def echo_response(response_id: str, status: str) -> dict[str, Any]:
    return {
        "audio": None,
        "id": response_id,
        "max_output_tokens": "inf",
        "object": "realtime.response",
        "output": [],
        "output_modalities": ["audio"],
        "status": status,
        "status_details": None,
        "usage": None,
    }


def error_event(
    frame: int,
    message: str,
    code: str = "pipecat_backend_error",
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
        f"voice-agent Pipecat backend does not support client event {event_type}",
        code="pipecat_backend_unsupported",
    )


async def write_chunk(writer: asyncio.StreamWriter, chunk: dict[str, Any]) -> None:
    writer.write(json.dumps(chunk, separators=(",", ":")).encode("utf-8"))
    writer.write(b"\n")
    await writer.drain()


class EchoAudioProcessor(FrameProcessor):
    """Tiny Pipecat processor that turns input audio frames into output audio frames."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=frame.audio,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                ),
                direction,
            )
            return

        await self.push_frame(frame, direction)


class OutputCollector(FrameProcessor):
    """Collect output audio frames emitted by the Pipecat pipeline."""

    def __init__(self, output_queue: asyncio.Queue[OutputAudioRawFrame]):
        super().__init__()
        self._output_queue = output_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputAudioRawFrame):
            await self._output_queue.put(frame)

        await self.push_frame(frame, direction)


class PipecatEchoSession:
    def __init__(self, *, sample_rate: int, channels: int, timeout: float):
        self._sample_rate = sample_rate
        self._channels = channels
        self._timeout = timeout
        self._output_queue: asyncio.Queue[OutputAudioRawFrame] = asyncio.Queue()
        self._runner_task: asyncio.Task | None = None

        pipeline = Pipeline(
            [
                EchoAudioProcessor(),
                OutputCollector(self._output_queue),
            ]
        )
        self._task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=sample_rate,
                audio_out_sample_rate=sample_rate,
            ),
            enable_rtvi=False,
            enable_turn_tracking=False,
            idle_timeout_secs=None,
        )
        self._runner = PipelineRunner(handle_sigint=False)

    async def start(self) -> None:
        if self._runner_task is None:
            self._runner_task = asyncio.create_task(self._runner.run(self._task))
            await asyncio.sleep(0)

    async def process_audio(self, audio: bytes) -> OutputAudioRawFrame:
        await self.start()
        await self._task.queue_frame(
            InputAudioRawFrame(
                audio=audio,
                sample_rate=self._sample_rate,
                num_channels=self._channels,
            )
        )
        return await asyncio.wait_for(self._output_queue.get(), timeout=self._timeout)

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
    session: PipecatEchoSession,
    event: dict[str, Any],
    frame: int,
    chunk_len: int,
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
                    "response": echo_response(response_id, "in_progress"),
                },
            ),
        )

        audio_base64 = event.get("audio", "")
        if not isinstance(audio_base64, str):
            audio_base64 = ""

        try:
            input_audio = base64.b64decode(audio_base64, validate=False)
            output_frame = await session.process_audio(input_audio)
        except Exception as exc:
            frame += 1
            await write_chunk(
                writer,
                error_event(frame, f"Pipecat pipeline failed: {exc}"),
            )
            return frame

        output_audio = base64.b64encode(output_frame.audio).decode("ascii")
        for start in range(0, len(output_audio), chunk_len):
            frame += 1
            await write_chunk(
                writer,
                annotated_event(
                    frame,
                    {
                        "type": "response.output_audio.delta",
                        "event_id": f"event_{SESSION_ID}_{frame}",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": output_audio[start : start + chunk_len],
                    },
                ),
            )

        frame += 1
        await write_chunk(
            writer,
            annotated_event(
                frame,
                {
                    "type": "response.output_audio.done",
                    "event_id": f"event_{SESSION_ID}_{frame}",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
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
                    "response": echo_response(response_id, "completed"),
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
    session = PipecatEchoSession(
        sample_rate=args.sample_rate,
        channels=args.channels,
        timeout=args.pipeline_timeout,
    )
    try:
        while line := await reader.readline():
            event = json.loads(line)
            frame = await handle_event(writer, session, event, frame, args.chunk_len)
    finally:
        await session.close()


async def run_backend(args: argparse.Namespace) -> None:
    host, port = args.connect
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port)
        except OSError as exc:
            print(
                f"waiting for frontend bridge at tcp://{host}:{port}: {exc}",
                file=sys.stderr,
            )
            await asyncio.sleep(args.reconnect_delay)
            continue

        print(f"Connected Pipecat realtime backend to tcp://{host}:{port}")
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
