# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python echo backend for the voice-agent realtime bridge."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

DEFAULT_CONNECT = "127.0.0.1:8081"
DEFAULT_CHUNK_LEN = 64
SESSION_ID = "voice-agent-python-backend"


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
    args = parser.parse_args()
    if args.chunk_len < 1:
        parser.error("--chunk-len must be >= 1")
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


def unsupported_event(frame: int, event: dict[str, Any]) -> dict[str, Any]:
    event_type = event.get("type", "<missing type>")
    return annotated_event(
        frame,
        {
            "type": "error",
            "event_id": f"event_{SESSION_ID}_{frame}",
            "error": {
                "type": "invalid_request_error",
                "code": "echo_backend_unsupported",
                "message": (
                    "voice-agent echo backend does not support client event "
                    f"{event_type}"
                ),
                "param": None,
                "event_id": None,
            },
        },
    )


async def write_chunk(
    writer: asyncio.StreamWriter, chunk: dict[str, Any]
) -> None:
    writer.write(json.dumps(chunk, separators=(",", ":")).encode("utf-8"))
    writer.write(b"\n")
    await writer.drain()


async def handle_event(
    writer: asyncio.StreamWriter,
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

        audio = event.get("audio", "")
        if not isinstance(audio, str):
            audio = ""

        for start in range(0, len(audio), chunk_len):
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
                        "delta": audio[start : start + chunk_len],
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
    chunk_len: int,
) -> None:
    frame = 0
    while line := await reader.readline():
        event = json.loads(line)
        frame = await handle_event(writer, event, frame, chunk_len)


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

        print(f"Connected realtime backend to tcp://{host}:{port}")
        try:
            await serve_connection(reader, writer, args.chunk_len)
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
