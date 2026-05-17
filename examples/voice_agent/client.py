# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal WebSocket client for Dynamo's experimental /v1/realtime endpoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import wave
from pathlib import Path
from typing import Any

try:
    import websockets
except ImportError as exc:  # pragma: no cover - exercised by humans.
    raise SystemExit(
        "Missing dependency: websockets\n"
        "Install it with: uv pip install --python /path/to/python "
        "-r examples/voice_agent/requirements.txt"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8080/v1/realtime",
        help="Dynamo realtime WebSocket URL.",
    )
    parser.add_argument(
        "--model", default="echo", help="Realtime model name for session.update."
    )
    parser.add_argument(
        "--text",
        default="hello realtime",
        help="Text to base64-encode into input_audio_buffer.append for the echo backend.",
    )
    parser.add_argument(
        "--audio-file",
        type=Path,
        help="16-bit PCM WAV file to send as input_audio_buffer.append audio.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the next server frame.",
    )
    return parser.parse_args()


def load_audio(args: argparse.Namespace) -> bytes:
    if args.audio_file is None:
        return args.text.encode("utf-8")

    with wave.open(str(args.audio_file), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("--audio-file must be 16-bit PCM WAV")
        if wav.getnchannels() != 1:
            raise ValueError("--audio-file must be mono PCM WAV")
        return wav.readframes(wav.getnframes())


async def recv_event(ws: Any, timeout: float) -> dict[str, Any]:
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError("timed out waiting for a realtime event") from exc

    if isinstance(raw, bytes):
        raise ValueError("Dynamo realtime endpoint should not send binary frames")

    event = json.loads(raw)
    if event.get("type") == "error":
        error = event.get("error") or {}
        message = error.get("message") or event
        raise RuntimeError(f"server error event: {message}")
    return event


async def expect_event(ws: Any, event_type: str, timeout: float) -> dict[str, Any]:
    event = await recv_event(ws, timeout)
    actual = event.get("type")
    if actual != event_type:
        raise RuntimeError(f"expected {event_type}, got {actual}: {event}")
    return event


async def run_client(args: argparse.Namespace) -> None:
    audio = base64.b64encode(load_audio(args)).decode("ascii")

    async with websockets.connect(args.url) as ws:
        await expect_event(ws, "session.created", args.timeout)

        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"type": "realtime", "model": args.model},
                }
            )
        )
        await expect_event(ws, "session.updated", args.timeout)

        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": audio,
                }
            )
        )

        echoed_audio = []
        text_output = []
        input_transcript = None
        while True:
            event = await recv_event(ws, args.timeout)
            event_type = event.get("type")
            if event_type == "response.output_audio.delta":
                echoed_audio.append(event.get("delta", ""))
            elif event_type == "response.output_text.delta":
                text_output.append(event.get("delta", ""))
            elif event_type == "conversation.item.input_audio_transcription.completed":
                input_transcript = event.get("transcript")
            elif event_type == "response.done":
                if text_output:
                    print("".join(text_output))
                elif input_transcript is not None:
                    print(input_transcript)
                else:
                    decoded = base64.b64decode("".join(echoed_audio)).decode(
                        "utf-8", errors="replace"
                    )
                    print(decoded)
                return


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(run_client(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"client failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
