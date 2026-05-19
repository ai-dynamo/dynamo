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
        "-r examples/voice_agent/client/requirements.txt"
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
        help="16-bit mono PCM WAV file to read as input_audio_buffer.append PCM audio.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.0,
        help=(
            "Split --audio-file into multiple input_audio_buffer.append events "
            "of this many seconds before sending input_audio_buffer.commit. "
            "0 sends the whole file as one append event."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the next server frame.",
    )
    return parser.parse_args()


def load_audio_chunks(args: argparse.Namespace) -> list[bytes]:
    if args.audio_file is None:
        return [args.text.encode("utf-8")]

    with wave.open(str(args.audio_file), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("--audio-file must be 16-bit PCM WAV")
        if wav.getnchannels() != 1:
            raise ValueError("--audio-file must be mono PCM WAV")
        if args.chunk_seconds <= 0:
            return [wav.readframes(wav.getnframes())]

        frames_per_chunk = max(1, int(wav.getframerate() * args.chunk_seconds))
        chunks = []
        while chunk := wav.readframes(frames_per_chunk):
            chunks.append(chunk)
        return chunks


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


class StreamingPrinter:
    def __init__(self):
        self.printed = False
        self.ends_with_newline = True

    def write(self, text: str) -> None:
        if not text:
            return
        sys.stdout.write(text)
        sys.stdout.flush()
        self.printed = True
        self.ends_with_newline = text.endswith("\n")

    def finish_turn(self) -> None:
        if self.printed and not self.ends_with_newline:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.ends_with_newline = True


async def run_client(args: argparse.Namespace) -> None:
    audio_chunks = load_audio_chunks(args)
    printer = StreamingPrinter()

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

        for audio in audio_chunks:
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio).decode("ascii"),
                    }
                )
            )

        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        audio_base64 = ""
        streamed_output = False
        input_transcript = None
        while True:
            event = await recv_event(ws, args.timeout)
            event_type = event.get("type")
            if event_type == "response.output_audio.delta":
                audio_base64 += event.get("delta", "")
                decode_len = (len(audio_base64) // 4) * 4
                if decode_len:
                    decoded = base64.b64decode(
                        audio_base64[:decode_len],
                        validate=False,
                    ).decode("utf-8", errors="replace")
                    printer.write(decoded)
                    streamed_output = True
                    audio_base64 = audio_base64[decode_len:]
            elif event_type == "response.output_text.delta":
                printer.write(event.get("delta", ""))
                streamed_output = True
            elif event_type == "conversation.item.input_audio_transcription.completed":
                input_transcript = event.get("transcript")
            elif event_type == "response.done":
                if audio_base64:
                    padded_audio = audio_base64 + "=" * (-len(audio_base64) % 4)
                    decoded = base64.b64decode(padded_audio).decode(
                        "utf-8", errors="replace"
                    )
                    printer.write(decoded)
                    streamed_output = True
                if streamed_output:
                    printer.finish_turn()
                elif input_transcript is not None:
                    print(input_transcript)
                break


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
