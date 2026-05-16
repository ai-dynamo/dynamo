# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal WebSocket client for Dynamo's current experimental /v1/realtime endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
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
        "--model", default="echo", help="Model name to send in the request."
    )
    parser.add_argument("--text", default="hello realtime", help="User text to send.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the next server frame.",
    )
    return parser.parse_args()


def extract_delta_text(message: dict[str, Any]) -> tuple[str, bool]:
    data = message.get("data", message)
    choices = data.get("choices") or []
    text_parts: list[str] = []
    finished = False

    for choice in choices:
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        if choice.get("finish_reason") == "stop":
            finished = True

    return "".join(text_parts), finished


async def run_client(args: argparse.Namespace) -> None:
    request = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.text}],
    }

    async with websockets.connect(args.url) as ws:
        await ws.send(json.dumps(request))

        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"timed out waiting for a frame from {args.url}"
                ) from exc

            if isinstance(raw, bytes):
                continue

            message = json.loads(raw)
            delta, finished = extract_delta_text(message)
            if delta:
                print(delta, end="", flush=True)
            if finished:
                print()
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
