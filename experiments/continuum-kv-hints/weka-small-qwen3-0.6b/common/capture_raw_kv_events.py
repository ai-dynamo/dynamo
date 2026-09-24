#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture vLLM's raw ZMQ KV-event stream as JSONL."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from typing import Any

import msgspec
import zmq


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--topic", default="kv-events")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, args.topic.encode())
    socket.connect(args.endpoint)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        while running:
            if not dict(poller.poll(250)).get(socket):
                continue
            topic, sequence, payload = socket.recv_multipart()
            record = {
                "topic": topic.decode(errors="replace"),
                "sequence": int.from_bytes(sequence, byteorder="big"),
                "payload": _json_safe(msgspec.msgpack.decode(payload)),
            }
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()

    socket.close(linger=0)


if __name__ == "__main__":
    main()
