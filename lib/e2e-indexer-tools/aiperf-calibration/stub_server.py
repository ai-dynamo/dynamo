#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trivially fast OpenAI-compatible chat-completions stub for load-generator calibration.

It never parses the JSON body: it reads `max_completion_tokens`/`max_tokens` and `"stream"`
with byte searches and answers immediately. Streaming responses carry one token per SSE event
(like Dynamo's frontend), a usage chunk, and `[DONE]`. With --itl-ms 0 the whole stream is one
write with Content-Length; with --itl-ms > 0 it is chunked and paced.

Several processes share the port through SO_REUSEPORT; each can be pinned to one CPU.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import socket
import sys
import time

MAX_TOKENS_RE = re.compile(rb'"max(?:_completion)?_tokens"\s*:\s*(\d+)')
STREAM_RE = re.compile(rb'"stream"\s*:\s*true')

ROLE_EVENT = (
    b'data: {"id":"stub","object":"chat.completion.chunk","created":0,"model":"stub",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
)
TOKEN_EVENT = (
    b'data: {"id":"stub","object":"chat.completion.chunk","created":0,"model":"stub",'
    b'"choices":[{"index":0,"delta":{"content":" tok"},"finish_reason":null}]}\n\n'
)
FINISH_EVENT = (
    b'data: {"id":"stub","object":"chat.completion.chunk","created":0,"model":"stub",'
    b'"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n'
)
DONE_EVENT = b"data: [DONE]\n\n"


def usage_event(prompt_tokens: int, completion_tokens: int) -> bytes:
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    payload = {
        "id": "stub",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "stub",
        "choices": [],
        "usage": usage,
    }
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def nonstream_body(prompt_tokens: int, completion_tokens: int) -> bytes:
    payload = {
        "id": "stub",
        "object": "chat.completion",
        "created": 0,
        "model": "stub",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": " tok" * completion_tokens},
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return json.dumps(payload, separators=(",", ":")).encode()


class Stats:
    requests = 0
    body_bytes = 0
    out_tokens = 0


_stream_cache: dict[int, bytes] = {}


def stream_middle(n: int) -> bytes:
    body = _stream_cache.get(n)
    if body is None:
        body = ROLE_EVENT + TOKEN_EVENT * n + FINISH_EVENT
        if len(_stream_cache) < 4096:
            _stream_cache[n] = body
    return body


class StubProtocol(asyncio.Protocol):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.buf = bytearray()
        self.transport: asyncio.Transport | None = None
        self.busy = False

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]
        sock = transport.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def data_received(self, data: bytes) -> None:
        self.buf += data
        if not self.busy:
            self._drain()

    def _drain(self) -> None:
        while True:
            head_end = self.buf.find(b"\r\n\r\n")
            if head_end < 0:
                return
            head = bytes(self.buf[:head_end])
            lines = head.split(b"\r\n")
            parts = lines[0].split(b" ")
            if len(parts) < 2:
                self.transport.close()
                return
            method, path = parts[0], parts[1]
            length = 0
            for line in lines[1:]:
                key, _, value = line.partition(b":")
                if key.strip().lower() == b"content-length":
                    length = int(value.strip())
            total = head_end + 4 + length
            if len(self.buf) < total:
                return
            body = bytes(self.buf[head_end + 4 : total])
            del self.buf[:total]
            if self.args.itl_ms > 0 and method == b"POST":
                self.busy = True
                asyncio.ensure_future(self._paced(path, body))
                return
            self.transport.write(self._respond(method, path, body))

    def _respond(self, method: bytes, path: bytes, body: bytes) -> bytes:
        if method == b"GET":
            if path.endswith(b"/models"):
                payload = b'{"object":"list","data":[{"id":"stub","object":"model"}]}'
                return _http(200, b"application/json", payload)
            if path.startswith(b"/metrics"):
                return _http(200, b"text/plain; version=0.0.4", b"stub_up 1\n")
            return _http(200, b"text/plain", b"ok")
        n, prompt_tokens, stream = self._parse(body)
        Stats.requests += 1
        Stats.body_bytes += len(body)
        Stats.out_tokens += n
        if not stream:
            return _http(200, b"application/json", nonstream_body(prompt_tokens, n))
        payload = stream_middle(n) + usage_event(prompt_tokens, n) + DONE_EVENT
        return _http(200, b"text/event-stream", payload)

    def _parse(self, body: bytes) -> tuple[int, int, bool]:
        m = MAX_TOKENS_RE.search(
            body, max(0, len(body) - 4096)
        ) or MAX_TOKENS_RE.search(body)
        n = int(m.group(1)) if m else self.args.default_osl
        n = max(1, min(n, self.args.max_osl))
        stream = STREAM_RE.search(body, max(0, len(body) - 4096)) is not None or (
            STREAM_RE.search(body) is not None
        )
        return n, len(body) // 4, stream

    async def _paced(self, path: bytes, body: bytes) -> None:
        n, prompt_tokens, stream = self._parse(body)
        Stats.requests += 1
        Stats.body_bytes += len(body)
        Stats.out_tokens += n
        t = self.transport
        if not stream:
            await asyncio.sleep(self.args.itl_ms * n / 1000.0)
            t.write(_http(200, b"application/json", nonstream_body(prompt_tokens, n)))
        else:
            t.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n"
            )
            t.write(_chunk(ROLE_EVENT))
            delay = self.args.itl_ms / 1000.0
            for _ in range(n):
                await asyncio.sleep(delay)
                if t.is_closing():
                    return
                t.write(_chunk(TOKEN_EVENT))
            t.write(
                _chunk(FINISH_EVENT + usage_event(prompt_tokens, n) + DONE_EVENT)
                + b"0\r\n\r\n"
            )
        self.busy = False
        if self.buf:
            self._drain()


def _chunk(data: bytes) -> bytes:
    return b"%x\r\n" % len(data) + data + b"\r\n"


def _http(status: int, ctype: bytes, payload: bytes) -> bytes:
    return (
        b"HTTP/1.1 %d OK\r\nContent-Type: %s\r\nContent-Length: %d\r\nConnection: keep-alive\r\n\r\n"
        % (status, ctype, len(payload))
        + payload
    )


async def report_loop(args: argparse.Namespace, idx: int) -> None:
    last = (time.monotonic(), 0, 0, os.times())
    while True:
        await asyncio.sleep(args.report_s)
        now = time.monotonic()
        t = os.times()
        dt = now - last[0]
        cpu = (t.user + t.system - last[3].user - last[3].system) / dt
        print(
            json.dumps(
                {
                    "proc": idx,
                    "t": time.time(),
                    "rps": round((Stats.requests - last[1]) / dt, 1),
                    "in_MBps": round((Stats.body_bytes - last[2]) / dt / 1e6, 2),
                    "cpu": round(cpu, 3),
                    "total": Stats.requests,
                }
            ),
            flush=True,
        )
        last = (now, Stats.requests, Stats.body_bytes, t)


def serve(args: argparse.Namespace, idx: int, cpu: int | None) -> None:
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    try:
        import uvloop  # type: ignore[import-not-found]

        uvloop.install()
    except ImportError:
        pass
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((args.host, args.port))
    sock.listen(4096)
    server = loop.run_until_complete(
        loop.create_server(lambda: StubProtocol(args), sock=sock)
    )
    if args.report_s > 0:
        loop.create_task(report_loop(args, idx))
    loop.add_signal_handler(signal.SIGTERM, loop.stop)
    try:
        loop.run_forever()
    finally:
        server.close()
        print(json.dumps({"proc": idx, "final_total": Stats.requests}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cpus", default="", help="comma list; one process per CPU")
    p.add_argument(
        "--procs", type=int, default=1, help="process count when --cpus is empty"
    )
    p.add_argument("--itl-ms", type=float, default=0.0)
    p.add_argument("--default-osl", type=int, default=128)
    p.add_argument("--max-osl", type=int, default=16384)
    p.add_argument("--report-s", type=float, default=5.0)
    args = p.parse_args()
    cpus = [int(c) for c in args.cpus.split(",") if c] or [None] * args.procs
    children = []
    for idx, cpu in enumerate(cpus):
        pid = os.fork()
        if pid == 0:
            serve(args, idx, cpu)
            os._exit(0)
        children.append(pid)

    def stop(*_: object) -> None:
        for pid in children:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for pid in children:
        os.waitpid(pid, 0)
    sys.exit(0)


if __name__ == "__main__":
    main()
