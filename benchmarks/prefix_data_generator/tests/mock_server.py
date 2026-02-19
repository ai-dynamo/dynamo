#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple mock server to receive and log requests from aiperf loadgen.

Prints each request with timestamp, useful for debugging request patterns,
verifying session ordering, and inspecting delays between requests.

Usage:
    python mock_server.py --port 8000

Then run aiperf pointing to this server:
    python agent_benchmark.py --input-dataset trace.jsonl --url http://localhost:8000
"""

import argparse
import asyncio
import json
import time
from datetime import datetime

from aiohttp import web


class RequestLogger:
    """Tracks and logs incoming requests with timing information."""

    def __init__(self, output_file: str | None = None, payload_file: str | None = None):
        self.request_count = 0
        self.start_time = None
        self.session_last_seen: dict[str, float] = {}
        self.output_file = output_file
        self.payload_file = payload_file
        self._file_handle = None
        self._payload_handle = None

        if output_file:
            self._file_handle = open(output_file, "w")
        if payload_file:
            self._payload_handle = open(payload_file, "w")

    def close(self):
        """Close the output files if open."""
        if self._file_handle:
            self._file_handle.close()
        if self._payload_handle:
            self._payload_handle.close()

    def log_request(self, request_data: dict, headers: dict) -> None:
        """Log a request with timestamp and relevant metadata."""
        now = time.time()
        if self.start_time is None:
            self.start_time = now

        self.request_count += 1
        elapsed = now - self.start_time
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Extract useful fields
        model = request_data.get("model", "unknown")
        messages = request_data.get("messages", [])
        # Support both max_completion_tokens (new) and max_tokens (legacy)
        max_tokens = request_data.get(
            "max_completion_tokens", request_data.get("max_tokens", "N/A")
        )

        # Check for session tracking via x-correlation-id header
        correlation_id = headers.get(
            "x-correlation-id", headers.get("X-Correlation-Id")
        )

        # Calculate delay since last request from same session
        session_delay = None
        if correlation_id and correlation_id in self.session_last_seen:
            session_delay = (now - self.session_last_seen[correlation_id]) * 1000  # ms
        if correlation_id:
            self.session_last_seen[correlation_id] = now

        # Build log line
        log_parts = [
            f"[{timestamp}]",
            f"#{self.request_count:4d}",
            f"elapsed={elapsed:7.2f}s",
            f"model={model}",
            f"msgs={len(messages)}",
            f"max_tokens={max_tokens}",
        ]

        if correlation_id:
            # Truncate for display
            short_id = correlation_id[:8] if len(correlation_id) > 8 else correlation_id
            log_parts.append(f"session={short_id}")

        if session_delay is not None:
            log_parts.append(f"delay={session_delay:.0f}ms")

        log_line = " | ".join(log_parts)
        print(log_line)

        if self._file_handle:
            self._file_handle.write(log_line + "\n")
            self._file_handle.flush()

        # Dump full payload to separate file
        if self._payload_handle:
            payload_entry = {
                "request_number": self.request_count,
                "timestamp": timestamp,
                "elapsed_s": elapsed,
                "correlation_id": correlation_id,
                "headers": {
                    k: v
                    for k, v in headers.items()
                    if k.lower() not in ("authorization", "accept")
                },
                "body": request_data,
            }
            self._payload_handle.write(json.dumps(payload_entry) + "\n")
            self._payload_handle.flush()


logger: RequestLogger = None  # Initialized in main()


async def health_check(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({"status": "ok"})


async def chat_completions(request: web.Request) -> web.StreamResponse:
    """Handle /v1/chat/completions requests."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Log the request
    headers = dict(request.headers)
    logger.log_request(body, headers)

    # Check if streaming is requested
    stream = body.get("stream", False)
    # Support both max_completion_tokens (new) and max_tokens (legacy)
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens", 10))

    if stream:
        # Return a streaming response
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        # Send a minimal streaming response
        # Each chunk simulates a token being generated
        for i in range(max_tokens):
            chunk = {
                "id": f"chatcmpl-mock-{logger.request_count}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.get("model", "mock"),
                "choices": [
                    {
                        "index": 0,
                        "delta": (
                            {"content": "x"}
                            if i > 0
                            else {"role": "assistant", "content": ""}
                        ),
                        "finish_reason": None,
                    }
                ],
            }
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await asyncio.sleep(0.001)  # Small delay between tokens

        # Send final chunk with finish_reason
        final_chunk = {
            "id": f"chatcmpl-mock-{logger.request_count}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": body.get("model", "mock"),
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        await response.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response
    else:
        # Non-streaming response
        return web.json_response(
            {
                "id": f"chatcmpl-mock-{logger.request_count}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "mock"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "x" * max_tokens},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": max_tokens,
                    "total_tokens": 10 + max_tokens,
                },
            }
        )


async def models_list(request: web.Request) -> web.Response:
    """Handle /v1/models endpoint."""
    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "id": "mock-model",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mock",
                }
            ],
        }
    )


def create_app() -> web.Application:
    """Create the aiohttp application."""
    app = web.Application()
    app.router.add_get("/health", health_check)
    app.router.add_get("/v1/models", models_list)
    app.router.add_post("/v1/chat/completions", chat_completions)
    return app


def main():
    global logger

    parser = argparse.ArgumentParser(
        description="Mock server to receive and log aiperf requests"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mock_server_logs.txt",
        help="Output file for request logs (default: mock_server_logs.txt)",
    )
    parser.add_argument(
        "--payload-output",
        type=str,
        default="mock_server_payloads.jsonl",
        help="Output file for full request payloads in JSONL format (default: mock_server_payloads.jsonl)",
    )
    args = parser.parse_args()

    # Initialize logger with output files
    logger = RequestLogger(output_file=args.output, payload_file=args.payload_output)

    print(f"Starting mock server on {args.host}:{args.port}")
    print(f"Logging requests to: {args.output}")
    print(f"Dumping payloads to: {args.payload_output}")
    print("Waiting for requests from aiperf...")
    print("-" * 80)

    try:
        app = create_app()
        web.run_app(app, host=args.host, port=args.port, print=None)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
