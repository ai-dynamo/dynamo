#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP shim from KVBM hub prefill dispatch to a Dynamo prefill endpoint.

The current hub dispatcher posts tokenized prefill work to `/v1/completions`.
For Experiment E we still want the prefill execution to go through
`python -m dynamo.vllm`, whose prefill worker registers a Dynamo internal
`ModelType.Prefill` endpoint rather than an OpenAI HTTP completions route.
This shim preserves the hub's HTTP contract and forwards accepted requests to
the configured Dynamo prefill endpoint as `PreprocessedRequest` dictionaries.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from typing import Any

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()

DEFAULT_MAX_BODY_BYTES = 1024 * 1024
DEFAULT_READ_TIMEOUT_SECONDS = 10.0
DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 300.0


class PublicHTTPError(Exception):
    def __init__(self, status: int, error: str):
        super().__init__(error)
        self.status = status
        self.error = error


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be positive")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _json_response(status: int, payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    reason = {
        200: "OK",
        400: "Bad Request",
        404: "Not Found",
        500: "Internal Server Error",
    }.get(status, "OK")
    headers = [
        f"HTTP/1.1 {status} {reason}",
        "content-type: application/json",
        f"content-length: {len(body)}",
        "connection: close",
        "",
        "",
    ]
    return "\r\n".join(headers).encode("ascii") + body


def _decode_json_object(body: bytes) -> dict[str, Any]:
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


async def _read_http_request(
    reader: asyncio.StreamReader,
    max_body_bytes: int,
    read_timeout_seconds: float,
) -> tuple[str, str, bytes]:
    try:
        header_blob = await asyncio.wait_for(
            reader.readuntil(b"\r\n\r\n"),
            timeout=read_timeout_seconds,
        )
    except TimeoutError as exc:
        raise PublicHTTPError(400, "request_timeout") from exc
    except asyncio.LimitOverrunError as exc:
        raise PublicHTTPError(400, "request_headers_too_large") from exc
    except asyncio.IncompleteReadError as exc:
        raise PublicHTTPError(400, "malformed_request") from exc

    header_text = header_blob.decode("iso-8859-1")
    lines = header_text.split("\r\n")
    try:
        method, path, _version = lines[0].split(" ", 2)
    except ValueError as exc:
        raise PublicHTTPError(400, "malformed_request") from exc

    content_length = 0
    for line in lines[1:]:
        if not line:
            continue
        name, _, value = line.partition(":")
        if name.lower() == "content-length":
            try:
                content_length = int(value.strip())
            except ValueError as exc:
                raise PublicHTTPError(400, "invalid_content_length") from exc
            if content_length < 0:
                raise PublicHTTPError(400, "invalid_content_length")
            if content_length > max_body_bytes:
                raise PublicHTTPError(400, "request_body_too_large")
    if content_length:
        try:
            body = await asyncio.wait_for(
                reader.readexactly(content_length),
                timeout=read_timeout_seconds,
            )
        except TimeoutError as exc:
            raise PublicHTTPError(400, "request_timeout") from exc
        except asyncio.IncompleteReadError as exc:
            raise PublicHTTPError(400, "incomplete_request_body") from exc
    else:
        body = b""
    return method.upper(), path, body


def _payload_to_preprocessed(payload: dict[str, Any], model: str) -> dict[str, Any]:
    prompt = payload.get("prompt")
    if not isinstance(prompt, list) or not all(isinstance(t, int) for t in prompt):
        raise ValueError("prompt must be a list of token ids")

    extra_args = {}
    if "kv_transfer_params" in payload:
        extra_args["kv_transfer_params"] = payload["kv_transfer_params"]

    max_tokens = payload.get("max_tokens", 1)
    if not isinstance(max_tokens, int) or max_tokens < 1:
        max_tokens = 1

    return {
        "model": payload.get("model") or model,
        "token_ids": prompt,
        "stop_conditions": {"max_tokens": max_tokens, "min_tokens": 1},
        "sampling_options": {
            "temperature": payload.get("temperature", 0),
        },
        "output_options": {},
        "eos_token_ids": [],
        "annotations": [],
        "extra_args": extra_args or None,
    }


async def _collect_prefill(client: Any, request: dict[str, Any]) -> list[Any]:
    items: list[Any] = []
    stream = await client.generate(request)
    async for item in stream:
        if hasattr(item, "is_error") and item.is_error():
            comments = item.comments() if hasattr(item, "comments") else []
            raise RuntimeError("; ".join(comments) or "Dynamo prefill returned error")
        data = item.data() if hasattr(item, "data") else item
        if data is not None:
            items.append(data)
    return items


async def _serve(runtime: DistributedRuntime) -> None:
    model = os.environ["MODEL"]
    endpoint_path = os.environ["KVBM_HUB_PREFILL_ENDPOINT"]
    host = os.environ.get("KVBM_HUB_PREFILL_SHIM_HOST", "127.0.0.1")
    port = int(os.environ.get("KVBM_HUB_PREFILL_SHIM_PORT", "8001"))
    max_body_bytes = _env_int(
        "KVBM_HUB_PREFILL_SHIM_MAX_BODY_BYTES",
        DEFAULT_MAX_BODY_BYTES,
    )
    read_timeout_seconds = _env_float(
        "KVBM_HUB_PREFILL_SHIM_READ_TIMEOUT_SECONDS",
        DEFAULT_READ_TIMEOUT_SECONDS,
    )
    upstream_timeout_seconds = _env_float(
        "KVBM_HUB_PREFILL_SHIM_UPSTREAM_TIMEOUT_SECONDS",
        DEFAULT_UPSTREAM_TIMEOUT_SECONDS,
    )

    endpoint = runtime.endpoint(endpoint_path)
    client = await endpoint.client()
    await client.wait_for_instances()

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            method, path, body = await _read_http_request(
                reader,
                max_body_bytes,
                read_timeout_seconds,
            )
            if method == "GET" and path == "/v1/models":
                response = _json_response(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": model,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "nvidia",
                            }
                        ],
                    },
                )
            elif method == "POST" and path == "/v1/completions":
                payload = _decode_json_object(body)
                preprocessed = _payload_to_preprocessed(payload, model)
                outputs = await asyncio.wait_for(
                    _collect_prefill(client, preprocessed),
                    timeout=upstream_timeout_seconds,
                )
                response = _json_response(
                    200,
                    {
                        "id": f"cmpl-prefill-{int(time.time() * 1000)}",
                        "object": "text_completion",
                        "model": model,
                        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                        "dynamo_prefill_output_count": len(outputs),
                    },
                )
            else:
                response = _json_response(404, {"error": "not found"})
        except PublicHTTPError as exc:
            response = _json_response(exc.status, {"error": exc.error})
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            print(f"[prefill-shim] invalid request: {exc}", flush=True)
            response = _json_response(400, {"error": "invalid_request"})
        except TimeoutError:
            print("[prefill-shim] upstream request timed out", flush=True)
            response = _json_response(500, {"error": "upstream_request_timeout"})
        except Exception as exc:  # noqa: BLE001 - return failure to the hub.
            traceback.print_exc()
            print(
                f"[prefill-shim] upstream request failed: {type(exc).__name__}",
                flush=True,
            )
            response = _json_response(500, {"error": "upstream_request_failed"})

        writer.write(response)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, host, port)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(
        "[prefill-shim] serving /v1/completions for "
        f"{endpoint_path} on {addrs} "
        f"max_body_bytes={max_body_bytes} "
        f"read_timeout_seconds={read_timeout_seconds} "
        f"upstream_timeout_seconds={upstream_timeout_seconds}",
        flush=True,
    )
    async with server:
        await server.serve_forever()


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    await _serve(runtime)


if __name__ == "__main__":
    asyncio.run(worker())
