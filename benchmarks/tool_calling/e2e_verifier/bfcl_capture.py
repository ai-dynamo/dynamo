# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe the pinned BFCL handler's HTTP traffic without changing its behavior."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

TRACE_FILE = "bfcl-http-trace.jsonl"
FAILURES_FILE = "bfcl-http-failures.jsonl"


@contextmanager
def capture_http(base_handler: type, session: type, output_dir: Path):
    """Associate concurrent requests with their BFCL case and preserve raw bodies."""
    context: ContextVar[dict[str, Any] | None] = ContextVar("bfcl_case", default=None)
    lock = threading.Lock()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = output_dir / TRACE_FILE
    failures = output_dir / FAILURES_FILE
    trace.touch()
    failures.touch()
    original_inference = base_handler.inference
    original_send = session.send

    def write(record: dict[str, Any], *, failed: bool = False) -> None:
        encoded = json.dumps(record, ensure_ascii=False) + "\n"
        with lock:
            with trace.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
            if failed:
                with failures.open("a", encoding="utf-8") as stream:
                    stream.write(encoded)

    @wraps(original_inference)
    def inference(handler, test_entry, *args, **kwargs):
        token = context.set({"case_id": test_entry["id"], "sequence": 0})
        try:
            return original_inference(handler, test_entry, *args, **kwargs)
        finally:
            context.reset(token)

    @wraps(original_send)
    def send(client, request, **kwargs):
        case = context.get()
        if case is None:
            return original_send(client, request, **kwargs)
        case["sequence"] += 1
        body = request.body or b""
        if isinstance(body, str):
            body = body.encode("utf-8")
        record = {
            "capture_id": str(uuid.uuid4()),
            "case_id": case["case_id"],
            "request_sequence": case["sequence"],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "request": {
                "method": request.method,
                "url": request.url,
                "content_type": request.headers.get("Content-Type"),
                "body": body.decode("utf-8"),
                "body_sha256": hashlib.sha256(body).hexdigest(),
            },
        }
        write({**record, "event": "request"})
        started = time.monotonic()
        try:
            response = original_send(client, request, **kwargs)
        except Exception as exc:
            # Evidence includes transport failures; preserve the original exception.
            write(
                {**record, "event": "transport_error", "error": repr(exc)},
                failed=True,
            )
            raise
        record["response"] = {
            "status": response.status_code,
            "body": response.text,
            "content_type": response.headers.get("Content-Type"),
            "request_id": response.headers.get("X-Request-ID"),
            "body_sha256": hashlib.sha256(response.content).hexdigest(),
        }
        record["duration_seconds"] = time.monotonic() - started
        write({**record, "event": "response"}, failed=response.status_code >= 400)
        return response

    base_handler.inference = inference
    session.send = send
    try:
        yield
    finally:
        base_handler.inference = original_inference
        session.send = original_send


def main() -> None:
    from bfcl.__main__ import cli
    from bfcl.model_handler.base_handler import BaseHandler
    from requests import Session

    with capture_http(BaseHandler, Session, Path("/results")):
        cli()


if __name__ == "__main__":
    main()
