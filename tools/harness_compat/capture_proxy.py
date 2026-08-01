#!/usr/bin/env python3
"""Transparent HTTP capture proxy for coding-harness discovery runs.

It forwards request bytes unchanged. Artifacts intentionally retain protocol
shape rather than prompts, responses, or credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_HOP_BY_HOP_HEADERS = {"connection", "content-length", "host", "transfer-encoding"}
_SECRET_HEADERS = {"authorization", "x-api-key", "api-key", "cookie", "set-cookie"}
_VERBATIM_HEADERS = {
    "accept",
    "anthropic-beta",
    "anthropic-version",
    "content-type",
    "openai-beta",
    "user-agent",
    "x-stainless-lang",
    "x-stainless-os",
    "x-stainless-package-version",
    "x-stainless-runtime",
}


def _digest(value: str) -> dict[str, Any]:
    return {"sha256_12": hashlib.sha256(value.encode()).hexdigest()[:12], "length": len(value)}


def _sanitize_headers(headers: Any) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for name, value in headers.items():
        key = name.lower()
        if key in _SECRET_HEADERS:
            sanitized[key] = "[redacted]"
        elif key in _VERBATIM_HEADERS:
            sanitized[key] = value
        else:
            sanitized[key] = _digest(value)
    return sanitized


def _content_types(value: Any) -> list[str]:
    if isinstance(value, str):
        return ["text"]
    if isinstance(value, list):
        return sorted(
            {
                str(block.get("type", "object")) if isinstance(block, dict) else type(block).__name__
                for block in value
            }
        )
    if value is None:
        return ["null"]
    return [type(value).__name__]


def _tool_result_error_count(value: Any) -> int:
    if not isinstance(value, list):
        return 0
    return sum(
        block.get("type") == "tool_result" and block.get("is_error") is True
        for block in value
        if isinstance(block, dict)
    )


def _request_shape(body: bytes) -> dict[str, Any]:
    """Return non-content-bearing request discriminators for Responses or Messages."""
    if not body:
        return {"body": "empty"}
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return {"body": "non_json", "length": len(body)}
    if not isinstance(value, dict):
        return {"body": type(value).__name__}

    shape: dict[str, Any] = {"top_level_keys": sorted(value)}
    for key in ("stream", "model", "max_tokens", "max_output_tokens"):
        if key in value:
            shape[key] = _digest(value[key]) if key == "model" and isinstance(value[key], str) else value[key]
    if isinstance(value.get("input"), list):
        input_items: list[dict[str, Any]] = []
        for item in value["input"]:
            if not isinstance(item, dict):
                input_items.append({"type": type(item).__name__})
                continue
            item_shape: dict[str, Any] = {
                "type": item.get("type"),
                "role": item.get("role"),
                "content_types": _content_types(item.get("content")),
            }
            errors = _tool_result_error_count(item.get("content"))
            if errors:
                item_shape["tool_result_error_count"] = errors
            input_items.append(item_shape)
        shape["input_items"] = input_items
    if isinstance(value.get("messages"), list):
        messages: list[dict[str, Any]] = []
        for item in value["messages"]:
            if not isinstance(item, dict):
                messages.append({"type": type(item).__name__})
                continue
            item_shape = {
                "role": item.get("role"),
                "content_types": _content_types(item.get("content")),
            }
            errors = _tool_result_error_count(item.get("content"))
            if errors:
                item_shape["tool_result_error_count"] = errors
            messages.append(item_shape)
        shape["messages"] = messages
    if isinstance(value.get("tools"), list):
        shape["tool_types"] = [
            item.get("type", "object") if isinstance(item, dict) else type(item).__name__
            for item in value["tools"]
        ]
        # Tool names are protocol metadata, not prompts, arguments, or schemas. They
        # distinguish a newly advertised native harness capability from an unchanged
        # generic `function` tool type without retaining any tool implementation.
        shape["tool_names"] = sorted(
            item["name"]
            for item in value["tools"]
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        )
    if isinstance(value.get("output_config"), dict):
        shape["output_config_keys"] = sorted(value["output_config"])
    if isinstance(value.get("text"), dict):
        shape["text_keys"] = sorted(value["text"])
    return shape


def _error_shape(body: bytes, truncated: bool) -> dict[str, Any]:
    """Record a JSON error's stable discriminators without preserving its text."""
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return {"body": "non_json", "length": len(body), "truncated": truncated}
    if not isinstance(value, dict):
        return {"body": type(value).__name__, "truncated": truncated}
    error = value.get("error", value)
    if not isinstance(error, dict):
        return {"top_level_keys": sorted(value), "truncated": truncated}
    shape: dict[str, Any] = {"top_level_keys": sorted(value), "error_keys": sorted(error), "truncated": truncated}
    for key in ("type", "code"):
        if key in error:
            shape[f"error_{key}"] = error[key]
    if isinstance(error.get("message"), str):
        shape["error_message"] = _digest(error["message"])
    return shape


class _Recorder:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        record["timestamp_unix_ms"] = round(time.time() * 1000)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self._path.open("a", encoding="utf-8") as output:
                output.write(line + "\n")


class _CaptureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        upstream: str,
        record_path: Path,
        inject_status: int | None,
        inject_at_request: int,
        truncate_sse_after_events: int | None,
        truncate_sse_at_request: int,
    ):
        super().__init__(address, _CaptureHandler)
        self.upstream = upstream.rstrip("/")
        self.recorder = _Recorder(record_path)
        self.inject_status = inject_status
        self.inject_at_request = inject_at_request
        self.truncate_sse_after_events = truncate_sse_after_events
        self.truncate_sse_at_request = truncate_sse_at_request
        self._request_count = 0
        self._request_lock = threading.Lock()

    def request_number(self) -> int:
        with self._request_lock:
            self._request_count += 1
            return self._request_count


class _CaptureHandler(BaseHTTPRequestHandler):
    server: _CaptureServer

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _record_sse(self, request_id: str, pending: bytes, chunk: bytes) -> tuple[bytes, int]:
        pending += chunk
        event_count = 0
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            if line.startswith(b"event:"):
                event_count += 1
                self.server.recorder.write(
                    {
                        "kind": "sse_event",
                        "request_id": request_id,
                        "event": line[6:].decode("utf-8", errors="replace").strip(),
                    }
                )
            elif line.startswith(b"data:"):
                try:
                    data = json.loads(line[5:].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(data, dict):
                    record: dict[str, Any] = {
                        "kind": "sse_data",
                        "request_id": request_id,
                        "type": data.get("type"),
                        "keys": sorted(data),
                    }
                    # These discriminators are protocol metadata, never model
                    # text or tool arguments. They make a failed coding reach
                    # signal distinguishable from a Messages translation fault.
                    if isinstance(data.get("stop_reason"), str):
                        record["stop_reason"] = data["stop_reason"]
                    delta = data.get("delta")
                    if isinstance(delta, dict) and isinstance(delta.get("stop_reason"), str):
                        record["stop_reason"] = delta["stop_reason"]
                    block = data.get("content_block")
                    if isinstance(block, dict):
                        if isinstance(block.get("type"), str):
                            record["content_block_type"] = block["type"]
                        if isinstance(block.get("name"), str):
                            record["tool_name"] = block["name"]
                    self.server.recorder.write(record)
        return pending, event_count

    def _inject_error(self, request_id: str, status: int) -> None:
        """Return a native endpoint-shaped error without contacting Dynamo.

        This is deliberately proxy-local fault injection: it tests the installed
        harness's decoding and retry behavior while the request/response shape is
        captured. It is not a claim about Dynamo's own error generation.
        """
        if self.path.startswith("/v1/messages"):
            error_type = {
                400: "invalid_request_error",
                401: "authentication_error",
                403: "permission_error",
                404: "not_found_error",
                409: "conflict_error",
                429: "rate_limit_error",
                529: "overloaded_error",
            }.get(status, "api_error")
            payload = {"type": "error", "error": {"type": error_type, "message": "compat injected error"}}
        else:
            error_type = {
                400: "invalid_request_error",
                401: "invalid_api_key",
                403: "insufficient_permissions",
                404: "model_not_found",
                409: "conflict_error",
                429: "rate_limit_error",
            }.get(status, "server_error")
            payload = {
                "error": {"message": "compat injected error", "type": error_type, "code": "compat_injected"}
            }
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if status == 429:
            self.send_header("Retry-After", "1")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.server.recorder.write(
            {
                "kind": "response",
                "request_id": request_id,
                "status": status,
                "headers": {"content-type": "application/json"},
                "injected": True,
            }
        )
        self.server.recorder.write(
            {
                "kind": "response_error",
                "request_id": request_id,
                "shape": _error_shape(body, False),
                "injected": True,
            }
        )

    def _forward(self) -> None:
        request_id = hashlib.sha256(f"{time.time_ns()}-{threading.get_ident()}".encode()).hexdigest()[:16]
        request_number = self.server.request_number()
        body_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(body_length) if body_length else b""
        self.server.recorder.write(
            {
                "kind": "request",
                "request_id": request_id,
                "method": self.command,
                "path": self.path,
                "request_number": request_number,
                "headers": _sanitize_headers(self.headers),
                "shape": _request_shape(body),
            }
        )

        if self.server.inject_status is not None and request_number == self.server.inject_at_request:
            self.server.recorder.write(
                {
                    "kind": "fault_injected",
                    "request_id": request_id,
                    "request_number": request_number,
                    "fault": "http_status",
                    "status": self.server.inject_status,
                }
            )
            self._inject_error(request_id, self.server.inject_status)
            return

        request = urllib.request.Request(
            url=self.server.upstream + self.path,
            data=body if body else None,
            method=self.command,
        )
        for name, value in self.headers.items():
            if name.lower() not in _HOP_BY_HOP_HEADERS:
                request.add_header(name, value)
        if body:
            request.add_header("Content-Length", str(len(body)))

        try:
            response = urllib.request.urlopen(request, timeout=900)
        except urllib.error.HTTPError as error:
            response = error
        except urllib.error.URLError as error:
            self.server.recorder.write(
                {"kind": "proxy_error", "request_id": request_id, "error": str(error.reason)}
            )
            self.send_error(502, "upstream unavailable")
            return

        try:
            self.send_response(response.status)
            for name, value in response.headers.items():
                if name.lower() not in _HOP_BY_HOP_HEADERS:
                    self.send_header(name, value)
            self.end_headers()
            self.server.recorder.write(
                {
                    "kind": "response",
                    "request_id": request_id,
                    "status": response.status,
                    "headers": _sanitize_headers(response.headers),
                }
            )
            pending = b""
            is_sse = "text/event-stream" in response.headers.get("Content-Type", "")
            truncate_sse = (
                is_sse
                and self.server.truncate_sse_after_events is not None
                and request_number == self.server.truncate_sse_at_request
            )
            sse_events_seen = 0
            error_body = bytearray()
            error_truncated = False
            read_size = 1 if truncate_sse else 8192
            while chunk := response.read(read_size):
                if is_sse:
                    pending, new_events = self._record_sse(request_id, pending, chunk)
                    sse_events_seen += new_events
                if response.status >= 400:
                    remaining = 64 * 1024 - len(error_body)
                    if remaining > 0:
                        error_body.extend(chunk[:remaining])
                    error_truncated |= len(chunk) > remaining
                self.wfile.write(chunk)
                self.wfile.flush()
                if truncate_sse and sse_events_seen >= self.server.truncate_sse_after_events:
                    self.server.recorder.write(
                        {
                            "kind": "fault_injected",
                            "request_id": request_id,
                            "request_number": request_number,
                            "fault": "sse_truncation",
                            "after_sse_events": sse_events_seen,
                        }
                    )
                    self.close_connection = True
                    return
            if response.status >= 400:
                self.server.recorder.write(
                    {
                        "kind": "response_error",
                        "request_id": request_id,
                        "shape": _error_shape(bytes(error_body), error_truncated),
                    }
                )
        except (BrokenPipeError, ConnectionResetError):
            self.server.recorder.write({"kind": "client_disconnect", "request_id": request_id})
        finally:
            response.close()

    do_DELETE = _forward
    do_GET = _forward
    do_PATCH = _forward
    do_POST = _forward
    do_PUT = _forward


def _parse_address(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise argparse.ArgumentTypeError("address must be HOST:PORT")
    try:
        return host, int(port)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", type=_parse_address, required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument(
        "--inject-status",
        type=int,
        choices=(400, 401, 403, 404, 409, 429, 500, 502, 503, 529),
        help="Return one endpoint-shaped HTTP error instead of forwarding a selected request.",
    )
    parser.add_argument(
        "--truncate-sse-after-events",
        type=int,
        help="Close one streamed response after this many SSE event lines, without a terminal event.",
    )
    parser.add_argument(
        "--truncate-sse-at-request",
        type=int,
        default=1,
        help="1-indexed streamed request number to truncate; applies only with --truncate-sse-after-events.",
    )
    parser.add_argument(
        "--inject-at-request",
        type=int,
        default=1,
        help="1-indexed request number to inject; applies only with --inject-status.",
    )
    args = parser.parse_args()
    parsed = urlsplit(args.upstream)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        parser.error("--upstream must be an HTTP URL")
    if args.inject_at_request < 1:
        parser.error("--inject-at-request must be positive")
    if args.truncate_sse_after_events is not None and args.truncate_sse_after_events < 1:
        parser.error("--truncate-sse-after-events must be positive")
    if args.truncate_sse_at_request < 1:
        parser.error("--truncate-sse-at-request must be positive")
    if args.inject_status is not None and args.truncate_sse_after_events is not None:
        parser.error("--inject-status and --truncate-sse-after-events are mutually exclusive")
    args.record.parent.mkdir(parents=True, exist_ok=True)
    server = _CaptureServer(
        args.listen,
        args.upstream,
        args.record,
        args.inject_status,
        args.inject_at_request,
        args.truncate_sse_after_events,
        args.truncate_sse_at_request,
    )
    print(
        f"capture proxy: http://{args.listen[0]}:{args.listen[1]} -> {args.upstream}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
