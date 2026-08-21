#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a pinned Omnigent Codex harness against Dynamo or a local capture endpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PINNED_OMNIGENT_COMMIT = "733234c303af7254597f99b14bda058878d3e8ca"
PINNED_UV_VERSION = "0.11.8"
CAPTURE_REPLY = "Omnigent reached the Dynamo Responses API."


@dataclass(frozen=True)
class Invocation:
    """A fully isolated Omnigent invocation."""

    command: tuple[str, ...]
    environment: dict[str, str]
    omnigent_repo: Path
    runtime_root: Path


@dataclass(frozen=True)
class Execution:
    """One Omnigent turn plus explicit local-host cleanup."""

    result: subprocess.CompletedProcess[str]
    cleanup: subprocess.CompletedProcess[str]
    codex_temp_clean: bool


@dataclass(frozen=True)
class CapturedRequest:
    """One HTTP request observed by the local protocol capture server."""

    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any] | None


class CaptureState:
    """Thread-safe request storage shared by capture handler instances."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: list[CapturedRequest] = []

    def append(self, request: CapturedRequest) -> None:
        with self._lock:
            self._requests.append(request)

    def snapshot(self) -> list[CapturedRequest]:
        with self._lock:
            return list(self._requests)


class CaptureServer(ThreadingHTTPServer):
    """Local OpenAI Responses API capture endpoint."""

    daemon_threads = True

    def __init__(self, state: CaptureState) -> None:
        super().__init__(("127.0.0.1", 0), CaptureHandler)
        self.state = state


class CaptureHandler(BaseHTTPRequestHandler):
    """Serve the minimal OpenAI surface exercised by Omnigent's Codex harness."""

    server: CaptureServer

    def log_message(self, message_format: str, *args: Any) -> None:
        del message_format, args

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._record(None)
        if urlsplit(self.path).path == "/v1/models":
            self._send_json({"object": "list", "data": []})
            return
        self._send_json({"error": {"message": "not found"}}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        raw_length = self.headers.get("content-length", "0")
        try:
            content_length = int(raw_length)
        except ValueError:
            content_length = 0
        raw_body = self.rfile.read(content_length)
        try:
            parsed = json.loads(raw_body) if raw_body else None
        except json.JSONDecodeError:
            parsed = None
        body = parsed if isinstance(parsed, dict) else None
        self._record(body)
        if self.path == "/v1/responses":
            model = str(body.get("model", "capture-model")) if body else "capture-model"
            payload = _responses_sse(CAPTURE_REPLY, model).encode()
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._send_json({"error": {"message": "not found"}}, status=404)

    def _record(self, body: dict[str, Any] | None) -> None:
        self.server.state.append(
            CapturedRequest(
                method=self.command,
                path=self.path,
                headers={key.lower(): value for key, value in self.headers.items()},
                body=body,
            )
        )

    def _send_json(self, value: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _responses_sse(text: str, model: str) -> str:
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    message_id = f"msg_{uuid.uuid4().hex[:12]}"
    now = int(time.time())
    message = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text}],
    }
    usage = {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18}
    completed = {
        "id": response_id,
        "object": "response",
        "created_at": now,
        "completed_at": now,
        "status": "completed",
        "model": model,
        "output": [message],
        "parallel_tool_calls": True,
        "tools": [],
        "tool_choice": "auto",
        "usage": usage,
    }
    created = {**completed, "completed_at": None, "status": "in_progress", "output": []}
    events = [
        ("response.created", {"response": created}),
        ("response.output_item.added", {"output_index": 0, "item": message}),
        (
            "response.output_text.delta",
            {
                "output_index": 0,
                "item_id": message_id,
                "content_index": 0,
                "delta": text,
            },
        ),
        (
            "response.output_text.done",
            {
                "output_index": 0,
                "item_id": message_id,
                "content_index": 0,
                "text": text,
            },
        ),
        ("response.output_item.done", {"output_index": 0, "item": message}),
        ("response.completed", {"response": completed}),
    ]
    chunks = []
    for sequence_number, (event_type, event_data) in enumerate(events):
        data = {"type": event_type, "sequence_number": sequence_number, **event_data}
        chunks.append(f"event: {event_type}\ndata: {json.dumps(data)}\n\n")
    return "".join(chunks)


def normalize_base_url(value: str) -> str:
    """Normalize a Dynamo frontend URL to an OpenAI-compatible ``/v1`` root."""

    root = value.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    parsed = urlsplit(root)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--base-url must be an absolute HTTP(S) URL")
    return f"{root}/v1"


def provider_config(base_url: str, model: str, api_key_env: str) -> dict[str, Any]:
    """Build Omnigent's vendor-neutral provider config without embedding a secret."""

    return {
        "auto_open_conversation": False,
        "telemetry": False,
        "providers": {
            "dynamo": {
                "kind": "gateway",
                "default": True,
                "openai": {
                    "base_url": normalize_base_url(base_url),
                    "api_key_ref": f"env:{api_key_env}",
                    "wire_api": "responses",
                    "models": {"default": model},
                },
            }
        },
    }


def validate_checkout(repo: Path) -> None:
    """Require the audited Omnigent source revision and a clean checkout."""

    resolved = repo.resolve()
    if not (resolved / "pyproject.toml").is_file():
        raise ValueError(f"--omnigent-repo is not an Omnigent checkout: {resolved}")
    commit = _git_output(resolved, "rev-parse", "HEAD")
    if commit != PINNED_OMNIGENT_COMMIT:
        raise ValueError(
            f"Omnigent checkout is at {commit}, expected {PINNED_OMNIGENT_COMMIT}"
        )
    status = _git_output(
        resolved, "status", "--porcelain=v1", "--untracked-files=normal"
    )
    if status:
        raise ValueError("Omnigent checkout must be clean for a reproducible run")


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_invocation(
    *,
    omnigent_repo: Path,
    cwd: Path,
    runtime_root: Path,
    base_url: str,
    model: str,
    prompt: str,
    api_key_env: str,
) -> Invocation:
    """Materialize isolated Omnigent state and return its pinned headless command."""

    if not model.strip():
        raise ValueError("--model must not be empty")
    if not prompt.strip():
        raise ValueError("--prompt must not be empty")
    resolved_cwd = cwd.resolve()
    if not resolved_cwd.is_dir():
        raise ValueError(f"--cwd is not a directory: {resolved_cwd}")

    config_root = runtime_root / "config"
    data_root = runtime_root / "data"
    home_root = runtime_root / "home"
    codex_root = runtime_root / "codex-source"
    cache_root = runtime_root / "cache"
    for path in (config_root, data_root, home_root, codex_root, cache_root):
        path.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "config.yaml"
    config_path.write_text(
        json.dumps(provider_config(base_url, model, api_key_env), indent=2) + "\n",
        encoding="utf-8",
    )

    environment = dict(os.environ)
    if not environment.get(api_key_env):
        environment[api_key_env] = "dummy"
    environment[f"OMNIGENT_{api_key_env}"] = environment[api_key_env]
    environment.update(
        {
            "CODEX_HOME": str(codex_root),
            "HOME": str(home_root),
            "HARNESS_CODEX_CWD": str(resolved_cwd),
            "NO_BROWSER": "1",
            "OMNIGENT_CONFIG_HOME": str(config_root),
            "OMNIGENT_DATA_DIR": str(data_root),
            "OMNIGENT_DISABLE_TELEMETRY": "true",
            "OMNIGENT_NO_UPDATE_CHECK": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_CACHE_DIR": str(cache_root),
        }
    )
    command = _omnigent_command(omnigent_repo) + (
        "run",
        "--harness",
        "codex",
        "--model",
        model,
        "-p",
        prompt,
        "--no-log",
    )
    return Invocation(
        command=command,
        environment=environment,
        omnigent_repo=omnigent_repo.resolve(),
        runtime_root=runtime_root,
    )


def _omnigent_command(omnigent_repo: Path) -> tuple[str, ...]:
    return (
        "uvx",
        "--from",
        f"uv=={PINNED_UV_VERSION}",
        "uv",
        "run",
        "--isolated",
        "--frozen",
        "--no-dev",
        "--project",
        str(omnigent_repo.resolve()),
        "python",
        "-m",
        "omnigent",
    )


def execute_invocation(
    invocation: Invocation,
    *,
    cwd: Path,
    timeout: float,
    capture_output: bool,
) -> Execution:
    """Run one turn and stop the isolated Omnigent host before returning."""

    resolved_cwd = cwd.resolve()
    codex_temp = resolved_cwd / ".codex-tmp"
    codex_temp_preexisting = codex_temp.exists()
    result: subprocess.CompletedProcess[str] | None = None
    try:
        result = subprocess.run(
            invocation.command,
            cwd=resolved_cwd,
            env=invocation.environment,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
        )
    finally:
        cleanup = subprocess.run(
            _omnigent_command(invocation.omnigent_repo) + ("stop",),
            cwd=resolved_cwd,
            env=invocation.environment,
            capture_output=True,
            text=True,
            timeout=min(timeout, 60.0),
        )
    if result is None:
        raise RuntimeError("Omnigent exited without a subprocess result")
    codex_temp_clean = _remove_empty_codex_temp(codex_temp, codex_temp_preexisting)
    return Execution(result=result, cleanup=cleanup, codex_temp_clean=codex_temp_clean)


def _remove_empty_codex_temp(path: Path, preexisting: bool) -> bool:
    if preexisting:
        return True
    try:
        path.rmdir()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def assess_capture(
    requests: list[CapturedRequest],
    *,
    expected_model: str,
    expected_api_key: str,
) -> dict[str, Any]:
    """Summarize protocol compatibility and the known lifecycle gap."""

    response_requests = [
        request for request in requests if request.path == "/v1/responses"
    ]
    response_thread_ids = [
        request.headers.get("thread-id", "") for request in response_requests
    ]
    thread_ids = sorted({thread_id for thread_id in response_thread_ids if thread_id})
    authorization_ok = bool(response_requests) and all(
        request.headers.get("authorization") == f"Bearer {expected_api_key}"
        for request in response_requests
    )
    models_ok = bool(response_requests) and all(
        request.body is not None and request.body.get("model") == expected_model
        for request in response_requests
    )
    responses_wire_ok = bool(response_requests) and all(
        request.body is not None and request.body.get("stream") is True
        for request in response_requests
    )
    session_affinity_ok = bool(response_requests) and all(response_thread_ids)
    terminal_request_seen = any(
        request.headers.get("x-dynamo-session-final", "").lower() == "true"
        for request in requests
    )
    protocol_compatible = all(
        (authorization_ok, models_ok, responses_wire_ok, session_affinity_ok)
    )
    return {
        "protocol_compatible": protocol_compatible,
        "responses_request_count": len(response_requests),
        "responses_wire_ok": responses_wire_ok,
        "bearer_auth_ok": authorization_ok,
        "model_ok": models_ok,
        "thread_ids": thread_ids,
        "unique_thread_count": len(thread_ids),
        "session_affinity_ok": session_affinity_ok,
        "persistent_thread_reuse_verified": False,
        "session_final_seen": terminal_request_seen,
        "lifecycle_qualified": protocol_compatible and terminal_request_seen,
        "observed_paths": sorted({request.path for request in requests}),
        "response_request_summaries": [
            {
                "thread_id": request.headers.get("thread-id"),
                "turn_metadata": request.headers.get("x-codex-turn-metadata"),
                "user_agent": request.headers.get("user-agent"),
                "input_excerpt": _input_excerpt(request.body),
            }
            for request in response_requests
        ],
    }


def _input_excerpt(body: dict[str, Any] | None) -> str | None:
    if body is None or "input" not in body:
        return None
    serialized = json.dumps(body["input"], separators=(",", ":"))
    return serialized[:240]


def _run_capture(args: argparse.Namespace) -> int:
    validate_checkout(args.omnigent_repo)
    state = CaptureState()
    server = CaptureServer(state)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    capture_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with tempfile.TemporaryDirectory(
            prefix=".omnigent-well-lit-", dir=args.cwd.resolve()
        ) as runtime_dir:
            invocation = build_invocation(
                omnigent_repo=args.omnigent_repo,
                cwd=args.cwd,
                runtime_root=Path(runtime_dir),
                base_url=capture_url,
                model=args.model,
                prompt=args.prompt,
                api_key_env=args.api_key_env,
            )
            execution = execute_invocation(
                invocation,
                cwd=args.cwd,
                timeout=args.timeout,
                capture_output=True,
            )
            expected_api_key = invocation.environment[args.api_key_env]
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
    validate_checkout(args.omnigent_repo)

    assessment = assess_capture(
        state.snapshot(),
        expected_model=args.model,
        expected_api_key=expected_api_key,
    )
    assessment.update(
        {
            "omnigent_commit": PINNED_OMNIGENT_COMMIT,
            "uv_version": PINNED_UV_VERSION,
            "command_exit_code": execution.result.returncode,
            "assistant_reply_seen": CAPTURE_REPLY in execution.result.stdout,
            "cleanup_exit_code": execution.cleanup.returncode,
            "codex_temp_clean": execution.codex_temp_clean,
            "stderr": execution.result.stderr.strip(),
            "cleanup_stderr": execution.cleanup.stderr.strip(),
        }
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))
    clean = execution.cleanup.returncode == 0 and execution.codex_temp_clean
    return (
        0
        if assessment["protocol_compatible"]
        and execution.result.returncode == 0
        and clean
        else 1
    )


def _run_dynamo(args: argparse.Namespace) -> int:
    validate_checkout(args.omnigent_repo)
    with tempfile.TemporaryDirectory(
        prefix=".omnigent-well-lit-", dir=args.cwd.resolve()
    ) as runtime_dir:
        invocation = build_invocation(
            omnigent_repo=args.omnigent_repo,
            cwd=args.cwd,
            runtime_root=Path(runtime_dir),
            base_url=args.base_url,
            model=args.model,
            prompt=args.prompt,
            api_key_env=args.api_key_env,
        )
        execution = execute_invocation(
            invocation,
            cwd=args.cwd,
            timeout=args.timeout,
            capture_output=False,
        )
    validate_checkout(args.omnigent_repo)
    if execution.result.returncode != 0:
        return execution.result.returncode
    if execution.cleanup.returncode != 0 or not execution.codex_temp_clean:
        return 1
    return 0


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--omnigent-repo", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument(
        "--prompt", default="Reply with exactly: Omnigent Dynamo smoke passed"
    )
    parser.add_argument("--api-key-env", default="DYNAMO_API_KEY")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    common = _common_parser()
    run_parser = subparsers.add_parser("run", parents=[common])
    run_parser.add_argument("--base-url", required=True)
    subparsers.add_parser("capture", parents=[common])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.action == "capture":
        return _run_capture(args)
    return _run_dynamo(args)


if __name__ == "__main__":
    sys.exit(main())
