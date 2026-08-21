#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a pinned Omnigent Codex harness against Dynamo or a local capture endpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PINNED_OMNIGENT_COMMIT = "733234c303af7254597f99b14bda058878d3e8ca"
PINNED_UV_VERSION = "0.11.8"
PINNED_CODEX_VERSION = "0.147.0"
DYNAMO_API_KEY_ENV = "DYNAMO_API_KEY"
CODEX_VERSION_TIMEOUT_SECONDS = 5.0
CODEX_TEMP_CLEANUP_TIMEOUT_SECONDS = 5.0
CAPTURE_REPLY = "Omnigent reached the Dynamo Responses API."
SAFE_CODEX_SANDBOX_LABELS = frozenset(
    {"bwrap", "landlock", "read-only", "seatbelt", "workspace-write"}
)
SCOPED_CLEANUP_CODE = """
from omnigent.cli import _list_daemon_records, _terminate_daemon
from omnigent.host.local_server import stop_local_omnigent_server

failures = []
records = _list_daemon_records()
for record in records:
    try:
        _terminate_daemon(record, force=True)
    except Exception as exc:
        failures.append(f"{record.pid}: {exc}")
stop_local_omnigent_server()
if failures:
    raise RuntimeError("; ".join(failures))
print(f"Stopped {len(records)} invocation daemon(s) and the invocation server.")
"""
PARENT_ENV_ALLOWLIST = (
    "COMSPEC",
    "CURL_CA_BUNDLE",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "PATHEXT",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "WINDIR",
)


@dataclass(frozen=True)
class Invocation:
    """A fully isolated Omnigent invocation."""

    command: tuple[str, ...]
    environment: dict[str, str]
    omnigent_repo: Path
    runtime_root: Path
    launch_cwd: Path
    sandbox_backend: str


@dataclass(frozen=True)
class ProcessStatus:
    """Structured outcome for a run or cleanup subprocess."""

    returncode: int | None
    stdout: str
    stderr: str
    error: str | None = None
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None and not self.timed_out


@dataclass(frozen=True)
class Execution:
    """One Omnigent turn plus explicit local-host cleanup."""

    result: ProcessStatus
    cleanup: ProcessStatus
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


def provider_config(base_url: str, model: str) -> dict[str, Any]:
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
                    "api_key_ref": f"env:{DYNAMO_API_KEY_ENV}",
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


def _minimal_parent_environment(
    source_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy only non-secret process settings required by pinned launch tools."""

    source = os.environ if source_environment is None else source_environment
    environment = {key: source[key] for key in PARENT_ENV_ALLOWLIST if source.get(key)}
    environment.setdefault("PATH", os.defpath)
    return environment


def resolve_codex_cli(
    explicit_path: Path | None,
    source_environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve one Codex executable and require the audited exact version."""

    environment = _minimal_parent_environment(source_environment)
    if explicit_path is None:
        discovered = shutil.which("codex", path=environment["PATH"])
        if discovered is None:
            raise ValueError(
                "Codex CLI was not found on PATH; install codex-cli 0.147.0 or "
                "pass --codex-bin /absolute/path/to/codex"
            )
        candidate = Path(discovered)
    else:
        candidate = explicit_path.expanduser()
    resolved = candidate.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"--codex-bin is not an executable file: {resolved}")
    try:
        result = subprocess.run(
            (str(resolved), "--version"),
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=CODEX_VERSION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            f"Codex version probe timed out after {CODEX_VERSION_TIMEOUT_SECONDS:g}s: "
            f"{resolved}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not run Codex version probe {resolved}: {exc}"
        ) from exc
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    match = re.search(r"\bcodex-cli\s+(\d+\.\d+\.\d+)\b", output)
    observed = match.group(1) if match is not None else "unrecognized"
    if result.returncode != 0 or observed != PINNED_CODEX_VERSION:
        raise ValueError(
            f"Codex CLI {resolved} reported {observed!r}; expected exact version "
            f"{PINNED_CODEX_VERSION} (probe exit code {result.returncode})"
        )
    return resolved


def _safe_sandbox_backend(environment: Mapping[str, str]) -> str:
    """Select a hard sandbox backend and refuse an unsandboxed fallback."""

    if sys.platform == "darwin":
        return "darwin_seatbelt"
    if sys.platform.startswith("linux"):
        if shutil.which("bwrap", path=environment["PATH"]) is None:
            raise ValueError(
                "Omnigent verification requires bubblewrap on Linux; install bwrap "
                "instead of falling back to danger-full-access"
            )
        return "linux_bwrap"
    raise ValueError(
        f"No audited Omnigent verification sandbox is available on {sys.platform!r}"
    )


def _capability_prompt(prompt: str, capability: str) -> str:
    if capability == "verify":
        policy = (
            "Verification-only task. Inspect and test within the workspace, but do not "
            "create, edit, rename, or delete files."
        )
    elif capability == "act":
        policy = (
            "Workspace edits are explicitly authorized for this task. Keep every change "
            "within the selected workspace and report each changed file."
        )
    else:
        raise ValueError(f"unsupported capability: {capability}")
    return f"{policy}\n\n{prompt}"


def build_invocation(
    *,
    omnigent_repo: Path,
    cwd: Path,
    runtime_root: Path,
    launch_cwd: Path,
    base_url: str,
    model: str,
    prompt: str,
    codex_path: Path,
    capability: str,
    source_environment: Mapping[str, str] | None = None,
) -> Invocation:
    """Materialize isolated Omnigent state and return its pinned headless command."""

    if not model.strip():
        raise ValueError("--model must not be empty")
    if not prompt.strip():
        raise ValueError("--prompt must not be empty")
    resolved_cwd = cwd.resolve()
    if not resolved_cwd.is_dir():
        raise ValueError(f"--cwd is not a directory: {resolved_cwd}")
    resolved_launch_cwd = launch_cwd.resolve()
    if not resolved_launch_cwd.is_dir():
        raise ValueError(
            f"invocation launch directory is not a directory: {resolved_launch_cwd}"
        )
    if not resolved_launch_cwd.is_relative_to(resolved_cwd):
        raise ValueError("invocation launch directory must be inside --cwd")

    config_root = runtime_root / "config"
    data_root = runtime_root / "data"
    home_root = runtime_root / "home"
    codex_root = runtime_root / "codex-source"
    cache_root = runtime_root / "cache"
    temp_root = runtime_root / "tmp"
    for path in (
        config_root,
        data_root,
        home_root,
        codex_root,
        cache_root,
        temp_root,
    ):
        path.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "config.yaml"
    config_path.write_text(
        json.dumps(provider_config(base_url, model), indent=2) + "\n",
        encoding="utf-8",
    )

    source = os.environ if source_environment is None else source_environment
    environment = _minimal_parent_environment(source)
    credential = source.get(DYNAMO_API_KEY_ENV, "").strip() or "dummy"
    sandbox = {
        "type": _safe_sandbox_backend(environment),
        "write_paths": [str(resolved_cwd), str(codex_root), str(temp_root)],
        "allow_network": True,
        "env_passthrough": [],
    }
    os_environment = {
        "type": "caller_process",
        "cwd": str(resolved_cwd),
        "sandbox": sandbox,
        "fork": False,
    }
    agent_path = config_root / "dynamo-omnigent-agent"
    agent_path.mkdir()
    (agent_path / "config.yaml").write_text(
        json.dumps(
            {
                "spec_version": 1,
                "name": "dynamo-omnigent",
                "description": "Pinned Codex through a Dynamo Responses endpoint.",
                "prompt": "You are Codex running through Omnigent and Dynamo.",
                "skills": "none",
                "executor": {
                    "type": "omnigent",
                    "model": model,
                    "config": {"harness": "codex"},
                },
                "os_env": os_environment,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    environment.update(
        {
            "CODEX_HOME": str(codex_root),
            DYNAMO_API_KEY_ENV: credential,
            "HOME": str(home_root),
            "HARNESS_CODEX_CWD": str(resolved_cwd),
            "HARNESS_CODEX_ENABLE_WEB_SEARCH": "0",
            "HARNESS_CODEX_SKILLS_FILTER": json.dumps("none"),
            "NO_BROWSER": "1",
            "OMNIGENT_CONFIG_HOME": str(config_root),
            "OMNIGENT_CODEX_PATH": str(codex_path.resolve()),
            "OMNIGENT_DATA_DIR": str(data_root),
            f"OMNIGENT_{DYNAMO_API_KEY_ENV}": credential,
            "OMNIGENT_DISABLE_TELEMETRY": "true",
            "OMNIGENT_NO_UPDATE_CHECK": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TEMP": str(temp_root),
            "TMP": str(temp_root),
            "TMPDIR": str(temp_root),
            "UV_CACHE_DIR": str(cache_root),
            "UV_NO_PROGRESS": "1",
        }
    )
    command = _omnigent_command(omnigent_repo) + (
        "run",
        str(agent_path),
        "-p",
        _capability_prompt(prompt, capability),
        "--server",
        "local",
        "--harness",
        "codex",
        "--model",
        model,
        "--no-log",
    )
    return Invocation(
        command=command,
        environment=environment,
        omnigent_repo=omnigent_repo.resolve(),
        runtime_root=runtime_root,
        launch_cwd=resolved_launch_cwd,
        sandbox_backend=sandbox["type"],
    )


def _omnigent_command(omnigent_repo: Path) -> tuple[str, ...]:
    return _uv_python_command(omnigent_repo) + ("-m", "omnigent")


def _uv_python_command(omnigent_repo: Path) -> tuple[str, ...]:
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
    )


def _scoped_cleanup_command(omnigent_repo: Path) -> tuple[str, ...]:
    return _uv_python_command(omnigent_repo) + ("-c", SCOPED_CLEANUP_CODE)


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
    if codex_temp.exists():
        raise ValueError(
            f"refusing to run with pre-existing Codex temporary state: {codex_temp}"
        )
    launch_cwd = invocation.launch_cwd
    result = _run_process(
        invocation.command,
        cwd=launch_cwd,
        environment=invocation.environment,
        capture_output=capture_output,
        timeout=timeout,
    )
    cleanup = _run_process(
        _scoped_cleanup_command(invocation.omnigent_repo),
        cwd=launch_cwd,
        environment=invocation.environment,
        capture_output=True,
        timeout=min(timeout, 60.0),
    )
    codex_temp_clean = _remove_empty_codex_temp(codex_temp)
    return Execution(result=result, cleanup=cleanup, codex_temp_clean=codex_temp_clean)


def _run_process(
    command: tuple[str, ...],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    capture_output: bool,
    timeout: float,
) -> ProcessStatus:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=dict(environment),
            capture_output=capture_output,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return ProcessStatus(
            returncode=None,
            stdout=_subprocess_text(exc.stdout),
            stderr=_subprocess_text(exc.stderr),
            error=f"timed out after {timeout:g}s",
            timed_out=True,
        )
    except (OSError, ValueError) as exc:
        return ProcessStatus(
            returncode=None,
            stdout="",
            stderr="",
            error=f"{type(exc).__name__}: {exc}",
        )
    except KeyboardInterrupt:
        return ProcessStatus(
            returncode=130,
            stdout="",
            stderr="",
            error="interrupted by keyboard",
        )
    return ProcessStatus(
        returncode=result.returncode,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def execution_diagnostic(
    execution: Execution,
    runtime_removed: bool,
    sensitive_values: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a bounded structured record for operator-visible cleanup failures."""

    return {
        "command": {
            "exit_code": execution.result.returncode,
            "timed_out": execution.result.timed_out,
            "error": execution.result.error,
            "stderr": _redact(execution.result.stderr.strip(), sensitive_values),
        },
        "cleanup": {
            "attempted": True,
            "exit_code": execution.cleanup.returncode,
            "timed_out": execution.cleanup.timed_out,
            "error": execution.cleanup.error,
            "stdout": _redact(execution.cleanup.stdout.strip(), sensitive_values),
            "stderr": _redact(execution.cleanup.stderr.strip(), sensitive_values),
        },
        "codex_temp_clean": execution.codex_temp_clean,
        "disposable_runtime_removed": runtime_removed,
    }


def _redact(value: str, sensitive_values: Sequence[str]) -> str:
    redacted = value
    for sensitive in sensitive_values:
        if sensitive and sensitive != "dummy":
            redacted = redacted.replace(sensitive, "[REDACTED]")
    return redacted


def _remove_empty_codex_temp(path: Path) -> bool:
    deadline = time.monotonic() + CODEX_TEMP_CLEANUP_TIMEOUT_SECONDS
    while True:
        try:
            path.rmdir()
        except FileNotFoundError:
            return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)
            continue
        return True


def _observed_sandbox_mode(request: CapturedRequest) -> str | None:
    raw = request.headers.get("x-codex-turn-metadata", "")
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(metadata, dict):
        return None
    sandbox = metadata.get("sandbox")
    return sandbox if isinstance(sandbox, str) and sandbox else None


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
    sandbox_modes = [_observed_sandbox_mode(request) for request in response_requests]
    safe_sandbox_observed = bool(response_requests) and all(
        mode in SAFE_CODEX_SANDBOX_LABELS for mode in sandbox_modes
    )
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
        "observed_sandbox_modes": sorted(
            {mode for mode in sandbox_modes if mode is not None}
        ),
        "safe_sandbox_observed": safe_sandbox_observed,
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


def _capture_is_successful(
    assessment: Mapping[str, Any], execution: Execution, runtime_removed: bool
) -> bool:
    return bool(
        assessment["protocol_compatible"]
        and assessment["safe_sandbox_observed"]
        and assessment["assistant_reply_seen"]
        and execution.result.ok
        and execution.cleanup.ok
        and execution.codex_temp_clean
        and runtime_removed
    )


def _run_capture(args: argparse.Namespace) -> int:
    validate_checkout(args.omnigent_repo)
    codex_path = resolve_codex_cli(args.codex_bin)
    state = CaptureState()
    server = CaptureServer(state)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    capture_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with (
            tempfile.TemporaryDirectory(prefix="omnigent-well-lit-") as runtime_dir,
            tempfile.TemporaryDirectory(
                prefix=".omnigent-well-lit-launch-", dir=args.cwd.resolve()
            ) as launch_dir,
        ):
            invocation = build_invocation(
                omnigent_repo=args.omnigent_repo,
                cwd=args.cwd,
                runtime_root=Path(runtime_dir),
                launch_cwd=Path(launch_dir),
                base_url=capture_url,
                model=args.model,
                prompt=args.prompt,
                codex_path=codex_path,
                capability=args.capability,
            )
            execution = execute_invocation(
                invocation,
                cwd=args.cwd,
                timeout=args.timeout,
                capture_output=True,
            )
            expected_api_key = invocation.environment[DYNAMO_API_KEY_ENV]
        runtime_removed = (
            not Path(runtime_dir).exists() and not Path(launch_dir).exists()
        )
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
            "codex_version": PINNED_CODEX_VERSION,
            "capability": args.capability,
            "requested_sandbox_backend": invocation.sandbox_backend,
            "requested_codex_sandbox_mode": "workspace-write",
            "approval_policy": "never",
            "command_exit_code": execution.result.returncode,
            "assistant_reply_seen": CAPTURE_REPLY in execution.result.stdout,
            "cleanup_exit_code": execution.cleanup.returncode,
            "codex_temp_clean": execution.codex_temp_clean,
            "execution": execution_diagnostic(
                execution,
                runtime_removed,
                (invocation.environment[DYNAMO_API_KEY_ENV],),
            ),
        }
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0 if _capture_is_successful(assessment, execution, runtime_removed) else 1


def _run_dynamo(args: argparse.Namespace) -> int:
    validate_checkout(args.omnigent_repo)
    codex_path = resolve_codex_cli(args.codex_bin)
    with (
        tempfile.TemporaryDirectory(prefix="omnigent-well-lit-") as runtime_dir,
        tempfile.TemporaryDirectory(
            prefix=".omnigent-well-lit-launch-", dir=args.cwd.resolve()
        ) as launch_dir,
    ):
        invocation = build_invocation(
            omnigent_repo=args.omnigent_repo,
            cwd=args.cwd,
            runtime_root=Path(runtime_dir),
            launch_cwd=Path(launch_dir),
            base_url=args.base_url,
            model=args.model,
            prompt=args.prompt,
            codex_path=codex_path,
            capability=args.capability,
        )
        execution = execute_invocation(
            invocation,
            cwd=args.cwd,
            timeout=args.timeout,
            capture_output=False,
        )
    runtime_removed = not Path(runtime_dir).exists() and not Path(launch_dir).exists()
    validate_checkout(args.omnigent_repo)
    success = all(
        (
            execution.result.ok,
            execution.cleanup.ok,
            execution.codex_temp_clean,
            runtime_removed,
        )
    )
    if success:
        return 0
    print(
        json.dumps(
            {
                "omnigent_execution": execution_diagnostic(
                    execution,
                    runtime_removed,
                    (invocation.environment[DYNAMO_API_KEY_ENV],),
                )
            },
            indent=2,
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    if execution.result.returncode not in (None, 0):
        return execution.result.returncode
    return 124 if execution.result.timed_out else 1


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--omnigent-repo", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument(
        "--prompt", default="Reply with exactly: Omnigent Dynamo smoke passed"
    )
    parser.add_argument(
        "--capability",
        choices=("verify", "act"),
        default="verify",
        help="Default verify forbids edits in the prompt; act explicitly authorizes workspace edits.",
    )
    parser.add_argument(
        "--codex-bin",
        type=Path,
        help=f"Codex CLI executable; must report exact version {PINNED_CODEX_VERSION}.",
    )
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
