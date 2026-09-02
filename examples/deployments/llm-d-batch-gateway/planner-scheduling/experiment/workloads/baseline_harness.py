#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run an evidence-preserving Batch Gateway workload against an existing stack."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime as dt
import hashlib
import io
import json
import math
import platform
import re
import secrets
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = EXPERIMENT_ROOT.parents[5]
DEFAULT_DATASET = (
    WORKSPACE_ROOT / "datasets" / "gsm8k" / "batch-gateway" / "gsm8k-main-test.jsonl"
)
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[a-z0-9][a-z0-9-]{0,47}$")
CONTROLLER_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z-planner-loop-[a-f0-9]{6}$")
RUN_KINDS = {"baseline", "planner-controlled", "planner-native"}
NATIVE_PLANNER_DEFAULT_POD_NAME_REGEX = r"planner"
NATIVE_PLANNER_DEFAULT_DECISION_LOG_REGEX = r"Batch scheduling decision:"
NATIVE_PLANNER_DEFAULT_MIN_DECISION_LOGS = 2
KUBERNETES_DNS_SUBDOMAIN_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?$")
SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
HF_CREDENTIAL_RE = re.compile(r"\bhf_[A-Za-z0-9][A-Za-z0-9_%.-]{15,}")
BEARER_RE = re.compile(r"(?i)(\bBearer\s+)[^\s\"']+")
URL_RE = re.compile(r"[a-z][a-z0-9+.-]*://[^\s\"'<>]+", re.IGNORECASE)
HF_ASSIGNMENT_RE = re.compile(
    r"(?im)(\b(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\b\s*[:=]\s*)[^\s,}\]]+"
)
HF_JSON_ENV_RE = re.compile(
    r'(?is)("name"\s*:\s*"(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN)"\s*,\s*'
    r'"value"\s*:\s*")[^"]*(")'
)
HF_YAML_ENV_RE = re.compile(
    r"(?im)(-\s+name:\s*(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\s*\n"
    r"(?:\s+[^\n]+\n)*?\s+value:\s*)[^\n]+"
)


class HarnessError(RuntimeError):
    """Expected harness failure with a user-actionable message."""


def utc_now() -> dt.datetime:
    """Return a timezone-aware UTC timestamp."""
    return dt.datetime.now(dt.timezone.utc)


def isoformat_utc(value: dt.datetime | None = None) -> str:
    """Format an RFC 3339 timestamp using a Z suffix."""
    current = value or utc_now()
    return (
        current.astimezone(dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def make_run_id(
    now: dt.datetime | None = None,
    random_suffix: str | None = None,
    *,
    kind: str = "baseline",
) -> str:
    """Create an unambiguous UTC run identifier."""
    if kind not in RUN_KINDS:
        raise HarnessError(f"unsupported run kind: {kind}")
    current = (now or utc_now()).astimezone(dt.timezone.utc)
    suffix = random_suffix or secrets.token_hex(3)
    run_id = f"{current.strftime('%Y%m%dT%H%M%SZ')}-{kind}-{suffix.lower()}"
    if not RUN_ID_RE.fullmatch(run_id):
        raise HarnessError(f"generated invalid run ID: {run_id}")
    return run_id


def control_plane_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Describe who owns batch admission for this workload run."""
    if args.run_kind == "baseline":
        return {
            "mode": "stock",
            "standalone_controller_run_id": None,
            "native_planner": None,
        }
    if args.run_kind == "planner-controlled":
        return {
            "mode": "standalone-controller",
            "standalone_controller_run_id": args.paired_controller_run_id,
            "native_planner": None,
        }
    if args.run_kind == "planner-native":
        return {
            "mode": "native-planner",
            "standalone_controller_run_id": None,
            "native_planner": {
                "pod_name_regex": args.native_planner_pod_name_regex,
                "configmap": args.native_planner_configmap,
                "decision_log_regex": args.native_planner_decision_log_regex,
                "minimum_decision_logs": args.native_planner_min_decision_logs,
            },
        }
    raise HarnessError(f"unsupported run kind: {args.run_kind}")


def safe_name(value: str, fallback: str = "artifact") -> str:
    """Return a path-safe artifact name."""
    sanitized = SAFE_NAME_RE.sub("-", value.strip()).strip("-.")
    return sanitized[:96] or fallback


def safe_url(value: str) -> str:
    """Remove credentials, query parameters, and fragments from a recorded URL."""
    try:
        parsed = urllib.parse.urlsplit(value)
        hostname = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return "<redacted-url>"
    if not parsed.scheme or not hostname:
        return "<redacted-url>"
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    if port is not None:
        hostname = f"{hostname}:{port}"
    return urllib.parse.urlunsplit(
        (parsed.scheme, hostname, parsed.path, "<redacted>" if parsed.query else "", "")
    )


def redact_text(value: str) -> str:
    """Redact credential-shaped strings before they reach an artifact or console."""
    result = URL_RE.sub(lambda match: safe_url(match.group(0)), value)
    result = HF_JSON_ENV_RE.sub(r"\1<redacted>\2", result)
    result = HF_YAML_ENV_RE.sub(r"\1<redacted>", result)
    result = HF_ASSIGNMENT_RE.sub(r"\1<redacted>", result)
    result = HF_CREDENTIAL_RE.sub("<redacted-hugging-face-credential>", result)
    result = BEARER_RE.sub(r"\1<redacted>", result)
    return result


def write_text(path: Path, value: str) -> None:
    """Write sanitized UTF-8 text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redact_text(value), encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, sanitized JSON."""
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(file_handle: TextIO, value: Any) -> None:
    """Append one sanitized JSON object and flush it."""
    serialized = redact_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    file_handle.write(serialized + "\n")
    file_handle.flush()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Return a nearest-rank percentile for a non-empty sequence."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


class TeeTextIO(io.TextIOBase):
    """Mirror sanitized harness output to the console and a raw log."""

    def __init__(self, console: TextIO, artifact: TextIO) -> None:
        self.console = console
        self.artifact = artifact
        self._lock = threading.Lock()

    def write(self, value: str) -> int:
        sanitized = redact_text(value)
        with self._lock:
            self.console.write(sanitized)
            self.console.flush()
            self.artifact.write(sanitized)
            self.artifact.flush()
        return len(value)

    def flush(self) -> None:
        with self._lock:
            self.console.flush()
            self.artifact.flush()

    def isatty(self) -> bool:
        return self.console.isatty()


@dataclasses.dataclass(frozen=True)
class CapturedCommand:
    """One subprocess invocation and its preserved result."""

    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def capture_command(
    directory: Path,
    name: str,
    argv: Sequence[str],
    *,
    timeout_seconds: float = 30,
) -> CapturedCommand:
    """Run a command without a shell and preserve stdout, stderr, and exit code."""
    artifact_name = safe_name(name)
    command = [str(part) for part in argv]
    directory.mkdir(parents=True, exist_ok=True)
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except FileNotFoundError as error:
        exit_code = 127
        stdout = ""
        stderr = str(error)
    except subprocess.TimeoutExpired as error:
        exit_code = 124
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        stderr += f"\ncommand timed out after {timeout_seconds:.1f}s"
        timed_out = True

    result = CapturedCommand(command, exit_code, stdout, stderr, timed_out)
    write_text(directory / f"{artifact_name}.stdout", stdout)
    write_text(directory / f"{artifact_name}.stderr", stderr)
    write_text(directory / f"{artifact_name}.exit-code", f"{exit_code}\n")
    write_json(
        directory / f"{artifact_name}.command.json",
        {
            "argv": command,
            "exit_code": exit_code,
            "timed_out": timed_out,
        },
    )
    return result


def git_state(repo_root: Path, artifact_directory: Path) -> dict[str, Any]:
    """Capture revision and dirty paths without copying user diffs."""
    root = capture_command(
        artifact_directory,
        "repo-root",
        ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
    )
    if root.exit_code != 0 or not root.stdout.strip():
        detail = redact_text(root.stderr.strip()) or f"exit code {root.exit_code}"
        raise HarnessError(f"Git repository discovery failed: {detail}")
    resolved_root = Path(root.stdout.strip()).expanduser().resolve()
    if not resolved_root.is_dir():
        raise HarnessError(
            f"Git repository discovery returned a missing path: {resolved_root}"
        )
    revision = capture_command(
        artifact_directory,
        "revision",
        ["git", "-C", str(resolved_root), "rev-parse", "--verify", "HEAD^{commit}"],
    )
    revision_value = revision.stdout.strip()
    if (
        revision.exit_code != 0
        or "\n" in revision_value
        or re.fullmatch(r"[0-9a-fA-F]{40,64}", revision_value) is None
    ):
        detail = (
            redact_text(revision.stderr.strip()) or f"exit code {revision.exit_code}"
        )
        raise HarnessError(f"Git revision capture failed: {detail}")
    status = capture_command(
        artifact_directory,
        "status",
        [
            "git",
            "-C",
            str(resolved_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
    )
    if status.exit_code != 0:
        detail = redact_text(status.stderr.strip()) or f"exit code {status.exit_code}"
        raise HarnessError(f"Git status capture failed: {detail}")
    paths = [line[3:] for line in status.stdout.splitlines() if len(line) >= 4]
    return {
        "repo_root": str(resolved_root),
        "revision": revision_value,
        "dirty": bool(paths),
        "dirty_paths": paths,
    }


def _sanitized_url_flag_value(flag: str, value: str) -> str:
    """Sanitize one URL option value while preserving a metrics endpoint name."""
    if flag == "--metrics-url":
        name, separator, url = value.partition("=")
        if not separator:
            return "<redacted-metrics-url>"
        return f"{name}={safe_url(url)}"
    return safe_url(value)


def sanitized_command(argv: Sequence[str]) -> str:
    """Render a command while protecting URL credentials and queries."""
    safe_argv: list[str] = []
    url_flags = {"--batch-base-url", "--online-base-url", "--metrics-url"}
    pending_url_flag: str | None = None
    for item in argv:
        if pending_url_flag is not None:
            safe_argv.append(_sanitized_url_flag_value(pending_url_flag, item))
            pending_url_flag = None
            continue
        option, separator, value = item.partition("=")
        if separator and option in url_flags:
            safe_argv.append(f"{option}={_sanitized_url_flag_value(option, value)}")
            continue
        safe_argv.append(item)
        if item in url_flags:
            pending_url_flag = item
    return redact_text(shlex.join(safe_argv))


class RunContext:
    """Own raw-run metadata and top-level logs."""

    def __init__(self, args: argparse.Namespace, argv: Sequence[str]) -> None:
        self.args = args
        self.started = utc_now()
        self.run_id = make_run_id(self.started, kind=args.run_kind)
        self.run_dir = args.experiment_root / "results" / "raw" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.command = sanitized_command(argv)
        self.source = git_state(args.repo_root, self.run_dir / "source-state")
        self.stdout_file = (self.run_dir / "stdout.log").open(
            "w", encoding="utf-8", buffering=1
        )
        self.stderr_file = (self.run_dir / "stderr.log").open(
            "w", encoding="utf-8", buffering=1
        )
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = TeeTextIO(self.original_stdout, self.stdout_file)
        sys.stderr = TeeTextIO(self.original_stderr, self.stderr_file)
        control_plane = control_plane_metadata(args)
        self.metadata: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "kind": "preflight" if args.preflight_only else args.run_kind,
            "requested_run_kind": args.run_kind,
            "control_plane": control_plane,
            "status": "running",
            "started": isoformat_utc(self.started),
            "ended": None,
            "exit_code": None,
            "working_directory": str(Path.cwd()),
            "command": self.command,
            "source": self.source,
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "executable": sys.executable,
            },
            "configuration": {
                "namespace": args.namespace,
                "context": args.context or "current-context",
                "model": args.model,
                "batch_size": args.batch_size,
                "start_index": args.start_index,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "completion_window": args.completion_window,
                "poll_interval_seconds": args.poll_interval_seconds,
                "timeout_seconds": args.timeout_seconds,
                "batch_base_url": safe_url(args.batch_base_url),
                "online_base_url": safe_url(args.online_base_url),
                "online_rate": args.online_rate,
                "online_duration_seconds": args.online_duration_seconds,
                "online_max_inflight": args.online_max_inflight,
                "online_max_tokens": args.online_max_tokens,
                "expected_gate_types": args.expected_gate_type,
                "expected_worker_pool_id": args.expected_worker_pool_id,
                "paired_controller_run_id": args.paired_controller_run_id,
                "native_planner": control_plane["native_planner"],
                "pod_name_regex": args.pod_name_regex,
            },
            "inputs": {
                "dataset": str(args.dataset),
                "dataset_sha256": sha256_file(args.dataset),
            },
            "secret_handling": {
                "hugging_face_credentials": "not read",
                "kubernetes_secrets": "not queried",
                "http_authorization": "fixed non-credential placeholder",
            },
            "batch": {},
            "outputs": [],
            "notes": [],
        }
        self.write_metadata()

    def add_note(self, note: str) -> None:
        self.metadata["notes"].append(note)
        self.write_metadata()

    def set_batch(self, **fields: Any) -> None:
        self.metadata["batch"].update(fields)
        self.write_metadata()

    def write_metadata(self) -> None:
        write_json(self.run_dir / "metadata.json", self.metadata)
        outputs = "\n".join(
            f"  - {output}" for output in self.metadata.get("outputs", [])
        )
        if not outputs:
            outputs = "  - Pending"
        notes = "\n".join(f"  - {note}" for note in self.metadata.get("notes", []))
        if not notes:
            notes = "  - None"
        fence = chr(96) * 3
        markdown = f"""<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run {self.run_id}

- Status: {self.metadata["status"]}
- Run kind: {self.metadata["requested_run_kind"]}
- Batch control plane: {self.metadata["control_plane"]["mode"]}
- Started: {self.metadata["started"]}
- Ended: {self.metadata["ended"] or "pending"}
- Exit code: {self.metadata["exit_code"] if self.metadata["exit_code"] is not None else "pending"}
- Working directory: {self.metadata["working_directory"]}
- Source revision: {self.source["revision"]} ({"dirty" if self.source["dirty"] else "clean"})
- Host/runtime: {self.metadata["host"]["platform"]}; Python {self.metadata["host"]["python"]}
- Workload/config: workload-manifest.json; batch-input.jsonl
- Inputs: {self.args.dataset} ({self.metadata["inputs"]["dataset_sha256"]})

## Command

{fence}text
{self.command}
{fence}

## Environment

- NAMESPACE={self.args.namespace}
- Kubernetes context: {self.args.context or "current-context"}
- Model: {self.args.model}
- Hugging Face credential environment: not read or recorded

## Outputs

{outputs}

## Notes

{notes}
"""
        write_text(self.run_dir / "metadata.md", markdown)

    def finalize(self, exit_code: int) -> None:
        self.metadata["ended"] = isoformat_utc()
        self.metadata["exit_code"] = exit_code
        if exit_code == 130:
            self.metadata["status"] = "interrupted"
        elif exit_code == 0:
            self.metadata["status"] = "completed"
        else:
            self.metadata["status"] = "failed"
        write_text(self.run_dir / "exit_code.txt", f"{exit_code}\n")
        self.metadata["outputs"] = sorted(
            str(path.relative_to(self.run_dir))
            for path in self.run_dir.rglob("*")
            if path.is_file() and path.name not in {"metadata.json", "metadata.md"}
        )
        self.write_metadata()
        self.stdout_file.flush()
        self.stderr_file.flush()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.stdout_file.close()
        self.stderr_file.close()


def normalize_workload(
    source: Path,
    destination: Path,
    manifest_path: Path,
    *,
    batch_size: int,
    start_index: int,
    model: str,
    max_tokens: int,
    temperature: float,
    run_kind: str = "baseline",
) -> dict[str, Any]:
    """Write a deterministic fixed-configuration slice of a Batch JSONL source."""
    if run_kind not in RUN_KINDS:
        raise HarnessError(f"unsupported run kind: {run_kind}")
    custom_id_kind = "planner-native" if run_kind == "planner-native" else "baseline"
    selected: list[tuple[str, dict[str, Any]]] = []
    seen_source_ids: set[str] = set()
    source_position = 0
    with source.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise HarnessError(
                    f"dataset line {line_number} is invalid JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise HarnessError(f"dataset line {line_number} is not a JSON object")
            source_id = value.get("custom_id")
            if not isinstance(source_id, str) or not source_id:
                raise HarnessError(f"dataset line {line_number} has no custom_id")
            if source_id in seen_source_ids:
                raise HarnessError(f"dataset has duplicate custom_id {source_id!r}")
            seen_source_ids.add(source_id)
            if source_position >= start_index and len(selected) < batch_size:
                selected.append((source_id, value))
            source_position += 1
            if len(selected) >= batch_size:
                break

    if len(selected) != batch_size:
        raise HarnessError(
            f"requested {batch_size} records at index {start_index}, "
            f"but the dataset provided {len(selected)}"
        )

    mapping: list[dict[str, str]] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as output_file:
        for offset, (source_id, value) in enumerate(selected, start=1):
            if value.get("method") != "POST":
                raise HarnessError(f"{source_id} does not use method POST")
            if value.get("url") != "/v1/chat/completions":
                raise HarnessError(
                    f"{source_id} targets unsupported URL {value.get('url')!r}"
                )
            body = value.get("body")
            if not isinstance(body, dict):
                raise HarnessError(f"{source_id} has no object request body")
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise HarnessError(f"{source_id} has no messages")
            normalized_id = f"gsm8k-{custom_id_kind}-{start_index + offset:06d}"
            request = {
                "custom_id": normalized_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                },
            }
            append_jsonl(output_file, request)
            mapping.append(
                {"source_custom_id": source_id, "submitted_custom_id": normalized_id}
            )

    manifest = {
        "schema_version": "1.0",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": destination.name,
        "output_sha256": sha256_file(destination),
        "run_kind": run_kind,
        "custom_id_prefix": f"gsm8k-{custom_id_kind}-",
        "selection": {
            "start_index": start_index,
            "count": batch_size,
            "order": "source file order",
        },
        "fixed_request_configuration": {
            "method": "POST",
            "url": "/v1/chat/completions",
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        },
        "id_mapping": mapping,
    }
    write_json(manifest_path, manifest)
    return manifest


class JsonHttpClient:
    """Small urllib client with deterministic placeholder authentication."""

    def __init__(self, base_url: str, tenant: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.tenant = tenant
        self.timeout_seconds = timeout_seconds

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": "Bearer baseline-placeholder",
            "X-MaaS-Username": self.tenant,
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> bytes:
        headers = self.headers()
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            response_body = error.read().decode("utf-8", errors="replace")
            raise HarnessError(
                f"{method} {path} returned HTTP {error.code}: "
                f"{redact_text(response_body[:1000])}"
            ) from error
        except urllib.error.URLError as error:
            raise HarnessError(f"{method} {path} failed: {error.reason}") from error

    def json_request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        raw = self.request(
            method,
            path,
            body=body,
            content_type="application/json" if body is not None else None,
        )
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as error:
            raise HarnessError(f"{method} {path} returned invalid JSON") from error
        if not isinstance(result, dict):
            raise HarnessError(f"{method} {path} returned non-object JSON")
        return result


class BatchClient(JsonHttpClient):
    """OpenAI-compatible file and Batch API client."""

    def upload(self, path: Path) -> dict[str, Any]:
        boundary = f"planner-baseline-{uuid.uuid4().hex}"
        contents = path.read_bytes()
        parts = [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="purpose"\r\n\r\n',
            b"batch\r\n",
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{path.name}"\r\n'
            ).encode(),
            b"Content-Type: application/jsonl\r\n\r\n",
            contents,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        raw = self.request(
            "POST",
            "/v1/files",
            body=b"".join(parts),
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as error:
            raise HarnessError("file upload returned invalid JSON") from error
        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
            raise HarnessError(f"file upload returned no identifier: {result}")
        return result

    def create_batch(
        self,
        file_id: str,
        completion_window: str,
        request_count: int,
    ) -> dict[str, Any]:
        if (
            isinstance(request_count, bool)
            or not isinstance(request_count, int)
            or request_count <= 0
        ):
            raise HarnessError("request_count must be a positive integer")
        result = self.json_request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": completion_window,
                # Gateway v0.3 does not expose the uploaded file's line count in
                # request_counts until the first request finishes. Planner uses
                # this immutable declaration to avoid a cap=0 bootstrap deadlock.
                "metadata": {"planner_request_count": str(request_count)},
            },
        )
        if not isinstance(result.get("id"), str):
            raise HarnessError(f"batch creation returned no identifier: {result}")
        return result

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        return self.json_request("GET", f"/v1/batches/{batch_id}")

    def list_batches(self) -> dict[str, Any]:
        return self.json_request("GET", "/v1/batches?limit=1")

    def download_file(self, file_id: str) -> bytes:
        return self.request("GET", f"/v1/files/{file_id}/content")


def parse_request_counts(batch: dict[str, Any]) -> tuple[int, int, int]:
    """Return total, completed, and failed counts with schema validation."""
    counts = batch.get("request_counts")
    if counts is None:
        return 0, 0, 0
    if not isinstance(counts, dict):
        raise HarnessError("batch request_counts is not an object")
    values = tuple(counts.get(name, 0) for name in ("total", "completed", "failed"))
    if not all(isinstance(value, int) and value >= 0 for value in values):
        raise HarnessError(f"batch returned invalid request_counts: {counts}")
    total, completed, failed = values
    return total, completed, failed


def effective_request_total(
    batch: dict[str, Any], reported_total: int, expected_total: int
) -> tuple[int, str]:
    """Use the harness-declared total while Gateway v0.3 reports zero.

    The raw Gateway value remains separately recorded in every progress row.
    Once Gateway reports a positive total, it must match the immutable count
    supplied by this harness at job creation.
    """

    if reported_total > 0:
        if reported_total != expected_total:
            raise HarnessError(
                f"batch reported total {reported_total}, expected {expected_total}"
            )
        return reported_total, "gateway"

    metadata = batch.get("metadata")
    declared = (
        metadata.get("planner_request_count") if isinstance(metadata, dict) else None
    )
    if declared != str(expected_total):
        raise HarnessError(
            "batch reported total=0 without the expected planner_request_count metadata"
        )
    return expected_total, "planner_request_count"


def poll_batch(
    client: BatchClient,
    batch_id: str,
    output_path: Path,
    *,
    expected_total: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Poll a Batch job and preserve every progress observation."""
    started_mono = time.monotonic()
    prior_completed = 0
    prior_elapsed = 0.0
    last_status = ""
    with output_path.open("w", encoding="utf-8", buffering=1) as progress_file:
        while True:
            elapsed = time.monotonic() - started_mono
            if elapsed > timeout_seconds:
                raise HarnessError(
                    f"batch {batch_id} did not finish within {timeout_seconds:.1f}s"
                )
            batch = client.get_batch(batch_id)
            status = batch.get("status")
            if not isinstance(status, str):
                raise HarnessError(
                    f"batch {batch_id} returned invalid status: {status}"
                )
            reported_total, completed, failed = parse_request_counts(batch)
            total, total_source = effective_request_total(
                batch, reported_total, expected_total
            )
            delta_elapsed = elapsed - prior_elapsed
            delta_completed = completed - prior_completed
            observation = {
                "observed_at": isoformat_utc(),
                "elapsed_seconds": round(elapsed, 6),
                "status": status,
                "total": total,
                "reported_total": reported_total,
                "total_source": total_source,
                "completed": completed,
                "failed": failed,
                "remaining": max(0, total - completed - failed),
                "delta_completed": delta_completed,
                "interval_seconds": round(delta_elapsed, 6),
                "interval_completion_rate_rps": (
                    delta_completed / delta_elapsed if delta_elapsed > 0 else None
                ),
            }
            append_jsonl(progress_file, observation)
            if status != last_status or delta_completed > 0:
                print(
                    f"batch {batch_id}: {status} completed={completed}/{total} "
                    f"failed={failed} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                last_status = status
            if status in TERMINAL_STATUSES:
                write_json(output_path.with_name("terminal-batch.json"), batch)
                return batch
            prior_completed = completed
            prior_elapsed = elapsed
            remaining = timeout_seconds - elapsed
            time.sleep(min(poll_interval_seconds, max(0.0, remaining)))


class MetricsSampler:
    """Periodically snapshot explicitly configured metric URLs."""

    def __init__(
        self,
        run_dir: Path,
        endpoints: list[tuple[str, str]],
        interval_seconds: float,
    ) -> None:
        self.run_dir = run_dir
        self.endpoints = endpoints
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: str | None = None
        self.sample_count = 0
        self.failure_count = 0
        self.endpoint_counts: dict[str, dict[str, int]] = {
            name: {"successful_samples": 0, "failed_samples": 0}
            for name, _url in endpoints
        }

    def start(self) -> None:
        if not self.endpoints:
            return
        self.thread = threading.Thread(target=self._run, name="metrics-sampler")
        self.thread.start()

    def _snapshot(self) -> None:
        observed = utc_now()
        stamp = observed.strftime("%Y%m%dT%H%M%S.%fZ")
        for name, url in self.endpoints:
            endpoint_dir = self.run_dir / "metrics" / safe_name(name)
            try:
                request = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(request, timeout=10) as response:
                    payload = response.read().decode("utf-8", errors="replace")
                    status = getattr(response, "status", 200)
                if not payload.strip():
                    raise HarnessError("metrics endpoint returned an empty response")
                write_text(endpoint_dir / f"{stamp}.prom", payload)
                write_json(
                    endpoint_dir / f"{stamp}.json",
                    {
                        "observed_at": isoformat_utc(observed),
                        "url": safe_url(url),
                        "http_status": status,
                    },
                )
                self.sample_count += 1
                self.endpoint_counts[name]["successful_samples"] += 1
            except Exception as error:  # noqa: BLE001 - preserve sampler failures
                error_text = f"{type(error).__name__}: {redact_text(str(error))}"
                write_json(
                    endpoint_dir / f"{stamp}.error.json",
                    {
                        "observed_at": isoformat_utc(observed),
                        "url": safe_url(url),
                        "error_type": type(error).__name__,
                        "error": redact_text(str(error)),
                    },
                )
                self.failure_count += 1
                self.endpoint_counts[name]["failed_samples"] += 1
                self.error = (
                    f"{self.failure_count} configured metric sample(s) failed; "
                    f"latest endpoint={name}: {error_text}"
                )

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                self._snapshot()
                self.stop_event.wait(self.interval_seconds)
        except Exception as error:  # noqa: BLE001 - thread boundary records failures
            self.error = f"{type(error).__name__}: {redact_text(str(error))}"

    def stop(self) -> dict[str, Any]:
        if self.thread is None:
            return {"enabled": False, "samples": 0, "error": None}
        self.stop_event.set()
        self.thread.join(timeout=max(15.0, self.interval_seconds + 5.0))
        if self.thread.is_alive():
            self.error = "metrics sampler did not stop"
        missing_endpoints = sorted(
            name
            for name, counts in self.endpoint_counts.items()
            if counts["successful_samples"] == 0
        )
        if missing_endpoints and self.error is None:
            self.error = (
                "configured metric endpoints produced no successful samples: "
                + ", ".join(missing_endpoints)
            )
        summary = {
            "enabled": True,
            "samples": self.sample_count,
            "failed_samples": self.failure_count,
            "error": self.error,
            "endpoints": [
                {
                    "name": name,
                    "url": safe_url(url),
                    **self.endpoint_counts[name],
                }
                for name, url in self.endpoints
            ],
        }
        write_json(self.run_dir / "metrics-summary.json", summary)
        return summary


class OnlineLoadRunner:
    """Drive fixed streaming online requests at an open-loop constant rate."""

    def __init__(self, args: argparse.Namespace, run_dir: Path) -> None:
        self.args = args
        self.run_dir = run_dir
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.internal_error: str | None = None
        self.results: list[dict[str, Any]] = []
        self._results_lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._output_path = run_dir / "online-requests.jsonl"
        self._output_file: TextIO | None = None

    @property
    def enabled(self) -> bool:
        return self.args.online_rate > 0 and self.args.online_duration_seconds > 0

    def start(self) -> None:
        if not self.enabled:
            return
        self._output_file = self._output_path.open("w", encoding="utf-8", buffering=1)
        self.thread = threading.Thread(target=self._run, name="online-load")
        self.thread.start()

    def _record(self, result: dict[str, Any]) -> None:
        with self._results_lock:
            self.results.append(result)
        if self._output_file is not None:
            with self._file_lock:
                append_jsonl(self._output_file, result)

    def _request(self, index: int, scheduled_mono: float, started_mono: float) -> None:
        request_started = time.monotonic()
        result: dict[str, Any] = {
            "request_index": index,
            "scheduled_offset_seconds": scheduled_mono - started_mono,
            "started_at": isoformat_utc(),
            "queue_delay_ms": max(0.0, (request_started - scheduled_mono) * 1000),
            "http_status": None,
            "ok": False,
            "ttft_ms": None,
            "latency_ms": None,
            "content_type": None,
            "stream_protocol_seen": False,
            "parsed_event_count": 0,
            "malformed_event_count": 0,
            "content_seen": False,
            "done_seen": False,
            "events_after_done": 0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error_type": None,
            "error": None,
        }
        payload = {
            "model": self.args.model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Reply with ONLINE {index:08d} and no other text.",
                }
            ],
            "max_tokens": self.args.online_max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request = urllib.request.Request(
            f"{self.args.online_base_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer baseline-placeholder",
                "Content-Type": "application/json",
                "X-MaaS-Username": self.args.tenant,
            },
            method="POST",
        )
        first_content_mono: float | None = None
        try:
            with urllib.request.urlopen(
                request, timeout=self.args.request_timeout_seconds
            ) as response:
                result["http_status"] = getattr(response, "status", 200)
                headers = getattr(response, "headers", None)
                content_type = headers.get("Content-Type", "") if headers else ""
                result["content_type"] = content_type
                media_type = content_type.partition(";")[0].strip().lower()
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    result["stream_protocol_seen"] = True
                    event_time = time.monotonic()
                    raw_data = line[5:].strip()
                    if result["done_seen"]:
                        result["events_after_done"] += 1
                        continue
                    if raw_data == "[DONE]":
                        result["done_seen"] = True
                        continue
                    try:
                        event = json.loads(raw_data)
                    except json.JSONDecodeError:
                        result["malformed_event_count"] += 1
                        continue
                    if not isinstance(event, dict):
                        result["malformed_event_count"] += 1
                        continue
                    result["parsed_event_count"] += 1
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        result["prompt_tokens"] = usage.get("prompt_tokens")
                        result["completion_tokens"] = usage.get("completion_tokens")
                    choices = event.get("choices")
                    if isinstance(choices, list):
                        for choice in choices:
                            if not isinstance(choice, dict):
                                continue
                            delta = choice.get("delta")
                            if not isinstance(delta, dict):
                                continue
                            content = delta.get("content") or delta.get(
                                "reasoning_content"
                            )
                            if isinstance(content, str) and content:
                                result["content_seen"] = True
                                if first_content_mono is None:
                                    first_content_mono = event_time
                ended_mono = time.monotonic()
                if first_content_mono is not None:
                    result["ttft_ms"] = (first_content_mono - request_started) * 1000
                result["latency_ms"] = (ended_mono - request_started) * 1000
                protocol_errors = []
                if not 200 <= int(result["http_status"]) < 300:
                    protocol_errors.append(
                        f"unexpected HTTP status {result['http_status']}"
                    )
                if media_type != "text/event-stream":
                    protocol_errors.append(
                        f"expected text/event-stream, got {content_type or '<missing>'}"
                    )
                if not result["stream_protocol_seen"]:
                    protocol_errors.append("response contained no SSE data fields")
                if result["parsed_event_count"] == 0:
                    protocol_errors.append("response contained no valid JSON SSE event")
                if result["malformed_event_count"]:
                    protocol_errors.append(
                        "response contained malformed SSE JSON event data"
                    )
                if not result["content_seen"]:
                    protocol_errors.append("response contained no streamed content")
                if not result["done_seen"]:
                    protocol_errors.append("response contained no [DONE] marker")
                if result["events_after_done"]:
                    protocol_errors.append("response contained data after [DONE]")
                if protocol_errors:
                    result["error_type"] = "invalid_sse_response"
                    result["error"] = "; ".join(protocol_errors)
                else:
                    result["ok"] = True
        except urllib.error.HTTPError as error:
            ended_mono = time.monotonic()
            result["http_status"] = error.code
            result["latency_ms"] = (ended_mono - request_started) * 1000
            result["error_type"] = type(error).__name__
            result["error"] = f"HTTP {error.code}"
        except Exception as error:  # noqa: BLE001 - preserve per-request failures
            ended_mono = time.monotonic()
            result["latency_ms"] = (ended_mono - request_started) * 1000
            result["error_type"] = type(error).__name__
            result["error"] = redact_text(str(error))
        result["ended_at"] = isoformat_utc()
        self._record(result)

    def _run(self) -> None:
        started = time.monotonic()
        deadline = started + self.args.online_duration_seconds
        interval = 1.0 / self.args.online_rate
        semaphore = threading.BoundedSemaphore(self.args.online_max_inflight)

        def bounded_request(index: int, scheduled: float) -> None:
            try:
                self._request(index, scheduled, started)
            finally:
                semaphore.release()

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.args.online_max_inflight,
                thread_name_prefix="online-request",
            ) as executor:
                index = 0
                while not self.stop_event.is_set():
                    scheduled = started + index * interval
                    if scheduled >= deadline:
                        break
                    wait_seconds = scheduled - time.monotonic()
                    if wait_seconds > 0 and self.stop_event.wait(wait_seconds):
                        break
                    if not semaphore.acquire(blocking=False):
                        self._record(
                            {
                                "request_index": index,
                                "scheduled_offset_seconds": scheduled - started,
                                "started_at": None,
                                "ended_at": isoformat_utc(),
                                "queue_delay_ms": None,
                                "http_status": None,
                                "ok": False,
                                "ttft_ms": None,
                                "latency_ms": None,
                                "stream_protocol_seen": False,
                                "prompt_tokens": None,
                                "completion_tokens": None,
                                "error_type": "max_inflight",
                                "error": (
                                    "request skipped because max inflight was reached"
                                ),
                            }
                        )
                    else:
                        executor.submit(bounded_request, index, scheduled)
                    index += 1
        except Exception as error:  # noqa: BLE001 - thread boundary records failures
            self.internal_error = f"{type(error).__name__}: {redact_text(str(error))}"

    def finish(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        if self.thread is None:
            raise HarnessError("online load was enabled but never started")
        self.thread.join(
            timeout=self.args.online_duration_seconds
            + self.args.request_timeout_seconds
            + 30
        )
        if self.thread.is_alive():
            self.stop_event.set()
            self.internal_error = "online load thread did not finish"
            self.thread.join(timeout=self.args.request_timeout_seconds + 5)
        if self._output_file is not None:
            if self.thread.is_alive():
                self._output_file.flush()
            else:
                self._output_file.close()
                self._output_file = None
        successes = [result for result in self.results if result.get("ok")]
        ttft_values = [
            float(result["ttft_ms"])
            for result in successes
            if isinstance(result.get("ttft_ms"), (int, float))
        ]
        latency_values = [
            float(result["latency_ms"])
            for result in successes
            if isinstance(result.get("latency_ms"), (int, float))
        ]
        summary = {
            "enabled": True,
            "scheduled_requests": len(self.results),
            "successful_requests": len(successes),
            "failed_requests": len(self.results) - len(successes),
            "error_rate": (
                (len(self.results) - len(successes)) / len(self.results)
                if self.results
                else None
            ),
            "ttft_ms": {
                "count": len(ttft_values),
                "mean": statistics.fmean(ttft_values) if ttft_values else None,
                "p50": percentile(ttft_values, 0.50),
                "p95": percentile(ttft_values, 0.95),
                "p99": percentile(ttft_values, 0.99),
            },
            "latency_ms": {
                "count": len(latency_values),
                "mean": statistics.fmean(latency_values) if latency_values else None,
                "p50": percentile(latency_values, 0.50),
                "p95": percentile(latency_values, 0.95),
                "p99": percentile(latency_values, 0.99),
            },
            "internal_error": self.internal_error,
        }
        write_json(self.run_dir / "online-summary.json", summary)
        if self.internal_error:
            raise HarnessError(f"online load failed: {self.internal_error}")
        return summary

    def stop(self) -> None:
        self.stop_event.set()


class KubernetesEvidence:
    """Collect scoped read-only state, ConfigMaps, logs, images, and metrics."""

    def __init__(
        self, args: argparse.Namespace, run_dir: Path, run_started: dt.datetime
    ) -> None:
        self.args = args
        self.run_dir = run_dir
        self.run_started = run_started
        self.pod_pattern = re.compile(args.pod_name_regex)
        self.native_planner_pod_pattern = (
            re.compile(args.native_planner_pod_name_regex)
            if args.native_planner_pod_name_regex
            else None
        )
        self.native_planner_decision_pattern = (
            re.compile(args.native_planner_decision_log_regex)
            if args.native_planner_decision_log_regex
            else None
        )

    def kubectl(self, *args: str) -> list[str]:
        command = ["kubectl"]
        if self.args.context:
            command.extend(["--context", self.args.context])
        command.extend(args)
        return command

    def _selected_pod_names(self, inventory: str) -> list[str]:
        names = []
        for line in inventory.splitlines():
            name = line.strip().removeprefix("pod/")
            if name and self.pod_pattern.search(name):
                names.append(name)
        return sorted(set(names))

    def _configmap_names(self, pods: list[dict[str, Any]]) -> list[str]:
        names: set[str] = set()
        for pod in pods:
            spec = pod.get("spec", {})
            if not isinstance(spec, dict):
                continue
            for volume in spec.get("volumes", []):
                if not isinstance(volume, dict):
                    continue
                config_map = volume.get("configMap")
                if isinstance(config_map, dict) and isinstance(
                    config_map.get("name"), str
                ):
                    names.add(config_map["name"])
            containers = list(spec.get("containers", [])) + list(
                spec.get("initContainers", [])
            )
            for container in containers:
                if not isinstance(container, dict):
                    continue
                for env_from in container.get("envFrom", []):
                    if not isinstance(env_from, dict):
                        continue
                    reference = env_from.get("configMapRef")
                    if isinstance(reference, dict) and isinstance(
                        reference.get("name"), str
                    ):
                        names.add(reference["name"])
                for env in container.get("env", []):
                    if not isinstance(env, dict):
                        continue
                    value_from = env.get("valueFrom")
                    if not isinstance(value_from, dict):
                        continue
                    reference = value_from.get("configMapKeyRef")
                    if isinstance(reference, dict) and isinstance(
                        reference.get("name"), str
                    ):
                        names.add(reference["name"])
        return sorted(names)

    def _direct_gate_types(self, arguments: list[str]) -> list[str]:
        """Resolve the pinned chart's unambiguous legacy single-queue gate."""
        selected_transports = self._argument_option_values(
            arguments, "--message-queue-impl"
        )
        if selected_transports != ["redis-sortedset"]:
            return []
        present_options = {argument.partition("=")[0] for argument in arguments}
        if {
            "--redis.ss.queues-config",
            "--redis.ss.queues-config-file",
        }.intersection(present_options):
            return []
        gate_types = self._argument_option_values(arguments, "--redis.ss.gate-type")
        if (
            len(gate_types) != 1
            or re.fullmatch(r"[a-zA-Z0-9-]+", gate_types[0]) is None
        ):
            return []
        return [gate_types[0].lower()]

    def _argument_option_values(
        self, arguments: list[str], expected_option: str
    ) -> list[str]:
        """Read split and inline values for one exact command-line option."""
        values = []
        for index, argument in enumerate(arguments):
            option, separator, inline_value = argument.partition("=")
            if option != expected_option:
                continue
            if separator:
                values.append(inline_value)
            elif index + 1 < len(arguments):
                values.append(arguments[index + 1])
        return values

    def _configured_transport_queue(self, value: str) -> dict[str, Any] | None:
        """Return the single unambiguous queue from an Async transport config."""
        try:
            document = json.loads(value)
        except json.JSONDecodeError:
            return None
        if not isinstance(document, dict):
            return None
        queues = document.get("queues")
        if (
            not isinstance(queues, list)
            or len(queues) != 1
            or not isinstance(queues[0], dict)
        ):
            return None
        return queues[0]

    def _configured_legacy_redis_queue(self, value: str) -> dict[str, Any] | None:
        """Return the single queue from v0.9 redis.ss.queues-config JSON."""
        try:
            document = json.loads(value)
        except json.JSONDecodeError:
            return None
        if (
            not isinstance(document, list)
            or len(document) != 1
            or not isinstance(document[0], dict)
        ):
            return None
        return document[0]

    def _transport_config_gate_types(self, arguments: list[str]) -> list[str]:
        """Resolve a gate from the one inline/split Async transport config."""
        values = self._argument_option_values(arguments, "--transport-config")
        if len(values) != 1:
            return []
        return self._configured_transport_gate_types(values[0])

    def _configured_transport_gate_types(self, value: str) -> list[str]:
        """Resolve the queue gate from one transport-config document."""
        queue = self._configured_transport_queue(value)
        if queue is None:
            return []
        return self._configured_queue_gate_types(queue)

    def _configured_queue_gate_types(self, queue: dict[str, Any]) -> list[str]:
        """Resolve a structurally parsed queue's gate type."""
        gate_type = queue.get("gate_type", queue.get("gateType"))
        if (
            not isinstance(gate_type, str)
            or re.fullmatch(r"[a-zA-Z0-9-]+", gate_type) is None
        ):
            return []
        return [gate_type.lower()]

    def _transport_queue_pool_id(self, queue: dict[str, Any]) -> str | None:
        """Resolve the queue's effective pool, including Async's default."""
        pool_id = queue.get("worker_pool_id", queue.get("workerPoolId", "default"))
        if not isinstance(pool_id, str) or not pool_id:
            return None
        return pool_id

    def _configured_pool_gate_types(self, value: str) -> list[str]:
        """Resolve the effective gate for the one expected worker pool."""
        try:
            document = json.loads(value)
        except json.JSONDecodeError:
            return []
        if not isinstance(document, list):
            return []
        expected_pool = self.args.expected_worker_pool_id
        pools = [
            pool
            for pool in document
            if isinstance(pool, dict) and pool.get("id") == expected_pool
        ]
        if len(pools) != 1:
            return []

        gate = pools[0]
        for _depth in range(4):
            gate_type = gate.get("gate_type", gate.get("gateType"))
            if not isinstance(gate_type, str) or not gate_type:
                return []
            normalized = gate_type.lower()
            if normalized != "wait-on-refuse":
                return [normalized]
            gate_params = gate.get("gate_params", gate.get("gateParams"))
            if not isinstance(gate_params, dict):
                return []
            nested = gate_params.get("gate")
            if not isinstance(nested, dict):
                return []
            gate = nested
        return []

    def _is_llmd_async_pod(self, pod: dict[str, Any]) -> bool:
        """Identify the live llm-d Async pod without trusting unrelated resources."""
        metadata = pod.get("metadata", {})
        labels = metadata.get("labels", {})
        spec = pod.get("spec", {})
        identities = [
            metadata.get("name"),
            labels.get("app.kubernetes.io/name") if isinstance(labels, dict) else None,
            labels.get("app.kubernetes.io/instance")
            if isinstance(labels, dict)
            else None,
        ]
        for container in spec.get("containers", []):
            if not isinstance(container, dict):
                continue
            identities.extend(
                [
                    container.get("name"),
                    container.get("image"),
                    *container.get("command", []),
                ]
            )
        identity = " ".join(str(value) for value in identities if value).lower()
        return "llm-d-async" in identity or "async-dispatch" in identity

    def _is_llmd_async_container(self, container: dict[str, Any]) -> bool:
        """Identify the Async process rather than a sidecar in the same pod."""
        identities = [
            container.get("name"),
            container.get("image"),
            *container.get("command", []),
        ]
        identity = " ".join(str(value) for value in identities if value).lower()
        return "llm-d-async" in identity

    def _active_async_pod_names(self, pods: list[dict[str, Any]]) -> list[str]:
        """Return Running Async pods whose Async container is Ready."""
        names = []
        for pod in pods:
            if (
                not self._is_llmd_async_pod(pod)
                or pod.get("status", {}).get("phase") != "Running"
            ):
                continue
            statuses = {
                str(status.get("name")): status
                for status in pod.get("status", {}).get("containerStatuses", [])
                if isinstance(status, dict) and status.get("name")
            }
            ready_async_container = any(
                self._is_llmd_async_container(container)
                and statuses.get(str(container.get("name")), {}).get("ready") is True
                for container in pod.get("spec", {}).get("containers", [])
                if isinstance(container, dict)
            )
            name = pod.get("metadata", {}).get("name")
            if ready_async_container and isinstance(name, str) and name:
                names.append(name)
        return sorted(set(names))

    def _container_argument_paths(self, container: dict[str, Any]) -> list[str]:
        """Return absolute paths that the running container was told to consume."""
        values = [
            str(value)
            for value in list(container.get("command", []))
            + list(container.get("args", []))
        ]
        paths: set[str] = set()
        for index, value in enumerate(values):
            if value.startswith("/"):
                paths.add(value)
            _option, separator, inline_value = value.partition("=")
            if separator and inline_value.startswith("/"):
                paths.add(inline_value)
            if (
                value.startswith("-")
                and index + 1 < len(values)
                and values[index + 1].startswith("/")
            ):
                paths.add(values[index + 1])
        return sorted(paths)

    def _mounted_argument_configs(
        self,
        container: dict[str, Any],
        volumes: dict[str, dict[str, Any]],
        configmaps: dict[str, dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Read ConfigMap keys mounted at exact paths consumed by the process."""
        configs: list[dict[str, str]] = []
        for argument_path in self._container_argument_paths(container):
            for mount in container.get("volumeMounts", []):
                if not isinstance(mount, dict):
                    continue
                mount_path = str(mount.get("mountPath", "")).rstrip("/")
                if not mount_path:
                    continue
                relative_path = None
                sub_path = mount.get("subPath")
                if isinstance(sub_path, str) and argument_path == mount_path:
                    relative_path = sub_path
                elif argument_path.startswith(f"{mount_path}/"):
                    relative_path = argument_path[len(mount_path) + 1 :]
                if relative_path is None:
                    continue
                volume = volumes.get(str(mount.get("name")), {})
                config_map_ref = volume.get("configMap", {})
                items = config_map_ref.get("items", [])
                if items:
                    config_key = next(
                        (
                            str(item.get("key"))
                            for item in items
                            if isinstance(item, dict)
                            and item.get("path") == relative_path
                            and item.get("key")
                        ),
                        None,
                    )
                else:
                    config_key = relative_path
                if config_key is None:
                    continue
                config_map_name = config_map_ref.get("name")
                if not isinstance(config_map_name, str):
                    continue
                config_map = configmaps.get(config_map_name, {})
                data = config_map.get("data", {})
                config_value = data.get(config_key) if isinstance(data, dict) else None
                if not isinstance(config_value, str):
                    continue
                configs.append(
                    {
                        "value": config_value,
                        "configmap": config_map_name,
                        "key": config_key,
                        "path": argument_path,
                    }
                )
        return configs

    def _active_gate_evidence(
        self,
        pods: list[dict[str, Any]],
        configmaps: dict[str, dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Bind gate discovery to live Async args or the exact config key they use."""
        evidence: list[dict[str, str]] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        active_pods = set(self._active_async_pod_names(pods))
        for pod in pods:
            pod_name = str(pod.get("metadata", {}).get("name", ""))
            if pod_name not in active_pods:
                continue
            spec = pod.get("spec", {})
            volumes = {
                str(volume.get("name")): volume
                for volume in spec.get("volumes", [])
                if isinstance(volume, dict) and volume.get("name")
            }
            for container in spec.get("containers", []):
                if not isinstance(container, dict) or not self._is_llmd_async_container(
                    container
                ):
                    continue
                container_name = str(container.get("name", ""))
                arguments = [
                    str(value)
                    for value in list(container.get("command", []))
                    + list(container.get("args", []))
                ]
                mounted_configs = self._mounted_argument_configs(
                    container, volumes, configmaps
                )
                inline_transport_values = self._argument_option_values(
                    arguments, "--transport-config"
                )
                transport_file_paths = set(
                    self._argument_option_values(arguments, "--transport-config-file")
                )
                mounted_transport_configs = [
                    config
                    for config in mounted_configs
                    if config["path"] in transport_file_paths
                ]
                transport_sources = [
                    {
                        "value": value,
                        "source": "transport-config-queue",
                    }
                    for value in inline_transport_values
                ] + [
                    {
                        **config,
                        "source": "mounted-transport-config-key",
                    }
                    for config in mounted_transport_configs
                ]
                transport_queue = None
                run_kind = getattr(self.args, "run_kind", "baseline")
                planner_mode = run_kind in {"planner-controlled", "planner-native"}
                argument_options = {
                    argument.partition("=")[0] for argument in arguments
                }
                uses_new_transport = bool(
                    {
                        "--transport",
                        "--transport-config",
                        "--transport-config-file",
                    }.intersection(argument_options)
                )
                queue_source: dict[str, str] | None = None
                if uses_new_transport:
                    if len(transport_sources) != 1:
                        continue
                    transport_queue = self._configured_transport_queue(
                        transport_sources[0]["value"]
                    )
                    if transport_queue is None:
                        continue
                    queue_source = transport_sources[0]
                else:
                    legacy_queue_file_paths = set(
                        self._argument_option_values(
                            arguments, "--redis.ss.queues-config-file"
                        )
                    )
                    mounted_legacy_queue_configs = [
                        config
                        for config in mounted_configs
                        if config["path"] in legacy_queue_file_paths
                    ]
                    legacy_queue_sources = [
                        {
                            "value": value,
                            "source": "legacy-queues-config",
                        }
                        for value in self._argument_option_values(
                            arguments, "--redis.ss.queues-config"
                        )
                    ] + [
                        {
                            **config,
                            "source": "mounted-legacy-queues-config-key",
                        }
                        for config in mounted_legacy_queue_configs
                    ]
                    legacy_queue_options_present = bool(
                        {
                            "--redis.ss.queues-config",
                            "--redis.ss.queues-config-file",
                        }.intersection(argument_options)
                    )
                    if legacy_queue_options_present:
                        if (
                            self._argument_option_values(
                                arguments, "--message-queue-impl"
                            )
                            != ["redis-sortedset"]
                            or len(legacy_queue_sources) != 1
                        ):
                            continue
                        transport_queue = self._configured_legacy_redis_queue(
                            legacy_queue_sources[0]["value"]
                        )
                        if transport_queue is None:
                            continue
                        queue_source = legacy_queue_sources[0]

                if queue_source is not None:
                    queue_gate_types = self._configured_queue_gate_types(
                        transport_queue
                    )
                    if queue_gate_types and not planner_mode:
                        pool_id = self._transport_queue_pool_id(transport_queue)
                        source = queue_source["source"]
                        for gate_type in queue_gate_types:
                            config_map_name = queue_source.get("configmap", "")
                            config_key = queue_source.get("key", "")
                            key = (
                                pod_name,
                                container_name,
                                gate_type,
                                source,
                                f"{config_map_name}/{config_key}",
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            record = {
                                "pod": pod_name,
                                "container": container_name,
                                "gate_type": gate_type,
                                "source": source,
                                "pool_id": pool_id or "unknown",
                            }
                            for field in ("configmap", "key", "path"):
                                value = queue_source.get(field)
                                if isinstance(value, str):
                                    record[field] = value
                            evidence.append(record)
                        continue

                direct_gate_types = self._direct_gate_types(arguments)
                if direct_gate_types and not planner_mode and not uses_new_transport:
                    for gate_type in direct_gate_types:
                        key = (
                            pod_name,
                            container_name,
                            gate_type,
                            "container-args",
                            "",
                        )
                        if key not in seen:
                            seen.add(key)
                            evidence.append(
                                {
                                    "pod": pod_name,
                                    "container": container_name,
                                    "gate_type": gate_type,
                                    "source": "legacy-container-args",
                                    "pool_id": "legacy-single-queue",
                                }
                            )
                    continue

                pool_file_paths = set(
                    self._argument_option_values(arguments, "--pool-config-file")
                )
                mounted_pool_configs = [
                    config
                    for config in mounted_configs
                    if config["path"] in pool_file_paths
                ]
                if (
                    transport_queue is None
                    or self._transport_queue_pool_id(transport_queue)
                    != self.args.expected_worker_pool_id
                    or len(pool_file_paths) != 1
                    or len(mounted_pool_configs) != 1
                ):
                    continue
                for config in mounted_pool_configs:
                    for gate_type in self._configured_pool_gate_types(config["value"]):
                        key = (
                            pod_name,
                            container_name,
                            gate_type,
                            config["configmap"],
                            config["key"],
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        evidence.append(
                            {
                                "pod": pod_name,
                                "container": container_name,
                                "gate_type": gate_type,
                                "source": "mounted-configmap-key",
                                "pool_id": self.args.expected_worker_pool_id,
                                "configmap": config["configmap"],
                                "key": config["key"],
                                "path": config["path"],
                            }
                        )
        return evidence

    def _pod_images(self, pods: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records = []
        for pod in pods:
            metadata = pod.get("metadata", {})
            spec = pod.get("spec", {})
            status = pod.get("status", {})
            statuses = {
                value.get("name"): value
                for value in list(status.get("containerStatuses", []))
                + list(status.get("initContainerStatuses", []))
                if isinstance(value, dict)
            }
            for container in list(spec.get("initContainers", [])) + list(
                spec.get("containers", [])
            ):
                if not isinstance(container, dict):
                    continue
                container_status = statuses.get(container.get("name"), {})
                records.append(
                    {
                        "pod": metadata.get("name"),
                        "phase": status.get("phase"),
                        "container": container.get("name"),
                        "image": container.get("image"),
                        "image_id": container_status.get("imageID"),
                        "ready": container_status.get("ready"),
                        "restart_count": container_status.get("restartCount"),
                    }
                )
        return records

    def _metric_ports(self, pod: dict[str, Any]) -> list[tuple[str, int]]:
        endpoints: set[tuple[str, int]] = set()
        spec = pod.get("spec", {})
        for container in spec.get("containers", []):
            if not isinstance(container, dict):
                continue
            container_name = str(container.get("name", "container"))
            for port in container.get("ports", []):
                if not isinstance(port, dict):
                    continue
                number = port.get("containerPort")
                name = str(port.get("name", ""))
                if isinstance(number, int) and (
                    "metric" in name.lower() or number in {8080, 9090, 9091}
                ):
                    endpoints.add((f"{container_name}-{name or number}", number))
        return sorted(endpoints)

    def preflight(self) -> dict[str, Any]:
        directory = self.run_dir / "kubernetes" / "preflight"
        context = capture_command(
            directory, "current-context", self.kubectl("config", "current-context")
        )
        namespace = capture_command(
            directory,
            "namespace",
            self.kubectl("get", "namespace", self.args.namespace, "-o", "json"),
        )
        can_get = capture_command(
            directory,
            "can-get-pods",
            self.kubectl("auth", "can-i", "get", "pods", "-n", self.args.namespace),
        )
        version = capture_command(
            directory, "version", self.kubectl("version", "-o", "json")
        )
        failures = []
        if context.exit_code != 0:
            failures.append("cannot resolve Kubernetes context")
        if namespace.exit_code != 0:
            failures.append(f"namespace {self.args.namespace!r} is unavailable")
        if can_get.exit_code != 0 or can_get.stdout.strip().lower() != "yes":
            failures.append("current identity cannot read pods")
        if version.exit_code != 0:
            failures.append("cannot query Kubernetes version")
        if failures:
            raise HarnessError("Kubernetes preflight failed: " + "; ".join(failures))
        return self.capture_phase("start", include_logs=True)

    def capture_phase(self, phase: str, *, include_logs: bool) -> dict[str, Any]:
        directory = self.run_dir / "kubernetes" / phase
        inventory = capture_command(
            directory,
            "pod-inventory",
            self.kubectl("get", "pods", "-n", self.args.namespace, "-o", "name"),
        )
        if inventory.exit_code != 0:
            raise HarnessError(f"failed to list pods during {phase} evidence capture")
        names = self._selected_pod_names(inventory.stdout)
        if not names:
            raise HarnessError(
                f"no pods matched --pod-name-regex {self.args.pod_name_regex!r}"
            )

        pods: list[dict[str, Any]] = []
        evidence_commands: dict[str, int] = {"pod-inventory": inventory.exit_code}
        for name in names:
            artifact = f"pod-{name}"
            captured = capture_command(
                directory,
                artifact,
                self.kubectl(
                    "get", "pod", name, "-n", self.args.namespace, "-o", "json"
                ),
            )
            evidence_commands[artifact] = captured.exit_code
            if captured.exit_code != 0:
                continue
            try:
                pod = json.loads(captured.stdout)
            except json.JSONDecodeError:
                continue
            if isinstance(pod, dict):
                pods.append(pod)

        pod_images = self._pod_images(pods)
        write_json(directory / "selected-pods.json", {"items": pods})
        write_json(directory / "images.json", pod_images)

        native_planner_pods = [
            pod
            for pod in pods
            if self.native_planner_pod_pattern is not None
            and self.native_planner_pod_pattern.search(
                str(pod.get("metadata", {}).get("name", ""))
            )
        ]
        native_planner_pod_names = sorted(
            str(pod.get("metadata", {}).get("name", ""))
            for pod in native_planner_pods
            if pod.get("metadata", {}).get("name")
        )
        native_planner_configmaps = self._configmap_names(native_planner_pods)
        native_planner_log_matches = 0
        native_planner_log_commands: dict[str, int] = {}

        configmaps: dict[str, dict[str, Any]] = {}
        for name in self._configmap_names(pods):
            artifact = f"configmap-{name}"
            captured = capture_command(
                directory,
                artifact,
                self.kubectl(
                    "get", "configmap", name, "-n", self.args.namespace, "-o", "json"
                ),
            )
            evidence_commands[artifact] = captured.exit_code
            if captured.exit_code == 0:
                with contextlib.suppress(json.JSONDecodeError):
                    config_map = json.loads(captured.stdout)
                    if isinstance(config_map, dict):
                        configmaps[name] = config_map

        resource_inventory = capture_command(
            directory,
            "resource-inventory",
            self.kubectl(
                "get",
                "deployment,statefulset,daemonset,service",
                "-n",
                self.args.namespace,
                "-o",
                "name",
            ),
        )
        evidence_commands["resource-inventory"] = resource_inventory.exit_code
        if resource_inventory.exit_code == 0:
            for resource in resource_inventory.stdout.splitlines():
                resource = resource.strip()
                short_name = resource.rsplit("/", 1)[-1]
                if not resource or not self.pod_pattern.search(short_name):
                    continue
                artifact = f"resource-{resource.replace('/', '-')}"
                captured = capture_command(
                    directory,
                    artifact,
                    self.kubectl(
                        "get", resource, "-n", self.args.namespace, "-o", "yaml"
                    ),
                )
                evidence_commands[artifact] = captured.exit_code

        for pod in pods:
            pod_name = str(pod.get("metadata", {}).get("name", ""))
            if not pod_name:
                continue
            event_name = f"events-{pod_name}"
            event_capture = capture_command(
                directory,
                event_name,
                self.kubectl(
                    "get",
                    "events",
                    "-n",
                    self.args.namespace,
                    "--field-selector",
                    f"involvedObject.name={pod_name}",
                    "-o",
                    "json",
                ),
            )
            evidence_commands[event_name] = event_capture.exit_code
            if include_logs:
                log_args = [
                    "logs",
                    pod_name,
                    "-n",
                    self.args.namespace,
                    "--all-containers=true",
                    "--timestamps=true",
                ]
                if phase == "start":
                    log_args.append("--tail=200")
                else:
                    log_args.extend(
                        [
                            "--since-time",
                            self.run_started.isoformat().replace("+00:00", "Z"),
                        ]
                    )
                log_name = f"logs-{pod_name}"
                log_capture = capture_command(
                    directory, log_name, self.kubectl(*log_args), timeout_seconds=60
                )
                evidence_commands[log_name] = log_capture.exit_code
                if (
                    pod_name in native_planner_pod_names
                    and self.native_planner_decision_pattern is not None
                ):
                    native_planner_log_commands[log_name] = log_capture.exit_code
                    if log_capture.exit_code == 0:
                        native_planner_log_matches += len(
                            self.native_planner_decision_pattern.findall(
                                log_capture.stdout
                            )
                        )
                statuses = pod.get("status", {}).get("containerStatuses", [])
                for status in statuses:
                    if not isinstance(status, dict) or not status.get("restartCount"):
                        continue
                    container = status.get("name")
                    previous_name = f"logs-previous-{pod_name}-{container}"
                    previous = capture_command(
                        directory,
                        previous_name,
                        self.kubectl(
                            "logs",
                            pod_name,
                            "-n",
                            self.args.namespace,
                            "--container",
                            str(container),
                            "--previous",
                            "--timestamps=true",
                        ),
                        timeout_seconds=60,
                    )
                    evidence_commands[previous_name] = previous.exit_code

            for endpoint_name, port in self._metric_ports(pod):
                metric_name = f"metrics-{pod_name}-{endpoint_name}"
                proxy_path = (
                    f"/api/v1/namespaces/{self.args.namespace}/pods/"
                    f"{pod_name}:{port}/proxy/metrics"
                )
                metric_capture = capture_command(
                    directory,
                    metric_name,
                    self.kubectl("get", "--raw", proxy_path),
                    timeout_seconds=15,
                )
                evidence_commands[metric_name] = metric_capture.exit_code

        phases = [pod.get("status", {}).get("phase") for pod in pods]
        active_async_pods = self._active_async_pod_names(pods)
        active_gate_evidence = self._active_gate_evidence(pods, configmaps)
        gate_types = sorted({record["gate_type"] for record in active_gate_evidence})
        native_planner_summary = None
        if self.args.run_kind == "planner-native":
            expected_configmap = self.args.native_planner_configmap
            running_pods = sorted(
                str(pod.get("metadata", {}).get("name", ""))
                for pod in native_planner_pods
                if pod.get("status", {}).get("phase") == "Running"
            )
            native_planner_summary = {
                "expected_pod_name_regex": self.args.native_planner_pod_name_regex,
                "expected_configmap": expected_configmap,
                "expected_decision_log_regex": (
                    self.args.native_planner_decision_log_regex
                ),
                "minimum_decision_logs": self.args.native_planner_min_decision_logs,
                "matched_pods": native_planner_pod_names,
                "running_pods": running_pods,
                "mounted_configmaps": native_planner_configmaps,
                "expected_configmap_mounted": (
                    expected_configmap in native_planner_configmaps
                ),
                "expected_configmap_captured": expected_configmap in configmaps,
                "images": [
                    image
                    for image in pod_images
                    if image.get("pod") in native_planner_pod_names
                ],
                "decision_log_match_count": native_planner_log_matches,
                "log_commands": native_planner_log_commands,
            }
        summary = {
            "phase": phase,
            "captured_at": isoformat_utc(),
            "pod_name_regex": self.args.pod_name_regex,
            "selected_pods": names,
            "pod_phases": phases,
            "configmaps": sorted(configmaps),
            "active_async_pods": active_async_pods,
            "expected_worker_pool_id": self.args.expected_worker_pool_id,
            "gate_types": gate_types,
            "active_gate_evidence": active_gate_evidence,
            "commands": evidence_commands,
            "native_planner": native_planner_summary,
        }
        write_json(directory / "summary.json", summary)
        if not any(pod_phase == "Running" for pod_phase in phases):
            raise HarnessError("no selected evidence pod is Running")
        return summary


def parse_metrics_urls(values: list[str]) -> list[tuple[str, str]]:
    """Parse NAME=URL metric endpoints and reject duplicates or unsafe schemes."""
    endpoints = []
    names: set[str] = set()
    for value in values:
        if "=" not in value:
            raise HarnessError("--metrics-url must use NAME=URL")
        name, url = value.split("=", 1)
        name = safe_name(name)
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HarnessError(f"invalid metric URL: {safe_url(url)}")
        if parsed.username or parsed.password or parsed.query:
            raise HarnessError(
                "metric URLs must not contain credentials or query strings"
            )
        if name in names:
            raise HarnessError(f"duplicate metric endpoint name: {name}")
        names.add(name)
        endpoints.append((name, url))
    return endpoints


def validate_args(args: argparse.Namespace) -> None:
    """Validate configuration before creating traffic."""
    if not args.dataset.is_file():
        raise HarnessError(f"dataset does not exist: {args.dataset}")
    if args.batch_size <= 0:
        raise HarnessError("--batch-size must be positive")
    if args.start_index < 0:
        raise HarnessError("--start-index cannot be negative")
    if args.max_tokens <= 0 or args.online_max_tokens <= 0:
        raise HarnessError("token limits must be positive")
    if not 0 <= args.temperature <= 2:
        raise HarnessError("--temperature must be between 0 and 2")
    if args.poll_interval_seconds <= 0 or args.timeout_seconds <= 0:
        raise HarnessError("poll and timeout values must be positive")
    if args.metrics_interval_seconds <= 0 or args.request_timeout_seconds <= 0:
        raise HarnessError("metrics and request timeout values must be positive")
    if args.online_rate < 0 or args.online_duration_seconds < 0:
        raise HarnessError("online rate and duration cannot be negative")
    if (args.online_rate == 0) != (args.online_duration_seconds == 0):
        raise HarnessError(
            "set both --online-rate and --online-duration-seconds, or leave both zero"
        )
    if args.online_max_inflight <= 0:
        raise HarnessError("--online-max-inflight must be positive")
    if not args.model.strip():
        raise HarnessError("--model cannot be empty")
    if not args.expected_worker_pool_id.strip():
        raise HarnessError("--expected-worker-pool-id cannot be empty")
    if args.run_kind == "planner-controlled" and not args.paired_controller_run_id:
        raise HarnessError(
            "--paired-controller-run-id is required for planner-controlled runs"
        )
    if args.paired_controller_run_id and not CONTROLLER_RUN_ID_RE.fullmatch(
        args.paired_controller_run_id
    ):
        raise HarnessError("--paired-controller-run-id has an invalid run ID format")
    if args.run_kind != "planner-controlled" and args.paired_controller_run_id:
        raise HarnessError(
            "--paired-controller-run-id is valid only with --run-kind "
            "planner-controlled"
        )
    native_options = {
        "--native-planner-pod-name-regex": args.native_planner_pod_name_regex,
        "--native-planner-configmap": args.native_planner_configmap,
        "--native-planner-decision-log-regex": (args.native_planner_decision_log_regex),
        "--native-planner-min-decision-logs": (args.native_planner_min_decision_logs),
    }
    if args.run_kind == "planner-native":
        if not args.native_planner_configmap:
            raise HarnessError(
                "--native-planner-configmap is required for planner-native runs"
            )
        if len(
            args.native_planner_configmap
        ) > 253 or not KUBERNETES_DNS_SUBDOMAIN_RE.fullmatch(
            args.native_planner_configmap
        ):
            raise HarnessError(
                "--native-planner-configmap must be a Kubernetes DNS subdomain name"
            )
        if not args.native_planner_pod_name_regex.strip():
            raise HarnessError("--native-planner-pod-name-regex cannot be empty")
        if not args.native_planner_decision_log_regex.strip():
            raise HarnessError("--native-planner-decision-log-regex cannot be empty")
        if {value.lower() for value in args.expected_gate_type} != {
            "redis-leased-rate"
        }:
            raise HarnessError(
                "planner-native runs require --expected-gate-type redis-leased-rate"
            )
        if args.skip_gate_verification:
            raise HarnessError(
                "--skip-gate-verification is not allowed for planner-native runs"
            )
    elif any(value is not None for value in native_options.values()):
        supplied = ", ".join(
            name for name, value in native_options.items() if value is not None
        )
        raise HarnessError(f"{supplied} are valid only with --run-kind planner-native")
    if args.skip_cluster_preflight and not args.preflight_only:
        raise HarnessError(
            "--skip-cluster-preflight is allowed only with --preflight-only"
        )
    if args.skip_api_preflight and not args.preflight_only:
        raise HarnessError("--skip-api-preflight is allowed only with --preflight-only")
    try:
        re.compile(args.pod_name_regex)
    except re.error as error:
        raise HarnessError(f"invalid --pod-name-regex: {error}") from error
    for option, pattern in (
        ("--native-planner-pod-name-regex", args.native_planner_pod_name_regex),
        (
            "--native-planner-decision-log-regex",
            args.native_planner_decision_log_regex,
        ),
    ):
        if pattern is None:
            continue
        try:
            re.compile(pattern)
        except re.error as error:
            raise HarnessError(f"invalid {option}: {error}") from error


def validate_gate(summary: dict[str, Any], expected: list[str], skip: bool) -> None:
    """Require evidence that the live run uses an allowed dispatch gate."""
    if skip:
        return
    allowed = {value.lower() for value in expected}
    active_pods = summary.get("active_async_pods")
    if not isinstance(active_pods, list) or not active_pods:
        raise HarnessError("could not identify a live llm-d Async pod")
    records = summary.get("active_gate_evidence")
    if not isinstance(records, list) or not records:
        raise HarnessError(
            "could not bind a dispatch gate to a live llm-d Async container argument "
            "or the exact mounted config key it consumes; use "
            "--skip-gate-verification only after manually preserving the active config"
        )
    by_pod: dict[str, set[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        pod = record.get("pod")
        gate_type = record.get("gate_type")
        if isinstance(pod, str) and pod and isinstance(gate_type, str):
            by_pod.setdefault(pod, set()).add(gate_type.lower())
    if not by_pod:
        raise HarnessError("active gate evidence contains no live pod-bound gate types")
    mismatched = {}
    for pod in active_pods:
        if not isinstance(pod, str) or not pod:
            continue
        gate_types = by_pod.get(pod, set())
        if not gate_types.intersection(allowed):
            mismatched[pod] = sorted(gate_types)
    if mismatched:
        raise HarnessError(
            f"expected every live llm-d Async pod to expose one of {sorted(allowed)} "
            f"but found mismatches {mismatched}"
        )


def validate_native_planner_evidence(
    args: argparse.Namespace,
    summary: dict[str, Any],
    *,
    require_decisions: bool,
) -> None:
    """Prove that a native run is owned by a live, configured Planner."""
    if args.run_kind != "planner-native":
        return
    native = summary.get("native_planner")
    if not isinstance(native, dict):
        raise HarnessError("native Planner evidence is missing from Kubernetes capture")
    matched_pods = native.get("matched_pods")
    if not isinstance(matched_pods, list) or not matched_pods:
        raise HarnessError(
            "no selected pod matched --native-planner-pod-name-regex "
            f"{args.native_planner_pod_name_regex!r}"
        )
    running_pods = native.get("running_pods")
    if not isinstance(running_pods, list) or not running_pods:
        raise HarnessError("no matching native Planner pod is Running")
    if native.get("expected_configmap_mounted") is not True:
        raise HarnessError(
            f"native Planner pods do not mount ConfigMap "
            f"{args.native_planner_configmap!r}"
        )
    if native.get("expected_configmap_captured") is not True:
        raise HarnessError(
            f"could not capture native Planner ConfigMap "
            f"{args.native_planner_configmap!r}"
        )
    if not require_decisions:
        return
    observed = native.get("decision_log_match_count")
    minimum = args.native_planner_min_decision_logs
    if not isinstance(observed, int) or observed < minimum:
        raise HarnessError(
            "native Planner emitted fewer in-run batch scheduling decisions than "
            f"required: observed={observed!r}, required={minimum}"
        )


def validate_terminal_results(
    run_dir: Path,
    terminal: dict[str, Any],
    expected_custom_ids: set[str],
    client: BatchClient,
    allow_request_failures: bool,
) -> dict[str, Any]:
    """Download result files and validate terminal counts and custom IDs."""
    expected_count = len(expected_custom_ids)
    status = terminal.get("status")
    if status != "completed":
        raise HarnessError(f"batch reached terminal status {status!r}")
    total, completed, failed = parse_request_counts(terminal)
    if total != expected_count:
        raise HarnessError(
            f"terminal total {total} does not match input {expected_count}"
        )
    if completed + failed != total:
        raise HarnessError(
            f"terminal counts are incomplete: completed={completed}, failed={failed}, "
            f"total={total}"
        )
    if failed and not allow_request_failures:
        raise HarnessError(f"batch completed with {failed} failed requests")

    output_count = 0
    error_count = 0
    custom_ids: set[str] = set()
    for field, filename in (
        ("output_file_id", "batch-output.jsonl"),
        ("error_file_id", "batch-errors.jsonl"),
    ):
        file_id = terminal.get(field)
        if not file_id:
            continue
        if not isinstance(file_id, str):
            raise HarnessError(f"{field} is not a string")
        contents = client.download_file(file_id)
        try:
            result_text = contents.decode("utf-8")
        except UnicodeDecodeError as error:
            raise HarnessError(f"{filename} is not valid UTF-8") from error
        path = run_dir / filename
        count = 0
        for line_number, line in enumerate(result_text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise HarnessError(
                    f"{filename} line {line_number} is invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise HarnessError(f"{filename} line {line_number} is not an object")
            custom_id = value.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                raise HarnessError(f"{filename} line {line_number} has no custom_id")
            if custom_id in custom_ids:
                raise HarnessError(
                    f"result files contain duplicate custom_id {custom_id!r}"
                )
            custom_ids.add(custom_id)
            count += 1
        write_text(path, result_text)
        if field == "output_file_id":
            output_count = count
        else:
            error_count = count

    if output_count != completed or error_count != failed:
        raise HarnessError(
            f"downloaded result counts differ: output={output_count}/{completed}, "
            f"errors={error_count}/{failed}"
        )
    missing_custom_ids = expected_custom_ids - custom_ids
    unexpected_custom_ids = custom_ids - expected_custom_ids
    if missing_custom_ids or unexpected_custom_ids:
        raise HarnessError(
            "downloaded result custom_id set differs from the submitted workload: "
            f"missing_count={len(missing_custom_ids)}, "
            f"missing_sample={sorted(missing_custom_ids)[:10]}, "
            f"unexpected_count={len(unexpected_custom_ids)}, "
            f"unexpected_sample={sorted(unexpected_custom_ids)[:10]}"
        )
    validation = {
        "expected_total": expected_count,
        "terminal_total": total,
        "completed": completed,
        "failed": failed,
        "downloaded_output_lines": output_count,
        "downloaded_error_lines": error_count,
        "unique_custom_ids": len(custom_ids),
        "custom_id_set_matches": True,
        "valid": True,
    }
    write_json(run_dir / "result-validation.json", validation)
    return validation


def local_tool_versions(run_dir: Path) -> dict[str, Any]:
    """Capture non-secret local tool versions."""
    directory = run_dir / "local-preflight"
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "bash": None,
        "kubectl": None,
    }
    bash = capture_command(directory, "bash-version", ["bash", "--version"])
    if bash.exit_code == 0:
        versions["bash"] = bash.stdout.splitlines()[0] if bash.stdout else "unknown"
    kubectl = capture_command(
        directory,
        "kubectl-client-version",
        ["kubectl", "version", "--client", "-o", "json"],
    )
    if kubectl.exit_code == 0:
        with contextlib.suppress(json.JSONDecodeError):
            versions["kubectl"] = json.loads(kubectl.stdout)
    write_json(directory / "versions.json", versions)
    return versions


def execute(args: argparse.Namespace, context: RunContext) -> int:
    """Execute preflight or a full workload run."""
    print(f"run ID: {context.run_id}")
    print(f"raw results: {context.run_dir}")
    validate_args(args)
    metrics_endpoints = parse_metrics_urls(args.metrics_url)
    local_tool_versions(context.run_dir)

    workload = normalize_workload(
        args.dataset,
        context.run_dir / "batch-input.jsonl",
        context.run_dir / "workload-manifest.json",
        batch_size=args.batch_size,
        start_index=args.start_index,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        run_kind=args.run_kind,
    )
    submitted_custom_ids = {
        mapping["submitted_custom_id"] for mapping in workload["id_mapping"]
    }
    if len(submitted_custom_ids) != args.batch_size:
        raise HarnessError(
            "normalized workload did not produce the expected unique custom_id set"
        )
    context.metadata["inputs"]["submitted_sha256"] = workload["output_sha256"]
    context.write_metadata()
    print(
        f"prepared {args.batch_size} GSM8K requests "
        f"(sha256={workload['output_sha256']})"
    )

    evidence: KubernetesEvidence | None = None
    if not args.skip_cluster_preflight:
        evidence = KubernetesEvidence(args, context.run_dir, context.started)
        start_summary = evidence.preflight()
        validate_gate(
            start_summary, args.expected_gate_type, args.skip_gate_verification
        )
        context.metadata["kubernetes_start"] = start_summary
        context.write_metadata()
        validate_native_planner_evidence(args, start_summary, require_decisions=False)
        print(
            "Kubernetes preflight passed; gate types: "
            + ", ".join(start_summary["gate_types"])
        )
    else:
        context.add_note("Cluster preflight skipped for local-only validation.")

    batch_client = BatchClient(
        args.batch_base_url, args.tenant, args.request_timeout_seconds
    )
    if not args.skip_api_preflight:
        listed = batch_client.list_batches()
        write_json(context.run_dir / "batch-api-preflight.json", listed)
        print("Batch API preflight passed")
    else:
        context.add_note("Batch API preflight skipped for local-only validation.")

    if args.preflight_only:
        write_json(
            context.run_dir / "preflight-summary.json",
            {
                "status": "completed",
                "dataset_records_prepared": args.batch_size,
                "cluster_checked": not args.skip_cluster_preflight,
                "api_checked": not args.skip_api_preflight,
                "traffic_created": False,
                "requested_run_kind": args.run_kind,
                "control_plane": control_plane_metadata(args),
            },
        )
        print("preflight completed; no Batch job or online traffic was created")
        return 0

    metrics = MetricsSampler(
        context.run_dir, metrics_endpoints, args.metrics_interval_seconds
    )
    online = OnlineLoadRunner(args, context.run_dir)
    core_error: BaseException | None = None
    terminal: dict[str, Any] | None = None
    online_summary: dict[str, Any] = {"enabled": False}
    metrics_summary: dict[str, Any] = {"enabled": False}
    try:
        upload = batch_client.upload(context.run_dir / "batch-input.jsonl")
        write_json(context.run_dir / "file-upload.json", upload)
        batch = batch_client.create_batch(
            upload["id"], args.completion_window, args.batch_size
        )
        write_json(context.run_dir / "batch-created.json", batch)
        batch_id = batch["id"]
        context.set_batch(id=batch_id, created_at=isoformat_utc())
        print(f"created batch {batch_id}")

        metrics.start()
        online.start()
        terminal = poll_batch(
            batch_client,
            batch_id,
            context.run_dir / "progress.jsonl",
            expected_total=args.batch_size,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        validate_terminal_results(
            context.run_dir,
            terminal,
            submitted_custom_ids,
            batch_client,
            args.allow_request_failures,
        )
    except BaseException as error:  # noqa: BLE001 - cleanup must survive interrupts
        core_error = error
        online.stop()
    finally:
        try:
            online_summary = online.finish()
        except BaseException as error:  # noqa: BLE001 - preserve both failures
            if core_error is None:
                core_error = error
            else:
                context.add_note(f"Online load cleanup also failed: {error}")
        metrics_summary = metrics.stop()
        if metrics_summary.get("error") and core_error is None:
            core_error = HarnessError(
                f"metrics sampler failed: {metrics_summary['error']}"
            )
        if evidence is not None:
            try:
                end_summary = evidence.capture_phase("end", include_logs=True)
                validate_gate(
                    end_summary, args.expected_gate_type, args.skip_gate_verification
                )
                context.metadata["kubernetes_end"] = end_summary
                context.write_metadata()
                validate_native_planner_evidence(
                    args, end_summary, require_decisions=True
                )
            except BaseException as error:  # noqa: BLE001 - preserve both failures
                if core_error is None:
                    core_error = HarnessError(f"end evidence capture failed: {error}")
                else:
                    context.add_note(f"End evidence capture also failed: {error}")

    context.metadata["online_summary"] = online_summary
    context.metadata["metrics_summary"] = metrics_summary
    if terminal is not None:
        total, completed, failed = parse_request_counts(terminal)
        context.set_batch(
            terminal_status=terminal.get("status"),
            total=total,
            completed=completed,
            failed=failed,
        )
    context.write_metadata()
    if core_error is not None:
        raise core_error
    print(f"{args.run_kind} completed and terminal results validated")
    return 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit a deterministic GSM8K Batch job to an existing deployment and "
            "preserve progress, optional online load, and read-only evidence."
        )
    )
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--repo-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--context", help="kubectl context (default: current context)")
    parser.add_argument("--tenant", default="planner-poc-baseline")
    parser.add_argument(
        "--run-kind",
        choices=("baseline", "planner-controlled", "planner-native"),
        default="baseline",
        help="Treatment label stored in the run ID and machine-readable metadata",
    )
    parser.add_argument(
        "--paired-controller-run-id",
        help="Controller run paired with a planner-controlled workload",
    )
    parser.add_argument(
        "--native-planner-pod-name-regex",
        help=(
            "Regex identifying the native Planner pod inside the scoped "
            "Kubernetes evidence"
        ),
    )
    parser.add_argument(
        "--native-planner-configmap",
        help="ConfigMap that must be mounted and captured for a planner-native run",
    )
    parser.add_argument(
        "--native-planner-decision-log-regex",
        help="Regex proving that the native Planner made in-run batch decisions",
    )
    parser.add_argument(
        "--native-planner-min-decision-logs",
        type=positive_int,
        help="Minimum matching in-run native Planner tick decisions",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=positive_int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=positive_int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--completion-window", default="24h")
    parser.add_argument("--batch-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--online-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--online-rate", type=nonnegative_float, default=0.0)
    parser.add_argument(
        "--online-duration-seconds", type=nonnegative_float, default=0.0
    )
    parser.add_argument("--online-max-inflight", type=positive_int, default=32)
    parser.add_argument("--online-max-tokens", type=positive_int, default=16)
    parser.add_argument(
        "--metrics-url",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="Unauthenticated Prometheus exposition endpoint to snapshot periodically",
    )
    parser.add_argument("--metrics-interval-seconds", type=float, default=15.0)
    parser.add_argument(
        "--pod-name-regex",
        default=r"batch-gateway|async-dispatch|qwen3-0-6b-batch",
        help="Regex selecting pods and related resources for scoped evidence",
    )
    parser.add_argument(
        "--expected-gate-type",
        action="append",
        default=[],
        help="Allowed dispatch gate type; repeat for multiple allowed values",
    )
    parser.add_argument(
        "--expected-worker-pool-id",
        default="dynamo-batch",
        help="Worker pool whose active llm-d Async gate must be proven",
    )
    parser.add_argument(
        "--skip-gate-verification",
        action="store_true",
        help="Do not require gate type discovery after manually preserving config",
    )
    parser.add_argument("--allow-request-failures", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Prepare and validate inputs without submitting traffic",
    )
    parser.add_argument(
        "--skip-cluster-preflight",
        action="store_true",
        help="Local preflight only: do not invoke kubectl",
    )
    parser.add_argument(
        "--skip-api-preflight",
        action="store_true",
        help="Local preflight only: do not contact the Batch API",
    )
    args = parser.parse_args(argv)
    args.experiment_root = args.experiment_root.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.dataset = args.dataset.expanduser().resolve()
    if args.run_kind == "planner-native":
        if args.native_planner_pod_name_regex is None:
            args.native_planner_pod_name_regex = NATIVE_PLANNER_DEFAULT_POD_NAME_REGEX
        if args.native_planner_decision_log_regex is None:
            args.native_planner_decision_log_regex = (
                NATIVE_PLANNER_DEFAULT_DECISION_LOG_REGEX
            )
        if args.native_planner_min_decision_logs is None:
            args.native_planner_min_decision_logs = (
                NATIVE_PLANNER_DEFAULT_MIN_DECISION_LOGS
            )
    if not args.expected_gate_type:
        args.expected_gate_type = (
            ["redis-leased-rate"] if args.run_kind == "planner-native" else ["constant"]
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    parsed_argv = list(argv if argv is not None else sys.argv[1:])
    args = parse_args(parsed_argv)
    context: RunContext | None = None
    exit_code = 1

    def interrupt_handler(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt_handler)
    signal.signal(signal.SIGINT, interrupt_handler)
    try:
        validate_args(args)
        context = RunContext(
            args, [sys.executable, str(Path(__file__).resolve()), *parsed_argv]
        )
        exit_code = execute(args, context)
    except KeyboardInterrupt:
        exit_code = 130
        print(f"{args.run_kind} interrupted", file=sys.stderr)
    except HarnessError as error:
        exit_code = 1
        print(f"error: {error}", file=sys.stderr)
    except BaseException:  # noqa: BLE001 - persist an artifact for unexpected exits
        exit_code = 1
        traceback.print_exc()
    finally:
        if context is not None:
            context.finalize(exit_code)
            print(f"run {context.run_id} exited with code {exit_code}")
            print(f"artifacts: {context.run_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
