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

"""Run one live Planner Gym cell with an optional concurrent Batch job.

This is an outer evidence harness. It does not deploy, patch, scale, or delete
Kubernetes resources. Batch arms delegate submission and evidence capture to
``baseline_harness``; online-only reuses its read-only Kubernetes and metrics
collectors directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

import baseline_harness

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DYNAMO_REPO_ROOT = EXPERIMENT_ROOT.parents[4]
WORKAREA_ROOT = DYNAMO_REPO_ROOT.parent
DEFAULT_PLANNER_GYM_ROOT = WORKAREA_ROOT / "planner-batch-impact" / "gym"
DEFAULT_PLANNER_GYM_SCRIPT = (
    DEFAULT_PLANNER_GYM_ROOT / "scripts" / "run_match_config.py"
)
DEFAULT_PLANNER_GYM_PYTHON = DEFAULT_PLANNER_GYM_ROOT / ".venv" / "bin" / "python"
DEFAULT_AIPERF_EXECUTABLE = DEFAULT_PLANNER_GYM_ROOT / ".venv" / "bin" / "aiperf"
DEFAULT_BATCH_HARNESS = Path(__file__).resolve().with_name("baseline_harness.py")
ARMS = ("online-only", "stock", "planner-native")
CAMPAIGN_POD_NAME_REGEX = (
    r"^(batch-gateway-|async-dispatch-llm-d-async-|"
    r"qwen3-0-6b-batch-(frontend|worker|planner)-)"
)
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z-planner-gym-batch-impact-[a-f0-9]{6}$")
BATCH_CREATED_RE = re.compile(r"^created batch (?P<batch_id>\S+)\s*$")
HARNESS_EVENT_PREFIX = baseline_harness.HARNESS_EVENT_PREFIX
MAX_CHILD_LOG_BYTES = 16 * 1024 * 1024
LOG_TRUNCATION_MARKER = "\n<output truncated by campaign runner>\n"
CHILD_TERMINATION_GRACE_SECONDS = 30.0


class CampaignError(RuntimeError):
    """Expected campaign failure with a user-actionable message."""


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
    now: dt.datetime | None = None, random_suffix: str | None = None
) -> str:
    """Create a collision-resistant UTC run identifier."""
    current = (now or utc_now()).astimezone(dt.timezone.utc)
    suffix = (random_suffix or secrets.token_hex(3)).lower()
    run_id = f"{current.strftime('%Y%m%dT%H%M%SZ')}-planner-gym-batch-impact-{suffix}"
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise CampaignError(f"generated invalid run ID: {run_id}")
    return run_id


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _normalize_executable(value: str) -> str:
    """Make path-like executable values absolute without dereferencing venv links."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or value.startswith(".") or len(candidate.parts) > 1:
        return str(candidate.absolute())
    return value


def _resolved_executable(value: str) -> str:
    """Return the executable selected by PATH, when discoverable."""
    return shutil.which(value) or value


def _validate_plain_http_url(value: str, option: str) -> None:
    """Reject URL-embedded credentials that cannot be preserved safely."""
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise CampaignError(f"{option} is not a valid URL: {error}") from error
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise CampaignError(f"{option} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise CampaignError(
            f"{option} must not embed credentials, query parameters, or fragments"
        )
    if port is not None and not 1 <= port <= 65535:
        raise CampaignError(f"{option} contains an invalid port")


def _safe_command(argv: Sequence[str]) -> str:
    """Render a command while removing URL credentials and queries."""
    url_flags = {"--endpoint-url", "--batch-base-url", "--metrics-url"}
    safe_argv: list[str] = []
    pending_url = False
    for item in argv:
        if pending_url:
            name, separator, url = item.partition("=")
            safe_argv.append(
                f"{name}={baseline_harness.safe_url(url)}"
                if separator
                else baseline_harness.safe_url(item)
            )
            pending_url = False
            continue
        option, separator, value = item.partition("=")
        if separator and option in url_flags:
            if option == "--metrics-url":
                name, name_separator, url = value.partition("=")
                sanitized = (
                    f"{name}={baseline_harness.safe_url(url)}"
                    if name_separator
                    else "<redacted-metrics-url>"
                )
            else:
                sanitized = baseline_harness.safe_url(value)
            safe_argv.append(f"{option}={sanitized}")
            continue
        safe_argv.append(item)
        pending_url = item in url_flags
    return baseline_harness.redact_text(shlex.join(safe_argv))


def _write_json(path: Path, value: Any) -> None:
    baseline_harness.write_json(path, value)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        baseline_harness.append_jsonl(output, value)


def _file_provenance(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": baseline_harness.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _artifact_checksums(run_dir: Path) -> dict[str, str]:
    """Hash every completed run artifact except the checksum manifest itself."""
    manifest_path = run_dir / "artifact-checksums.json"
    return {
        str(path.relative_to(run_dir)): baseline_harness.sha256_file(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    }


def _match_documents(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a strict one-endpoint, one-trace, one-SLO Match Config."""
    endpoint_name = "batch-impact-endpoint"
    autoscaler_name = f"batch-impact-{args.arm}"
    endpoint_catalog = {
        "endpoints": [
            {
                "name": endpoint_name,
                "url": args.endpoint_url,
                "model": args.model,
                "endpoint_type": args.endpoint_type,
                "description": f"Existing endpoint measured by the {args.arm} arm.",
            }
        ]
    }
    slo: dict[str, Any] = {"name": args.slo_name}
    for name in ("ttft_ms", "itl_ms", "e2e_ms"):
        value = getattr(args, f"slo_{name}")
        if value is not None:
            slo[name] = value
    match_config = {
        "schema_version": 1,
        "name": f"planner-gym-batch-impact-{args.arm}",
        "description": "Measure one live online workload with the selected Batch arm.",
        "labels": {
            "arm": args.arm,
            "campaign": "planner-gym-batch-impact",
        },
        "backend": {
            "type": "real",
            "endpoint_catalog": "endpoint-catalog.json",
            "aiperf": {
                "executable": args.aiperf_executable,
                "tokenizer": args.tokenizer,
                "streaming": args.streaming,
                "timeout_s": args.aiperf_timeout_seconds,
                # AIPerf's Mooncake loader defaults to 512. Forward the
                # declared external-trace granularity so non-default traces
                # are replayed with the same hash-id semantics we validated.
                "extra_args": [
                    "--isl-block-size",
                    str(args.trace_block_size),
                    # AIPerf's auto-scaled RecordProcessors are spawned at
                    # PROFILE_START, after DatasetConfiguredNotification is
                    # published during PROFILE_CONFIGURE. Make one processor a
                    # required service so its subscription is registered before
                    # configuration and every completed request is processed.
                    "--record-processor-service-count",
                    "1",
                ],
            },
            "autoscalers": [
                {
                    "name": autoscaler_name,
                    "endpoint": endpoint_name,
                    "autoscaler_type": args.arm,
                    "declared_config": {
                        "batch_arm": args.arm,
                        "expected_gate_types": args.expected_gate_type,
                    },
                }
            ],
        },
        "evaluations": {
            "traces": [
                {
                    "name": args.trace_name,
                    "path": str(args.trace),
                    "block_size": args.trace_block_size,
                    "presorted": args.trace_presorted,
                }
            ],
            "defaults": {
                "seed": args.seed,
                "max_requests": args.max_requests,
                "arrival_speedup": args.arrival_speedup,
            },
        },
        "slo_profiles": [slo],
        "metrics": {
            "rank_by": "goodput_rps",
            "include": [
                "goodput_rps",
                "good_rate",
                "good_count",
                "request_throughput_rps",
                "completed_requests",
                "duration_s",
                "mean_ttft_ms",
                "p99_ttft_ms",
                "mean_itl_ms",
                "p99_itl_ms",
                "mean_e2e_ms",
                "p99_e2e_ms",
            ],
        },
        "execution": {
            "repetitions": 1,
            "fail_fast": True,
            "max_runs": 1,
        },
        "publish": {
            "artifact_root": "../planner-gym/artifacts",
            "destinations": [
                {"type": "console"},
                {"type": "json", "path": "../planner-gym/results.json"},
                {"type": "html", "path": "../planner-gym/report.html"},
            ],
        },
    }
    return endpoint_catalog, match_config


def _batch_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    """Build the delegated Batch harness command for one treatment arm."""
    if args.arm not in {"stock", "planner-native"}:
        raise CampaignError("the online-only arm does not have a Batch command")
    command = [
        args.batch_python,
        str(args.batch_harness),
        "--experiment-root",
        str(run_dir / "batch-harness"),
        "--repo-root",
        str(args.dynamo_repo_root),
        "--dataset",
        str(args.dataset),
        "--namespace",
        args.namespace,
        "--tenant",
        args.tenant,
        "--run-kind",
        "baseline" if args.arm == "stock" else "planner-native",
        "--model",
        args.model,
        "--batch-size",
        str(args.batch_size),
        "--start-index",
        str(args.batch_start_index),
        "--max-tokens",
        str(args.batch_max_tokens),
        "--temperature",
        str(args.batch_temperature),
        "--completion-window",
        args.completion_window,
        "--batch-base-url",
        args.batch_base_url,
        "--online-base-url",
        args.endpoint_url,
        "--request-timeout-seconds",
        str(args.batch_request_timeout_seconds),
        "--poll-interval-seconds",
        str(args.batch_poll_interval_seconds),
        "--timeout-seconds",
        str(args.batch_timeout_seconds),
        "--abort-cleanup-timeout-seconds",
        str(args.batch_abort_cleanup_timeout_seconds),
        "--metrics-interval-seconds",
        str(args.metrics_interval_seconds),
        "--pod-name-regex",
        args.pod_name_regex,
        "--expected-worker-pool-id",
        args.expected_worker_pool_id,
    ]
    if args.context:
        command.extend(["--context", args.context])
    for gate_type in args.expected_gate_type:
        command.extend(["--expected-gate-type", gate_type])
    for metrics_url in args.metrics_url:
        command.extend(["--metrics-url", metrics_url])
    if args.allow_request_failures:
        command.append("--allow-request-failures")
    if args.arm == "planner-native":
        command.extend(
            [
                "--native-planner-configmap",
                args.native_planner_configmap,
                "--native-planner-pod-name-regex",
                args.native_planner_pod_name_regex,
                "--native-planner-decision-log-regex",
                args.native_planner_decision_log_regex,
                "--native-planner-min-decision-logs",
                str(args.native_planner_min_decision_logs),
            ]
        )
    return command


class _BoundedSanitizedLog:
    """Write a sanitized child stream with a hard byte bound."""

    def __init__(self, path: Path, byte_limit: int = MAX_CHILD_LOG_BYTES) -> None:
        self.path = path
        self.byte_limit = byte_limit
        self.written = 0
        self.truncated = False
        self.sensitive_yaml_indent: int | None = None

    def _sanitize(self, value: str) -> str:
        """Redact sensitive YAML env entries even when streamed one line at a time."""
        lines = value.splitlines(keepends=True)
        for index, line in enumerate(lines):
            content = line.rstrip("\r\n")
            line_ending = line[len(content) :]
            name_match = baseline_harness.YAML_ENV_NAME_LINE_RE.fullmatch(content)
            if name_match is not None:
                if baseline_harness.SENSITIVE_KEY_RE.search(name_match.group("name")):
                    self.sensitive_yaml_indent = len(name_match.group("indent"))
                else:
                    self.sensitive_yaml_indent = None
                continue
            if self.sensitive_yaml_indent is None or not content.strip():
                continue
            indentation = len(content) - len(content.lstrip(" "))
            if indentation <= self.sensitive_yaml_indent:
                self.sensitive_yaml_indent = None
                continue
            value_match = baseline_harness.YAML_ENV_VALUE_LINE_RE.fullmatch(content)
            if value_match is not None:
                lines[index] = f"{value_match.group('prefix')}<redacted>{line_ending}"
                self.sensitive_yaml_indent = None
        return baseline_harness.redact_text("".join(lines))

    def write(self, output: TextIO, value: str) -> None:
        if self.truncated:
            return
        sanitized = self._sanitize(value)
        encoded = sanitized.encode("utf-8")
        remaining = self.byte_limit - self.written
        if len(encoded) <= remaining:
            output.write(sanitized)
            self.written += len(encoded)
            output.flush()
            return
        if remaining > 0:
            prefix = encoded[:remaining].decode("utf-8", errors="ignore")
            output.write(prefix)
            self.written += len(prefix.encode("utf-8"))
        output.write(LOG_TRUNCATION_MARKER)
        output.flush()
        self.truncated = True


@dataclasses.dataclass
class CapturedProcess:
    """A child process whose two output streams are drained concurrently."""

    name: str
    argv: list[str]
    process: subprocess.Popen[str]
    process_group_id: int | None
    termination_grace_seconds: float
    stdout_thread: threading.Thread
    stderr_thread: threading.Thread
    marker_event: threading.Event
    active_event: threading.Event
    dispatch_active_event: threading.Event
    marker_state: dict[str, Any]
    stream_errors: list[str]

    def finish_capture(self) -> None:
        self.stdout_thread.join(timeout=10)
        self.stderr_thread.join(timeout=10)
        if self.stdout_thread.is_alive() or self.stderr_thread.is_alive():
            raise CampaignError(f"{self.name} output capture did not finish")
        if self.stream_errors:
            raise CampaignError(
                f"{self.name} output capture failed: {self.stream_errors[0]}"
            )


@dataclasses.dataclass(frozen=True)
class BatchDispatchBarrier:
    """Observed proof that one remote Batch job is actively dispatching."""

    batch_id: str
    created_observed_at: str
    in_progress_at: str
    in_progress_observed_at: str
    in_progress_status: str
    dispatch_active_at: str
    dispatch_active_observed_at: str
    dispatch_active_status: str
    completed: int
    failed: int
    total: int


def _copy_stream(
    stream: TextIO,
    log_path: Path,
    stream_errors: list[str],
    *,
    marker_event: threading.Event | None = None,
    active_event: threading.Event | None = None,
    dispatch_active_event: threading.Event | None = None,
    marker_state: dict[str, Any] | None = None,
) -> None:
    writer = _BoundedSanitizedLog(log_path)
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as output:
            for line in stream:
                if marker_event is not None and marker_state is not None:
                    stripped = line.rstrip("\r\n")
                    if stripped.startswith(HARNESS_EVENT_PREFIX):
                        try:
                            event = json.loads(stripped[len(HARNESS_EVENT_PREFIX) :])
                        except (json.JSONDecodeError, TypeError):
                            event = None
                        if isinstance(event, dict):
                            event_name = event.get("event")
                            batch_id = event.get("batch_id")
                            observed_at = event.get("observed_at")
                            if isinstance(batch_id, str) and isinstance(
                                observed_at, str
                            ):
                                if event_name == "batch-created":
                                    marker_state["batch_id"] = batch_id
                                    marker_state["created_at"] = observed_at
                                    marker_state[
                                        "created_observed_at"
                                    ] = isoformat_utc()
                                    marker_event.set()
                                elif event_name == "batch-active":
                                    status = event.get("status")
                                    prior_batch_id = marker_state.get("batch_id")
                                    if (
                                        status == "in_progress"
                                        and prior_batch_id == batch_id
                                    ):
                                        marker_state["active_at"] = observed_at
                                        marker_state[
                                            "active_observed_at"
                                        ] = isoformat_utc()
                                        marker_state["active_status"] = status
                                        if active_event is not None:
                                            active_event.set()
                                elif event_name == "batch-dispatch-active":
                                    status = event.get("status")
                                    completed = event.get("completed")
                                    failed = event.get("failed")
                                    total = event.get("total")
                                    counts_are_valid = all(
                                        not isinstance(value, bool)
                                        and isinstance(value, int)
                                        and value >= 0
                                        for value in (completed, failed, total)
                                    )
                                    prior_batch_id = marker_state.get("batch_id")
                                    if (
                                        status == "in_progress"
                                        and prior_batch_id == batch_id
                                        and counts_are_valid
                                        and completed + failed > 0
                                        and completed + failed <= total
                                        and marker_state.get("active_status")
                                        == "in_progress"
                                    ):
                                        marker_state["dispatch_active_at"] = observed_at
                                        marker_state[
                                            "dispatch_active_observed_at"
                                        ] = isoformat_utc()
                                        marker_state["dispatch_active_status"] = status
                                        marker_state["dispatch_completed"] = completed
                                        marker_state["dispatch_failed"] = failed
                                        marker_state["dispatch_total"] = total
                                        if dispatch_active_event is not None:
                                            dispatch_active_event.set()
                                elif event_name == "batch-terminal":
                                    status = event.get("status")
                                    if isinstance(status, str):
                                        marker_state["terminal_status"] = status
                    match = BATCH_CREATED_RE.fullmatch(line.rstrip("\r\n"))
                    if match is not None and not marker_event.is_set():
                        marker_state["batch_id"] = match.group("batch_id")
                        marker_state["created_observed_at"] = isoformat_utc()
                        marker_event.set()
                writer.write(output, line)
    except (OSError, ValueError) as error:
        stream_errors.append(baseline_harness.redact_text(str(error)))
    finally:
        stream.close()


def _start_process(
    name: str,
    argv: Sequence[str],
    *,
    cwd: Path,
    log_dir: Path,
    watch_batch_marker: bool = False,
    termination_grace_seconds: float = CHILD_TERMINATION_GRACE_SECONDS,
) -> CapturedProcess:
    """Launch one command without a shell and begin bounded log capture."""
    log_dir.mkdir(parents=True, exist_ok=True)
    marker_event = threading.Event()
    active_event = threading.Event()
    dispatch_active_event = threading.Event()
    marker_state: dict[str, Any] = {}
    stream_errors: list[str] = []
    try:
        process = subprocess.Popen(
            [str(item) for item in argv],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=os.name == "posix",
        )
    except OSError as error:
        raise CampaignError(f"could not start {name}: {error}") from error
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise CampaignError(f"could not capture {name} output")
    stdout_thread = threading.Thread(
        target=_copy_stream,
        args=(process.stdout, log_dir / f"{name}.stdout.log", stream_errors),
        kwargs={
            "marker_event": marker_event if watch_batch_marker else None,
            "active_event": active_event if watch_batch_marker else None,
            "dispatch_active_event": (
                dispatch_active_event if watch_batch_marker else None
            ),
            "marker_state": marker_state if watch_batch_marker else None,
        },
        name=f"{name}-stdout-capture",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_copy_stream,
        args=(process.stderr, log_dir / f"{name}.stderr.log", stream_errors),
        name=f"{name}-stderr-capture",
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    return CapturedProcess(
        name=name,
        argv=[str(item) for item in argv],
        process=process,
        process_group_id=process.pid if os.name == "posix" else None,
        termination_grace_seconds=termination_grace_seconds,
        stdout_thread=stdout_thread,
        stderr_thread=stderr_thread,
        marker_event=marker_event,
        active_event=active_event,
        dispatch_active_event=dispatch_active_event,
        marker_state=marker_state,
        stream_errors=stream_errors,
    )


def _process_group_exists(process_group_id: int) -> bool:
    """Return whether an owned POSIX process group still has live members."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_tree(handle: CapturedProcess, signum: int) -> None:
    """Signal the complete child process tree, falling back to the leader."""
    if handle.process_group_id is not None:
        try:
            os.killpg(handle.process_group_id, signum)
        except ProcessLookupError:
            pass
        return
    if handle.process.poll() is not None:
        return
    try:
        handle.process.send_signal(signum)
    except ProcessLookupError:
        pass


def _process_tree_exists(handle: CapturedProcess) -> bool:
    if handle.process_group_id is not None:
        return _process_group_exists(handle.process_group_id)
    return handle.process.poll() is None


def _wait_for_process_tree_exit(
    handle: CapturedProcess, timeout_seconds: float
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _process_tree_exists(handle):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if handle.process.poll() is None:
            try:
                handle.process.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(min(0.05, remaining))
    return True


def _terminate_process(handle: CapturedProcess) -> None:
    """Terminate and reap an owned child tree before closing its capture."""
    if _process_tree_exists(handle):
        _signal_process_tree(handle, signal.SIGTERM)
        if not _wait_for_process_tree_exit(handle, handle.termination_grace_seconds):
            _signal_process_tree(handle, signal.SIGKILL)
            if not _wait_for_process_tree_exit(handle, 10.0):
                raise CampaignError(
                    f"{handle.name} process group did not exit after SIGKILL"
                )
    if handle.process.poll() is None:
        handle.process.wait(timeout=10)
    handle.finish_capture()


def _cleanup_processes(handles: Sequence[CapturedProcess]) -> list[str]:
    """Best-effort cleanup that attempts every child and retains diagnostics."""
    errors: list[str] = []
    for handle in handles:
        try:
            _terminate_process(handle)
        except BaseException as error:  # noqa: BLE001 - cleanup must continue
            errors.append(f"{handle.name}: {baseline_harness.redact_text(str(error))}")
    return errors


def _attach_cleanup_notes(error: BaseException, cleanup_errors: Sequence[str]) -> None:
    add_note = getattr(error, "add_note", None)
    if not callable(add_note):
        return
    for cleanup_error in cleanup_errors:
        add_note(f"child cleanup also failed: {cleanup_error}")


def _wait_one(handle: CapturedProcess, timeout_seconds: float) -> int:
    try:
        exit_code = handle.process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        failure = CampaignError(
            f"{handle.name} exceeded its {timeout_seconds:g}s deadline"
        )
        _attach_cleanup_notes(failure, _cleanup_processes([handle]))
        raise failure from error
    except BaseException as error:
        _attach_cleanup_notes(error, _cleanup_processes([handle]))
        raise
    try:
        handle.finish_capture()
    except BaseException as error:
        _attach_cleanup_notes(error, _cleanup_processes([handle]))
        raise
    return exit_code


def _wait_for_batch_dispatch_active(
    handle: CapturedProcess, timeout_seconds: float
) -> BatchDispatchBarrier:
    """Wait for positive Batch progress while the job remains in progress."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process(handle)
            raise CampaignError(
                "Batch harness did not prove positive in_progress dispatch within "
                f"{timeout_seconds:g}s; online load was not started"
            )
        dispatch_active_observed = handle.dispatch_active_event.wait(
            timeout=min(0.1, remaining)
        )
        terminal_status = handle.marker_state.get("terminal_status")
        if terminal_status is not None:
            _terminate_process(handle)
            raise CampaignError(
                f"Batch job reached terminal status {terminal_status!r} before "
                "online load started"
            )
        if dispatch_active_observed:
            if handle.process.poll() is not None:
                handle.finish_capture()
                raise CampaignError(
                    "Batch harness exited after reporting positive in_progress "
                    "dispatch but before online load started"
                )
            return BatchDispatchBarrier(
                batch_id=handle.marker_state["batch_id"],
                created_observed_at=handle.marker_state["created_observed_at"],
                in_progress_at=handle.marker_state["active_at"],
                in_progress_observed_at=handle.marker_state["active_observed_at"],
                in_progress_status=handle.marker_state["active_status"],
                dispatch_active_at=handle.marker_state["dispatch_active_at"],
                dispatch_active_observed_at=handle.marker_state[
                    "dispatch_active_observed_at"
                ],
                dispatch_active_status=handle.marker_state["dispatch_active_status"],
                completed=handle.marker_state["dispatch_completed"],
                failed=handle.marker_state["dispatch_failed"],
                total=handle.marker_state["dispatch_total"],
            )
        exit_code = handle.process.poll()
        if exit_code is not None:
            handle.finish_capture()
            raise CampaignError(
                "Batch harness exited with code "
                f"{exit_code} before proving positive in_progress dispatch"
            )


def _wait_concurrent(
    batch: CapturedProcess,
    planner_gym: CapturedProcess,
    timeout_seconds: float,
) -> tuple[int, int]:
    """Wait for both arms, terminating the peer on failure or campaign timeout."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        batch_exit = batch.process.poll()
        planner_exit = planner_gym.process.poll()
        if batch_exit is not None and planner_exit is not None:
            try:
                batch.finish_capture()
                planner_gym.finish_capture()
            except BaseException as error:
                _attach_cleanup_notes(error, _cleanup_processes([batch, planner_gym]))
                raise
            return batch_exit, planner_exit
        if batch_exit not in {None, 0}:
            try:
                _terminate_process(planner_gym)
                batch.finish_capture()
            except BaseException as error:
                _attach_cleanup_notes(error, _cleanup_processes([batch, planner_gym]))
                raise
            return batch_exit, planner_gym.process.returncode
        if planner_exit not in {None, 0}:
            try:
                _terminate_process(batch)
                planner_gym.finish_capture()
            except BaseException as error:
                _attach_cleanup_notes(error, _cleanup_processes([batch, planner_gym]))
                raise
            return batch.process.returncode, planner_exit
        if time.monotonic() >= deadline:
            failure = CampaignError(
                f"concurrent workload exceeded its {timeout_seconds:g}s deadline"
            )
            _attach_cleanup_notes(failure, _cleanup_processes([batch, planner_gym]))
            raise failure
        time.sleep(0.05)


class CampaignContext:
    """Own one immutable outer raw run and its machine-readable metadata."""

    def __init__(self, args: argparse.Namespace, argv: Sequence[str]) -> None:
        self.args = args
        self.started = utc_now()
        self.run_id = make_run_id(self.started)
        self.run_dir = args.experiment_root / "results" / "raw" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.run_dir / "logs" / "events.jsonl"
        self.metadata: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "kind": "planner-gym-batch-impact",
            "arm": args.arm,
            "status": "running",
            "started": isoformat_utc(self.started),
            "ended": None,
            "exit_code": None,
            "command": _safe_command(argv),
            "configuration": {
                "endpoint_url": baseline_harness.safe_url(args.endpoint_url),
                "batch_base_url": baseline_harness.safe_url(args.batch_base_url),
                "tenant": args.tenant,
                "namespace": args.namespace,
                "context": args.context or "current-context",
                "model": args.model,
                "endpoint_type": args.endpoint_type,
                "trace_name": args.trace_name,
                "trace_block_size": args.trace_block_size,
                "trace_presorted": args.trace_presorted,
                "seed": args.seed,
                "max_requests": args.max_requests,
                "arrival_speedup": args.arrival_speedup,
                "expected_gate_types": args.expected_gate_type,
                "expected_worker_pool_id": args.expected_worker_pool_id,
                "batch_size": args.batch_size if args.arm != "online-only" else None,
                "batch_abort_cleanup_timeout_seconds": (
                    args.batch_abort_cleanup_timeout_seconds
                    if args.arm != "online-only"
                    else None
                ),
            },
            "source": {},
            "inputs": {},
            "commands": {},
            "synchronization": {
                "batch_required": args.arm != "online-only",
                "batch_id": None,
                "batch_created_observed_at": None,
                "batch_active_observed_at": None,
                "batch_active_status": None,
                "batch_dispatch_active_at": None,
                "batch_dispatch_active_observed_at": None,
                "batch_dispatch_active_status": None,
                "batch_dispatch_completed": None,
                "batch_dispatch_failed": None,
                "batch_dispatch_total": None,
                "planner_gym_started_at": None,
            },
            "remote_cleanup": {
                "triggered": False,
                "batch_id": None,
                "succeeded": None,
            },
            "children": {},
            "error": None,
            "secret_handling": {
                "environment": "inherited but not recorded",
                "child_logs": "credential-shaped values and URL queries redacted",
                "kubernetes_secrets": "not queried by this outer runner",
            },
        }
        self.event("campaign-created", arm=args.arm)
        self.write_metadata()

    def event(self, event: str, **fields: Any) -> None:
        _append_jsonl(
            self.events_path,
            {"timestamp": isoformat_utc(), "event": event, **fields},
        )

    def write_metadata(self) -> None:
        _write_json(self.run_dir / "metadata.json", self.metadata)

    def finalize(self, exit_code: int, error: str | None = None) -> None:
        self.metadata["ended"] = isoformat_utc()
        self.metadata["exit_code"] = exit_code
        if exit_code == 0:
            self.metadata["status"] = "completed"
        elif exit_code == 130:
            self.metadata["status"] = "interrupted"
        else:
            self.metadata["status"] = "failed"
        self.metadata["error"] = (
            baseline_harness.redact_text(error) if error is not None else None
        )
        self.event("campaign-finished", exit_code=exit_code)
        self.write_metadata()
        _write_json(
            self.run_dir / "artifact-checksums.json",
            _artifact_checksums(self.run_dir),
        )


def validate_args(args: argparse.Namespace) -> None:
    """Validate all local inputs before creating a raw run or external traffic."""
    for option, path in (
        ("--experiment-root", args.experiment_root),
        ("--dynamo-repo-root", args.dynamo_repo_root),
        ("--planner-gym-root", args.planner_gym_root),
    ):
        if not path.is_dir():
            raise CampaignError(f"{option} is not a directory: {path}")
    for option, path in (
        ("--planner-gym-script", args.planner_gym_script),
        ("--batch-harness", args.batch_harness),
    ):
        if not path.is_file():
            raise CampaignError(f"{option} is not a file: {path}")
    if not args.trace.is_file():
        raise CampaignError(f"--trace is not a file: {args.trace}")
    if args.arm != "online-only" and (
        args.dataset is None or not args.dataset.is_file()
    ):
        raise CampaignError("--dataset must name an existing file for Batch arms")
    if not args.namespace.strip():
        raise CampaignError("--namespace cannot be empty")
    if not args.tenant.strip():
        raise CampaignError("--tenant cannot be empty")
    if not args.model.strip():
        raise CampaignError("--model cannot be empty")
    if not args.expected_worker_pool_id.strip():
        raise CampaignError("--expected-worker-pool-id cannot be empty")
    _validate_plain_http_url(args.endpoint_url, "--endpoint-url")
    _validate_plain_http_url(args.batch_base_url, "--batch-base-url")
    try:
        re.compile(args.pod_name_regex)
    except re.error as error:
        raise CampaignError(f"invalid --pod-name-regex: {error}") from error
    try:
        metrics_endpoints = baseline_harness.parse_metrics_urls(args.metrics_url)
    except baseline_harness.HarnessError as error:
        raise CampaignError(str(error)) from error
    required_metric_names = {"async", "frontend"}
    metric_names = {name for name, _url in metrics_endpoints}
    missing_metric_names = sorted(required_metric_names - metric_names)
    unexpected_metric_names = sorted(metric_names - required_metric_names)
    if len(metrics_endpoints) != 2 or missing_metric_names or unexpected_metric_names:
        diagnostics = []
        if missing_metric_names:
            diagnostics.append("missing: " + ", ".join(missing_metric_names))
        if unexpected_metric_names:
            diagnostics.append("unexpected: " + ", ".join(unexpected_metric_names))
        raise CampaignError(
            "--metrics-url requires exactly frontend=URL and async=URL"
            + ("; " + "; ".join(diagnostics) if diagnostics else "")
        )
    if all(
        value is None for value in (args.slo_ttft_ms, args.slo_itl_ms, args.slo_e2e_ms)
    ):
        raise CampaignError("at least one SLO threshold is required")
    if args.arm == "planner-native":
        if not args.native_planner_configmap:
            raise CampaignError(
                "--native-planner-configmap is required for planner-native"
            )
        if {value.lower() for value in args.expected_gate_type} != {
            "redis-leased-rate"
        }:
            raise CampaignError(
                "planner-native requires exactly --expected-gate-type redis-leased-rate"
            )
    elif any(
        value is not None
        for value in (
            args.native_planner_configmap,
            args.native_planner_pod_name_regex,
            args.native_planner_decision_log_regex,
            args.native_planner_min_decision_logs,
        )
    ):
        raise CampaignError(
            "native Planner evidence options are valid only for planner-native"
        )
    for executable_option, executable in (
        ("--planner-gym-python", args.planner_gym_python),
        ("--batch-python", args.batch_python),
        ("--aiperf-executable", args.aiperf_executable),
    ):
        if shutil.which(executable) is None and not Path(executable).is_file():
            raise CampaignError(
                f"{executable_option} is not executable or discoverable: {executable}"
            )


def _prepare_run(args: argparse.Namespace, context: CampaignContext) -> Path:
    config_dir = context.run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=False)
    endpoint_catalog, match_config = _match_documents(args)
    endpoint_path = config_dir / "endpoint-catalog.json"
    match_path = config_dir / "match-config.yaml"
    _write_json(endpoint_path, endpoint_catalog)
    _write_json(match_path, match_config)

    source_dir = context.run_dir / "source-state"
    context.metadata["source"] = {
        "dynamo": baseline_harness.git_state(
            args.dynamo_repo_root, source_dir / "dynamo"
        ),
        "planner_gym": baseline_harness.git_state(
            args.planner_gym_root, source_dir / "planner-gym"
        ),
    }
    input_paths = {
        "runner": Path(__file__).resolve(),
        "planner_gym_script": args.planner_gym_script,
        "batch_harness": args.batch_harness,
        "match_config": match_path,
        "endpoint_catalog": endpoint_path,
        "trace": args.trace,
    }
    if args.dataset is not None:
        input_paths["dataset"] = args.dataset
    context.metadata["inputs"] = {
        name: _file_provenance(path) for name, path in input_paths.items()
    }
    context.metadata["executables"] = {
        "planner_gym_python": _resolved_executable(args.planner_gym_python),
        "batch_python": _resolved_executable(args.batch_python),
        "aiperf": _resolved_executable(args.aiperf_executable),
    }
    context.write_metadata()
    return match_path


def _planner_command(args: argparse.Namespace, match_path: Path) -> list[str]:
    return [args.planner_gym_python, str(args.planner_gym_script), str(match_path)]


def _cancel_remote_batch(
    args: argparse.Namespace, run_dir: Path, batch_id: str
) -> dict[str, Any]:
    """Idempotently terminalize a remote job as the outer-runner backstop."""
    client = baseline_harness.BatchClient(
        args.batch_base_url,
        args.tenant,
        args.batch_request_timeout_seconds,
    )
    return baseline_harness.cancel_batch_to_terminal(
        client,
        batch_id,
        run_dir / "abort-cleanup",
        poll_interval_seconds=args.batch_poll_interval_seconds,
        timeout_seconds=args.batch_abort_cleanup_timeout_seconds,
        source="campaign-runner-backstop",
    )


def _online_only_evidence_args(args: argparse.Namespace) -> argparse.Namespace:
    """Adapt outer-runner options to the shared read-only evidence collector."""
    values = vars(args).copy()
    values["run_kind"] = "baseline"
    return argparse.Namespace(**values)


def execute(args: argparse.Namespace, context: CampaignContext) -> int:
    """Validate the one-cell matrix, then execute the selected treatment arm."""
    match_path = _prepare_run(args, context)
    logs = context.run_dir / "logs"
    planner_command = _planner_command(args, match_path)
    validate_command = [*planner_command, "--validate-only", "--print-matrix"]
    context.metadata["commands"]["planner_gym_validate"] = _safe_command(
        validate_command
    )
    context.metadata["commands"]["planner_gym"] = _safe_command(planner_command)
    context.write_metadata()

    context.event("planner-gym-validation-started")
    validation = _start_process(
        "planner-gym-validate",
        validate_command,
        cwd=args.planner_gym_root,
        log_dir=logs,
    )
    validation_exit = _wait_one(validation, args.validation_timeout_seconds)
    context.metadata["children"]["planner_gym_validate_exit_code"] = validation_exit
    context.event("planner-gym-validation-finished", exit_code=validation_exit)
    context.write_metadata()
    if validation_exit != 0:
        raise CampaignError(
            f"Planner Gym Match Config validation exited with code {validation_exit}"
        )

    if args.arm == "online-only":
        evidence_args = _online_only_evidence_args(args)
        evidence_root = context.run_dir / "online-evidence"
        evidence = baseline_harness.KubernetesEvidence(
            evidence_args, evidence_root, context.started
        )
        start_summary = evidence.preflight()
        context.metadata["kubernetes_start"] = start_summary
        metrics = baseline_harness.MetricsSampler(
            evidence_root,
            baseline_harness.parse_metrics_urls(args.metrics_url),
            args.metrics_interval_seconds,
        )
        planner: CapturedProcess | None = None
        planner_exit: int | None = None
        core_error: BaseException | None = None
        metrics.start()
        try:
            context.metadata["synchronization"][
                "planner_gym_started_at"
            ] = isoformat_utc()
            context.event("planner-gym-started")
            planner = _start_process(
                "planner-gym",
                planner_command,
                cwd=args.planner_gym_root,
                log_dir=logs,
            )
            planner_exit = _wait_one(planner, args.campaign_timeout_seconds)
            if planner_exit != 0:
                core_error = CampaignError(
                    f"Planner Gym exited with code {planner_exit}"
                )
        except BaseException as error:  # noqa: BLE001 - preserve evidence on abort
            core_error = error
        finally:
            metrics_summary = metrics.stop()
            context.metadata["metrics_summary"] = metrics_summary
            if metrics_summary.get("error") and core_error is None:
                core_error = CampaignError(
                    f"metrics sampler failed: {metrics_summary['error']}"
                )
            try:
                context.metadata["kubernetes_end"] = evidence.capture_phase(
                    "end", include_logs=True
                )
            except BaseException as error:  # noqa: BLE001 - retain primary error
                if core_error is None:
                    core_error = CampaignError(f"end evidence capture failed: {error}")
                else:
                    context.metadata.setdefault("evidence_errors", []).append(
                        baseline_harness.redact_text(str(error))
                    )
            context.metadata["children"]["planner_gym_exit_code"] = planner_exit
            context.event("planner-gym-finished", exit_code=planner_exit)
            context.write_metadata()
        if core_error is not None:
            raise core_error
        return 0

    batch_command = _batch_command(args, context.run_dir)
    context.metadata["commands"]["batch_harness"] = _safe_command(batch_command)
    context.write_metadata()
    context.event("batch-harness-started")
    batch = _start_process(
        "batch-harness",
        batch_command,
        cwd=args.dynamo_repo_root,
        log_dir=logs,
        watch_batch_marker=True,
        termination_grace_seconds=(
            args.batch_abort_cleanup_timeout_seconds
            + args.batch_request_timeout_seconds
            + 15.0
        ),
    )
    planner: CapturedProcess | None = None
    try:
        dispatch = _wait_for_batch_dispatch_active(
            batch, args.batch_start_timeout_seconds
        )
        planner_started_at = isoformat_utc()
        context.metadata["synchronization"].update(
            {
                "batch_id": dispatch.batch_id,
                "batch_created_observed_at": dispatch.created_observed_at,
                "batch_active_observed_at": dispatch.in_progress_observed_at,
                "batch_active_status": dispatch.in_progress_status,
                "batch_dispatch_active_at": dispatch.dispatch_active_at,
                "batch_dispatch_active_observed_at": (
                    dispatch.dispatch_active_observed_at
                ),
                "batch_dispatch_active_status": dispatch.dispatch_active_status,
                "batch_dispatch_completed": dispatch.completed,
                "batch_dispatch_failed": dispatch.failed,
                "batch_dispatch_total": dispatch.total,
                "planner_gym_started_at": planner_started_at,
            }
        )
        context.event("batch-created-observed", batch_id=dispatch.batch_id)
        context.event(
            "batch-active-observed",
            batch_id=dispatch.batch_id,
            status=dispatch.in_progress_status,
        )
        context.event(
            "batch-dispatch-active-observed",
            batch_id=dispatch.batch_id,
            status=dispatch.dispatch_active_status,
            completed=dispatch.completed,
            failed=dispatch.failed,
            total=dispatch.total,
        )
        context.event("planner-gym-started")
        context.write_metadata()
        planner = _start_process(
            "planner-gym",
            planner_command,
            cwd=args.planner_gym_root,
            log_dir=logs,
        )
        batch_exit, planner_exit = _wait_concurrent(
            batch, planner, args.campaign_timeout_seconds
        )
        context.metadata["children"].update(
            {
                "batch_harness_exit_code": batch_exit,
                "planner_gym_exit_code": planner_exit,
            }
        )
        if batch_exit != 0:
            raise CampaignError(f"Batch harness exited with code {batch_exit}")
        if planner_exit != 0:
            raise CampaignError(f"Planner Gym exited with code {planner_exit}")
    except BaseException as error:
        handles = [batch]
        if planner is not None:
            handles.append(planner)
        cleanup_errors = _cleanup_processes(handles)
        batch_id = batch.marker_state.get("batch_id")
        if batch_id is not None:
            context.metadata["synchronization"]["batch_id"] = batch_id
            try:
                remote_cleanup = _cancel_remote_batch(args, context.run_dir, batch_id)
            except BaseException as cleanup_error:  # noqa: BLE001 - retain primary
                remote_cleanup = {
                    "triggered": True,
                    "source": "campaign-runner-backstop",
                    "batch_id": batch_id,
                    "succeeded": False,
                    "error": baseline_harness.redact_text(str(cleanup_error)),
                }
            context.metadata["remote_cleanup"] = remote_cleanup
            context.event(
                "remote-batch-cleanup-finished",
                batch_id=batch_id,
                succeeded=remote_cleanup["succeeded"],
                terminal_status=remote_cleanup.get("terminal_status"),
            )
            if not remote_cleanup["succeeded"]:
                cleanup_errors.append(
                    "remote Batch cleanup: "
                    f"{remote_cleanup.get('error') or 'unknown error'}"
                )
        context.write_metadata()
        _attach_cleanup_notes(error, cleanup_errors)
        raise

    context.event("batch-harness-finished", exit_code=batch_exit)
    context.event("planner-gym-finished", exit_code=planner_exit)
    context.write_metadata()
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one live Planner Gym Match Config cell alone or concurrently "
            "with an evidence-preserving Batch Gateway job."
        )
    )
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--dynamo-repo-root", type=Path, default=DYNAMO_REPO_ROOT)
    parser.add_argument(
        "--planner-gym-root", type=Path, default=DEFAULT_PLANNER_GYM_ROOT
    )
    parser.add_argument(
        "--planner-gym-script", type=Path, default=DEFAULT_PLANNER_GYM_SCRIPT
    )
    parser.add_argument(
        "--planner-gym-python",
        default=str(DEFAULT_PLANNER_GYM_PYTHON),
        help="Python executable from the Planner Gym environment",
    )
    parser.add_argument("--aiperf-executable", default=str(DEFAULT_AIPERF_EXECUTABLE))
    parser.add_argument("--batch-harness", type=Path, default=DEFAULT_BATCH_HARNESS)
    parser.add_argument("--batch-python", default=sys.executable)
    parser.add_argument("--dataset", type=Path)

    parser.add_argument("--endpoint-url", required=True)
    parser.add_argument("--batch-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint-type", default="chat")
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--no-streaming", dest="streaming", action="store_false", default=True
    )
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="External Mooncake replay JSONL used for the one Match Config cell",
    )
    parser.add_argument("--trace-name", default="batch-impact-online")
    parser.add_argument("--trace-block-size", type=positive_int, default=512)
    parser.add_argument(
        "--trace-presorted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the external trace is already sorted by timestamp",
    )
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument(
        "--max-requests",
        type=positive_int,
        help="optional cap; omitted by default so every external-trace row runs",
    )
    parser.add_argument("--arrival-speedup", type=positive_float, default=1.0)
    parser.add_argument("--slo-name", default="interactive")
    parser.add_argument("--slo-ttft-ms", type=positive_float, default=300.0)
    parser.add_argument("--slo-itl-ms", type=positive_float, default=50.0)
    parser.add_argument("--slo-e2e-ms", type=positive_float)
    parser.add_argument("--aiperf-timeout-seconds", type=positive_float, default=900.0)

    parser.add_argument("--namespace", default="default")
    parser.add_argument("--context")
    parser.add_argument("--tenant", default="planner-poc-baseline")
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=4000,
        help="persistent backlog size for the 180-second comparison trace",
    )
    parser.add_argument("--batch-start-index", type=nonnegative_int, default=0)
    parser.add_argument("--batch-max-tokens", type=positive_int, default=128)
    parser.add_argument("--batch-temperature", type=finite_float, default=0.0)
    parser.add_argument("--completion-window", default="24h")
    parser.add_argument(
        "--batch-request-timeout-seconds", type=positive_float, default=120.0
    )
    parser.add_argument(
        "--batch-poll-interval-seconds", type=positive_float, default=2.0
    )
    parser.add_argument("--batch-timeout-seconds", type=positive_float, default=3600.0)
    parser.add_argument(
        "--batch-abort-cleanup-timeout-seconds",
        type=positive_float,
        default=120.0,
        help="Deadline for proving an aborted remote Batch job reached terminal state",
    )
    parser.add_argument(
        "--metrics-url",
        action="append",
        default=[],
        metavar="NAME=URL",
        help=(
            "repeat for the required frontend=URL and async=URL continuous "
            "evidence endpoints"
        ),
    )
    parser.add_argument("--metrics-interval-seconds", type=positive_float, default=15.0)
    parser.add_argument(
        "--pod-name-regex",
        default=CAMPAIGN_POD_NAME_REGEX,
    )
    parser.add_argument("--expected-gate-type", action="append", default=[])
    parser.add_argument("--expected-worker-pool-id", default="dynamo-batch")
    parser.add_argument("--allow-request-failures", action="store_true")
    parser.add_argument("--native-planner-configmap")
    parser.add_argument("--native-planner-pod-name-regex")
    parser.add_argument("--native-planner-decision-log-regex")
    parser.add_argument("--native-planner-min-decision-logs", type=positive_int)

    parser.add_argument(
        "--validation-timeout-seconds", type=positive_float, default=120.0
    )
    parser.add_argument(
        "--batch-start-timeout-seconds",
        type=positive_float,
        default=300.0,
        help=(
            "Deadline for observing positive Batch progress while in_progress "
            "before online load"
        ),
    )
    parser.add_argument(
        "--campaign-timeout-seconds", type=positive_float, default=5400.0
    )
    args = parser.parse_args(argv)

    for name in (
        "experiment_root",
        "dynamo_repo_root",
        "planner_gym_root",
        "planner_gym_script",
        "batch_harness",
        "dataset",
        "trace",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    for name in ("planner_gym_python", "batch_python", "aiperf_executable"):
        setattr(args, name, _normalize_executable(getattr(args, name)))
    if not args.expected_gate_type:
        if args.arm == "planner-native":
            args.expected_gate_type = ["redis-leased-rate"]
        elif args.arm == "stock":
            args.expected_gate_type = ["prometheus-query"]
    if args.arm == "planner-native":
        if args.native_planner_pod_name_regex is None:
            args.native_planner_pod_name_regex = (
                baseline_harness.NATIVE_PLANNER_DEFAULT_POD_NAME_REGEX
            )
        if args.native_planner_decision_log_regex is None:
            args.native_planner_decision_log_regex = (
                baseline_harness.NATIVE_PLANNER_DEFAULT_DECISION_LOG_REGEX
            )
        if args.native_planner_min_decision_logs is None:
            args.native_planner_min_decision_logs = (
                baseline_harness.NATIVE_PLANNER_DEFAULT_MIN_DECISION_LOGS
            )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    parsed_argv = list(argv if argv is not None else sys.argv[1:])
    args = parse_args(parsed_argv)
    context: CampaignContext | None = None
    exit_code = 1
    error_message: str | None = None

    def interrupt_handler(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt_handler)
    signal.signal(signal.SIGINT, interrupt_handler)
    try:
        validate_args(args)
        context = CampaignContext(
            args,
            [sys.executable, str(Path(__file__).resolve()), *parsed_argv],
        )
        print(f"run ID: {context.run_id}")
        print(f"raw results: {context.run_dir}")
        exit_code = execute(args, context)
    except KeyboardInterrupt:
        exit_code = 130
        error_message = "campaign interrupted"
        print(error_message, file=sys.stderr)
    except CampaignError as error:
        exit_code = 1
        error_message = str(error)
        print(f"error: {baseline_harness.redact_text(str(error))}", file=sys.stderr)
    except BaseException as error:  # noqa: BLE001 - preserve unexpected failures
        exit_code = 1
        error_message = str(error)
        print(
            baseline_harness.redact_text(traceback.format_exc()),
            file=sys.stderr,
        )
    finally:
        if context is not None:
            context.finalize(exit_code, error_message)
            print(f"run {context.run_id} exited with code {exit_code}")
            print(f"artifacts: {context.run_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
