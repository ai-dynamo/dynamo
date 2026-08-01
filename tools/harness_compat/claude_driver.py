#!/usr/bin/env python3
"""Native Claude Code stream-json driver for compatibility-lab scenarios.

The CLI remains responsible for Messages requests, tool execution, session
state, and compaction. The driver supplies native user events and observes the
structured CLI stream; it never constructs Anthropic API requests itself.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable


def _now_ms() -> int:
    return round(time.time() * 1000)


def _id_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def _client_version(executable: str) -> str:
    completed = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=15, check=False)
    output = (completed.stdout or completed.stderr).strip()
    return output if output else f"exit={completed.returncode}"


def _fingerprint(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {"keys": sorted(value)}
        for key in ("type", "subtype", "status", "stop_reason", "name"):
            if key in value:
                result[key] = value[key]
        request = value.get("request")
        if isinstance(request, dict):
            result["request_keys"] = sorted(request)
            for key in ("type", "subtype"):
                if isinstance(request.get(key), str):
                    result[f"request_{key}"] = request[key]
        return result
    if isinstance(value, list):
        return {"list_length": len(value), "items": [_fingerprint(item) for item in value[:8]]}
    return type(value).__name__


class ClaudeStream:
    def __init__(
        self,
        executable: str,
        command: list[str],
        environment: dict[str, str],
        artifact_dir: Path,
        workspace: Path,
    ):
        self._executable = executable
        self._command = command
        self._environment = environment
        self._artifact_dir = artifact_dir
        self._workspace = workspace
        self._process: asyncio.subprocess.Process | None = None
        self._events: list[dict[str, Any]] = []
        self._event_available = asyncio.Event()
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._log_path = artifact_dir / "harness.jsonl"

    def _record(self, direction: str, event: dict[str, Any]) -> None:
        record = {"timestamp_unix_ms": _now_ms(), "direction": direction, "shape": _fingerprint(event)}
        with self._log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    async def start(self) -> None:
        self._process = await asyncio.create_subprocess_exec(
            self._executable,
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # `--replay-user-messages` emits the original JSONL user event.
            # Long-context compaction probes intentionally make that event
            # larger than asyncio's 64 KiB default line limit.
            limit=4 * 1024 * 1024,
            env=self._environment,
            cwd=self._workspace,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def _read_stdout(self) -> None:
        assert self._process and self._process.stdout
        while line := await self._process.stdout.readline():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self._record("invalid_stdout", {"type": "non_json"})
                continue
            self._record("received", event)
            self._events.append(event)
            self._event_available.set()

    async def _read_stderr(self) -> None:
        assert self._process and self._process.stderr
        path = self._artifact_dir / "claude.stderr.log"
        with path.open("wb") as output:
            while chunk := await self._process.stderr.read(8192):
                output.write(chunk)

    async def user(self, content: str) -> None:
        assert self._process and self._process.stdin
        event = {"type": "user", "message": {"role": "user", "content": content}}
        self._record("sent", event)
        self._process.stdin.write(json.dumps(event).encode() + b"\n")
        await self._process.stdin.drain()

    async def close_input(self) -> None:
        """Signal that no more user JSONL events will arrive without killing Claude."""
        if self._process is None or self._process.stdin is None or self._process.stdin.is_closing():
            return
        self._process.stdin.close()
        with contextlib.suppress(BrokenPipeError):
            await self._process.stdin.wait_closed()

    async def result(self, after: int, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            for event in self._events[after:]:
                if event.get("type") == "result":
                    return event
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for Claude result event")
            if self._process is not None and self._process.returncode is not None:
                raise RuntimeError(f"Claude exited before result event (exit={self._process.returncode})")
            # Clear before checking/waiting so a result arriving during the check
            # cannot be left in _events with its wakeup signal discarded.
            self._event_available.clear()
            for event in self._events[after:]:
                if event.get("type") == "result":
                    return event
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._event_available.wait(), timeout=min(remaining, 0.5))

    async def event(
        self, after: int, predicate: Callable[[dict[str, Any]], bool], timeout_s: float
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            for event in self._events[after:]:
                if predicate(event):
                    return event
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for Claude stream event")
            if self._process is not None and self._process.returncode is not None:
                raise RuntimeError(f"Claude exited before stream event (exit={self._process.returncode})")
            self._event_available.clear()
            for event in self._events[after:]:
                if predicate(event):
                    return event
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._event_available.wait(), timeout=min(remaining, 0.5))

    def event_count(self) -> int:
        return len(self._events)

    @property
    def artifact_dir(self) -> Path:
        return self._artifact_dir

    def exit_code(self) -> int | None:
        return self._process.returncode if self._process is not None else None

    def result_event_count(self) -> int:
        return sum(event.get("type") == "result" for event in self._events)

    def has_event_type(self, event_type: str) -> bool:
        return any(event.get("type") == event_type for event in self._events)

    async def wait_exit(self, timeout_s: float) -> int:
        if self._process is None:
            raise RuntimeError("Claude process was not started")
        return await asyncio.wait_for(self._process.wait(), timeout=timeout_s)

    def tool_names(self) -> list[str]:
        names: list[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("type") == "tool_use" and isinstance(value.get("name"), str):
                    names.append(value["name"])
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        for event in self._events:
            visit(event)
        return names

    def agent_task_stats(self) -> dict[str, int]:
        """Return only lifecycle counts; task text and child output remain private."""
        stats = {"started": 0, "completed": 0, "errored": 0}
        for event in self._events:
            if event.get("type") != "system":
                continue
            subtype = event.get("subtype")
            if subtype == "task_started":
                stats["started"] += 1
            elif subtype == "task_notification":
                if event.get("status") == "completed":
                    stats["completed"] += 1
                else:
                    stats["errored"] += 1
        return stats

    def has_compact_boundary(self) -> bool:
        def visit(value: Any) -> bool:
            if isinstance(value, dict):
                if value.get("subtype") == "compact_boundary" or value.get("type") == "compact_boundary":
                    return True
                return any(visit(nested) for nested in value.values())
            if isinstance(value, list):
                return any(visit(nested) for nested in value)
            return False

        return any(visit(event) for event in self._events)

    async def close(self) -> None:
        if self._process is None:
            return
        if self._process.stdin is not None and not self._process.stdin.is_closing():
            self._process.stdin.close()
            with contextlib.suppress(BrokenPipeError):
                await self._process.stdin.wait_closed()
        if self._process.returncode is None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._process.wait(), timeout=15)
        if self._process.returncode is None:
            self._process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._process.wait(), timeout=10)
        if self._process.returncode is None:
            self._process.kill()
            await self._process.wait()
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


def _prepare_workspace(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text("# Arithmetic fixture\n\nThe checker expects an `answer.txt` file.\n", encoding="utf-8")
    (path / "adder.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left + right\n", encoding="utf-8"
    )


def _wire_has_tool_result_error(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") != "request" or not isinstance(row.get("shape"), dict):
            continue
        for message in row["shape"].get("messages", []):
            if isinstance(message, dict) and message.get("tool_result_error_count", 0) > 0:
                return True
    return False


def _agent_definition() -> str:
    return json.dumps(
        {
            "compat_inspector": {
                "description": "Inspect one small fixture file and report its behavior.",
                "prompt": "Read adder.py and return one sentence describing its behavior. Do not modify files.",
                "tools": ["Read"],
            }
        }
    )


def _record_background_event(artifact_dir: Path, subtype: str, status: str | None = None) -> None:
    """Record background lifecycle shape without retaining session output."""
    event: dict[str, Any] = {"type": "background_agent", "subtype": subtype}
    if status is not None:
        event["status"] = status
    record = {"timestamp_unix_ms": _now_ms(), "direction": "received", "shape": _fingerprint(event)}
    with (artifact_dir / "harness.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def _background_sessions(executable: str, environment: dict[str, str], workspace: Path) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [executable, "agents", "--json", "--all", "--cwd", str(workspace)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        env=environment,
        cwd=workspace,
    )
    if completed.returncode:
        raise RuntimeError("Claude background session listing failed")
    parsed = json.loads(completed.stdout)
    if not isinstance(parsed, list):
        raise RuntimeError("Claude background session listing was not an array")
    return [item for item in parsed if isinstance(item, dict)]


def _background_session_id(item: dict[str, Any]) -> str | None:
    for key in ("id", "agentId", "agent_id", "backgroundId", "background_id"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return None


def _background_session_status(item: dict[str, Any]) -> str:
    for key in ("status", "state"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return "unknown"


async def _run_background_scenario(
    executable: str,
    environment: dict[str, str],
    artifact_dir: Path,
    workspace: Path,
    model: str,
    root_tools: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Run Claude's native background service without copying task output into artifacts."""
    before_ids = {_background_session_id(item) for item in _background_sessions(executable, environment, workspace)}
    launch = subprocess.run(
        [
            executable,
            "--background",
            "--bare",
            "--dangerously-skip-permissions",
            "--model",
            model,
            "--tools",
            root_tools,
            "Use Bash to create background_agent.txt containing BACKGROUND followed by a newline, read it back, and finish.",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=environment,
        cwd=workspace,
    )
    if launch.returncode:
        launch_text = f"{launch.stdout}\n{launch.stderr}".lower()
        reason = (
            "credential_kind_unsupported"
            if "not supported for this credential kind" in launch_text
            else "client_preflight_rejected"
        )
        _record_background_event(artifact_dir, "preflight_rejected", reason)
        return {
            "background_launch_reason": reason,
            "background_file_exists": False,
            "reached": False,
        }
    match = re.search(r"backgrounded\W+([A-Za-z0-9_-]+)", launch.stdout)
    background_id = match.group(1) if match is not None else None
    _record_background_event(artifact_dir, "launched")
    deadline = time.monotonic() + timeout_s
    terminal = False
    status = "unknown"
    try:
        while True:
            sessions = _background_sessions(executable, environment, workspace)
            selected = next(
                (
                    item
                    for item in sessions
                    if _background_session_id(item) == background_id
                    or (background_id is None and _background_session_id(item) not in before_ids)
                ),
                None,
            )
            if selected is not None:
                if background_id is None:
                    background_id = _background_session_id(selected)
                status = _background_session_status(selected)
                _record_background_event(artifact_dir, "status", status)
                terminal = status.lower() in {"completed", "succeeded", "success", "failed", "error", "stopped", "cancelled"}
                if terminal:
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for Claude background session")
            await asyncio.sleep(1)
    finally:
        if background_id is not None and not terminal:
            subprocess.run(
                [executable, "stop", background_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
                env=environment,
                cwd=workspace,
            )
            _record_background_event(artifact_dir, "stopped")
    file_exists = (workspace / "background_agent.txt").exists()
    return {
        "background_session_sha256_12": _id_digest(background_id) if background_id is not None else None,
        "terminal_status": status,
        "background_file_exists": file_exists,
        "reached": status.lower() in {"completed", "succeeded", "success"} and file_exists,
    }


async def _run_scenario(
    stream: ClaudeStream, scenario: str, workspace: Path, result_timeout_s: float
) -> dict[str, Any]:
    baseline_prompt = (
        "Work as a coding agent in this small repository. Use Bash to inspect README.md and adder.py. "
        "Create answer.txt containing only 42 followed by a newline, read it back with Bash, then summarize the verification."
    )
    if scenario == "baseline":
        before = stream.event_count()
        await stream.user(baseline_prompt)
        result = await stream.result(before, result_timeout_s)
        tools = stream.tool_names()
        answer_exists = (workspace / "answer.txt").exists()
        return {
            "result_subtype": result.get("subtype"),
            "tool_names": tools,
            "answer_exists": answer_exists,
            "reached": result.get("subtype") == "success" and "Bash" in tools and answer_exists,
        }

    if scenario == "prompt_suggestions":
        first = await _run_scenario(stream, "baseline", workspace, result_timeout_s)
        # The feature event is optional at this point in the installed client.
        # A completed coding turn without it is a reached negative observation,
        # not a controller timeout.
        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
            await stream.event(
                0,
                lambda event: event.get("type") == "prompt_suggestion",
                min(result_timeout_s, 30),
            )
        return {
            "baseline_reached": first["reached"],
            "prompt_suggestion_observed": stream.has_event_type("prompt_suggestion"),
            "reached": first["reached"] and stream.has_event_type("prompt_suggestion"),
        }

    if scenario == "structured_output":
        before = stream.event_count()
        await stream.user(
            "Use Bash to inspect README.md and adder.py. Create answer.txt containing only 42 followed by a newline, "
            "read it back with Bash, then return the required structured result."
        )
        result = await stream.result(before, result_timeout_s)
        return {
            "result_subtype": result.get("subtype"),
            "answer_exists": (workspace / "answer.txt").exists(),
            "reached": result.get("subtype") == "success" and (workspace / "answer.txt").exists(),
        }

    if scenario == "tool_failure":
        before = stream.event_count()
        await stream.user(
            "Use Bash to run the command false. After it fails, use Bash to create tool_failure_recovered.txt containing "
            "RECOVERED followed by a newline, read it back, and then finish."
        )
        result = await stream.result(before, result_timeout_s)
        recovered = (workspace / "tool_failure_recovered.txt").exists()
        error_observed = _wire_has_tool_result_error(stream.artifact_dir / "wire.jsonl")
        return {
            "result_subtype": result.get("subtype"),
            "tool_result_error_observed": error_observed,
            "recovered_file_exists": recovered,
            "reached": result.get("subtype") == "success" and error_observed and recovered,
        }

    if scenario == "auto_compact":
        before = stream.event_count()
        padding = " context" * 25_000
        await stream.user(
            "Use Bash to create answer.txt containing 42 followed by a newline, read it back, and finish." + padding
        )
        first = await stream.result(before, result_timeout_s)
        after_first = stream.event_count()
        await stream.user("Use Bash to read answer.txt and state its contents.")
        second = await stream.result(after_first, result_timeout_s)
        return {
            "first_subtype": first.get("subtype"),
            "second_subtype": second.get("subtype"),
            "compact_boundary": stream.has_compact_boundary(),
            "reached": first.get("subtype") == "success"
            and second.get("subtype") == "success"
            and stream.has_compact_boundary(),
        }

    if scenario == "baseline_eof":
        await stream.user(baseline_prompt)
        await stream.close_input()
        exit_code = await stream.wait_exit(result_timeout_s)
        tools = stream.tool_names()
        answer_exists = (workspace / "answer.txt").exists()
        return {
            "exit_code": exit_code,
            "result_event_count": stream.result_event_count(),
            "tool_names": tools,
            "answer_exists": answer_exists,
            "reached": exit_code == 0 and "Bash" in tools and answer_exists,
        }

    if scenario == "agent":
        before = stream.event_count()
        await stream.user(
            "Use the Agent tool exactly once with subagent_type=compat_inspector. Wait for it to finish. Then stop "
            "immediately: do not call any more tools and respond with exactly SUBAGENT_COMPLETED."
        )
        result = await stream.result(before, result_timeout_s)
        tools = stream.tool_names()
        task_stats = stream.agent_task_stats()
        return {
            "result_subtype": result.get("subtype"),
            "tool_names": tools,
            "agent_task_stats": task_stats,
            "reached": result.get("subtype") == "success"
            # Claude Code reports native Agent execution through system task
            # lifecycle events. The stream's generic tool-use item currently
            # does not reliably carry the public Agent label.
            and task_stats["started"] >= 1
            and task_stats["completed"] >= 1
            and task_stats["errored"] == 0,
        }

    if scenario in {"mcp_tool", "mcp_tool_failure", "mcp_progress"}:
        failure = scenario == "mcp_tool_failure"
        tool_name = "mcp__fixture__fixture_failure" if failure else "mcp__fixture__fixture_answer"
        progress = scenario == "mcp_progress"
        result_path = workspace / ("mcp_tool_failure_recovered.txt" if failure else "mcp_progress.txt" if progress else "mcp_tool.txt")
        action = (
            "After it returns an error, use Bash to create mcp_tool_failure_recovered.txt containing RECOVERED, read that "
            "file back, and finish."
            if failure
            else f"Then use Bash to create {result_path.name} containing its result, read that file back, and finish."
        )
        before = stream.event_count()
        await stream.user(f"Use the {tool_name} MCP tool at least once. {action}")
        result = await stream.result(before, result_timeout_s)
        tools = stream.tool_names()
        mcp_tool_calls = tools.count(tool_name)
        trace_path = stream.artifact_dir / "mcp_transport.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
        progress_sent = trace.get("progress_sent") if isinstance(trace, dict) else None
        return {
            "result_subtype": result.get("subtype"),
            "mcp_failure": failure,
            "mcp_progress_sent": progress_sent,
            "mcp_tool_calls": mcp_tool_calls,
            "result_file_exists": result_path.exists(),
            "reached": result.get("subtype") == "success"
            and mcp_tool_calls >= 1
            and result_path.exists()
            and (not progress or progress_sent is True),
        }

    if scenario == "mcp_elicitation":
        tool_name = "mcp__fixture__fixture_elicitation"
        result_path = workspace / "mcp_elicitation.txt"
        before = stream.event_count()
        await stream.user(
            "Use the mcp__fixture__fixture_elicitation MCP tool exactly once. Then use Bash to create "
            "mcp_elicitation.txt containing its result, read that file back, and finish."
        )
        # Claude's stream-json input accepts user-text events only. A native MCP
        # elicitation therefore appears as an observable control request but
        # cannot be answered by this noninteractive transport. Stop at that
        # boundary rather than letting the fixture wait until the global timeout.
        control = await stream.event(
            before,
            lambda event: event.get("type") == "control_request",
            timeout_s=min(result_timeout_s, 90),
        )
        trace_path = stream.artifact_dir / "mcp_transport.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
        action = trace.get("elicitation_response_action") if isinstance(trace, dict) else None
        tools = stream.tool_names()
        mcp_tool_calls = tools.count(tool_name)
        return {
            "mcp_control_request": True,
            "mcp_control_request_type": (
                control.get("request", {}).get("subtype") if isinstance(control.get("request"), dict) else None
            ),
            "mcp_elicitation_action": action,
            "mcp_tool_calls": mcp_tool_calls,
            "result_file_exists": result_path.exists(),
            "reached": False,
        }

    if scenario == "agent_eof":
        await stream.user(
            "Use the Agent tool exactly once with subagent_type=compat_inspector. Wait for it to finish, then report "
            "the child result without reading adder.py yourself."
        )
        await stream.close_input()
        exit_code = await stream.wait_exit(result_timeout_s)
        task_stats = stream.agent_task_stats()
        return {
            "exit_code": exit_code,
            "result_event_count": stream.result_event_count(),
            "agent_task_stats": task_stats,
            "reached": exit_code == 0
            and task_stats["started"] >= 1
            and task_stats["completed"] >= 1
            and task_stats["errored"] == 0,
        }

    if scenario == "nested_agent":
        before = stream.event_count()
        await stream.user(
            "Use the Agent tool exactly once. Tell the child to try to delegate a fact to another Agent and report whether "
            "that capability is available. Wait for the child and report the result."
        )
        result = await stream.result(before, result_timeout_s)
        tools = stream.tool_names()
        task_stats = stream.agent_task_stats()
        return {
            "result_subtype": result.get("subtype"),
            "tool_names": tools,
            "agent_task_stats": task_stats,
            "reached": result.get("subtype") == "success"
            and task_stats["started"] >= 1
            and task_stats["completed"] >= 1
            and task_stats["errored"] == 0,
        }

    if scenario == "compact":
        before = stream.event_count()
        await stream.user(baseline_prompt)
        first = await stream.result(before, result_timeout_s)
        second_before = stream.event_count()
        await stream.user("/compact")
        second = await stream.result(second_before, result_timeout_s)
        final_before = stream.event_count()
        await stream.user("After compaction, use Bash to read answer.txt and state its contents.")
        final = await stream.result(final_before, result_timeout_s)
        compact_boundary = stream.has_compact_boundary()
        return {
            "first_subtype": first.get("subtype"),
            "compact_subtype": second.get("subtype"),
            "final_subtype": final.get("subtype"),
            "compact_boundary": compact_boundary,
            "reached": first.get("subtype") == "success"
            and final.get("subtype") == "success"
            and compact_boundary,
        }

    if scenario == "steer":
        before = stream.event_count()
        await stream.user(
            "Use Bash to inspect README.md and adder.py slowly, one step at a time. Do not finish until every inspection "
            "has been verified."
        )
        await stream.event(
            before,
            lambda event: event.get("type") == "system"
            and event.get("subtype") == "status"
            and event.get("status") == "requesting",
            min(result_timeout_s, 30),
        )
        await stream.user(
            "User steering: stop the prior inspection. Create claude_steering.txt containing STEERED followed by a newline, "
            "read it with Bash, and then finish."
        )
        # The print-mode JSONL protocol keeps a multi-user session open until
        # EOF. Both messages have been accepted while the first was active; EOF
        # now asks Claude to settle that queued/steered interaction and emit its
        # terminal result without a process-level cancellation.
        await stream.close_input()
        first = await stream.result(before, result_timeout_s)
        result_subtypes = [first.get("subtype")]
        steering_file_exists = (workspace / "claude_steering.txt").exists()
        if not steering_file_exists:
            after_first = stream.event_count()
            second = await stream.result(after_first, result_timeout_s)
            result_subtypes.append(second.get("subtype"))
            steering_file_exists = (workspace / "claude_steering.txt").exists()
        return {
            "result_subtypes": result_subtypes,
            "steering_file_exists": steering_file_exists,
            "reached": all(subtype == "success" for subtype in result_subtypes) and steering_file_exists,
        }

    raise ValueError(f"unknown scenario: {scenario}")


async def run(args: argparse.Namespace) -> int:
    artifact_dir = args.artifacts.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if (artifact_dir / "result.json").exists():
        raise FileExistsError(f"run directory already contains result.json: {artifact_dir}")
    workspace = artifact_dir / "workspace"
    claude_home = artifact_dir / "claude_home"
    _prepare_workspace(workspace)
    executable = shutil.which(args.claude) if os.sep not in args.claude else args.claude
    if not executable:
        raise FileNotFoundError(f"Claude executable not found: {args.claude}")
    environment = {
        **os.environ,
        "HOME": str(claude_home),
        "ANTHROPIC_BASE_URL": args.proxy_url.rstrip("/"),
        "ANTHROPIC_AUTH_TOKEN": "compat-lab-placeholder",
        "ANTHROPIC_API_KEY": "compat-lab-placeholder",
        "ANTHROPIC_MODEL": args.model,
        "ANTHROPIC_SMALL_FAST_MODEL": args.model,
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096",
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": str(args.auto_compact_pct),
    }
    session_id = str(uuid.uuid4())
    root_tools = "Bash,Read,Write,Edit,Agent"
    command = [
        "--bare",
        "--dangerously-skip-permissions",
        "--model",
        args.model,
        "--session-id",
        session_id,
        "--tools",
        root_tools,
        "--agents",
        _agent_definition(),
        "--print",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--replay-user-messages",
    ]
    if args.scenario in {"mcp_tool", "mcp_tool_failure", "mcp_elicitation", "mcp_progress"}:
        mcp_env: dict[str, str] = {}
        if args.scenario == "mcp_tool_failure":
            mcp_env["DYNAMO_COMPAT_FIXTURE_MCP_FAIL"] = "1"
        elif args.scenario == "mcp_elicitation":
            mcp_env = {
                "DYNAMO_COMPAT_FIXTURE_MCP_ELICIT": "1",
                "DYNAMO_COMPAT_FIXTURE_MCP_TRACE": str(artifact_dir / "mcp_transport.json"),
            }
        elif args.scenario == "mcp_progress":
            mcp_env = {
                "DYNAMO_COMPAT_FIXTURE_MCP_PROGRESS": "1",
                "DYNAMO_COMPAT_FIXTURE_MCP_TRACE": str(artifact_dir / "mcp_transport.json"),
            }
        mcp_config = json.dumps(
            {
                "mcpServers": {
                    "fixture": {
                        "command": sys.executable,
                        "args": [str(Path(__file__).with_name("fixture_mcp_server.py"))],
                        "env": mcp_env,
                    }
                }
            }
        )
        command.extend(["--mcp-config", mcp_config, "--strict-mcp-config"])
    if args.scenario == "structured_output":
        command.extend(
            [
                "--json-schema",
                json.dumps(
                    {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                        "additionalProperties": False,
                    }
                ),
            ]
        )
    if args.scenario == "agent_forwarded":
        command.append("--forward-subagent-text")
    if args.scenario == "prompt_suggestions":
        command.extend(["--prompt-suggestions", "true"])
    run_started_unix_ms = _now_ms()
    scenario = {
        "harness": "claude_code",
        "scenario": args.scenario,
        "model": args.model,
        "client_version": _client_version(executable),
        "proxy_url": args.proxy_url,
        "auto_compact_pct": args.auto_compact_pct,
        "result_timeout_s": args.result_timeout_s,
        "client_session_sha256_12": _id_digest(session_id),
        "root_tools": root_tools,
        "mcp_fixture_enabled": args.scenario in {"mcp_tool", "mcp_tool_failure", "mcp_elicitation", "mcp_progress"},
        "run_started_unix_ms": run_started_unix_ms,
    }
    (artifact_dir / "scenario.json").write_text(
        json.dumps(
            scenario,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    stream = ClaudeStream(executable, command, environment, artifact_dir, workspace)
    resumed_stream: ClaudeStream | None = None
    result: dict[str, Any]

    async def run_stream_scenario() -> dict[str, Any]:
        nonlocal resumed_stream
        await stream.start()
        if args.scenario in {"resume", "fork_session"}:
            first = await _run_scenario(stream, "baseline", workspace, args.result_timeout_s)
            await stream.close_input()
            with contextlib.suppress(asyncio.TimeoutError):
                await stream.wait_exit(30)
            await stream.close()
            resumed_command = command.copy()
            session_id_index = resumed_command.index("--session-id")
            del resumed_command[session_id_index : session_id_index + 2]
            resumed_command.extend(["--resume", session_id])
            if args.scenario == "fork_session":
                resumed_command.append("--fork-session")
            resumed_stream = ClaudeStream(executable, resumed_command, environment, artifact_dir, workspace)
            await resumed_stream.start()
            before = resumed_stream.event_count()
            await resumed_stream.user(
                "Use Bash to read answer.txt, create resumed_session.txt containing RESUMED followed by a newline, "
                "read it back, and finish."
            )
            second = await resumed_stream.result(before, args.result_timeout_s)
            resumed_file_exists = (workspace / "resumed_session.txt").exists()
            result = {
                "first_subtype": first.get("result_subtype"),
                "second_subtype": second.get("subtype"),
                "resumed_file_exists": resumed_file_exists,
                "reached": first.get("reached") and second.get("subtype") == "success" and resumed_file_exists,
            }
        else:
            scenario_name = "agent" if args.scenario == "agent_forwarded" else args.scenario
            result = await _run_scenario(stream, scenario_name, workspace, args.result_timeout_s)
        result["outcome"] = "pass" if result.pop("reached") else "not_reached"
        return result

    try:
        if args.scenario == "background":
            result = await asyncio.wait_for(
                _run_background_scenario(
                    executable,
                    environment,
                    artifact_dir,
                    workspace,
                    args.model,
                    root_tools,
                    args.result_timeout_s,
                ),
                timeout=args.result_timeout_s,
            )
            result["outcome"] = "pass" if result.pop("reached") else "not_reached"
        else:
            # A workflow can await more than one native terminal event (manual
            # compaction and process resume both do). Bound the whole workflow,
            # rather than accidentally granting the configured budget to every
            # individual wait.
            result = await asyncio.wait_for(run_stream_scenario(), timeout=args.result_timeout_s)
    except (TimeoutError, asyncio.TimeoutError) as error:
        result = {
            "outcome": "inconclusive",
            "error_type": type(error).__name__,
            "error": str(error),
            "agent_task_stats": stream.agent_task_stats(),
        }
    except RuntimeError as error:
        result = {
            "outcome": "inconclusive" if stream.exit_code() == 0 else "harness_failure",
            "error_type": type(error).__name__,
            "error": str(error),
            "exit_code": stream.exit_code(),
            "agent_task_stats": stream.agent_task_stats(),
        }
    except Exception as error:
        result = {"outcome": "harness_failure", "error_type": type(error).__name__, "error": str(error)}
    finally:
        if resumed_stream is not None:
            await resumed_stream.close()
        await stream.close()
    scenario["run_finished_unix_ms"] = _now_ms()
    (artifact_dir / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
    (artifact_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["outcome"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--claude", default="claude")
    parser.add_argument("--auto-compact-pct", type=int, default=1)
    parser.add_argument("--result-timeout-s", type=float, default=900)
    parser.add_argument(
        "--scenario",
        choices=("baseline", "structured_output", "prompt_suggestions", "tool_failure", "auto_compact", "baseline_eof", "agent", "agent_forwarded", "agent_eof", "nested_agent", "compact", "resume", "fork_session", "steer", "background", "mcp_tool", "mcp_tool_failure", "mcp_elicitation", "mcp_progress"),
        required=True,
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
