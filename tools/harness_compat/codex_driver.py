#!/usr/bin/env python3
"""Native Codex app-server driver for compatibility-lab scenarios.

This does not emulate Codex over HTTP. It asks the installed app-server to run
real turns, then uses its lifecycle RPCs for compaction, steering, and
interruption.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _now_ms() -> int:
    return round(time.time() * 1000)


def _id_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def _client_version(executable: str) -> str:
    completed = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=15, check=False)
    output = (completed.stdout or completed.stderr).strip()
    return output if output else f"exit={completed.returncode}"


def _fingerprint(value: Any) -> Any:
    """Preserve RPC shape without copying model/user text into the transcript."""
    if isinstance(value, dict):
        result: dict[str, Any] = {"keys": sorted(value)}
        for key in ("method", "type", "status", "code"):
            if key in value:
                result[key] = value[key]
        for key in ("threadId", "turnId", "id"):
            if isinstance(value.get(key), str):
                result[key] = value[key][:12]
        params = value.get("params")
        if isinstance(params, dict):
            if isinstance(params.get("threadId"), str):
                result["params_threadId"] = params["threadId"][:12]
            if isinstance(params.get("turnId"), str):
                result["params_turnId"] = params["turnId"][:12]
            item = params.get("item")
            if isinstance(item, dict):
                for key in ("type", "name", "status"):
                    if isinstance(item.get(key), str):
                        result[f"item_{key}"] = item[key]
            turn = params.get("turn")
            if isinstance(turn, dict):
                if isinstance(turn.get("id"), str):
                    result["params_turn_id"] = turn["id"][:12]
                if isinstance(turn.get("status"), str):
                    result["turn_status"] = turn["status"]
            error = params.get("error")
            if isinstance(error, dict):
                for key in ("code", "type"):
                    if isinstance(error.get(key), (str, int)):
                        result[f"error_{key}"] = error[key]
        return result
    if isinstance(value, list):
        return {"list_length": len(value), "items": [_fingerprint(item) for item in value[:8]]}
    return type(value).__name__


class AppServer:
    def __init__(
        self, executable: str, environment: dict[str, str], artifact_dir: Path, turn_timeout_s: float
    ):
        self._executable = executable
        self._environment = environment
        self._artifact_dir = artifact_dir
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._notifications: list[dict[str, Any]] = []
        self._notification_event = asyncio.Event()
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._log_path = artifact_dir / "harness.jsonl"
        self._approval_request_count = 0
        self._dynamic_tool_request_count = 0
        self._dynamic_tool_success = True
        self._user_input_request_count = 0
        self._mcp_elicitation_request_count = 0
        self.turn_timeout_s = turn_timeout_s

    def _record(self, direction: str, message: dict[str, Any]) -> None:
        record = {"timestamp_unix_ms": _now_ms(), "direction": direction, "shape": _fingerprint(message)}
        with self._log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    async def start(self) -> None:
        self._process = await asyncio.create_subprocess_exec(
            self._executable,
            "app-server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._environment,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def _read_stdout(self) -> None:
        assert self._process and self._process.stdout
        while line := await self._process.stdout.readline():
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._record("invalid_stdout", {"type": "non_json"})
                continue
            self._record("received", message)
            if "id" in message and isinstance(message["id"], (str, int)) and not isinstance(message["id"], bool):
                future = self._pending.pop(message["id"], None) if isinstance(message["id"], int) else None
                if future is not None:
                    future.set_result(message)
                    continue
                if isinstance(message.get("method"), str):
                    await self._accept_isolated_approval(message)
                    await self._answer_user_input(message)
                    await self._answer_mcp_elicitation(message)
                    await self._accept_dynamic_tool_call(message)
                    continue
            if "method" in message:
                self._notifications.append(message)
                self._notification_event.set()

    async def _accept_isolated_approval(self, message: dict[str, Any]) -> None:
        """Accept fixture-only tool approvals that forked threads can reintroduce.

        The root thread is created with `approvalPolicy: never`, but current
        Codex forks can issue the normal app-server approval callback again.
        The harness owns a disposable workspace, so accepting this callback is
        the native equivalent of its root policy, not a bypass for user work.
        """
        if message.get("method") not in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            return
        assert self._process and self._process.stdin
        response = {"jsonrpc": "2.0", "id": message["id"], "result": {"decision": "acceptForSession"}}
        self._approval_request_count += 1
        self._record("sent", response)
        self._process.stdin.write(json.dumps(response).encode() + b"\n")
        await self._process.stdin.drain()

    async def _accept_dynamic_tool_call(self, message: dict[str, Any]) -> None:
        if message.get("method") != "item/tool/call":
            return
        assert self._process and self._process.stdin
        response = {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "success": self._dynamic_tool_success,
                "contentItems": [
                    {"type": "inputText", "text": "42" if self._dynamic_tool_success else "fixture unavailable"}
                ],
            },
        }
        self._dynamic_tool_request_count += 1
        self._record("sent", response)
        self._process.stdin.write(json.dumps(response).encode() + b"\n")
        await self._process.stdin.drain()

    async def _answer_user_input(self, message: dict[str, Any]) -> None:
        if message.get("method") != "item/tool/requestUserInput":
            return
        params = message.get("params")
        questions = params.get("questions") if isinstance(params, dict) else None
        if not isinstance(questions, list):
            return
        answers: dict[str, dict[str, list[str]]] = {}
        for question in questions:
            if isinstance(question, dict) and isinstance(question.get("id"), str):
                answers[question["id"]] = {"answers": ["CONTINUE"]}
        assert self._process and self._process.stdin
        response = {"jsonrpc": "2.0", "id": message["id"], "result": {"answers": answers}}
        self._user_input_request_count += 1
        self._record("sent", response)
        self._process.stdin.write(json.dumps(response).encode() + b"\n")
        await self._process.stdin.drain()

    async def _answer_mcp_elicitation(self, message: dict[str, Any]) -> None:
        """Accept the fixture's non-secret form request without recording it."""
        if message.get("method") != "mcpServer/elicitation/request":
            return
        assert self._process and self._process.stdin
        response = {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {"action": "accept", "content": {"choice": "CONTINUE"}},
        }
        self._mcp_elicitation_request_count += 1
        self._record("sent", response)
        self._process.stdin.write(json.dumps(response).encode() + b"\n")
        await self._process.stdin.drain()

    async def _read_stderr(self) -> None:
        assert self._process and self._process.stderr
        path = self._artifact_dir / "codex-app-server.stderr.log"
        with path.open("wb") as output:
            while chunk := await self._process.stderr.read(8192):
                output.write(chunk)

    async def request(self, method: str, params: dict[str, Any], timeout_s: float = 60) -> dict[str, Any]:
        assert self._process and self._process.stdin
        request_id = self._next_id
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        self._record("sent", message)
        self._process.stdin.write(json.dumps(message).encode() + b"\n")
        await self._process.stdin.drain()
        try:
            response = await asyncio.wait_for(future, timeout=timeout_s)
        except BaseException:
            self._pending.pop(request_id, None)
            raise
        if "error" in response:
            raise RuntimeError(f"{method} failed: {response['error']}")
        return response["result"]

    async def notification(
        self, method: str, predicate: Callable[[dict[str, Any]], bool] | None = None, timeout_s: float = 600
    ) -> dict[str, Any]:
        predicate = predicate or (lambda _message: True)
        deadline = time.monotonic() + timeout_s
        while True:
            # Clear before scanning: a notification that arrives before or during
            # the scan is then observed either in the list or through this event.
            # Clearing after the scan can lose the final turn/completed wakeup.
            self._notification_event.clear()
            for index, message in enumerate(self._notifications):
                if message.get("method") == method and predicate(message):
                    return self._notifications.pop(index)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {method}")
            await asyncio.wait_for(self._notification_event.wait(), timeout=remaining)

    async def close(self) -> None:
        if self._process is None:
            return
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

    def tool_call_count(self, name: str) -> int:
        """Count named tool items reported by the native app-server."""
        count = 0

        def visit(value: Any) -> None:
            nonlocal count
            if isinstance(value, dict):
                if value.get("name") == name:
                    count += 1
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        for notification in self._notifications:
            visit(notification.get("params", {}))
        return count

    def error_summary(self) -> dict[str, Any]:
        """Summarize app-server errors without retaining their message text."""
        codes: set[str] = set()
        types: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if isinstance(value.get("code"), (str, int)):
                    codes.add(str(value["code"]))
                if isinstance(value.get("type"), str):
                    types.add(value["type"])
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        errors = [message for message in self._notifications if message.get("method") == "error"]
        for message in errors:
            visit(message.get("params", {}))
        return {"count": len(errors), "codes": sorted(codes), "types": sorted(types)}

    def completed_turn_status(self, thread_id: str) -> str | None:
        """Return a terminal status already observed for this isolated thread."""
        for message in reversed(self._notifications):
            if message.get("method") != "turn/completed":
                continue
            params = message.get("params")
            if not isinstance(params, dict) or params.get("threadId") != thread_id:
                continue
            turn = params.get("turn")
            if isinstance(turn, dict) and isinstance(turn.get("status"), str):
                return turn["status"]
        return None

    def approval_request_count(self) -> int:
        return self._approval_request_count

    def dynamic_tool_request_count(self) -> int:
        return self._dynamic_tool_request_count

    def user_input_request_count(self) -> int:
        return self._user_input_request_count

    def mcp_elicitation_request_count(self) -> int:
        return self._mcp_elicitation_request_count

    def set_dynamic_tool_success(self, success: bool) -> None:
        self._dynamic_tool_success = success

    def notification_count(self, method: str) -> int:
        return sum(message.get("method") == method for message in self._notifications)

    def command_exit_codes(self, thread_id: str) -> list[int]:
        """Return native command-result exit codes without retaining command text or output."""
        exit_codes: list[int] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key in ("exitCode", "exit_code"):
                    if isinstance(value.get(key), int) and not isinstance(value[key], bool):
                        exit_codes.append(value[key])
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        for message in self._notifications:
            if message.get("method") != "item/completed":
                continue
            params = message.get("params")
            if not isinstance(params, dict) or params.get("threadId") != thread_id:
                continue
            item = params.get("item")
            if isinstance(item, dict) and item.get("type") == "commandExecution":
                visit(item)
        return exit_codes

    def item_type_count(self, thread_id: str, item_type: str) -> int:
        count = 0
        for message in self._notifications:
            if message.get("method") not in {"item/started", "item/completed"}:
                continue
            params = message.get("params")
            if not isinstance(params, dict) or params.get("threadId") != thread_id:
                continue
            item = params.get("item")
            if isinstance(item, dict) and item.get("type") == item_type:
                count += 1
        return count


def _write_config(
    codex_home: Path,
    proxy_url: str,
    model: str,
    mcp_fixture: Path | None = None,
    mcp_failure: bool = False,
    mcp_elicitation: bool = False,
    mcp_progress: bool = False,
    mcp_trace: Path | None = None,
    request_user_input: bool = False,
) -> None:
    codex_home.mkdir(parents=True, exist_ok=True)
    mcp_config = ""
    if mcp_fixture is not None:
        mcp_env_vars: list[str] = []
        if mcp_failure:
            mcp_env_vars.append('DYNAMO_COMPAT_FIXTURE_MCP_FAIL = "1"')
        elif mcp_elicitation:
            mcp_env_vars.append('DYNAMO_COMPAT_FIXTURE_MCP_ELICIT = "1"')
            mcp_env_vars.append('DYNAMO_COMPAT_FIXTURE_MCP_OPENAI_FORM = "1"')
        elif mcp_progress:
            mcp_env_vars.append('DYNAMO_COMPAT_FIXTURE_MCP_PROGRESS = "1"')
        if mcp_trace is not None:
            mcp_env_vars.append(f"DYNAMO_COMPAT_FIXTURE_MCP_TRACE = {json.dumps(str(mcp_trace))}")
        mcp_env = f"env = {{ {', '.join(mcp_env_vars)} }}\n" if mcp_env_vars else ""
        mcp_config = f"""
[mcp_servers.fixture]
command = {json.dumps(sys.executable)}
args = [{json.dumps(str(mcp_fixture))}]
{mcp_env}startup_timeout_sec = 10
tool_timeout_sec = 30
"""
    user_input_feature = "experimental_request_user_input = true\n" if request_user_input else ""
    (codex_home / "config.toml").write_text(
        f"""model_provider = "local"
model = "{model}"
model_max_output_tokens = 4096

[features]
{user_input_feature}
[features.multi_agent_v2]
enabled = true
max_concurrent_threads_per_session = 4
non_code_mode_only = false

[model_providers.local]
name = "dynamo-compat-lab"
base_url = "{proxy_url.rstrip('/')}/v1"
wire_api = "responses"
env_key = "LOCAL_API_KEY"
{mcp_config}
""",
        encoding="utf-8",
    )


def _prepare_workspace(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text("# Arithmetic fixture\n\nThe checker expects an `answer.txt` file.\n", encoding="utf-8")
    (path / "adder.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left + right\n", encoding="utf-8"
    )


def _text_input(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def _journal_tool_call_count(codex_home: Path, tool_name: str) -> int:
    """Count native Codex function calls without retaining session text in artifacts.

    App-server item notifications intentionally omit function names in the current
    protocol. The isolated Codex journal has the authoritative response item type
    and name, so use it only as a semantic oracle after the turn completes.
    """
    count = 0
    for journal in (codex_home / "sessions").glob("**/*.jsonl"):
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "response_item":
                continue
            payload = record.get("payload")
            if isinstance(payload, dict) and payload.get("type") == "function_call" and payload.get("name") == tool_name:
                count += 1
    return count


def _journal_agent_error_count(codex_home: Path) -> int:
    """Count completed-agent error statuses emitted through collaboration tools."""
    count = 0

    def visit(value: Any) -> None:
        nonlocal count
        if isinstance(value, dict):
            status = value.get("agent_status")
            if isinstance(status, dict) and "errored" in status:
                count += 1
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for journal in (codex_home / "sessions").glob("**/*.jsonl"):
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict) or payload.get("type") != "function_call_output":
                continue
            output = payload.get("output")
            if not isinstance(output, str):
                continue
            try:
                visit(json.loads(output))
            except json.JSONDecodeError:
                continue
    return count


async def _start_turn(
    server: AppServer, thread_id: str, prompt: str, output_schema: dict[str, Any] | None = None
) -> str:
    params: dict[str, Any] = {"threadId": thread_id, "input": _text_input(prompt)}
    if output_schema is not None:
        params["outputSchema"] = output_schema
    result = await server.request("turn/start", params)
    return result["turn"]["id"]


async def _wait_turn(server: AppServer, thread_id: str, _turn_id: str) -> dict[str, Any]:
    event = await server.notification(
        "turn/completed",
        # The app-server can replace the turn ID as it materializes a compacted
        # thread. This driver runs one turn at a time in one isolated thread, so
        # the thread-scoped completion remains the stable lifecycle contract.
        lambda message: message.get("params", {}).get("threadId") == thread_id,
        timeout_s=server.turn_timeout_s,
    )
    return event["params"]["turn"]


async def _wait_for_journal_tool(
    server: AppServer,
    thread_id: str,
    journal_tool_count: Callable[[str], int],
    tool_name: str,
    minimum: int,
    timeout_s: float = 120,
) -> int:
    deadline = time.monotonic() + timeout_s
    while True:
        count = journal_tool_count(tool_name)
        if count >= minimum:
            return count
        terminal_status = server.completed_turn_status(thread_id)
        if terminal_status is not None:
            raise RuntimeError(
                f"turn reached terminal status {terminal_status} before native tool {tool_name} was observed"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for native tool {tool_name}")
        await asyncio.sleep(0.1)


async def _run_scenario(
    server: AppServer,
    thread_id: str,
    scenario: str,
    workspace: Path,
    journal_tool_count: Callable[[str], int],
    journal_agent_error_count: Callable[[], int],
) -> dict[str, Any]:
    baseline_prompt = (
        "Work as a coding agent in this small repository. Inspect README.md and adder.py. "
        "Use your shell tools, create answer.txt containing only 42 followed by a newline, read it back, "
        "then summarize the exact verification you performed."
    )
    if scenario == "baseline":
        turn_id = await _start_turn(server, thread_id, baseline_prompt)
        turn = await _wait_turn(server, thread_id, turn_id)
        answer_exists = (workspace / "answer.txt").exists()
        return {
            "turn_status": turn["status"],
            "answer_exists": answer_exists,
            "reached": turn["status"] == "completed" and answer_exists,
        }

    if scenario in {"dynamic_tool", "dynamic_namespace_tool", "dynamic_tool_failure"}:
        tool_name = "fixture.answer" if scenario == "dynamic_namespace_tool" else "fixture_answer"
        result_path = workspace / ("dynamic_tool_failure_recovered.txt" if scenario == "dynamic_tool_failure" else "dynamic_tool.txt")
        prompt = (
            f"Use the {tool_name} dynamic tool exactly once. Then use a shell tool to create {result_path.name} containing "
            "the tool result, read that file back, and finish."
        )
        if scenario == "dynamic_tool_failure":
            prompt = (
                "Use the fixture_answer dynamic tool exactly once. After it reports failure, use a shell tool to create "
                "dynamic_tool_failure_recovered.txt containing RECOVERED, read that file back, and finish."
            )
        turn_id = await _start_turn(
            server,
            thread_id,
            prompt,
        )
        turn = await _wait_turn(server, thread_id, turn_id)
        file_exists = result_path.exists()
        return {
            "turn_status": turn["status"],
            "dynamic_tool_requests": server.dynamic_tool_request_count(),
            "dynamic_tool_items": server.item_type_count(thread_id, "dynamicToolCall"),
            "result_file_exists": file_exists,
            "reached": turn["status"] == "completed"
            and server.dynamic_tool_request_count() == 1
            and file_exists,
        }

    if scenario in {"mcp_tool", "mcp_tool_failure", "mcp_progress"}:
        failure = scenario == "mcp_tool_failure"
        tool_name = "mcp__fixture__fixture_failure" if failure else "mcp__fixture__fixture_answer"
        progress = scenario == "mcp_progress"
        result_path = workspace / ("mcp_tool_failure_recovered.txt" if failure else "mcp_progress.txt" if progress else "mcp_tool.txt")
        action = (
            "After it returns an error, use a shell tool to create mcp_tool_failure_recovered.txt containing RECOVERED, "
            "read that file back, and finish."
            if failure
            else f"Then use a shell tool to create {result_path.name} containing its result, read that file back, and finish."
        )
        turn_id = await _start_turn(
            server,
            thread_id,
            f"Use the {tool_name} MCP tool exactly once. {action}",
        )
        turn = await _wait_turn(server, thread_id, turn_id)
        mcp_tool_item_events = server.item_type_count(thread_id, "mcpToolCall")
        return {
            "turn_status": turn["status"],
            "mcp_failure": failure,
            "mcp_progress_events": server.notification_count("item/mcpToolCall/progress"),
            "mcp_tool_item_events": mcp_tool_item_events,
            "result_file_exists": result_path.exists(),
            "reached": turn["status"] == "completed"
            and mcp_tool_item_events == 2
            and result_path.exists()
            and (not progress or server.notification_count("item/mcpToolCall/progress") >= 1),
        }

    if scenario == "mcp_elicitation":
        result_path = workspace / "mcp_elicitation.txt"
        turn_id = await _start_turn(
            server,
            thread_id,
            "Use the mcp__fixture__fixture_elicitation MCP tool exactly once. Then use a shell tool to create "
            "mcp_elicitation.txt containing its result, read that file back, and finish.",
        )
        turn = await _wait_turn(server, thread_id, turn_id)
        mcp_tool_item_events = server.item_type_count(thread_id, "mcpToolCall")
        return {
            "turn_status": turn["status"],
            "mcp_elicitation_requests": server.mcp_elicitation_request_count(),
            "mcp_tool_item_events": mcp_tool_item_events,
            "result_file_exists": result_path.exists(),
            "reached": turn["status"] == "completed"
            and server.mcp_elicitation_request_count() == 1
            and mcp_tool_item_events == 2
            and result_path.exists(),
        }

    if scenario == "request_user_input":
        result_path = workspace / "user_input.txt"
        turn_id = await _start_turn(
            server,
            thread_id,
            "Use the request_user_input tool exactly once to ask one non-secret question with one option. After the user "
            "answers, use a shell tool to create user_input.txt containing USER_INPUT_RECEIVED, read it back, and finish.",
        )
        turn = await _wait_turn(server, thread_id, turn_id)
        return {
            "turn_status": turn["status"],
            "user_input_requests": server.user_input_request_count(),
            "result_file_exists": result_path.exists(),
            "reached": turn["status"] == "completed"
            and server.user_input_request_count() == 1
            and result_path.exists(),
        }

    if scenario == "collaboration_plan":
        turn_id = await _start_turn(
            server,
            thread_id,
            "Use a shell tool to read README.md. Do not modify any files. Then give a concise implementation plan for "
            "the arithmetic fixture and finish.",
        )
        turn = await _wait_turn(server, thread_id, turn_id)
        return {
            "turn_status": turn["status"],
            "command_execution_items": server.item_type_count(thread_id, "commandExecution"),
            "reached": turn["status"] == "completed"
            and server.item_type_count(thread_id, "commandExecution") >= 1,
        }

    if scenario == "approval_auto_review":
        turn_id = await _start_turn(
            server,
            thread_id,
            "Use a shell tool to create approval_review.txt containing APPROVED, read it back, and finish.",
        )
        turn = await _wait_turn(server, thread_id, turn_id)
        file_exists = (workspace / "approval_review.txt").exists()
        review_started = server.notification_count("item/autoApprovalReview/started")
        review_completed = server.notification_count("item/autoApprovalReview/completed")
        return {
            "turn_status": turn["status"],
            "auto_review_started": review_started,
            "auto_review_completed": review_completed,
            "approval_requests": server.approval_request_count(),
            "result_file_exists": file_exists,
            "reached": turn["status"] == "completed"
            and review_started >= 1
            and review_completed >= 1
            and file_exists,
        }

    if scenario == "structured_output":
        turn_id = await _start_turn(
            server,
            thread_id,
            "Return the required structured result with status set to OK.",
            {
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
                "additionalProperties": False,
            },
        )
        turn = await _wait_turn(server, thread_id, turn_id)
        return {
            "turn_status": turn["status"],
            # This is a response-format probe. Ordinary coding tool-loop
            # coverage remains B0/C1; requiring a tool here would conflate
            # JSON-schema support with MiniMax's discretionary tool choice.
            "reached": turn["status"] == "completed",
        }

    if scenario == "expected_error":
        turn_id = await _start_turn(server, thread_id, "Reply with exactly OK.")
        turn = await _wait_turn(server, thread_id, turn_id)
        error_summary = server.error_summary()
        if turn["status"] == "completed":
            disposition = "recovered_after_error" if error_summary["count"] else "transparent_retry"
        else:
            disposition = "terminal_failure"
        return {
            "turn_status": turn["status"],
            "app_server_error": error_summary,
            "disposition": disposition,
            # The proxy's wire record proves injection. Native Codex can either
            # surface it here or retry it below the app-server protocol.
            "reached": turn["status"] in {"completed", "failed"},
        }

    if scenario == "tool_failure":
        turn_id = await _start_turn(
            server,
            thread_id,
            "Use a shell tool to run false. After that command fails, use a shell tool to create "
            "tool_failure_recovered.txt containing RECOVERED, read it back, and finish.",
        )
        turn = await _wait_turn(server, thread_id, turn_id)
        exit_codes = server.command_exit_codes(thread_id)
        recovered = (workspace / "tool_failure_recovered.txt").exists()
        return {
            "turn_status": turn["status"],
            "command_exit_codes": exit_codes,
            "recovered_file_exists": recovered,
            "reached": turn["status"] == "completed" and 1 in exit_codes and recovered,
        }

    if scenario == "goal_lifecycle":
        turn_id = await _start_turn(
            server,
            thread_id,
            "Use create_goal exactly once to create a goal to verify this arithmetic fixture. Then use get_goal exactly "
            "once. Use a shell tool to create goal_verified.txt containing GOAL_VERIFIED, read it back, then use "
            "update_goal exactly once to mark the goal complete and finish.",
        )
        turn = await _wait_turn(server, thread_id, turn_id)
        create_count = journal_tool_count("create_goal")
        get_count = journal_tool_count("get_goal")
        update_count = journal_tool_count("update_goal")
        file_exists = (workspace / "goal_verified.txt").exists()
        return {
            "turn_status": turn["status"],
            "create_goal_calls": create_count,
            "get_goal_calls": get_count,
            "update_goal_calls": update_count,
            "result_file_exists": file_exists,
            "reached": turn["status"] == "completed"
            and create_count == 1
            and get_count >= 1
            and update_count == 1
            and file_exists,
        }

    if scenario == "error_recovery":
        failed_turn = await _start_turn(server, thread_id, baseline_prompt)
        first = await _wait_turn(server, thread_id, failed_turn)
        follow_up = await _start_turn(
            server, thread_id, "Create error_recovery.txt containing RECOVERED, read it with a tool, then finish."
        )
        second = await _wait_turn(server, thread_id, follow_up)
        recovery_file_exists = (workspace / "error_recovery.txt").exists()
        return {
            "first_turn_status": first["status"],
            "follow_up_status": second["status"],
            "recovery_file_exists": recovery_file_exists,
            "reached": first["status"] == "failed"
            and second["status"] == "completed"
            and recovery_file_exists,
        }

    if scenario == "subagent":
        turn_id = await _start_turn(
            server,
            thread_id,
            "Use spawn_agent exactly once to ask a child agent to inspect adder.py and identify its behavior. "
            "Wait for that child with wait_agent, then report the child result and do not inspect adder.py yourself.",
        )
        turn = await _wait_turn(server, thread_id, turn_id)
        spawn_count = journal_tool_count("spawn_agent")
        wait_count = journal_tool_count("wait_agent")
        agent_error_count = journal_agent_error_count()
        return {
            "turn_status": turn["status"],
            "spawn_agent_calls": spawn_count,
            "wait_agent_calls": wait_count,
            "agent_error_count": agent_error_count,
            # The journal aggregates the primary thread and its descendants.
            # A child can legitimately delegate further, so prove the requested
            # primary delegation happened without treating successful nested work
            # as a failure of this scenario.
            "reached": turn["status"] == "completed"
            and spawn_count >= 1
            and wait_count >= 1
            and agent_error_count == 0,
        }

    if scenario == "nested_subagent":
        turn_id = await _start_turn(
            server,
            thread_id,
            "Use spawn_agent exactly once. Tell the child to use spawn_agent exactly once, wait for its child, "
            "and return the grandchild result. Wait for the first child before you answer.",
        )
        turn = await _wait_turn(server, thread_id, turn_id)
        spawn_count = journal_tool_count("spawn_agent")
        agent_error_count = journal_agent_error_count()
        return {
            "turn_status": turn["status"],
            "spawn_agent_calls": spawn_count,
            "agent_error_count": agent_error_count,
            "reached": turn["status"] == "completed" and spawn_count >= 2 and agent_error_count == 0,
        }

    if scenario == "parallel_subagents":
        turn_id = await _start_turn(
            server,
            thread_id,
            "Start two child agents with different task_name values: one inspects README.md and the other inspects adder.py. "
            "Start both before waiting for either, then wait for both and report both results.",
        )
        turn = await _wait_turn(server, thread_id, turn_id)
        spawn_count = journal_tool_count("spawn_agent")
        agent_error_count = journal_agent_error_count()
        return {
            "turn_status": turn["status"],
            "spawn_agent_calls": spawn_count,
            "agent_error_count": agent_error_count,
            "reached": turn["status"] == "completed" and spawn_count >= 2 and agent_error_count == 0,
        }

    if scenario == "compact":
        first_turn = await _start_turn(server, thread_id, baseline_prompt)
        await _wait_turn(server, thread_id, first_turn)
        compact_result = await server.request("thread/compact/start", {"threadId": thread_id})
        follow_up = await _start_turn(server, thread_id, "After compaction, read answer.txt with a tool and state its contents.")
        turn = await _wait_turn(server, thread_id, follow_up)
        return {
            "turn_status": turn["status"],
            "compact_result": _fingerprint(compact_result),
            "reached": turn["status"] == "completed",
        }

    if scenario == "rollback":
        first_turn = await _start_turn(server, thread_id, baseline_prompt)
        first = await _wait_turn(server, thread_id, first_turn)
        rollback = await server.request("thread/rollback", {"threadId": thread_id, "numTurns": 1})
        follow_up = await _start_turn(
            server, thread_id, "Create rollback_follow_up.txt containing ROLLED_BACK, read it with a shell tool, and finish."
        )
        second = await _wait_turn(server, thread_id, follow_up)
        follow_up_file_exists = (workspace / "rollback_follow_up.txt").exists()
        return {
            "first_turn_status": first["status"],
            "rollback_result": _fingerprint(rollback),
            "follow_up_status": second["status"],
            "follow_up_file_exists": follow_up_file_exists,
            "reached": first["status"] == "completed"
            and second["status"] == "completed"
            and follow_up_file_exists,
        }

    if scenario == "thread_fork":
        root_turn = await _start_turn(server, thread_id, baseline_prompt)
        root_first = await _wait_turn(server, thread_id, root_turn)
        fork = await server.request("thread/fork", {"threadId": thread_id, "lastTurnId": root_turn})
        fork_id = fork["thread"]["id"]
        fork_turn = await _start_turn(
            server, fork_id, "Use a shell tool to read answer.txt, create fork.txt containing FORKED, then read it back."
        )
        fork_completed = await _wait_turn(server, fork_id, fork_turn)
        # Forked threads retain an isolated workspace view. Capture the branch
        # result before resuming the root, which may restore its own view.
        fork_file_exists_before_root = (workspace / "fork.txt").exists()
        root_follow_up = await _start_turn(
            server,
            thread_id,
            "Use a shell tool to read answer.txt, create root_after_fork.txt containing ROOT, then read it back.",
        )
        root_completed = await _wait_turn(server, thread_id, root_follow_up)
        return {
            "root_first_status": root_first["status"],
            "fork_status": fork_completed["status"],
            "root_follow_up_status": root_completed["status"],
            "approval_requests": server.approval_request_count(),
            "fork_file_exists_before_root": fork_file_exists_before_root,
            "root_follow_up_file_exists": (workspace / "root_after_fork.txt").exists(),
            "reached": root_first["status"] == "completed"
            and fork_completed["status"] == "completed"
            and root_completed["status"] == "completed"
            and fork_file_exists_before_root
            and (workspace / "root_after_fork.txt").exists(),
        }

    if scenario == "detached_review":
        review = await server.request(
            "review/start",
            {
                "threadId": thread_id,
                "delivery": "detached",
                "target": {"type": "custom", "instructions": "Review adder.py for correctness and report concise findings."},
            },
        )
        review_thread_id = review["reviewThreadId"]
        turn = await _wait_turn(server, review_thread_id, review["turn"]["id"])
        entered = server.item_type_count(review_thread_id, "enteredReviewMode")
        exited = server.item_type_count(review_thread_id, "exitedReviewMode")
        return {
            "review_turn_status": turn["status"],
            "detached_thread": review_thread_id != thread_id,
            "entered_review_mode": entered,
            "exited_review_mode": exited,
            "reached": turn["status"] == "completed" and review_thread_id != thread_id and entered >= 1 and exited >= 1,
        }

    if scenario == "subagent_compact":
        first_turn = await _start_turn(
            server,
            thread_id,
            "Use spawn_agent exactly once to ask a child agent to inspect adder.py and identify its behavior. "
            "Wait for that child with wait_agent, then report the child result and do not inspect adder.py yourself.",
        )
        first = await _wait_turn(server, thread_id, first_turn)
        spawn_count = journal_tool_count("spawn_agent")
        agent_error_count = journal_agent_error_count()
        compact_result = await server.request("thread/compact/start", {"threadId": thread_id})
        follow_up = await _start_turn(
            server,
            thread_id,
            "After compaction, create post_agent_compaction.txt containing OK, read it with a tool, then finish.",
        )
        second = await _wait_turn(server, thread_id, follow_up)
        follow_up_file_exists = (workspace / "post_agent_compaction.txt").exists()
        return {
            "first_turn_status": first["status"],
            "spawn_agent_calls": spawn_count,
            "agent_error_count": agent_error_count,
            "compact_result": _fingerprint(compact_result),
            "follow_up_status": second["status"],
            "follow_up_file_exists": follow_up_file_exists,
            "reached": first["status"] == "completed"
            and spawn_count >= 1
            and agent_error_count == 0
            and second["status"] == "completed"
            and follow_up_file_exists,
        }

    if scenario == "invalid_lifecycle":
        completed_turn = await _start_turn(server, thread_id, "Reply with exactly OK.")
        first = await _wait_turn(server, thread_id, completed_turn)
        rejected = False
        try:
            await server.request("turn/interrupt", {"threadId": thread_id, "turnId": completed_turn})
        except RuntimeError:
            rejected = True
        follow_up = await _start_turn(
            server, thread_id, "Create stale_lifecycle.txt containing OK, read it with a tool, then finish."
        )
        second = await _wait_turn(server, thread_id, follow_up)
        follow_up_file_exists = (workspace / "stale_lifecycle.txt").exists()
        return {
            "first_turn_status": first["status"],
            "stale_interrupt_rejected": rejected,
            "follow_up_status": second["status"],
            "follow_up_file_exists": follow_up_file_exists,
            "reached": first["status"] == "completed"
            and rejected
            and second["status"] == "completed"
            and follow_up_file_exists,
        }

    if scenario in {"steer", "steer_after_tool", "interrupt"}:
        active_turn = await _start_turn(
            server,
            thread_id,
            "Use shell tools to inspect every file in the repository one by one, explaining your findings slowly before "
            "you finish. Do not skip tool use.",
        )
        # `turn/start` returns before the native lifecycle notification. Waiting
        # for it makes this an actual active-turn control test, rather than a
        # race between the two JSON-RPC writes.
        await server.notification(
            "turn/started",
            lambda message: message.get("params", {}).get("threadId") == thread_id,
        )
        initial_tool_reached = False
        if scenario == "steer_after_tool":
            await _wait_for_journal_tool(server, thread_id, journal_tool_count, "exec_command", 1)
            initial_tool_reached = True
        if scenario in {"steer", "steer_after_tool"}:
            lifecycle_result = await server.request(
                "turn/steer",
                {
                    "threadId": thread_id,
                    "expectedTurnId": active_turn,
                    "input": _text_input("Stop the current investigation. Instead create steering.txt containing STEERED, read it, and finish."),
                },
            )
        else:
            lifecycle_result = await server.request("turn/interrupt", {"threadId": thread_id, "turnId": active_turn})
        first = await _wait_turn(server, thread_id, active_turn)
        follow_up = await _start_turn(
            server, thread_id, "Create post_lifecycle.txt containing OK, read it with a tool, then finish."
        )
        second = await _wait_turn(server, thread_id, follow_up)
        steering_file_exists = (workspace / "steering.txt").exists()
        follow_up_file_exists = (workspace / "post_lifecycle.txt").exists()
        return {
            "first_turn_status": first["status"],
            "lifecycle_result": _fingerprint(lifecycle_result),
            "initial_tool_reached": initial_tool_reached,
            "steering_file_exists": steering_file_exists,
            "follow_up_status": second["status"],
            "follow_up_file_exists": follow_up_file_exists,
            "reached": second["status"] == "completed"
            and follow_up_file_exists
            and (scenario not in {"steer", "steer_after_tool"} or steering_file_exists)
            and (scenario != "steer_after_tool" or initial_tool_reached)
            and (scenario != "interrupt" or first["status"] == "interrupted"),
        }

    raise ValueError(f"unknown scenario: {scenario}")


async def run(args: argparse.Namespace) -> int:
    artifact_dir = args.artifacts.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if (artifact_dir / "result.json").exists():
        raise FileExistsError(f"run directory already contains result.json: {artifact_dir}")
    workspace = artifact_dir / "workspace"
    codex_home = artifact_dir / "codex_home"
    _prepare_workspace(workspace)
    mcp_fixture = Path(__file__).with_name("fixture_mcp_server.py") if args.scenario in {"mcp_tool", "mcp_tool_failure", "mcp_elicitation", "mcp_progress"} else None
    _write_config(
        codex_home,
        args.proxy_url,
        args.model,
        mcp_fixture=mcp_fixture,
        mcp_failure=args.scenario == "mcp_tool_failure",
        mcp_elicitation=args.scenario == "mcp_elicitation",
        mcp_progress=args.scenario == "mcp_progress",
        mcp_trace=artifact_dir / "mcp_transport.json" if args.scenario in {"mcp_elicitation", "mcp_progress"} else None,
        request_user_input=args.scenario == "request_user_input",
    )
    executable = shutil.which(args.codex) if os.sep not in args.codex else args.codex
    if not executable:
        raise FileNotFoundError(f"Codex executable not found: {args.codex}")
    environment = {
        **os.environ,
        "CODEX_HOME": str(codex_home),
        "HOME": str(codex_home),
        "LOCAL_API_KEY": "compat-lab-placeholder",
    }
    scenario = {
        "harness": "codex",
        "scenario": args.scenario,
        "model": args.model,
        "client_version": _client_version(executable),
        "proxy_url": args.proxy_url,
        "turn_timeout_s": args.turn_timeout_s,
        "run_started_unix_ms": _now_ms(),
    }
    (artifact_dir / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
    server = AppServer(executable, environment, artifact_dir, args.turn_timeout_s)
    resumed_server: AppServer | None = None
    result: dict[str, Any]
    try:
        await server.start()
        initialize_capabilities: dict[str, bool] = {}
        if args.scenario in {
            "collaboration_plan",
            "dynamic_tool",
            "dynamic_namespace_tool",
            "dynamic_tool_failure",
            "approval_auto_review",
            "mcp_tool",
            "mcp_tool_failure",
            "mcp_elicitation",
            "mcp_progress",
            "request_user_input",
        }:
            initialize_capabilities["experimentalApi"] = True
        if args.scenario == "mcp_elicitation":
            initialize_capabilities["mcpServerOpenaiFormElicitation"] = True
        await server.request(
            "initialize",
            {
                "clientInfo": {"name": "dynamo-harness-compat-lab", "version": "0"},
                "capabilities": initialize_capabilities,
            },
        )
        thread_start_params: dict[str, Any] = {
            "cwd": str(workspace),
            "model": args.model,
            "modelProvider": "local",
            "approvalPolicy": "never",
            "sandbox": "danger-full-access",
        }
        if args.scenario in {"dynamic_tool", "dynamic_namespace_tool", "dynamic_tool_failure"}:
            if args.scenario in {"dynamic_tool", "dynamic_tool_failure"}:
                thread_start_params["dynamicTools"] = [
                    {
                        "type": "function",
                        "name": "fixture_answer",
                        "description": "Return the fixed arithmetic fixture answer.",
                        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
                    }
                ]
            else:
                thread_start_params["dynamicTools"] = [
                    {
                        "type": "namespace",
                        "name": "fixture",
                        "description": "Arithmetic fixture utilities.",
                        "tools": [
                            {
                                "type": "function",
                                "name": "answer",
                                "description": "Return the fixed arithmetic fixture answer.",
                                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
                            }
                        ],
                    }
                ]
        if args.scenario == "approval_auto_review":
            thread_start_params.update(
                {
                    "approvalPolicy": "on-request",
                    "approvalsReviewer": "auto_review",
                    "sandbox": "read-only",
                }
            )
        thread = await server.request(
            "thread/start",
            thread_start_params,
        )
        scenario["client_session_sha256_12"] = _id_digest(thread["thread"]["id"])
        (artifact_dir / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
        thread_id = thread["thread"]["id"]
        if args.scenario == "thread_resume":
            first = await _run_scenario(
                server,
                thread_id,
                "baseline",
                workspace,
                lambda tool_name: _journal_tool_call_count(codex_home, tool_name),
                lambda: _journal_agent_error_count(codex_home),
            )
            await server.close()
            resumed_server = AppServer(executable, environment, artifact_dir, args.turn_timeout_s)
            await resumed_server.start()
            await resumed_server.request(
                "initialize",
                {"clientInfo": {"name": "dynamo-harness-compat-lab", "version": "0"}, "capabilities": {}},
            )
            resumed = await resumed_server.request("thread/resume", {"threadId": thread_id})
            second_turn = await _start_turn(
                resumed_server,
                thread_id,
                "Use a shell tool to read answer.txt, create resumed_thread.txt containing RESUMED, then read it back.",
            )
            second = await _wait_turn(resumed_server, thread_id, second_turn)
            resumed_file_exists = (workspace / "resumed_thread.txt").exists()
            result = {
                "first_turn_status": first.get("turn_status"),
                "resumed_thread_matches": resumed["thread"]["id"] == thread_id,
                "second_turn_status": second["status"],
                "resumed_file_exists": resumed_file_exists,
                "reached": first.get("reached")
                and resumed["thread"]["id"] == thread_id
                and second["status"] == "completed"
                and resumed_file_exists,
            }
        else:
            if args.scenario == "dynamic_tool_failure":
                server.set_dynamic_tool_success(False)
            collaboration_update: dict[str, Any] | None = None
            if args.scenario == "collaboration_plan":
                collaboration_update = await server.request(
                    "thread/settings/update",
                    {
                        "threadId": thread_id,
                        "collaborationMode": {
                            "mode": "plan",
                            "settings": {"model": args.model, "developer_instructions": None},
                        },
                    },
                )
            result = await _run_scenario(
                server,
                thread_id,
                args.scenario,
                workspace,
                lambda tool_name: _journal_tool_call_count(codex_home, tool_name),
                lambda: _journal_agent_error_count(codex_home),
            )
            if collaboration_update is not None:
                result["collaboration_update"] = _fingerprint(collaboration_update)
        result["outcome"] = "pass" if result.pop("reached") else "not_reached"
    except (TimeoutError, asyncio.TimeoutError) as error:
        result = {
            "outcome": "inconclusive",
            "first_divergent_boundary": "native_turn_timeout",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    except Exception as error:
        result = {"outcome": "harness_failure", "error_type": type(error).__name__, "error": str(error)}
    finally:
        if resumed_server is not None:
            await resumed_server.close()
        await server.close()
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
    parser.add_argument("--codex", default="codex")
    parser.add_argument(
        "--turn-timeout-s",
        type=float,
        default=600,
        help="Bound one native Codex turn so model-driven agent expansion cannot run indefinitely.",
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "baseline",
            "collaboration_plan",
            "dynamic_tool",
            "dynamic_namespace_tool",
            "dynamic_tool_failure",
            "approval_auto_review",
            "mcp_tool",
            "mcp_tool_failure",
            "mcp_progress",
            "mcp_elicitation",
            "request_user_input",
            "structured_output",
            "expected_error",
            "tool_failure",
            "goal_lifecycle",
            "error_recovery",
            "subagent",
            "nested_subagent",
            "parallel_subagents",
            "compact",
            "rollback",
            "thread_fork",
            "thread_resume",
            "detached_review",
            "subagent_compact",
            "invalid_lifecycle",
            "steer",
            "steer_after_tool",
            "interrupt",
        ),
        required=True,
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
