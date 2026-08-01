#!/usr/bin/env python3
"""Drive one real Claude Code interactive cancellation/steering scenario.

The non-interactive JSONL API accepts user guidance but has no documented
cancel control. This driver uses a disposable pseudo-terminal so Escape is
handled by the installed Claude Code client itself. Terminal output is never
persisted; artifacts contain only action timings and protocol shape from the
separate capture proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import pexpect


def _now_ms() -> int:
    return round(time.time() * 1000)


def _id_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def _client_version(executable: str) -> str:
    completed = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=15, check=False)
    return ((completed.stdout or completed.stderr).strip()) or f"exit={completed.returncode}"


def _prepare_workspace(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text("# Interactive cancellation fixture\n", encoding="utf-8")
    (path / "adder.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left + right\n", encoding="utf-8"
    )


def _write_record(path: Path, action: str, **fields: Any) -> None:
    record = {"timestamp_unix_ms": _now_ms(), "action": action, **fields}
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def _send_enter(child: pexpect.spawn, text: str = "") -> None:
    """Send terminal Enter as CR; the full-screen renderer does not treat LF as a submit key."""
    child.send(text + "\r")


def _request_count(wire: Path) -> int:
    if not wire.exists():
        return 0
    count = 0
    for line in wire.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        count += row.get("kind") == "request"
    return count


def _wait_for_request(wire: Path, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _request_count(wire) > 0:
            return True
        time.sleep(0.1)
    return False


def _wait_for_sse_event(wire: Path, event: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if wire.exists():
            for line in wire.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("kind") == "sse_event" and row.get("event") == event:
                    return True
        time.sleep(0.1)
    return False


def _wait_for_file(path: Path, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.2)
    return False


def _terminal_state(child: pexpect.spawn, timeout_s: float) -> dict[str, bool | int | None]:
    """Classify the terminal startup state without retaining rendered text."""
    deadline = time.monotonic() + timeout_s
    captured = ""
    while time.monotonic() < deadline:
        try:
            captured += child.read_nonblocking(size=4096, timeout=0.2)
        except pexpect.TIMEOUT:
            continue
        except pexpect.EOF:
            break
    normalized = captured.lower()
    return {
        "alive": child.isalive(),
        "exit_status": child.exitstatus,
        "signal_status": child.signalstatus,
        "output_seen": bool(captured),
        "mentions_claude": "claude" in normalized,
        "mentions_api_key": "api key" in normalized or "api_key" in normalized,
        "mentions_login": "login" in normalized or "sign in" in normalized,
        "mentions_trust": "trust" in normalized,
        "mentions_permission": "permission" in normalized,
        "mentions_bypass_permissions": "bypass permissions" in normalized,
        "mentions_prompt": (
            "what can i help" in normalized
            or "how can i help" in normalized
            or "press enter" in normalized
            or "continue" in normalized
        ),
        "mentions_security_notes": "security" in normalized,
        "mentions_workspace": "accessing workspace" in normalized,
        "mentions_workspace_trust": "yes, i trust this folder" in normalized,
        "mentions_input_cursor": "❯" in captured,
    }


def _wait_for_chat_input(child: pexpect.spawn, timeout_s: float) -> dict[str, bool | int | None]:
    deadline = time.monotonic() + timeout_s
    state = _terminal_state(child, 0.2)
    while time.monotonic() < deadline:
        if state["mentions_input_cursor"]:
            return state
        state = _terminal_state(child, 0.2)
    return state


def run(args: argparse.Namespace) -> int:
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    if (artifacts / "result.json").exists():
        raise FileExistsError(f"run directory already contains result.json: {artifacts}")
    workspace = artifacts / "workspace"
    claude_home = artifacts / "claude_home"
    _prepare_workspace(workspace)
    executable = shutil.which(args.claude) if os.sep not in args.claude else args.claude
    if not executable:
        raise FileNotFoundError(f"Claude executable not found: {args.claude}")
    session_id = str(uuid.uuid4())
    scenario = {
        "harness": "claude_code_interactive",
        "scenario": args.scenario,
        "model": args.model,
        "client_version": _client_version(executable),
        "client_session_sha256_12": _id_digest(session_id),
        "proxy_url": args.proxy_url,
        "run_started_unix_ms": _now_ms(),
    }
    (artifacts / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
    actions = artifacts / "harness.jsonl"
    dummy_api_key = "sk-ant-api03-compatibility-lab-placeholder"
    environment = {
        **os.environ,
        "HOME": str(claude_home),
        "ANTHROPIC_BASE_URL": args.proxy_url.rstrip("/"),
        # Interactive Claude validates the API-key shape before it opens its
        # terminal session. This is a deliberately non-secret syntactic dummy;
        # Dynamo ignores it and the capture proxy redacts authorization headers.
        "ANTHROPIC_AUTH_TOKEN": dummy_api_key,
        "ANTHROPIC_API_KEY": dummy_api_key,
        "ANTHROPIC_MODEL": args.model,
        "ANTHROPIC_SMALL_FAST_MODEL": args.model,
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096",
        "NO_COLOR": "1",
        "TERM": "xterm-256color",
    }
    command = [
        executable,
        "--bare",
        "--dangerously-skip-permissions",
        "--model",
        args.model,
        "--session-id",
        session_id,
        "--tools",
        "Bash,Read,Write,Edit,Agent",
    ]
    child: pexpect.spawn | None = None
    result: dict[str, Any]
    try:
        child = pexpect.spawn(
            command[0],
            command[1:],
            cwd=str(workspace),
            env=environment,
            encoding="utf-8",
            timeout=10,
            echo=False,
        )
        child.setwinsize(32, 120)
        _write_record(actions, "interactive_started", terminal_state=_terminal_state(child, 5))
        # If a startup acknowledgement is presented, an empty Enter can dismiss
        # it. If the normal prompt is already active this is an inert empty line.
        _send_enter(child)
        _write_record(actions, "startup_enter_sent")
        post_enter_state = _terminal_state(child, 2)
        _write_record(actions, "post_enter_terminal_state", terminal_state=post_enter_state)
        chat_input_state: dict[str, bool | int | None] | None = None
        if post_enter_state["mentions_api_key"]:
            # A fresh interactive HOME first presents a choice to use the
            # already-supplied API key. The second choice (No) is selected by
            # default, so move to Yes and confirm it.
            child.send("1\r")
            _write_record(actions, "api_key_onboarding_accepted")
            post_api_key_state = _terminal_state(child, 2)
            _write_record(actions, "post_api_key_terminal_state", terminal_state=post_api_key_state)
            for acknowledgement in range(2):
                if not post_api_key_state["mentions_security_notes"]:
                    break
                _send_enter(child)
                _write_record(actions, "security_notes_acknowledged", attempt=acknowledgement + 1)
                post_api_key_state = _terminal_state(child, 2)
                _write_record(actions, "post_security_terminal_state", terminal_state=post_api_key_state)
            if post_api_key_state["mentions_workspace"] or post_api_key_state["mentions_workspace_trust"]:
                # This workspace is created by the driver under its isolated
                # artifact directory, so accepting its selected Yes choice is
                # within the test's explicit scope.
                _send_enter(child)
                _write_record(actions, "workspace_trust_accepted")
                post_api_key_state = _terminal_state(child, 2)
                _write_record(actions, "post_trust_terminal_state", terminal_state=post_api_key_state)
            if post_api_key_state["mentions_bypass_permissions"] or post_api_key_state["mentions_permission"]:
                # The warning defaults to "No, exit". Select its explicit
                # accept option for this disposable, isolated lab only.
                child.send("2\r")
                _write_record(actions, "bypass_permissions_accepted")
                chat_input_state = _terminal_state(child, 2)
                _write_record(actions, "post_bypass_terminal_state", terminal_state=chat_input_state)
        if chat_input_state is None or not chat_input_state["mentions_input_cursor"]:
            chat_input_state = _wait_for_chat_input(child, 20)
        _write_record(actions, "chat_input_state", terminal_state=chat_input_state)
        _send_enter(
            child,
            "Use Bash to inspect README.md and adder.py slowly, one step at a time. Do not finish until every inspection "
            "has been verified."
        )
        _write_record(actions, "initial_user_message_sent")
        # The interactive renderer can defer its first network turn while it
        # finishes its startup animation and input focus transition.
        # This TUI can retain the first carriage return while it finishes its
        # input-focus handoff. A second empty Enter commits the already-typed
        # prompt; if the first Enter did commit it, the empty input is ignored
        # while the turn is active.
        initial_submit_window_s = min(args.request_timeout_s, 5)
        request_seen = _wait_for_request(artifacts / "wire.jsonl", initial_submit_window_s)
        if not request_seen:
            _send_enter(child)
            _write_record(actions, "initial_submit_retry_sent")
            request_seen = _wait_for_request(
                artifacts / "wire.jsonl", max(0, args.request_timeout_s - initial_submit_window_s)
            )
        _write_record(actions, "request_observed", observed=request_seen)
        if not request_seen:
            result = {"request_observed_before_escape": False, "steering_file_exists": False, "reached": False}
        else:
            message_started = _wait_for_sse_event(artifacts / "wire.jsonl", "message_start", args.request_timeout_s)
            _write_record(actions, "message_start_observed", observed=message_started)
            if not message_started:
                result = {
                    "request_observed_before_escape": True,
                    "message_started_before_escape": False,
                    "steering_file_exists": False,
                    "reached": False,
                }
            else:
                child.send("\x1b")
                _write_record(actions, "escape_sent")
                first_message_stopped = _wait_for_sse_event(artifacts / "wire.jsonl", "message_stop", args.request_timeout_s)
                _write_record(actions, "post_escape_message_stop_observed", observed=first_message_stopped)
                post_escape_state = _wait_for_chat_input(child, 20)
                _write_record(actions, "post_escape_chat_input_state", terminal_state=post_escape_state)
                if not first_message_stopped or not post_escape_state["mentions_input_cursor"]:
                    result = {
                        "request_observed_before_escape": True,
                        "message_started_before_escape": True,
                        "post_escape_message_stop": first_message_stopped,
                        "post_escape_input_ready": post_escape_state["mentions_input_cursor"],
                        "steering_file_exists": False,
                        "reached": False,
                    }
                else:
                    _send_enter(
                        child,
                        "Create interactive_steering.txt containing STEERED followed by a newline, read it with Bash, and then finish.",
                    )
                    _write_record(actions, "steering_user_message_sent")
                    steering_submit_window_s = min(args.request_timeout_s, 5)
                    steering_file_exists = _wait_for_file(workspace / "interactive_steering.txt", steering_submit_window_s)
                    if not steering_file_exists:
                        _send_enter(child)
                        _write_record(actions, "steering_submit_retry_sent")
                        steering_file_exists = _wait_for_file(
                            workspace / "interactive_steering.txt", max(0, args.request_timeout_s - steering_submit_window_s)
                        )
                    _write_record(actions, "steering_file_observed", exists=steering_file_exists)
                    result = {
                        "request_observed_before_escape": request_seen,
                        "message_started_before_escape": message_started,
                        "post_escape_message_stop": first_message_stopped,
                        "post_escape_input_ready": post_escape_state["mentions_input_cursor"],
                        "steering_file_exists": steering_file_exists,
                        "reached": request_seen
                        and message_started
                        and first_message_stopped
                        and post_escape_state["mentions_input_cursor"]
                        and steering_file_exists,
                    }
    except Exception as error:
        result = {"outcome": "harness_failure", "error_type": type(error).__name__, "error": str(error)}
    finally:
        if child is not None and child.isalive():
            _send_enter(child, "/exit")
            try:
                child.expect(pexpect.EOF, timeout=10)
            except (pexpect.TIMEOUT, pexpect.EOF):
                child.terminate(force=True)
        scenario["run_finished_unix_ms"] = _now_ms()
        (artifacts / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
    if "outcome" not in result:
        result["outcome"] = "pass" if result.pop("reached") else "not_reached"
    (artifacts / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["outcome"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--claude", default="claude")
    parser.add_argument("--scenario", choices=("cancel_steer",), required=True)
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=150,
        help="Bound each real terminal interaction while allowing Claude's startup/input handoff to settle.",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
