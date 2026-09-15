#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate paired custom-encoder demo runs over a shared filesystem."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_SIDES = ("control", "dynamo-vllm")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_REMOTE_HEARTBEAT_TTL_SECONDS = 15.0


def _validate_name(value: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "must contain only letters, digits, dot, underscore, or hyphen"
        )
    return value


def _state_paths(state_dir: Path, session_id: str) -> tuple[Path, Path]:
    session_dir = state_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir / "state.json", session_dir / "state.lock"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"next_round": 1, "rounds": []}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"invalid coordinator state at {path}")
    state.setdefault("next_round", 1)
    state.setdefault("rounds", [])
    return state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _with_locked_state(state_path: Path, lock_path: Path, update: Any) -> Any:
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = _load_state(state_path)
        result = update(state)
        _write_state(state_path, state)
        return result


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _prune_abandoned_rounds(state: dict[str, Any]) -> None:
    current_hostname = socket.gethostname()
    now = time.time()
    retained: list[dict[str, Any]] = []
    for round_state in state["rounds"]:
        if round_state.get("start_time") is not None:
            retained.append(round_state)
            continue
        participants = round_state.get("participants", {})
        live: dict[str, Any] = {}
        for side, participant in participants.items():
            participant_hostname = participant.get("hostname")
            if participant_hostname == current_hostname:
                is_live = _pid_alive(int(participant.get("pid", -1)))
            elif participant_hostname:
                last_seen = float(
                    participant.get(
                        "heartbeat_at", participant.get("registered_at", 0.0)
                    )
                )
                is_live = now - last_seen <= _REMOTE_HEARTBEAT_TTL_SECONDS
            else:
                # State written by an older coordinator cannot safely use a
                # PID for cross-node liveness. Let it age out quickly.
                registered_at = float(participant.get("registered_at", 0.0))
                is_live = now - registered_at <= _REMOTE_HEARTBEAT_TTL_SECONDS
            if is_live:
                live[side] = participant
        if live:
            round_state["participants"] = live
            retained.append(round_state)
    state["rounds"] = retained


def _register_waiter(
    state: dict[str, Any], side: str, lead_seconds: float
) -> tuple[int, str]:
    _prune_abandoned_rounds(state)
    selected: dict[str, Any] | None = None
    for candidate in reversed(state["rounds"]):
        if candidate.get("mode", "parallel") != "parallel":
            continue
        if candidate.get("start_time") is not None:
            continue
        participants = candidate.setdefault("participants", {})
        if side not in participants:
            selected = candidate
            break

    if selected is None:
        round_id = int(state["next_round"])
        state["next_round"] = round_id + 1
        selected = {
            "id": round_id,
            "mode": "parallel",
            "created_at": time.time(),
            "participants": {},
            "results": {},
            "start_time": None,
        }
        state["rounds"].append(selected)

    token = f"{side}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    now = time.time()
    selected["participants"][side] = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "token": token,
        "registered_at": now,
        "heartbeat_at": now,
    }
    if all(name in selected["participants"] for name in _SIDES):
        selected["start_time"] = time.time() + lead_seconds
    return int(selected["id"]), token


def _register_serial_waiter(state: dict[str, Any], side: str) -> tuple[int, str]:
    _prune_abandoned_rounds(state)
    selected: dict[str, Any] | None = None
    for candidate in reversed(state["rounds"]):
        if candidate.get("mode") != "serial":
            continue
        results = candidate.get("results", {})
        if all(name in results for name in _SIDES):
            continue
        participants = candidate.setdefault("participants", {})
        if side not in participants:
            selected = candidate
            break

    if selected is None:
        round_id = int(state["next_round"])
        state["next_round"] = round_id + 1
        selected = {
            "id": round_id,
            "mode": "serial",
            "created_at": time.time(),
            "participants": {},
            "results": {},
            "start_time": None,
        }
        state["rounds"].append(selected)

    token = f"{side}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    now = time.time()
    selected["participants"][side] = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "token": token,
        "registered_at": now,
        "heartbeat_at": now,
    }
    if side == "control" and selected.get("start_time") is None:
        # A started serial round is retained while the control benchmark runs,
        # even though this short-lived coordinator process has returned.
        selected["start_time"] = now
    return int(selected["id"]), token


def _find_round(state: dict[str, Any], round_id: int) -> dict[str, Any]:
    for round_state in state["rounds"]:
        if int(round_state["id"]) == round_id:
            return round_state
    raise ValueError(f"coordinator round {round_id} not found")


def _wait_start(args: argparse.Namespace) -> int:
    state_path, lock_path = _state_paths(args.state_dir, args.session_id)
    round_id, token = _with_locked_state(
        state_path,
        lock_path,
        lambda state: _register_waiter(state, args.side, args.lead_seconds),
    )
    print(
        f"{args.side}: warmup complete; waiting for the other terminal...",
        file=sys.stderr,
        flush=True,
    )
    deadline = time.monotonic() + args.timeout
    start_time: float | None = None
    while time.monotonic() < deadline:

        def inspect(state: dict[str, Any]) -> float | None:
            round_state = _find_round(state, round_id)
            participant = round_state.get("participants", {}).get(args.side)
            if participant is None or participant.get("token") != token:
                raise RuntimeError("coordinator registration was replaced")
            participant["heartbeat_at"] = time.time()
            value = round_state.get("start_time")
            return None if value is None else float(value)

        start_time = _with_locked_state(state_path, lock_path, inspect)
        if start_time is not None:
            break
        time.sleep(0.2)
    if start_time is None:
        raise TimeoutError("timed out waiting for the other demo terminal")

    print(
        f"Both sides ready. Starting paired round {round_id}...",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(max(0.0, start_time - time.time()))
    print(json.dumps({"round_id": round_id, "start_time": start_time}))
    return 0


def _wait_turn(args: argparse.Namespace) -> int:
    state_path, lock_path = _state_paths(args.state_dir, args.session_id)
    round_id, token = _with_locked_state(
        state_path,
        lock_path,
        lambda state: _register_serial_waiter(state, args.side),
    )
    if args.side == "control":
        print("Sequential control has the H100 first.", file=sys.stderr, flush=True)
        print(json.dumps({"round_id": round_id}))
        return 0

    print(
        "Optimized side ready; waiting for sequential control to finish...",
        file=sys.stderr,
        flush=True,
    )
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:

        def inspect(state: dict[str, Any]) -> bool:
            round_state = _find_round(state, round_id)
            participant = round_state.get("participants", {}).get(args.side)
            if participant is None or participant.get("token") != token:
                raise RuntimeError("coordinator registration was replaced")
            participant["heartbeat_at"] = time.time()
            return "control" in round_state.get("results", {})

        if _with_locked_state(state_path, lock_path, inspect):
            print(
                f"Control complete. Starting optimized round {round_id}...",
                file=sys.stderr,
                flush=True,
            )
            print(json.dumps({"round_id": round_id}))
            return 0
        time.sleep(0.2)
    raise TimeoutError("timed out waiting for sequential control")


def _result_summary(result_path: Path, wall_seconds: float) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    request_count = int(result["request_count"]["avg"])
    if request_count < 1 or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("invalid request count or full-process wall time")
    return {
        "artifact": str(result_path.parent),
        "request_count": request_count,
        "errors": len(result.get("error_summary", [])),
        "request_throughput": float(result["request_throughput"]["avg"]),
        "output_throughput": float(result["output_token_throughput"]["avg"]),
        "average_e2e_ms": float(result["request_latency"]["avg"]),
        "p99_e2e_ms": float(result["request_latency"]["p99"]),
        "wall_seconds": wall_seconds,
        "full_process_throughput": request_count / wall_seconds,
    }


def _combined(round_id: int, results: dict[str, Any]) -> dict[str, Any]:
    control = results["control"]
    test = results["dynamo-vllm"]
    return {
        "round_id": round_id,
        "control": control,
        "dynamo_vllm": test,
        "comparison": {
            "request_throughput_gain_pct": 100.0
            * (test["request_throughput"] / control["request_throughput"] - 1.0),
            "full_process_throughput_gain_pct": 100.0
            * (
                test["full_process_throughput"] / control["full_process_throughput"]
                - 1.0
            ),
            "average_e2e_reduction_pct": 100.0
            * (1.0 - test["average_e2e_ms"] / control["average_e2e_ms"]),
            "p99_e2e_reduction_pct": 100.0
            * (1.0 - test["p99_e2e_ms"] / control["p99_e2e_ms"]),
        },
    }


def _submit_result(args: argparse.Namespace) -> int:
    state_path, lock_path = _state_paths(args.state_dir, args.session_id)
    summary = _result_summary(args.result_path, args.wall_seconds)

    def submit(state: dict[str, Any]) -> None:
        round_state = _find_round(state, args.round_id)
        if args.side not in round_state.get("participants", {}):
            raise ValueError(
                f"{args.side} did not participate in round {args.round_id}"
            )
        round_state.setdefault("results", {})[args.side] = summary

    _with_locked_state(state_path, lock_path, submit)
    print(
        f"{args.side}: waiting for the paired result...",
        file=sys.stderr,
        flush=True,
    )
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:

        def inspect(state: dict[str, Any]) -> dict[str, Any] | None:
            results = _find_round(state, args.round_id).get("results", {})
            if all(side in results for side in _SIDES):
                return _combined(args.round_id, results)
            return None

        combined = _with_locked_state(state_path, lock_path, inspect)
        if combined is not None:
            print(json.dumps(combined, sort_keys=True))
            return 0
        time.sleep(0.2)
    raise TimeoutError("timed out waiting for the paired benchmark result")


def _abort_session(args: argparse.Namespace) -> int:
    """Wake all waiters by removing the active rounds for one session."""
    state_path, lock_path = _state_paths(args.state_dir, args.session_id)

    def abort(state: dict[str, Any]) -> list[int]:
        round_ids = [int(round_state["id"]) for round_state in state["rounds"]]
        state["rounds"] = []
        return round_ids

    aborted = _with_locked_state(state_path, lock_path, abort)
    print(json.dumps({"aborted_rounds": aborted}, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state-dir", type=Path, required=True)
    common.add_argument("--session-id", type=_validate_name, required=True)
    common.add_argument("--side", choices=_SIDES, required=True)
    common.add_argument("--timeout", type=float, default=600.0)

    wait_parser = subparsers.add_parser("wait-start", parents=[common])
    wait_parser.add_argument("--lead-seconds", type=float, default=5.0)
    wait_parser.set_defaults(handler=_wait_start)

    turn_parser = subparsers.add_parser("wait-turn", parents=[common])
    turn_parser.set_defaults(handler=_wait_turn)

    result_parser = subparsers.add_parser("submit-result", parents=[common])
    result_parser.add_argument("--round-id", type=int, required=True)
    result_parser.add_argument("--result-path", type=Path, required=True)
    result_parser.add_argument("--wall-seconds", type=float, required=True)
    result_parser.set_defaults(handler=_submit_result)

    abort_parser = subparsers.add_parser("abort-session")
    abort_parser.add_argument("--state-dir", type=Path, required=True)
    abort_parser.add_argument("--session-id", type=_validate_name, required=True)
    abort_parser.set_defaults(handler=_abort_session)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(f"demo coordinator error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
