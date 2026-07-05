#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one evaluator command independently of a kubectl exec stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_ROOT = Path("/artifacts/glm52-nscale/.campaign-run.lock")
INVOCATION_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_if_absent(path: Path, payload: dict[str, Any]) -> bool:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def process_start_ticks(pid: int) -> int:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        value = proc_stat.read_text(encoding="utf-8")
        close = value.rfind(")")
        if close < 0:
            raise RuntimeError(f"invalid /proc stat for pid {pid}")
        fields = value[close + 2 :].split()
        return int(fields[19])
    started = subprocess.check_output(
        ["ps", "-o", "lstart=", "-p", str(pid)],
        text=True,
    ).strip()
    if not started:
        raise ProcessLookupError(pid)
    return int(hashlib.sha256(started.encode()).hexdigest()[:16], 16)


def process_matches(pid: int, start_ticks: int) -> bool:
    try:
        return process_start_ticks(pid) == start_ticks
    except (
        FileNotFoundError,
        ProcessLookupError,
        PermissionError,
        RuntimeError,
        subprocess.CalledProcessError,
    ):
        return False


def validate_invocation(value: str) -> str:
    if not INVOCATION_PATTERN.fullmatch(value):
        raise RuntimeError("invalid invocation id")
    return value


def validate_state_dir(value: str) -> Path:
    state_dir = Path(value)
    if not state_dir.is_absolute():
        raise RuntimeError("state directory must be absolute")
    root = Path(os.environ.get("GLM52_REMOTE_COMMAND_ROOT", DEFAULT_ROOT)).resolve()
    if state_dir.parent.resolve() != root or state_dir.name != "command":
        raise RuntimeError(f"state directory must be {root / 'command'}")
    return state_dir


def normalize_exit_code(return_code: int) -> int:
    if return_code < 0:
        return 128 + (-return_code)
    return min(return_code, 255)


def command_document(invocation_id: str, argv: list[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "invocation_id": invocation_id,
        "argv": argv,
    }


def acquire_lock(
    state_dir: Path,
    invocation_id: str,
    *,
    variant: str,
    campaign_phase: str,
    attestation: str,
    argv_sha256: str,
    deployment_sha256: str,
    acquired_at: str,
) -> dict[str, Any]:
    lock_dir = state_dir.parent
    owner = {
        "schema_version": 2,
        "variant": variant,
        "campaign_phase": campaign_phase,
        "attestation": attestation,
        "invocation_id": invocation_id,
        "argv_sha256": argv_sha256,
        "deployment_sha256": deployment_sha256,
        "acquired_at": acquired_at,
    }

    def validate_owner(path: Path) -> None:
        if load_json(path / "owner.json") != owner:
            raise RuntimeError("campaign lock is owned by another invocation")

    staging = lock_dir.with_name(f".{lock_dir.name}.{invocation_id}.preparing")
    while True:
        if lock_dir.exists():
            validate_owner(lock_dir)
            break
        try:
            staging.mkdir(mode=0o700)
            atomic_json(staging / "owner.json", owner)
        except FileExistsError:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if lock_dir.exists():
                    validate_owner(lock_dir)
                    break
                if (staging / "owner.json").is_file():
                    validate_owner(staging)
                    break
                time.sleep(0.01)
            else:
                shutil.rmtree(staging, ignore_errors=True)
                continue
        if lock_dir.exists():
            break
        try:
            os.rename(staging, lock_dir)
            break
        except OSError:
            if lock_dir.exists():
                validate_owner(lock_dir)
                shutil.rmtree(staging, ignore_errors=True)
                break
            if staging.exists():
                continue
            raise
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "acquired",
        "invocation_id": invocation_id,
    }


def release_lock(state_dir: Path, invocation_id: str) -> dict[str, Any]:
    lock_dir = state_dir.parent
    owner = load_json(lock_dir / "owner.json")
    if owner.get("invocation_id") != invocation_id:
        raise RuntimeError("refusing to release a lock owned by another invocation")
    shutil.rmtree(lock_dir)
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "released",
        "invocation_id": invocation_id,
    }


def verify_command(state_dir: Path, invocation_id: str, argv: list[str]) -> None:
    expected = command_document(invocation_id, argv)
    deadline = time.monotonic() + 5
    while not (state_dir / "command.json").is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError("existing detached command is incomplete")
        time.sleep(0.01)
    actual = load_json(state_dir / "command.json")
    if actual != expected:
        raise RuntimeError("existing detached command does not match this invocation")


def status_payload(state_dir: Path, invocation_id: str) -> dict[str, Any]:
    command = load_json(state_dir / "command.json")
    if command.get("invocation_id") != invocation_id:
        raise RuntimeError("detached command invocation id mismatch")

    status_path = state_dir / "status.json"
    if status_path.is_file():
        status = load_json(status_path)
        if status.get("invocation_id") != invocation_id:
            raise RuntimeError("detached command status invocation id mismatch")
        exit_code = status.get("exit_code")
        if not isinstance(exit_code, int) or not 0 <= exit_code <= 255:
            raise RuntimeError("detached command status has invalid exit code")
        result = {
            "schema_version": SCHEMA_VERSION,
            "state": "finished",
            "invocation_id": invocation_id,
            "exit_code": exit_code,
            "finished_at": status.get("finished_at"),
        }
        for key in (
            "cancelled",
            "driver_error",
            "forwarded_signal",
            "terminated_by_guard",
        ):
            if key in status:
                result[key] = status[key]
        return result

    launch_path = state_dir / "launch.json"
    if not launch_path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "state": "starting",
            "invocation_id": invocation_id,
        }
    launch = load_json(launch_path)
    if launch.get("invocation_id") != invocation_id:
        raise RuntimeError("detached command launch invocation id mismatch")
    pid = launch.get("driver_pid")
    start_ticks = launch.get("driver_start_ticks")
    if not isinstance(pid, int) or not isinstance(start_ticks, int):
        raise RuntimeError("detached command launch identity is invalid")
    if process_matches(pid, start_ticks):
        return {
            "schema_version": SCHEMA_VERSION,
            "state": "running",
            "invocation_id": invocation_id,
            "driver_pid": pid,
            "started_at": launch.get("started_at"),
        }

    # The supervisor publishes status immediately before exiting. A probe can
    # observe the process exit between its first status read and this identity
    # check, so allow the atomic status rename to become visible before
    # classifying the run as lost.
    for _ in range(20):
        if status_path.is_file():
            return status_payload(state_dir, invocation_id)
        time.sleep(0.01)

    started_path = state_dir / "started.json"
    child_running = False
    if started_path.is_file():
        started = load_json(started_path)
        child_pid = started.get("command_pid")
        child_ticks = started.get("command_start_ticks")
        child_running = (
            isinstance(child_pid, int)
            and isinstance(child_ticks, int)
            and process_matches(child_pid, child_ticks)
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "orphaned" if child_running else "lost",
        "invocation_id": invocation_id,
    }


def start_command(
    state_dir: Path, invocation_id: str, argv: list[str]
) -> dict[str, Any]:
    if not argv:
        raise RuntimeError("detached command argv must not be empty")
    staging = state_dir.with_name(f".{state_dir.name}.{invocation_id}.preparing")
    while not state_dir.exists():
        try:
            staging.mkdir(mode=0o700)
            atomic_json(
                staging / "command.json",
                command_document(invocation_id, argv),
            )
        except FileExistsError:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if state_dir.exists():
                    break
                if (staging / "command.json").is_file():
                    verify_command(staging, invocation_id, argv)
                    break
                time.sleep(0.01)
            else:
                shutil.rmtree(staging, ignore_errors=True)
                continue
        if state_dir.exists():
            break
        try:
            os.rename(staging, state_dir)
        except OSError:
            if not state_dir.exists():
                if staging.exists():
                    continue
                raise
            shutil.rmtree(staging, ignore_errors=True)
    verify_command(state_dir, invocation_id, argv)

    current = status_payload(state_dir, invocation_id)
    if current["state"] in {"running", "finished", "orphaned", "lost"}:
        return current

    subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_run",
            "--state-dir",
            str(state_dir),
            "--invocation-id",
            invocation_id,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        current = status_payload(state_dir, invocation_id)
        if current["state"] != "starting":
            return current
        time.sleep(0.05)
    raise RuntimeError("detached command supervisor did not start")


def process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_process_group(process_group: int, timeout: float = 5) -> bool:
    if not process_group_exists(process_group):
        return True
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_group_exists(process_group):
            return True
        time.sleep(0.05)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_group_exists(process_group):
            return True
        time.sleep(0.05)
    return False


def stop_child(child: subprocess.Popen[bytes], timeout: float = 5) -> bool:
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return False
    if process_group_exists(child.pid):
        return stop_process_group(child.pid, timeout)
    return True


def publish_cancelled_status(state_dir: Path, invocation_id: str) -> None:
    atomic_json_if_absent(
        state_dir / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "invocation_id": invocation_id,
            "exit_code": 143,
            "finished_at": utc_now(),
            "cancelled": True,
            "terminated_by_guard": True,
        },
    )


def run_command(state_dir: Path, invocation_id: str) -> int:
    command = load_json(state_dir / "command.json")
    if command.get("invocation_id") != invocation_id:
        return 125
    argv = command.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(v, str) for v in argv)
    ):
        return 125

    launch = {
        "schema_version": SCHEMA_VERSION,
        "invocation_id": invocation_id,
        "driver_pid": os.getpid(),
        "driver_start_ticks": process_start_ticks(os.getpid()),
        "started_at": utc_now(),
    }
    try:
        (state_dir / "claimed").mkdir(mode=0o700)
    except FileExistsError:
        return 0
    atomic_json(state_dir / "launch.json", launch)
    if (state_dir / "cancel.json").is_file() or (state_dir / "status.json").is_file():
        publish_cancelled_status(state_dir, invocation_id)
        return 0

    output_path = state_dir / "output.log"
    descriptor = os.open(
        output_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    child: subprocess.Popen[bytes] | None = None
    forwarded_signal: int | None = None

    def forward(signum: int, _frame: object) -> None:
        nonlocal forwarded_signal
        forwarded_signal = signum
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    exit_code = 125
    driver_error: str | None = None
    safe_terminal = True
    try:
        with os.fdopen(descriptor, "ab", buffering=0) as output:
            try:
                child = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
                try:
                    child_start_ticks = process_start_ticks(child.pid)
                except (
                    FileNotFoundError,
                    ProcessLookupError,
                    subprocess.CalledProcessError,
                ):
                    child_start_ticks = 0
                atomic_json(
                    state_dir / "started.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "invocation_id": invocation_id,
                        "command_pid": child.pid,
                        "command_start_ticks": child_start_ticks,
                        "started_at": utc_now(),
                    },
                )
                if forwarded_signal is not None and child.poll() is None:
                    os.killpg(child.pid, forwarded_signal)
                exit_code = normalize_exit_code(child.wait())
                if process_group_exists(child.pid):
                    driver_error = (
                        "command process group remained active after child exit"
                    )
                    exit_code = 125
                    safe_terminal = stop_process_group(child.pid)
            except FileNotFoundError as error:
                if child is None:
                    exit_code = 127
                else:
                    driver_error = f"{type(error).__name__}: {error}"
                    exit_code = 125
                    safe_terminal = stop_child(child)
            except PermissionError as error:
                if child is None:
                    exit_code = 126
                else:
                    driver_error = f"{type(error).__name__}: {error}"
                    exit_code = 125
                    safe_terminal = stop_child(child)
            except BaseException as error:  # noqa: BLE001
                driver_error = f"{type(error).__name__}: {error}"
                exit_code = 125
                if child is not None:
                    safe_terminal = stop_child(child)
    finally:
        cancelled = (
            forwarded_signal is not None or (state_dir / "cancel.json").is_file()
        )
        if cancelled:
            exit_code = 143
        if not safe_terminal:
            atomic_json(
                state_dir / "driver-error.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "invocation_id": invocation_id,
                    "driver_error": driver_error or "command process group survived",
                    "recorded_at": utc_now(),
                },
            )
            return 0
        status: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "invocation_id": invocation_id,
            "exit_code": exit_code,
            "finished_at": utc_now(),
        }
        if driver_error is not None:
            status["driver_error"] = driver_error
        if forwarded_signal is not None:
            status["forwarded_signal"] = forwarded_signal
        if cancelled:
            status["cancelled"] = True
        atomic_json_if_absent(state_dir / "status.json", status)
    return 0


def terminate_command(
    state_dir: Path,
    invocation_id: str,
    timeout: float,
) -> dict[str, Any]:
    cancel = {
        "schema_version": SCHEMA_VERSION,
        "invocation_id": invocation_id,
        "requested_at": utc_now(),
    }
    if not atomic_json_if_absent(state_dir / "cancel.json", cancel):
        existing_cancel = load_json(state_dir / "cancel.json")
        if existing_cancel.get("invocation_id") != invocation_id:
            raise RuntimeError("detached command cancellation invocation id mismatch")
    current = status_payload(state_dir, invocation_id)
    if current["state"] == "finished":
        return current

    def identities() -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for name, pid_key, ticks_key in (
            ("started.json", "command_pid", "command_start_ticks"),
            ("launch.json", "driver_pid", "driver_start_ticks"),
        ):
            path = state_dir / name
            if not path.is_file():
                continue
            document = load_json(path)
            pid = document.get(pid_key)
            ticks = document.get(ticks_key)
            if isinstance(pid, int) and isinstance(ticks, int):
                result.append((pid, ticks))
        return result

    def signal_live(signum: int) -> None:
        for pid, ticks in identities():
            if process_matches(pid, ticks) or process_group_exists(pid):
                try:
                    os.killpg(pid, signum)
                except ProcessLookupError:
                    pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        signal_live(signal.SIGTERM)
        current = status_payload(state_dir, invocation_id)
        if current["state"] == "finished":
            return current
        time.sleep(0.1)

    signal_live(signal.SIGKILL)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        live = [
            (pid, ticks)
            for pid, ticks in identities()
            if process_matches(pid, ticks) or process_group_exists(pid)
        ]
        if not live:
            publish_cancelled_status(state_dir, invocation_id)
            return status_payload(state_dir, invocation_id)
        signal_live(signal.SIGKILL)
        time.sleep(0.1)
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "orphaned",
        "invocation_id": invocation_id,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="operation", required=True)
    for operation in (
        "acquire",
        "release",
        "start",
        "status",
        "terminate",
        "_run",
    ):
        command = subparsers.add_parser(operation)
        command.add_argument("--state-dir", required=True)
        command.add_argument("--invocation-id", required=True)
        if operation == "acquire":
            command.add_argument("--variant", required=True)
            command.add_argument("--campaign-phase", required=True)
            command.add_argument("--attestation", required=True)
            command.add_argument("--argv-sha256", required=True)
            command.add_argument("--deployment-sha256", required=True)
            command.add_argument("--acquired-at", required=True)
        if operation == "start":
            command.add_argument("argv", nargs=argparse.REMAINDER)
        if operation == "terminate":
            command.add_argument("--timeout", type=float, default=10.0)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        invocation_id = validate_invocation(args.invocation_id)
        state_dir = validate_state_dir(args.state_dir)
        if args.operation == "acquire":
            emit(
                acquire_lock(
                    state_dir,
                    invocation_id,
                    variant=args.variant,
                    campaign_phase=args.campaign_phase,
                    attestation=args.attestation,
                    argv_sha256=args.argv_sha256,
                    deployment_sha256=args.deployment_sha256,
                    acquired_at=args.acquired_at,
                )
            )
        elif args.operation == "release":
            emit(release_lock(state_dir, invocation_id))
        elif args.operation == "start":
            argv = args.argv
            if argv[:1] == ["--"]:
                argv = argv[1:]
            emit(start_command(state_dir, invocation_id, argv))
        elif args.operation == "status":
            emit(status_payload(state_dir, invocation_id))
        elif args.operation == "terminate":
            emit(terminate_command(state_dir, invocation_id, args.timeout))
        else:
            return run_command(state_dir, invocation_id)
    except BaseException as error:  # noqa: BLE001
        emit(
            {
                "schema_version": SCHEMA_VERSION,
                "state": "error",
                "invocation_id": getattr(args, "invocation_id", None),
                "error": f"{type(error).__name__}: {error}",
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
