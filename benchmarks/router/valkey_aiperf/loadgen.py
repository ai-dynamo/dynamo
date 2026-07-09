# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.router.router_process import ValkeyModuleProcess, ValkeySentinelProcess

from .valkey import info_fields, valkey_info

def stop_aiperf_process_group(process: subprocess.Popen[str]) -> None:
    """Bound shutdown of aiperf and all client workers it launched."""

    if process.poll() is not None:
        return
    for signal_to_send, grace_seconds in (
        (signal.SIGINT, 20),
        (signal.SIGTERM, 10),
        (signal.SIGKILL, 5),
    ):
        try:
            os.killpg(process.pid, signal_to_send)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            continue


def run_aiperf(
    command: list[str],
    *,
    run_dir: Path,
    log_path: Path,
    timeout_seconds: int,
    fault_hook: Callable[[threading.Event], dict[str, Any]] | None = None,
    fault_join_timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Run aiperf in its own process group so a stuck finalizer is recoverable."""

    started = time.monotonic()
    timed_out = False
    hook_stop = threading.Event()
    fault_result: dict[str, Any] = {"status": "not_requested"}
    hook_thread: threading.Thread | None = None
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            env=os.environ.copy(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        if fault_hook is not None:
            fault_result = {"status": "waiting"}

            def run_fault_hook() -> None:
                nonlocal fault_result
                try:
                    fault_result = fault_hook(hook_stop)
                except Exception as error:  # Preserve aiperf artifacts for diagnosis.
                    fault_result = {
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }

            hook_thread = threading.Thread(
                target=run_fault_hook,
                name="valkey-primary-fault-injection",
                daemon=True,
            )
            hook_thread.start()
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_aiperf_process_group(process)
            returncode = process.returncode
        except BaseException:
            stop_aiperf_process_group(process)
            raise
        finally:
            hook_stop.set()

    if hook_thread is not None:
        # A hook which already killed the primary must finish proving (or
        # timing out) promotion before ExitStack starts tearing down Sentinel
        # and Valkey. Its own deadline remains the authoritative bound.
        hook_thread.join(timeout=fault_join_timeout_seconds)
        if hook_thread.is_alive():
            fault_result = {
                "status": "error",
                "error": "fault injection thread did not stop after aiperf exited",
            }

    return {
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "fault_injection": fault_result,
    }


def is_completed_profiling_record(record: Any) -> bool:
    if not isinstance(record, Mapping):
        return False
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        return False
    if metadata.get("benchmark_phase") != "profiling":
        return False
    if record.get("error") is not None or metadata.get("was_cancelled"):
        return False
    start = metadata.get("request_start_ns")
    end = metadata.get("request_end_ns")
    return isinstance(start, int) and isinstance(end, int) and end >= start


def inject_primary_kill_after_profiling_starts(
    stop: threading.Event,
    *,
    records_path: Path,
    completed_records: int,
    primary: ValkeyModuleProcess,
    sentinels: list[ValkeySentinelProcess],
    promoted_port: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """SIGKILL the primary with live requests, then prove quorum promotion."""

    observed_records = 0
    while not stop.is_set():
        if records_path.is_file():
            with records_path.open("r", encoding="utf-8", errors="replace") as records:
                observed_records = 0
                for line in records:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if is_completed_profiling_record(record):
                        observed_records += 1
                        if observed_records >= completed_records:
                            break
            if observed_records >= completed_records:
                break
        time.sleep(0.05)
    else:
        return {
            "status": "not_triggered",
            "observed_completed_records": observed_records,
            "reason": "aiperf exited before fault injection",
        }

    if primary.proc is None or primary.proc.poll() is not None:
        raise RuntimeError("Valkey primary exited before SIGKILL injection")
    killed_pid = primary.proc.pid
    killed_at_wall = datetime.now().astimezone().isoformat()
    killed_at = time.monotonic()
    promotion_deadline = killed_at + timeout_seconds
    os.kill(killed_pid, signal.SIGKILL)

    last_votes: list[tuple[str, int] | None] = []
    last_role = "unavailable"
    while time.monotonic() < promotion_deadline:
        last_votes = [sentinel.get_master_addr(timeout=0.5) for sentinel in sentinels]
        agreeing = sum(vote == ("127.0.0.1", promoted_port) for vote in last_votes)
        try:
            last_role = info_fields(valkey_info(promoted_port, "replication")).get(
                "role", "unknown"
            )
        except (OSError, RuntimeError, ValueError):
            last_role = "unavailable"
        if agreeing >= 2 and last_role == "master":
            promoted_at = time.monotonic()
            return {
                "status": "promoted",
                "killed_pid": killed_pid,
                "killed_at": killed_at_wall,
                "observed_completed_records": observed_records,
                "promoted_port": promoted_port,
                "sentinel_votes": last_votes,
                "promotion_seconds": promoted_at - killed_at,
                "promoted_at": datetime.now().astimezone().isoformat(),
            }
        time.sleep(0.05)
    return {
        "status": "promotion_timeout",
        "killed_pid": killed_pid,
        "killed_at": killed_at_wall,
        "observed_completed_records": observed_records,
        "promoted_port": promoted_port,
        "sentinel_votes": last_votes,
        "promoted_role": last_role,
        "timeout_seconds": timeout_seconds,
    }
