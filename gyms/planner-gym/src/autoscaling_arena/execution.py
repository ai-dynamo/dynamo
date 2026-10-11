# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-isolated match execution with wall-time and process-tree RSS limits."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import multiprocessing
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

import psutil

RunTarget = Callable[[Any, Any, Any], dict[str, Any]]


def run_isolated(
    config: Any,
    item: Any,
    context: Any,
    *,
    timeout_s: float | None = None,
    max_memory_mb: float | None = None,
    target: RunTarget | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Execute one run in a spawned process; monitor all of its descendants.

    The caller owns scheduling and checkpoint reuse. Results travel through an
    atomic JSON file, so a large report cannot deadlock a multiprocessing pipe.
    RSS is sampled every 50 ms; the limit is an execution guard, not a cgroup
    reservation, and very short allocation spikes may occur between samples.
    """

    for name, value in (("timeout_s", timeout_s), ("max_memory_mb", max_memory_mb)):
        if value is not None and (
            isinstance(value, bool) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    runner = importlib.import_module("autoscaling_arena.match_runner")
    workers = context.session_root / "workers"
    workers.mkdir(parents=True, exist_ok=True)
    prefix = hashlib.sha256(item.run_id.encode()).hexdigest()[:16] + "-"
    worker_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=workers))
    result_path = worker_dir / "result.json"
    log_path = worker_dir / "worker.log"
    process = multiprocessing.get_context("spawn").Process(
        target=_worker,
        args=(config, item, context, worker_dir, target),
        name="planner-gym-run",
    )
    started = time.monotonic()
    peak_rss = 0
    failure: tuple[str, str] | None = None
    cleaned_up = False
    log_sanitized = False
    try:
        process.start()
        while process.is_alive():
            if cancel_event is not None and cancel_event.is_set():
                failure = ("RunCancelledError", "run cancelled by the caller")
                break
            rss = _tree_rss(process.pid)
            peak_rss = max(peak_rss, rss)
            elapsed = time.monotonic() - started
            if timeout_s is not None and elapsed >= timeout_s:
                failure = (
                    "RunTimeoutError",
                    f"run exceeded wall-time limit of {timeout_s:g} s",
                )
                break
            if max_memory_mb is not None and rss > max_memory_mb * 1024 * 1024:
                failure = (
                    "RunMemoryLimitError",
                    f"process-tree RSS exceeded limit of {max_memory_mb:g} MiB",
                )
                break
            process.join(timeout=0.05)
        if failure is not None:
            _terminate_tree(process, worker_dir)
            cleaned_up = True
        process.join()
        if failure is None and process.exitcode != 0:
            failure = (
                "RunWorkerExitError",
                f"worker exited with code {process.exitcode}",
            )
        if failure is None:
            try:
                result = json.loads(result_path.read_text())
            except (OSError, ValueError) as exc:
                failure = ("RunWorkerResultError", f"cannot read worker result: {exc}")
            else:
                if not isinstance(result, dict) or result.get("run_id") != item.run_id:
                    failure = (
                        "RunWorkerResultError",
                        "worker returned an invalid run identity",
                    )
        if failure is not None:
            result = runner._failure_result(
                item,
                artifact_dir=context.session_root / "runs" / item.run_id,
                error_type=failure[0],
                message=runner._redact_exception_message(config, failure[1]),
            )
        if not cleaned_up:
            _terminate_tree(process, worker_dir)
            cleaned_up = True
        _sanitize_log(log_path, config, runner)
        log_sanitized = True
        result["execution"] = {
            **result.get("execution", {}),
            "wall_time_s": time.monotonic() - started,
            "peak_rss_mb": peak_rss / (1024 * 1024),
            "timeout_s": timeout_s,
            "max_memory_mb": max_memory_mb,
            "isolated": True,
        }
        result.setdefault("artifacts", {})["worker_log"] = str(log_path)
        return result
    finally:
        original_exception = sys.exception()
        # A backend may leave helpers alive even after returning a result.
        # Also clean up if the parent receives KeyboardInterrupt or an error.
        try:
            if process.pid is not None and not cleaned_up:
                _terminate_tree(process, worker_dir)
                process.join(timeout=2)
        finally:
            try:
                if not log_sanitized:
                    try:
                        _sanitize_log(log_path, config, runner)
                    except BaseException:
                        # Fail closed if redaction itself fails. Preserve an
                        # in-flight interrupt/error after removing the raw log.
                        log_path.unlink(missing_ok=True)
                        log_path.with_suffix(".redacted").unlink(missing_ok=True)
                        if original_exception is None:
                            raise
            finally:
                process.close()


def _worker(config, item, context, worker_dir: Path, target: RunTarget | None) -> None:
    os.setsid()
    (worker_dir / "process-group").write_text(str(os.getpid()))
    runner = importlib.import_module("autoscaling_arena.match_runner")
    with (worker_dir / "worker.log").open("w", buffering=1) as log:
        os.dup2(log.fileno(), sys.stdout.fileno())
        os.dup2(log.fileno(), sys.stderr.fileno())
        try:
            execute = target if target is not None else runner._run_one_item
            result = execute(config, item, context)
        except Exception as exc:
            # A failed child becomes an explicit per-run result; the parent
            # retains the failure while continuing other independently run jobs.
            result = runner._failure_result(
                item,
                artifact_dir=context.session_root / "runs" / item.run_id,
                error_type=type(exc).__name__,
                message=runner._redact_exception_message(config, str(exc)),
            )
        result = runner._sanitize_json(result)
        temporary = worker_dir / "result.json.tmp"
        temporary.write_text(json.dumps(result, allow_nan=False))
        os.replace(temporary, worker_dir / "result.json")
        sys.stdout.flush()
        sys.stderr.flush()


def _tree_rss(pid: int) -> int:
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


def _terminate_tree(process, worker_dir: Path) -> None:
    try:
        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    has_group = (worker_dir / "process-group").exists()
    if has_group:
        # Helpers can be reparented after a worker exits. Include its process
        # group even when the original parent no longer exists, and wait for
        # those helpers to exit before reporting successful cleanup.
        known = {descendant.pid for descendant in descendants}
        for candidate in psutil.process_iter():
            if candidate.pid == process.pid or candidate.pid in known:
                continue
            try:
                if os.getpgid(candidate.pid) == process.pid:
                    descendants.append(candidate)
            except (ProcessLookupError, PermissionError):
                continue
        # The marker is written only after setsid(), so this cannot signal the
        # caller's process group during the child's startup race.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.is_alive():
        process.terminate()
    for descendant in descendants:
        try:
            descendant.terminate()
        except psutil.NoSuchProcess:
            continue
    process.join(timeout=0.5)
    if has_group:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.is_alive():
        process.kill()
    _, alive = psutil.wait_procs(descendants, timeout=0.5)
    for descendant in alive:
        try:
            descendant.kill()
        except psutil.NoSuchProcess:
            continue


def _sanitize_log(path: Path, config: Any, runner: Any) -> None:
    if not path.exists():
        return
    temporary = path.with_suffix(".redacted")
    try:
        with path.open(errors="replace") as source, temporary.open("w") as destination:
            for line in source:
                destination.write(runner._redact_exception_message(config, line))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
