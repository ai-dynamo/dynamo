# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU process probes for isolation, bounded execution, and descendant cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from autoscaling_arena import execution, match_runner
from autoscaling_arena.execution import run_isolated
from autoscaling_arena.match_config import load_match_config

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.timeout(20),
]


@dataclass(frozen=True)
class _Config:
    backend: None = None

    def to_dict(self):
        return {"api_key": "synthetic-secret"}


@pytest.fixture
def isolated_inputs(tmp_path: Path):
    item = SimpleNamespace(
        run_id="0001-synthetic",
        backend="sim",
        autoscaler="fixed",
        workload="flat",
        sla="interactive",
        repetition=0,
        seed=7,
    )
    return _Config(), item, SimpleNamespace(session_root=tmp_path)


def _success(config, item, context):
    print("stdout synthetic-secret", flush=True)
    print("stderr synthetic-secret", file=sys.stderr, flush=True)
    return {
        "run_id": item.run_id,
        "status": "ok",
        "payload": "x" * (2 * 1024 * 1024),
        "execution": {"setup_wall_s": 0.125, "replay_wall_s": 0.25},
    }


def _error(config, item, context):
    raise ValueError("failed with synthetic-secret")


def _crash(config, item, context):
    os._exit(17)


def _wait(config, item, context):
    threading.Event().wait(30)
    return {"run_id": item.run_id, "status": "ok"}


def _interruptible(config, item, context):
    print("stdout synthetic-secret", flush=True)
    print("stderr synthetic-secret", file=sys.stderr, flush=True)
    (context.session_root / "interrupt-ready.pid").write_text(str(os.getpid()))
    threading.Event().wait(30)
    return {"run_id": item.run_id, "status": "ok"}


def _descendant(config, item, context):
    # The supervisor must clean this child even when termination interrupts
    # the normal Popen context-manager cleanup in this worker.
    with subprocess.Popen(
        [sys.executable, "-c", "import threading; threading.Event().wait(30)"]
    ) as child:
        (context.session_root / "descendant.pid").write_text(str(child.pid))
        threading.Event().wait(30)
    return {"run_id": item.run_id, "status": "ok"}


def _orphan(config, item, context):
    child_pid = os.fork()
    if child_pid == 0:
        threading.Event().wait(30)
        os._exit(0)
    (context.session_root / "descendant.pid").write_text(str(child_pid))
    return {"run_id": item.run_id, "status": "ok"}


def test_large_result_uses_file_and_logs_are_redacted(isolated_inputs):
    result = run_isolated(*isolated_inputs, target=_success, timeout_s=10)

    assert result["status"] == "ok"
    assert len(result["payload"]) == 2 * 1024 * 1024
    assert result["execution"]["isolated"] is True
    assert result["execution"]["wall_time_s"] > 0
    assert result["execution"]["peak_rss_mb"] > 0
    assert result["execution"]["setup_wall_s"] == 0.125
    assert result["execution"]["replay_wall_s"] == 0.25
    log = Path(result["artifacts"]["worker_log"]).read_text()
    assert "stdout <redacted>" in log
    assert "stderr <redacted>" in log
    assert "synthetic-secret" not in log


def test_child_exception_is_a_sanitized_failure(isolated_inputs):
    result = run_isolated(*isolated_inputs, target=_error, timeout_s=10)

    assert result["status"] == "failed"
    assert result["error"] == {
        "type": "ValueError",
        "message": "failed with <redacted>",
    }


def test_abrupt_child_exit_is_reported(isolated_inputs):
    result = run_isolated(*isolated_inputs, target=_crash, timeout_s=10)

    assert result["status"] == "failed"
    assert result["error"]["type"] == "RunWorkerExitError"
    assert "17" in result["error"]["message"]


def test_timeout_terminates_worker_and_descendant(isolated_inputs):
    result = run_isolated(*isolated_inputs, target=_descendant, timeout_s=2)

    assert result["error"]["type"] == "RunTimeoutError"
    assert result["execution"]["wall_time_s"] < 8
    pid_path = isolated_inputs[2].session_root / "descendant.pid"
    assert pid_path.exists(), "worker did not reach descendant creation"
    pid = int(pid_path.read_text())
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )


def test_rss_limit_terminates_worker(isolated_inputs):
    result = run_isolated(*isolated_inputs, target=_wait, timeout_s=10, max_memory_mb=1)

    assert result["error"]["type"] == "RunMemoryLimitError"
    assert result["execution"]["peak_rss_mb"] > 1
    assert result["execution"]["wall_time_s"] < 8


def test_success_also_cleans_lingering_helpers(isolated_inputs):
    result = run_isolated(*isolated_inputs, target=_orphan, timeout_s=10)

    assert result["status"] == "ok"
    pid = int((isolated_inputs[2].session_root / "descendant.pid").read_text())
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )


def test_caller_cancellation_stops_active_worker(isolated_inputs):
    cancelled = threading.Event()
    timer = threading.Timer(0.5, cancelled.set)
    timer.start()
    try:
        result = run_isolated(
            *isolated_inputs, target=_wait, timeout_s=10, cancel_event=cancelled
        )
    finally:
        timer.cancel()
        timer.join()

    assert result["error"]["type"] == "RunCancelledError"
    assert result["execution"]["wall_time_s"] < 4


@pytest.mark.parametrize("redaction_fails", [False, True])
def test_parent_sigint_cleans_worker_and_never_retains_raw_log(
    isolated_inputs, monkeypatch, redaction_fails
):
    root = isolated_inputs[2].session_root
    ready = root / "interrupt-ready.pid"
    tree_rss = execution._tree_rss

    def interrupt_when_ready(pid):
        rss = tree_rss(pid)
        if ready.exists():
            # Deliver a real SIGINT to the supervising process, after the
            # spawned worker has flushed both log streams. This exercises
            # BaseException cleanup rather than the cancellation Event path.
            os.kill(os.getpid(), signal.SIGINT)
        return rss

    monkeypatch.setattr(execution, "_tree_rss", interrupt_when_ready)
    if redaction_fails:

        def fail_redaction(path, config, runner):
            raise OSError("synthetic redaction failure")

        monkeypatch.setattr(execution, "_sanitize_log", fail_redaction)
    with pytest.raises(KeyboardInterrupt):
        run_isolated(*isolated_inputs, target=_interruptible, timeout_s=10)

    assert ready.is_file()
    assert not psutil.pid_exists(int(ready.read_text()))
    logs = list((root / "workers").glob("*/worker.log"))
    if redaction_fails:
        assert not logs
    else:
        assert len(logs) == 1
        content = logs[0].read_text()
        assert "synthetic-secret" not in content
        assert "stdout <redacted>" in content
        assert "stderr <redacted>" in content


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_limits_fail_before_starting_worker(isolated_inputs, value):
    with pytest.raises(ValueError, match="finite and positive"):
        run_isolated(*isolated_inputs, timeout_s=value)
    assert not (isolated_inputs[2].session_root / "workers").exists()


def test_parallel_builtin_trace_reader_survives_other_worker_write(
    tmp_path, monkeypatch
):
    config = load_match_config(
        Path(__file__).resolve().parents[1] / "configs/match.controlled.example.yaml"
    )
    first = next(config.iter_runs())
    second = replace(first, index=2, run_id="0002-sim-second-flat")
    context = match_runner._ExecutionContext(
        "parallel-probe", tmp_path, tmp_path / "external"
    )
    first_written = threading.Event()
    second_opened = threading.Event()
    first_read = threading.Event()
    worker = threading.local()
    original_open = Path.open

    def delayed_open(path, mode="r", *args, **kwargs):
        handle = original_open(path, mode, *args, **kwargs)
        if mode == "w" and path.name == "flat.jsonl" and worker.index == 2:
            second_opened.set()
            if not first_read.wait(timeout=5):
                handle.close()
                raise RuntimeError("first worker did not read its trace")
        return handle

    def consume_trace(config, item, run_context):
        worker.index = item.index
        run_dir = run_context.session_root / "runs" / item.run_id
        run_dir.mkdir(parents=True)
        if item.index == 2 and not first_written.wait(timeout=5):
            raise RuntimeError("first worker did not write its trace")
        trace = match_runner._materialize_trace(
            run_context,
            workload_name="flat",
            seed=7,
            max_requests=16,
            arrival_speedup=1.0,
            speedup_is_materialized=False,
        )
        if item.index == 1:
            first_written.set()
            if not second_opened.wait(timeout=5):
                raise RuntimeError("second worker did not start writing")
            try:
                rows = trace.read_text().splitlines()
            finally:
                first_read.set()
        else:
            rows = trace.read_text().splitlines()
        return {
            "run_id": item.run_id,
            "status": "ok",
            "trace": trace,
            "rows": rows,
        }

    monkeypatch.setattr(Path, "open", delayed_open)
    monkeypatch.setattr(match_runner, "_run_sim_item", consume_trace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(match_runner._run_one_item, config, item, context)
            for item in (first, second)
        ]
        results = [future.result() for future in futures]

    assert [result["status"] for result in results] == ["ok", "ok"], results
    assert [len(result["rows"]) for result in results] == [16, 16]
    assert results[0]["rows"] == results[1]["rows"]
    assert results[0]["trace"] != results[1]["trace"]
    for item, result in zip((first, second), results):
        assert result["trace"].is_relative_to(tmp_path / "runs" / item.run_id)
