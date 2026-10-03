# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from learned_routing.pool import EvalPool, Job
from learned_routing.slots import SlotPool

HOLDER = """
import json, sys, time
from learned_routing.slots import SlotPool
pool = SlotPool(sys.argv[1], int(sys.argv[2]), poll_s=0.01)
for _ in range(3):
    with pool.slot("t") as slot:
        start = time.time()
        time.sleep(0.12)
        end = time.time()
    print(json.dumps({"slot": slot.index, "start": start, "end": end}), flush=True)
"""


def check_intervals(intervals: list[dict], capacity: int) -> None:
    for i, a in enumerate(intervals):
        # The maximum concurrency is attained at some interval's start: count holders there.
        holding = [b for b in intervals if b["start"] <= a["start"] < b["end"]]
        assert len(holding) <= capacity, holding
        for b in intervals[i + 1 :]:
            if a["slot"] == b["slot"]:
                assert a["end"] <= b["start"] or b["end"] <= a["start"], (a, b)


def test_slots_cap_concurrency_across_processes(tmp_path):
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", HOLDER, str(tmp_path), "3"],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(6)
    ]
    intervals = []
    for proc in procs:
        out, _ = proc.communicate(timeout=60)
        assert proc.returncode == 0
        intervals += [json.loads(line) for line in out.splitlines()]
    assert len(intervals) == 18
    check_intervals(intervals, 3)
    # with 6 processes competing for 3 slots, the pool must actually have been saturated
    peak = max(
        sum(1 for b in intervals if b["start"] <= a["start"] < b["end"])
        for a in intervals
    )
    assert peak == 3


def test_slots_cap_concurrency_across_threads_of_one_process(tmp_path):
    pool = SlotPool(tmp_path, 2, poll_s=0.01)
    intervals, lock = [], threading.Lock()

    def hold():
        for _ in range(3):
            with pool.slot("thread") as slot:
                start = time.time()
                time.sleep(0.05)
                end = time.time()
            with lock:
                intervals.append({"slot": slot.index, "start": start, "end": end})

    threads = [threading.Thread(target=hold) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(intervals) == 18
    check_intervals(intervals, 2)


def test_a_killed_holder_releases_its_slot(tmp_path):
    script = (
        "import sys, time\nfrom learned_routing.slots import SlotPool\n"
        "s = SlotPool(sys.argv[1], 1).acquire('victim')\nprint('held', flush=True)\ntime.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)], stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout.readline().strip() == "held"
    pool = SlotPool(tmp_path, 1)
    assert pool.try_acquire() is None
    status = pool.status()
    assert status[0]["busy"] and str(proc.pid) in status[0]["holder"]
    proc.kill()  # exact child PID
    proc.wait()
    slot = pool.try_acquire()
    assert slot is not None
    slot.release()


def test_pool_survives_worker_timeouts_and_crashes(tmp_path):
    slots = SlotPool(tmp_path / "slots", 2, poll_s=0.01)
    pool = EvalPool(
        slots, concurrency=1, log_dir=tmp_path / "logs", job_dir=tmp_path / "jobs"
    )

    def job(kind, seconds=0.0, timeout=30.0, name="j"):
        payload = {
            "job_id": name,
            "replay": {"kind": kind, "seconds": seconds},
            "record": {"name": name},
        }
        return Job(payload, timeout_s=timeout)

    try:
        first = pool.run([job("_test_sleep", name="a")])[0]
        assert first.status == "done" and first.record["name"] == "a"
        pid = first.record["worker_pid"]
        timed_out = pool.run([job("_test_sleep", seconds=30.0, timeout=1.0, name="b")])[
            0
        ]
        assert timed_out.status == "timeout"
        assert (
            timed_out.record["error"].startswith("timeout")
            and timed_out.record["name"] == "b"
        )
        crashed = pool.run([job("_test_crash", name="c")])[0]
        assert crashed.status == "crashed" and crashed.record["error"].startswith(
            "worker_crashed"
        )
        after = pool.run([job("_test_sleep", name="d")])[0]
        assert after.status == "done" and after.record["worker_pid"] != pid
        assert not any(row["busy"] for row in slots.status())  # every slot released
    finally:
        pool.close()


def test_pool_reuses_worker_processes_across_runs(tmp_path):
    slots = SlotPool(tmp_path / "slots", 1, poll_s=0.01)
    pool = EvalPool(
        slots, concurrency=1, log_dir=tmp_path / "logs", job_dir=tmp_path / "jobs"
    )
    try:
        pids = set()
        for name in "abc":
            payload = {"job_id": name, "replay": {"kind": "_test_sleep"}, "record": {}}
            outcome = pool.run([Job(payload, timeout_s=30.0)])[0]
            pids.add(outcome.record["worker_pid"])
        assert len(pids) == 1
    finally:
        pool.close()
    assert Path(tmp_path / "logs").exists()


def test_idle_worker_exit_is_resent_not_recorded_as_a_crash(tmp_path):
    slots = SlotPool(tmp_path / "slots", 1, poll_s=0.01)
    pool = EvalPool(
        slots,
        concurrency=1,
        log_dir=tmp_path / "logs",
        job_dir=tmp_path / "jobs",
        env={"LR_WORKER_IDLE_EXIT_S": "0.3"},
    )
    try:
        first = pool.run(
            [
                Job(
                    {"job_id": "a", "replay": {"kind": "_test_sleep"}, "record": {}},
                    30.0,
                )
            ]
        )[0]
        time.sleep(1.0)  # the worker exits idle
        second = pool.run(
            [
                Job(
                    {"job_id": "b", "replay": {"kind": "_test_sleep"}, "record": {}},
                    30.0,
                )
            ]
        )[0]
        assert first.status == second.status == "done"
        assert second.record["worker_pid"] != first.record["worker_pid"]
    finally:
        pool.close()


def test_job_sent_to_an_exited_worker_is_resent(tmp_path, monkeypatch):
    from learned_routing import pool as pool_module

    slots = SlotPool(tmp_path / "slots", 1, poll_s=0.01)
    pool = EvalPool(
        slots,
        concurrency=1,
        log_dir=tmp_path / "logs",
        job_dir=tmp_path / "jobs",
        env={"LR_WORKER_IDLE_EXIT_S": "0.3"},
    )
    try:
        first = pool.run(
            [
                Job(
                    {"job_id": "a", "replay": {"kind": "_test_sleep"}, "record": {}},
                    30.0,
                )
            ]
        )[0]
        time.sleep(
            1.0
        )  # the worker has exited idle, but the pool still believes it is alive
        monkeypatch.setattr(
            pool_module.WorkerProcess,
            "alive",
            property(lambda self: self.proc is not None),
        )
        second = pool.run(
            [
                Job(
                    {"job_id": "b", "replay": {"kind": "_test_sleep"}, "record": {}},
                    30.0,
                )
            ]
        )[0]
        assert second.status == "done" and second.record["error"] is None
        assert second.record["worker_pid"] != first.record["worker_pid"]
    finally:
        pool.close()
