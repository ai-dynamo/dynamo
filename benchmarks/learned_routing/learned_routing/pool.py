# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run replay jobs in worker subprocesses, each holding a machine-wide slot while it runs.

- Up to ``concurrency`` threads; each owns at most one persistent worker subprocess
  (:mod:`learned_routing.worker`) and takes a slot from :class:`~learned_routing.slots.SlotPool`
  for the duration of each job (startup included), so the machine-wide cap holds across agents.
- Every job has a timeout. A worker that times out is killed by its exact PID (it is our own
  child) and replaced; a worker that dies mid-job is replaced. Both become error records.
- A worker reports the bindings build ID it would load; a mismatch with the parent's build ID
  (bindings rebuilt mid-run) fails the job instead of caching results under the wrong build.
- Workers are recycled after ``max_jobs_per_worker`` jobs or ``max_worker_rss_mib`` peak RSS, and
  exit by themselves after ``LR_WORKER_IDLE_EXIT_S`` (default 300 s) without a job. Each job is
  acknowledged on receipt, so a worker that exits idle just as a job arrives is restarted and the
  job resent instead of being recorded as a crash.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from learned_routing.slots import SlotPool

READY_TIMEOUT_S = 180.0
FIRST_WAVE_GRACE_S = 2.0


def isolation_flags() -> list[str]:
    """The parent's import-isolation flags, so a worker resolves imports exactly as its parent.

    A bundle's ``run.sh`` runs ``python -S`` with ``PYTHONPATH=site``; a worker started without
    ``-S`` would silently import the interpreter's site-packages instead (build audit B1).
    """
    if sys.flags.isolated:
        return ["-I", *(["-S"] if sys.flags.no_site else [])]
    flags = []
    if sys.flags.no_site:
        flags.append("-S")
    if sys.flags.no_user_site:
        flags.append("-s")
    if sys.flags.ignore_environment:
        flags.append("-E")
    if getattr(sys.flags, "safe_path", False):
        flags.append("-P")
    return flags


@dataclass
class Job:
    payload: dict  # the worker job (job_id, replay, scoring, record, ...); result_path is per thread
    timeout_s: float
    label: str = ""


@dataclass
class JobOutcome:
    job: Job
    record: dict | None
    status: str  # done | error | timeout | crashed | skipped
    slot: int | None = None
    waited_s: float = 0.0
    extra: dict = field(default_factory=dict)


class WorkerProcess:
    def __init__(self, python: str, log_path: Path, env: dict):
        self.python = python
        self.log_path = log_path
        self.env = env
        self.proc: subprocess.Popen | None = None
        self.jobs = 0
        self.buffer = b""
        self.ready_info: dict = {}

    def _readline(self, deadline: float) -> bytes | None:
        """One protocol line, or None on timeout; b"" on EOF."""
        assert self.proc is not None and self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            readable, _, _ = select.select([fd], [], [], min(remaining, 5.0))
            if not readable:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                return b""
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        try:
            self.proc = subprocess.Popen(
                [self.python, *isolation_flags(), "-m", "learned_routing.worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                env=self.env,
                close_fds=True,
            )
        finally:
            log.close()
        self.buffer = b""
        self.jobs = 0
        line = self._readline(time.monotonic() + READY_TIMEOUT_S)
        if not line:
            self.kill()
            raise RuntimeError(f"worker failed to start (see {self.log_path})")
        self.ready_info = json.loads(line)

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def run(self, payload: dict, timeout_s: float) -> str:
        """done | timeout | crashed | lost (exited before taking the job; safe to resend)."""
        assert self.proc is not None and self.proc.stdin is not None
        deadline = time.monotonic() + timeout_s
        try:
            self.proc.stdin.write((json.dumps(payload) + "\n").encode())
            self.proc.stdin.flush()
        except BrokenPipeError:
            self.kill()
            return "lost"
        acked = False
        while True:
            line = self._readline(deadline)
            if line is None:
                self.kill()
                return "timeout"
            if line == b"":
                self.kill()
                return "crashed" if acked else "lost"
            message = json.loads(line)
            if message.get("job_id") != payload["job_id"]:
                self.kill()
                return "crashed"
            if message.get("status") == "ack":
                acked = True
                self.jobs += 1
                continue
            return "done"

    def kill(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.kill()  # SIGKILL to this exact child PID
        self.proc.wait()
        self.proc = None

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.kill()
        self.proc = None


class EvalPool:
    def __init__(
        self,
        slots: SlotPool,
        *,
        concurrency: int,
        log_dir: Path,
        job_dir: Path,
        build_id: str | None = None,
        build_check_dir: Path | None = None,
        python: str | None = None,
        env: dict | None = None,
        max_jobs_per_worker: int = 400,
        max_worker_rss_mib: float = 1536.0,
    ):
        self.slots = slots
        self.concurrency = max(1, concurrency)
        self.log_dir = Path(log_dir)
        self.job_dir = Path(job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.build_id = build_id
        self.python = python or sys.executable
        base_env = dict(os.environ)
        base_env.setdefault("DYN_LOG", "warn")
        base_env["PYTHONDONTWRITEBYTECODE"] = "1"
        base_env.update(env or {})
        if build_id and build_check_dir:
            base_env["LR_BUILD_CHECK_DIR"] = str(build_check_dir)
        self.env = base_env
        self.max_jobs_per_worker = max_jobs_per_worker
        self.max_worker_rss_mib = max_worker_rss_mib
        self._local = threading.local()
        self._workers: list[WorkerProcess] = []
        self._lock = threading.Lock()
        self._counter = 0
        # One long-lived executor, so thread-local workers survive across run() calls.
        self._executor = ThreadPoolExecutor(
            max_workers=self.concurrency, thread_name_prefix="lr-eval"
        )

    def _worker(self) -> WorkerProcess:
        worker = getattr(self._local, "worker", None)
        if worker is None:
            with self._lock:
                self._counter += 1
                index = self._counter
            log = self.log_dir / f"worker-{os.getpid()}-{index:03d}.log"
            worker = WorkerProcess(self.python, log, self.env)
            self._local.worker = worker
            with self._lock:
                self._workers.append(worker)
        return worker

    def _run_one(
        self, job: Job, deadline: float | None, stop: threading.Event
    ) -> JobOutcome:
        if stop.is_set() or (deadline is not None and time.monotonic() >= deadline):
            return JobOutcome(job, None, "skipped")
        waited = time.monotonic()
        slot = None
        while slot is None:
            slot = self.slots.try_acquire(job.label)
            if slot is not None:
                break
            if stop.is_set() or (deadline is not None and time.monotonic() >= deadline):
                return JobOutcome(
                    job, None, "skipped", waited_s=time.monotonic() - waited
                )
            time.sleep(self.slots.poll_s)
        waited = time.monotonic() - waited
        try:
            worker = self._worker()
            if not worker.alive:
                problem = self._start(worker)
                if problem:
                    return self._failure(job, "error", slot.index, waited, problem)
            payload = dict(job.payload)
            payload["result_path"] = str(
                self.job_dir / f"{os.getpid()}-{threading.get_ident()}.json"
            )
            status = worker.run(payload, job.timeout_s)
            # "lost": the worker exited idle before taking the job, so resend it once.
            if status == "lost":
                problem = self._start(worker)
                if problem:
                    return self._failure(job, "error", slot.index, waited, problem)
                status = worker.run(payload, job.timeout_s)
                if status == "lost":
                    status = "crashed"
        except Exception as exc:  # worker start failures become error records
            return self._failure(
                job, "error", slot.index, waited, f"{type(exc).__name__}: {exc}"
            )
        finally:
            slot.release()
        if status == "timeout":
            return self._failure(
                job,
                "timeout",
                slot.index,
                waited,
                f"timeout: no result within {job.timeout_s:.0f} s",
            )
        if status == "crashed":
            return self._failure(
                job, "crashed", slot.index, waited, "worker_crashed: see worker log"
            )
        record = json.loads(Path(payload["result_path"]).read_text())
        record["slot"] = slot.index
        if (
            worker.jobs >= self.max_jobs_per_worker
            or (record.get("worker_peak_rss_mib") or 0.0) > self.max_worker_rss_mib
        ):
            worker.close()
        return JobOutcome(
            job, record, "error" if record.get("error") else "done", slot.index, waited
        )

    def _start(self, worker: WorkerProcess) -> str | None:
        """Start ``worker``; an error message if its bindings build differs from the run's."""
        worker.start()
        built = worker.ready_info.get("build_id")
        if self.build_id and built not in (None, self.build_id):
            worker.kill()
            return (
                f"bindings_changed: worker build {built} != run build {self.build_id}; "
                "restart the run"
            )
        return None

    def _failure(
        self, job: Job, status: str, slot: int | None, waited: float, error: str
    ) -> JobOutcome:
        worker = getattr(self._local, "worker", None)
        record = dict(job.payload["record"])
        record.update(
            error=error, slot=slot, worker_log=str(worker.log_path) if worker else None
        )
        return JobOutcome(job, record, status, slot, waited)

    def run(
        self,
        jobs: Sequence[Job],
        *,
        deadline: float | None = None,
        on_outcome: Callable[[JobOutcome], None] | None = None,
    ) -> list[JobOutcome]:
        """Run ``jobs``; ``on_outcome`` sees each outcome as it completes. Returns them in order."""
        if not jobs:
            return []
        stop = threading.Event()
        outcomes: list[JobOutcome | None] = [None] * len(jobs)
        # Progress guarantee: the first wave may start up to FIRST_WAVE_GRACE_S after the
        # deadline, so a caller whose deadline has (nearly) passed still completes some jobs.
        first_wave = (
            None
            if deadline is None
            else max(deadline, time.monotonic() + FIRST_WAVE_GRACE_S)
        )
        futures = {
            self._executor.submit(
                self._run_one,
                job,
                first_wave if index < self.concurrency else deadline,
                stop,
            ): index
            for index, job in enumerate(jobs)
        }
        try:
            for future in as_completed(futures):
                outcome = future.result()
                outcomes[futures[future]] = outcome
                if on_outcome is not None:
                    on_outcome(outcome)
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
        return [o for o in outcomes if o is not None]

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            workers, self._workers = self._workers, []
        for worker in workers:
            worker.close()

    def __enter__(self) -> "EvalPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
