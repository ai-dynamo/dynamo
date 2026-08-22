#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-parallel test runner (used by conftest.py, not invoked directly).

Runs pytest tests as independent subprocesses with VRAM-aware scheduling.
Each test gets CUDA_VISIBLE_DEVICES and KV cache overrides
(_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES / _PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS)
so the engine allocates only its declared VRAM budget.

Usage (always via pytest):
    pytest --max-vram-gib=6 -n auto -m "gpu_1 and vllm" tests/serve/
    pytest --max-vram-gib=6 -n 4 -sv -m "gpu_1 and vllm" tests/serve/

Flags:
    --max-vram-gib=N   Only run tests with profiled_vram_gib <= N
    -n N / -n auto     Run N tests concurrently (auto = GPU budget / smallest test)
    -s                 Stream subprocess output live with [wN] prefixes
    -v / -vv           Passed through to subprocesses for verbose test names

A 10-second cooldown between launches avoids the vLLM profiling race
(bug #10643). Tests that fail due to profiling race are retried up to 3 times.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Below the sys.path insert above, which is what makes these resolvable.
from tests.utils.vram_utils import (  # noqa: E402
    VRAM_MULTI_PROC_MARGIN,
    auto_worker_count,
    detect_gpus,
    effective_cpu_budget,
    load_test_meta,
)


@dataclass
class _TestEntry:
    """A test scheduled for GPU-parallel execution."""

    id: str
    name: str
    profiled_gib: float
    timeout: float
    requested_vllm_kv_cache_bytes: int | None = None
    requested_sglang_kv_tokens: int | None = None
    requested_sglang_vram_gib: float | None = None
    requested_trtllm_kv_tokens: int | None = None
    requested_trtllm_vram_gib: float | None = None
    skip_reason: str | None = None
    w_id: int = 0
    assigned_gpu: int | None = None
    retries: int = 0
    # When this test first started, across all its attempts. Each relaunch takes
    # a fresh per-child start_time, so without this a test that fails just under
    # its deadline gets a brand-new full budget on every retry and can burn the
    # whole step without the watchdog ever firing.
    first_start_time: float | None = None

    @property
    def est_duration(self) -> float:
        """Estimated runtime in seconds.

        Repo convention sets ``@pytest.mark.timeout`` to ~3x a test's measured
        runtime, so ``timeout / 3`` is the best a-priori duration estimate. The
        scheduler orders longest-first (LPT) on this value to minimize makespan.
        """
        return self.timeout / 3.0


@dataclass
class _CompletedTest:
    """Result record for a finished test subprocess."""

    test: _TestEntry
    duration: float
    passed: bool
    skipped: bool = False
    skip_reason: str | None = None
    fail_reason: str | None = None
    # Never started: the job ran out of time or GPU capacity. Reported apart
    # from a failure because the test did not fail, but still counted against
    # the run, since a suite that did not finish must not come out green.
    not_run: bool = False


@dataclass
class _TentativeGpu:
    """Scratch copy of GPU budget/free state used during scheduling."""

    budget: float
    free: float
    count: int


@dataclass
class _GpuState:
    """Per-GPU bookkeeping for VRAM budget tracking."""

    index: int
    total_gib: float
    budget_multi: float
    budget_used: float = 0.0
    running_count: int = 0


@dataclass
class _RunningTest:
    """State for a test subprocess currently executing on a GPU."""

    proc: subprocess.Popen[str]
    test: _TestEntry
    start_time: float
    # Process-group id, captured at launch. Not derived from the pid on demand:
    # once the direct child is reaped, getpgid(pid) fails even while surviving
    # grandchildren still sit in the group holding VRAM.
    pgid: int
    captured: list[str] = field(default_factory=list)
    reader_thread: threading.Thread | None = None
    watchdog_reason: str | None = None
    # When the watchdog first signalled this child. Escalation is timed from
    # here rather than counted, because repeating a signal does nothing: the
    # kernel already holds it pending.
    kill_started_at: float | None = None


def _print(msg: str = "") -> None:
    """Print to stderr so pytest doesn't capture it."""
    print(msg, file=sys.stderr, flush=True)


def _fmt_req(test: _TestEntry) -> str:
    """Format the resource request value for display."""
    if test.requested_sglang_kv_tokens is not None:
        return f"req_kv_tokens={int(test.requested_sglang_kv_tokens)}"
    if test.requested_sglang_vram_gib is not None:
        return f"req_vram={test.requested_sglang_vram_gib:.1f} GiB"
    if test.requested_trtllm_kv_tokens is not None:
        return f"req_kv_tokens={int(test.requested_trtllm_kv_tokens)}"
    if test.requested_trtllm_vram_gib is not None:
        return f"req_vram={test.requested_trtllm_vram_gib:.1f} GiB"
    if test.requested_vllm_kv_cache_bytes is not None:
        gib = int(test.requested_vllm_kv_cache_bytes) / (1024**3)
        return f"req_kv={gib:.2f} GiB"
    return "req_kv=None"


_JUNIT_DIR = os.path.join(tempfile.gettempdir(), "gpu_parallel_junit")
_JUNIT_COMBINED = os.path.join(_JUNIT_DIR, "combined.xml")


def _junit_path(test_name: str) -> str:
    """Where a child is told to write its --junitxml, keyed on the node id."""
    safe_name = test_name.replace("/", "_").replace("::", "__")
    return os.path.join(_JUNIT_DIR, f"{safe_name}.xml")


def _mangle_test_address(node_id: str) -> tuple[str, str]:
    """Split a node id the way pytest's own junitxml writer does.

    Mirrors _pytest.junitxml.mangle_test_address so a synthesized entry lines up
    with a real one: "tests/a/b.py::T::t[p]" -> ("tests.a.b.T", "t[p]"). A plain
    rpartition("::") would yield "tests/a/b.py", which pytest never emits, and
    would mis-split any parametrized id containing "::" -- IPv6 literals such as
    "::1" appear in this repo's ids.
    """
    path, open_bracket, params = node_id.partition("[")
    names = path.split("::")
    names[0] = names[0].replace("/", ".").removesuffix(".py")
    names[-1] += open_bracket + params
    return ".".join(names[:-1]), names[-1]


def _write_watchdog_junit(test: _TestEntry, duration: float, reason: str) -> None:
    """Write the JUnit entry a watchdog-killed child never got to write.

    SIGKILL means pytest never reaches its own --junitxml write, so without this
    the test is absent from the aggregated report entirely -- a silent hole in
    Datadog test visibility and CI test reports rather than a visible failure.

    Nothing in here is allowed to escape into the scheduling loop: an exception
    there orphans every other running child holding its VRAM, which is a far
    worse outcome than a missing report line.
    """
    try:
        import xml.etree.ElementTree as ET

        classname, name = _mangle_test_address(test.name)
        # _aggregate_junit_xml skips any file whose root is not a <testsuite> and
        # sums these counters off it, so the wrapper is required even for one case.
        # Its own name attribute is discarded there, so it is not worth inventing.
        suite = ET.Element(
            "testsuite", {"tests": "1", "failures": "1", "time": f"{duration:.3f}"}
        )
        case = ET.SubElement(
            suite,
            "testcase",
            {"classname": classname, "name": name, "time": f"{duration:.3f}"},
        )
        ET.SubElement(
            case, "failure", {"message": reason, "type": "WatchdogTimeout"}
        ).text = reason
        os.makedirs(_JUNIT_DIR, exist_ok=True)
        # encoding="utf-8" rather than "unicode": the latter writes through the
        # locale's preferred encoding, which is US-ASCII under LC_ALL=C, and a
        # non-ASCII node id then raises UnicodeEncodeError.
        ET.ElementTree(suite).write(
            _junit_path(test.name), encoding="utf-8", xml_declaration=True
        )
    except Exception as exc:  # noqa: BLE001 - a report write must never escape
        _print(f"[watchdog] could not write a JUnit entry for {test.name}: {exc}")


def _parse_junit_skipped(junit_path: str) -> str | None:
    """Check JUnit XML for a skipped test. Returns skip reason or None."""
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(junit_path)
    except (ET.ParseError, FileNotFoundError):
        return None
    root = tree.getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return None
    for tc in suite.findall("testcase"):
        skip_el = tc.find("skipped")
        if skip_el is not None:
            return skip_el.get("message", "skipped")
    return None


def _aggregate_junit_xml(junit_dir: str) -> str | None:
    """Merge per-test JUnit XML files into one combined testsuite."""
    import xml.etree.ElementTree as ET

    xmls = sorted(Path(junit_dir).glob("*.xml"))
    xmls = [x for x in xmls if x.name != "combined.xml"]
    if not xmls:
        return None

    total_tests = total_errors = total_failures = 0
    total_time = 0.0
    testcases = []

    for xml_path in xmls:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        root = tree.getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            continue
        total_tests += int(suite.get("tests", 0))
        total_errors += int(suite.get("errors", 0))
        total_failures += int(suite.get("failures", 0))
        total_time += float(suite.get("time", 0))
        testcases.extend(suite.findall("testcase"))

    combined = ET.Element(
        "testsuite",
        {
            "name": "gpu-parallel",
            "tests": str(total_tests),
            "errors": str(total_errors),
            "failures": str(total_failures),
            "time": f"{total_time:.3f}",
        },
    )
    for tc in testcases:
        combined.append(tc)

    out = _JUNIT_COMBINED
    ET.ElementTree(combined).write(out, encoding="unicode", xml_declaration=True)
    return out


def _collect_tests(pytest_args: list[str], max_vram_gib: float) -> list[str]:
    """Run pytest --collect-only to get test IDs, filtered by --max-vram-gib."""
    _strip_exact = {"-v", "-vv", "-vvv", "--verbose", "-s", "--capture=no"}
    collect_args = []
    for a in pytest_args:
        if a in _strip_exact:
            continue
        if a.startswith("-") and not a.startswith("--") and "v" in a:
            stripped = a.replace("v", "")
            if stripped != "-":
                collect_args.append(stripped)
            continue
        collect_args.append(a)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        f"--max-vram-gib={max_vram_gib}",
        "--collect-only",
        "-q",
        *collect_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    test_ids = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if ".py::" in line and not line.startswith(" "):
            test_ids.append(line)
    return test_ids


def _get_gpu_used_gib(gpu_index: int = 0) -> float:
    """Query actual GPU memory used via pynvml."""
    try:
        import pynvml
    except ImportError:
        return 0.0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return mem.used / (1024**3)
    except pynvml.NVMLError:
        return 0.0


_RETRYABLE_INIT_MARKERS = [
    "Error in memory profiling",  # vLLM profiling race assertion
    "Free memory on device",  # not enough free VRAM at startup
    "Engine core initialization failed",  # engine init crash
    "exited with code 0 while waiting for health check",  # engine started but died during init
    "exited with code -15 while waiting for health check",  # SIGTERM during init
    "exited with code -9 while waiting for health check",  # SIGKILL (OOM killer) during init
]
_MAX_RETRIES = 3

# Last-resort deadline behind the child's own --timeout, for when pytest-timeout
# is swallowed by a C-level block and the stuck child stalls the whole run.
_WATCHDOG_GRACE_S = 120
# Held back from the job budget so the orchestrator can still kill what is
# running, write the summary and aggregate JUnit before the runner is torn down.
_WATCHDOG_REPORT_RESERVE_S = 60
# How long a killed child gets to shut down cleanly on SIGTERM -- releasing
# /dev/shm segments, NATS subscriptions and etcd leases -- before it is forced.
# A SIGKILLed tree leaves those behind and the next test on the slot then fails
# for reasons invisible in its own log.
_WATCHDOG_TERM_GRACE_S = 5.0
# Total patience after the deadline before the lane is retired. SIGKILL lands in
# microseconds when it can land at all, so a tree still alive a few seconds past
# the first one is in uninterruptible sleep -- a CUDA/NCCL ioctl is the usual
# cause -- and waiting longer only delays the tests that could still run
# elsewhere. Kept short because being wrong now costs one lane, not the run.
_WATCHDOG_GIVE_UP_S = 8.0


def _watchdog_attempts() -> int:
    """How many of a child's own --timeout windows to allow before killing it.

    An attempt here is one full --timeout window. A child runs its test once,
    plus once more per Datadog Auto Test Retry, and ddtrace calls
    _reset_pytest_timeout() before each retry -- so every attempt gets a fresh
    window rather than sharing one, and the budget has to clear all of them.

    DD_CIVISIBILITY_FLAKY_RETRY_COUNT is a count of *retries*, not attempts:
    ddtrace computes ``retries_so_far = len(test.test_runs) - 1``. So the shipped
    COUNT=2 in shared-test.yml means three attempts, not two. With retries turned
    off -- nightly-ci.yml does this on nine jobs, several of which also run the
    GPU-parallel stage -- a child makes exactly one attempt, and allowing more
    would leave a stalled slot sitting for twice as long as anything justifies.
    """
    if os.environ.get("DD_CIVISIBILITY_FLAKY_RETRY_ENABLED", "").strip().lower() in (
        "0",
        "false",
    ):
        return 1
    try:
        return max(1, int(os.environ["DD_CIVISIBILITY_FLAKY_RETRY_COUNT"]) + 1)
    except (KeyError, ValueError):
        return 3  # the shipped retry count of 2, plus the initial attempt


def _job_budget_s() -> float | None:
    """Wall-clock the GPU stage gets before the runner kills the whole step.

    A deadline past this never fires -- the runner dies first and the watchdog
    never gets a turn, which is the failure this exists to prevent. GitHub
    exposes no variable for `timeout-minutes`, so shared-test.yml passes it
    through explicitly, and always has a value to pass because its own input
    carries a default.

    None means there is no step wall, which is the truth for a local run rather
    than a value worth guessing: assuming one would kill healthy tests in a long
    local session for a deadline that does not exist.
    """
    try:
        return float(os.environ["GPU_TEST_TIMEOUT_MINUTES"]) * 60
    except (KeyError, ValueError):
        return None


def _watchdog_kill_at(
    start_time: float, timeout: float, attempts: int, step_deadline: float | None
) -> float:
    """When to give up on a child, as a monotonic timestamp.

    Two independent bounds, whichever comes first:

    - what the test itself could legitimately need: every attempt burning a full
      --timeout, plus grace for interpreter start, collection and teardown;
    - what the job has left, when it is running under one. The step is torn down
      at `step_deadline` whatever happens, so a child still running just before
      it cannot finish. Killing it first turns an opaque runner timeout into a
      named FAILED test, and is what keeps a test guarded when its own deadline
      lands past the step cap -- the 1800s tests in a 30-minute pipeline -- or
      when it started so late in a serial lane that its own window never had
      room to expire.
    """
    kill_at = start_time + timeout * attempts + _WATCHDOG_GRACE_S
    if step_deadline is None:
        return kill_at
    return min(kill_at, step_deadline - _WATCHDOG_REPORT_RESERVE_S)


def _signal_process_tree(pid: int, pgid: int, sig: int) -> None:
    """Signal a child and everything it started. Never blocks, never raises.

    Signals the process group first, which reaches descendants that have already
    been reparented to init and so are invisible to a ppid walk. Engines that
    call setsid() escape their parent's group, so the psutil descendants are
    swept too -- snapshotted before signalling, since the walk stops finding them
    once the intermediate processes die.

    Deliberately does not wait: psutil's wait() reaps through os.waitpid, which
    makes the later Popen.poll() raise ChildProcessError and report status 0 --
    a killed test reading as a pass. Reaping stays with Popen; the scheduling
    loop observes the exit on its next pass.
    """
    try:
        descendants = psutil.Process(pid).children(recursive=True)
    except (psutil.Error, OSError):
        descendants = []
    try:
        os.killpg(pgid, sig)
    except OSError:
        pass
    for proc in descendants:
        try:
            proc.send_signal(sig)
        except (psutil.Error, OSError):
            pass


def _process_group_alive(pgid: int) -> bool:
    """Whether anything is left in the child's process group.

    Kept off Popen.poll(), which only ever describes the direct child: a tree
    whose pytest died while an engine grandchild survived still holds VRAM, and
    that is the case worth knowing about before the slot is handed on.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal
    except OSError:
        return False
    return True


def _capture_output(pipe, captured: list[str], prefix: str | None = None) -> None:
    """Read all lines from a pipe into `captured`. Runs in a thread.

    If prefix is set, also prints each line live (-s mode).
    """
    for line in iter(pipe.readline, ""):
        line = line.rstrip("\n")
        if line:
            captured.append(line)
            if prefix is not None:
                _print(f"{prefix} {line}")
    pipe.close()


def _parse_cuda_visible(raw: str | None, available: list[dict]) -> list[int]:
    """Parse CUDA_VISIBLE_DEVICES value into a list of physical GPU indices.

    Semantics match CUDA:
      None (unset)   → all GPUs visible
      ""  (empty)    → no GPUs visible
      "0,1"          → those specific GPUs

    Raises ValueError on UUID/MIG tokens (not supported by the scheduler).
    """
    avail_indices = [g["index"] for g in available]
    if raw is None:
        return avail_indices
    if raw.strip() == "":
        return []
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError:
            raise ValueError(
                f"Unsupported CUDA_VISIBLE_DEVICES token {part!r}; "
                "only integer GPU indices are supported by the scheduler"
            )
        if idx not in avail_indices:
            raise ValueError(f"GPU {idx} not found (available: {avail_indices})")
        indices.append(idx)
    return indices


def _priority_key(test: _TestEntry) -> tuple[bool, float, float]:
    """Scheduling-priority sort key (use with ``reverse=True``).

    Ordered so that, highest priority first:
      1. VRAM tests (``profiled_gib > 0``) come before zero-VRAM fillers, so the
         memory-bound tests own the schedule and fillers only mop up spare slots.
      2. Longest ``est_duration`` (= timeout/3) first (LPT) to minimize makespan.
      3. Largest VRAM first, so a big test anchors an empty GPU at the full
         single-proc cap and smaller tests pack alongside it.

    Defined as a module function so ``run_parallel`` and its tests share one
    definition and can't drift.
    """
    return (test.profiled_gib > 0, test.est_duration, test.profiled_gib)


def _select_launches(
    pending: list[_TestEntry],
    gpu_states: dict[int, _GpuState],
    actual_free: dict[int, float],
    num_slots: int,
    running_count: int,
) -> list[tuple[int, int]]:
    """Pick which pending tests to launch this pass, and on which GPU.

    Pure (no NVML / no subprocesses): the caller passes the live per-GPU budget
    state and the actual free VRAM (from nvidia-smi). ``pending`` must already be
    in scheduling-priority order (VRAM tests by longest est_duration / largest
    VRAM first, zero-VRAM fillers last -- see the sort in ``run_parallel``).

    Returns a list of ``(pending_index, gpu_index)`` to launch now, honoring:

      * ``num_slots`` -- global cap on concurrently running subprocesses.
      * Per-GPU VRAM budget with two independent gates (same as before): a test
        fits only if BOTH the reserved-budget sum AND the actual nvidia-smi
        usage leave room under the cap. The cap is the full card for the first
        test on an idle GPU, then ``budget_multi`` once it hosts 2+ (reserving
        the multi-process margin for CUDA context overhead).
      * Pairing -- best-fit places each VRAM test on the GPU with the most free
        budget, so a large test that anchored an empty GPU gets backfilled with
        smaller tests up to the budget instead of running alone.
      * Anti-starvation -- when the highest-priority VRAM test does not fit, the
        GPU where it is closest to fitting is *reserved* for it. Lower-priority
        tests may still backfill that GPU, but only up to ``cap - required`` so
        that once the current occupants free, the reserved test is guaranteed to
        fit (the backfill we add now can never sum past the space it needs).
        Zero-VRAM fillers bypass the budget gates entirely (they allocate no
        memory) so transient memory pressure can't strand an otherwise-free slot.
    """
    tentative = {
        gi: _TentativeGpu(
            budget=gs.budget_used,
            free=actual_free[gi],
            count=gs.running_count,
        )
        for gi, gs in gpu_states.items()
    }
    # GPU -> required GiB of a blocked higher-priority VRAM test, and the budget
    # we have since added to that GPU via lower-priority backfill. Backfill is
    # capped at cap - required so the reserved test still fits once occupants free.
    reserved_req: dict[int, float] = {}
    backfill_added: dict[int, float] = {}
    to_launch: list[tuple[int, int]] = []

    def _cap(gi: int) -> float:
        # First test on an idle GPU may use the whole card; once it hosts 2+,
        # reserve the multi-process margin for CUDA context overhead.
        gs = gpu_states[gi]
        return gs.total_gib if tentative[gi].count < 1 else gs.budget_multi

    for idx, test in enumerate(pending):
        if running_count + len(to_launch) >= num_slots:
            break

        # Zero-VRAM filler: no budget impact, just needs a free slot. Place on
        # the least-loaded GPU for balance; never reserves and is never blocked.
        if test.profiled_gib <= 0:
            gi = min(gpu_states, key=lambda g: tentative[g].count)
            to_launch.append((idx, gi))
            tentative[gi].count += 1
            continue

        # VRAM test: best-fit on the GPU with the most free budget that passes
        # both gates and respects any reservation.
        best_gi: int | None = None
        best_avail = -1.0
        for gi, gs in gpu_states.items():
            ts = tentative[gi]
            cap = _cap(gi)
            avail = cap - ts.budget
            if avail < test.profiled_gib:
                continue  # reserved-budget gate
            actual_used = gs.total_gib - ts.free
            if actual_used + test.profiled_gib > cap:
                continue  # actual-usage gate (catches init-time spikes)
            if gi in reserved_req and (
                backfill_added[gi] + test.profiled_gib > cap - reserved_req[gi]
            ):
                continue  # would crowd out the reserved higher-priority test
            if avail > best_avail:
                best_gi, best_avail = gi, avail

        if best_gi is not None:
            to_launch.append((idx, best_gi))
            tentative[best_gi].budget += test.profiled_gib
            tentative[best_gi].free -= test.profiled_gib
            tentative[best_gi].count += 1
            if best_gi in reserved_req:
                backfill_added[best_gi] += test.profiled_gib
            continue

        # Blocked: reserve the GPU where this test is closest to fitting (most
        # free budget), unless that GPU is already held for an even-higher-
        # priority test. Keep scanning -- smaller tests may still fit elsewhere
        # or backfill under the reservation, and fillers keep filling slots.
        cand: int | None = None
        cand_avail = -1.0
        for gi in gpu_states:
            if gi in reserved_req:
                continue
            a = _cap(gi) - tentative[gi].budget
            if a > cand_avail:
                cand, cand_avail = gi, a
        if cand is not None:
            reserved_req[cand] = test.profiled_gib
            backfill_added[cand] = 0.0

    return to_launch


def run_parallel(
    test_ids: list[str],
    meta: dict[str, dict],
    max_vram_gib: float,
    num_slots: int,
    gpu_indices: list[int] | None = None,
    extra_pytest_args: list[str] | None = None,
    stream: bool = False,
    parent_basetemp: str | None = None,
) -> int:
    """Run tests in parallel with VRAM-aware scheduling across multiple GPUs.

    Flags (mimic pytest semantics):
      -s       Stream subprocess output live with [wN] prefixes.
      -v/-vv   Passed through to subprocesses for verbose test names / diffs.
               No effect on the orchestrator's output.

    Without -s, output is buffered and printed after each test completes.
    Returns exit code: 0 if all pass, 1 if any fail.
    """
    gpus = detect_gpus()
    if not gpus:
        _print("ERROR: No GPUs detected")
        return 1

    # xdist resolves `-n auto` from os.cpu_count(), which reports HOST cores and
    # ignores Docker --cpus -- so num_slots can be many times the real CPU budget
    # (e.g. 32 under --cpus=4). Combined with the zero-VRAM filler backfill that
    # would oversubscribe the CPU and slow every concurrent test. Cap at the
    # container's true CPU budget; raise NUM_CPUS (or --cpus) to allow more.
    cpu_budget = effective_cpu_budget()
    if num_slots > cpu_budget:
        _print(
            f"Capping concurrency: {num_slots} -> {cpu_budget} slots "
            f"(CPU budget; raise NUM_CPUS to allow more)"
        )
        num_slots = cpu_budget
    num_slots = max(1, num_slots)

    if gpu_indices is None:
        gpu_indices = [g["index"] for g in gpus]

    gpu_by_idx = {g["index"]: g for g in gpus}
    gpu_states: dict[int, _GpuState] = {}
    for gi in gpu_indices:
        if gi not in gpu_by_idx:
            _print(
                f"ERROR: GPU{gi} not found "
                f"(available: {[g['index'] for g in gpus]})"
            )
            return 1
        total = gpu_by_idx[gi]["total_mib"] / 1024.0
        gpu_states[gi] = _GpuState(
            index=gi,
            total_gib=total,
            budget_multi=total * (1.0 - VRAM_MULTI_PROC_MARGIN),
        )

    tests: list[_TestEntry] = []
    for tid in test_ids:
        m = meta.get(tid, {})
        tests.append(
            _TestEntry(
                id=tid,
                name=tid,
                profiled_gib=m.get("profiled_vram_gib", max_vram_gib),
                requested_vllm_kv_cache_bytes=m.get("requested_vllm_kv_cache_bytes"),
                timeout=m.get("timeout", 600),
                requested_sglang_kv_tokens=m.get("requested_sglang_kv_tokens"),
                requested_sglang_vram_gib=m.get("requested_sglang_vram_gib"),
                requested_trtllm_kv_tokens=m.get("requested_trtllm_kv_tokens"),
                requested_trtllm_vram_gib=m.get("requested_trtllm_vram_gib"),
                skip_reason=m.get("skip_reason"),
            )
        )

    # Separate skip-marked tests — they won't actually run, so don't
    # validate KV markers or consume GPU budget.
    skipped_tests = [t for t in tests if t.skip_reason is not None]
    tests = [t for t in tests if t.skip_reason is None]

    # Scheduling priority (highest first):
    #   1. VRAM tests (profiled_gib > 0) before zero-VRAM fillers, so the
    #      memory-bound tests own the schedule; fillers only mop up spare slots
    #      (they never consume the GPU budget, so they must not crowd a VRAM
    #      test out of a concurrency slot).
    #   2. Longest est_duration (= timeout/3) first (LPT) to minimize makespan.
    #   3. Largest VRAM first, so a big test anchors an empty GPU at the full
    #      single-proc cap and smaller tests pack into the remaining budget
    #      alongside it instead of the big test running alone on the tail.
    tests.sort(key=_priority_key, reverse=True)

    # Reject tests without a KV marker — without explicit memory control
    # they'd each grab the engine's default (e.g. vLLM 90%) and OOM when
    # run concurrently. Tests with profiled_gib=0 are exempt (mock/CPU-only).
    no_kv = [
        t
        for t in tests
        if t.requested_vllm_kv_cache_bytes is None
        and t.requested_sglang_kv_tokens is None
        and t.requested_sglang_vram_gib is None
        and t.requested_trtllm_kv_tokens is None
        and t.requested_trtllm_vram_gib is None
        and t.profiled_gib > 0
    ]
    if no_kv:
        _print(
            f"\nERROR: {len(no_kv)} test(s) lack a requested_vllm_kv_cache_bytes, "
            f"requested_sglang_kv_tokens, requested_sglang_vram_gib, "
            f"requested_trtllm_kv_tokens, or requested_trtllm_vram_gib marker "
            f"and cannot run in parallel:"
        )
        for t in no_kv:
            _print(f"  {t.name}")
        _print("\nAdd the appropriate marker via profile_pytest.py, " "then rerun.")
        return 1

    # Identify tests in metadata that exceed the VRAM budget
    test_id_set = set(test_ids)
    over_budget = []
    for nodeid, m in meta.items():
        if nodeid not in test_id_set:
            profiled = m.get("profiled_vram_gib")
            if profiled is not None and profiled > max_vram_gib:
                over_budget.append((nodeid, profiled))

    # Assign permanent worker IDs (w0, w1, ...) to all tests including skipped
    all_tests = tests + skipped_tests
    for idx, test in enumerate(all_tests):
        test.w_id = idx

    os.makedirs(_JUNIT_DIR, exist_ok=True)
    # Children mkdir their nested --basetemp non-recursively, so the root
    # has to exist first.
    if parent_basetemp:
        os.makedirs(parent_basetemp, exist_ok=True)

    # --- Plan header ---
    n_run = len(tests)
    n_skip = len(skipped_tests)
    count_str = f"{n_run} tests"
    if n_skip:
        count_str += f", {n_skip} skipped"

    if len(gpu_states) == 1:
        gi = next(iter(gpu_states))
        gs = gpu_states[gi]
        _print(
            f"\nGPU parallel: {count_str}, {num_slots} concurrent slots, "
            f"GPU{gi} ({gs.total_gib:.0f} GiB, "
            f"{gs.budget_multi:.0f} GiB multi-proc budget)"
        )
    else:
        gpu_list = ",".join(str(gi) for gi in sorted(gpu_states))
        sizes = {int(gs.total_gib) for gs in gpu_states.values()}
        budgets = {int(gs.budget_multi) for gs in gpu_states.values()}
        if len(sizes) == 1 and len(budgets) == 1:
            size_str = (
                f"{next(iter(sizes))} GiB each, "
                f"{next(iter(budgets))} GiB multi-proc budget"
            )
        else:
            size_str = ", ".join(
                f"GPU{gi}: {gs.total_gib:.0f}/{gs.budget_multi:.0f} GiB"
                for gi, gs in sorted(gpu_states.items())
            )
        _print(
            f"\nGPU parallel: {count_str}, {num_slots} concurrent slots, "
            f"GPUs {gpu_list} ({size_str})"
        )

    _print()
    for test in tests:
        _print(
            f"[w{test.w_id}] {test.name}  "
            f"profiled={test.profiled_gib:.1f} GiB, "
            f"{_fmt_req(test)}, "
            f"timeout={int(test.timeout)}s"
        )
    if over_budget:
        _print()
        _print(
            f"Over budget ({len(over_budget)} -- profiled > max_vram_gib {max_vram_gib:.0f} GiB):"
        )
        for name, profiled in sorted(over_budget, key=lambda x: x[1], reverse=True):
            _print(f"  {name}  (profiled={profiled:.1f} GiB)")
    _print()

    # --- Report skip-marked tests immediately (like xdist SKIPPED) ---
    completed: list[_CompletedTest] = []
    for test in skipped_tests:
        _print(f"[w{test.w_id}] {test.name} SKIPPED" f" - {test.skip_reason}")
        completed.append(
            _CompletedTest(
                test=test,
                duration=0,
                passed=False,
                skipped=True,
                skip_reason=test.skip_reason,
            )
        )

    # --- Scheduling state ---
    t0 = time.monotonic()
    pending = list(tests)
    running: dict[int, _RunningTest] = {}
    next_status = t0 + 10
    # vLLM needs a stagger because --gpu-memory-utilization triggers a memory
    # profiling step that snapshots free memory — concurrent launches corrupt
    # each other's snapshots (bug #10643). SGLang uses --max-total-tokens
    # which is deterministic, so no stagger is needed.
    _VLLM_LAUNCH_STAGGER_S = 5.0
    last_vllm_launch: dict[int, float] = {}  # gpu_index -> monotonic timestamp

    def _build_status_lines(now: float) -> list[str]:
        """Build per-GPU status lines for periodic output."""
        elapsed = int(now - t0)
        lines = []
        for gi in sorted(gpu_states):
            gs = gpu_states[gi]
            actual = _get_gpu_used_gib(gi)
            workers = sorted(
                w for w, run_info in running.items() if run_info.test.assigned_gpu == gi
            )
            wstr = ", ".join(
                f"w{w}({int(now - running[w].start_time)}s)" for w in workers
            )
            part = f"GPU{gi}: {actual:.1f}/{gs.total_gib:.0f} GiB"
            if wstr:
                part += f" [{wstr}]"
            lines.append(f"[elapsed {elapsed}s] {part}")
        return lines

    def _launch_test(test: _TestEntry, env_base: dict) -> _RunningTest:
        """Build env, spawn subprocess, start output streamer thread."""
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(test.assigned_gpu)
        if test.requested_sglang_kv_tokens is not None:
            env["_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS"] = str(
                int(test.requested_sglang_kv_tokens)
            )
        elif test.requested_trtllm_kv_tokens is not None:
            env["_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS"] = str(
                int(test.requested_trtllm_kv_tokens)
            )
        elif test.requested_trtllm_vram_gib is not None:
            gib_to_bytes = int(test.requested_trtllm_vram_gib * 1024**3)
            env["_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES"] = str(gib_to_bytes)
        elif test.requested_vllm_kv_cache_bytes is not None:
            env["_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES"] = str(
                int(test.requested_vllm_kv_cache_bytes)
            )

        safe_name = test.name.replace("/", "_").replace("::", "__")
        # Give each child a unique COVERAGE_FILE so its session-end combine
        # (pytest-cov collapses data_suffix=True shards into the unsuffixed
        # path) doesn't clobber siblings. Harmless when pytest-cov is not
        # active — the env var is just unread.
        #
        # Key only on w_id, which is a permanent, globally-unique per-test index
        # (assigned via enumerate() above), so it already guarantees uniqueness.
        # Do NOT append the test node-id: pytest-cov adds its own
        # ".<hostname>.<pid>.<random>" parallel suffix (plus a temp suffix during
        # the atomic save), so a long node-id here can push the filename past the
        # filesystem's 255-byte NAME_MAX and crash the run with
        # "OSError: [Errno 36] File name too long". The w_id->test mapping stays
        # recoverable from the orchestrator log and the per-test JUnit filenames.
        parent_cov_file = env.get("COVERAGE_FILE")
        if parent_cov_file:
            env["COVERAGE_FILE"] = f"{parent_cov_file}.w{test.w_id}"
        junit_path = _junit_path(test.name)
        has_tb = extra_pytest_args and any(
            a.startswith("--tb") for a in extra_pytest_args
        )
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test.id,
            "-x",
            *([] if has_tb else ["--tb=short"]),
            f"--timeout={int(test.timeout)}",
            f"--junitxml={junit_path}",
        ]
        if extra_pytest_args:
            cmd.extend(extra_pytest_args)
        # Give each child a unique --basetemp under the parent's root.
        # pytest rmtrees the given basetemp at session startup, so a shared
        # root would let siblings wipe each other's tmp_path trees mid-run.
        # Prefix with w{w_id} because safe_name normalizes "/" and "::" and
        # is not guaranteed collision-free across parametrized ids.
        # Appended after extra_pytest_args so the orchestrator's per-child
        # path wins over any --basetemp that a caller stuffs into
        # extra_pytest_args (pytest uses argparse semantics: last value wins).
        if parent_basetemp:
            child_basetemp = os.path.join(parent_basetemp, f"w{test.w_id}-{safe_name}")
            cmd.extend(["--basetemp", child_basetemp])

        # start_new_session makes the child a process-group leader, so the
        # watchdog can signal it and everything it spawned with one killpg --
        # including descendants already reparented to init, which a ppid walk
        # cannot see. It also detaches the child from the orchestrator's
        # terminal, so a Ctrl-C no longer reaches it; the try/finally around the
        # scheduling loop is what cleans up in that case.
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        started = time.monotonic()
        if test.first_start_time is None:
            test.first_start_time = started
        run_info = _RunningTest(proc=proc, test=test, start_time=started, pgid=proc.pid)
        w_id = test.w_id
        stream_prefix = f"[w{w_id}]" if stream else None
        t = threading.Thread(
            target=_capture_output,
            args=(proc.stdout, run_info.captured, stream_prefix),
            daemon=True,
        )
        t.start()
        run_info.reader_thread = t
        return run_info

    def _close_child_pipe(run_info: _RunningTest) -> None:
        """Release the read end once the reader thread is done with it.

        _capture_output only returns at EOF, which needs every holder of the
        inherited write end to exit -- so after a kill that leaves a survivor,
        the join times out and the thread never runs its own close(). Closing
        here keeps a long run from leaking one fd per kill.
        """
        if run_info.reader_thread is not None:
            run_info.reader_thread.join(timeout=5)
        try:
            if run_info.proc.stdout is not None:
                run_info.proc.stdout.close()
        except OSError:
            pass

    def _record_unreaped(run_info: _RunningTest, now: float) -> None:
        """Book a child that outlived its kills.

        The normal completion path cannot: it is keyed on Popen.poll(), which
        never returns for a process stuck in uninterruptible sleep.
        """
        test = run_info.test
        duration = now - run_info.start_time
        reason = run_info.watchdog_reason or "killed by the orchestrator"
        _close_child_pipe(run_info)
        if not stream:
            for line in list(run_info.captured):
                _print(f"[w{test.w_id}] {line}")
        _print(f"[w{test.w_id}] {test.name} FAILED [{duration:.0f}s]")
        if not os.path.exists(_junit_path(test.name)):
            _write_watchdog_junit(test, duration, reason)
        completed.append(
            _CompletedTest(
                test=test, duration=duration, passed=False, fail_reason=reason
            )
        )

    env_base = os.environ.copy()
    watchdog_attempts = _watchdog_attempts()
    job_budget = _job_budget_s()
    step_deadline = None if job_budget is None else t0 + job_budget
    # Set when the run cannot continue: a child would not die, so its VRAM is
    # never coming back. Every job that runs this stage has a single GPU, so
    # there is nowhere else to put the remaining tests.
    abort_reason: str | None = None

    def _latest_start(now: float) -> float | None:
        """Last moment a test may be launched and still have time to finish."""
        if step_deadline is None:
            return None
        return step_deadline - _WATCHDOG_REPORT_RESERVE_S

    try:
        while pending or running:
            now = time.monotonic()

            # Kill anything past its deadline; the completion check below reaps it
            # and frees its GPU budget for the queued tests. Signals are sent
            # without waiting -- a blocking wait here would stall every other
            # child's reaping, and reaping through psutil would make Popen.poll()
            # report a false success for the test we just killed.
            for w_id, run_info in list(running.items()):
                if run_info.proc.poll() is not None:
                    continue
                test = run_info.test
                kill_at = _watchdog_kill_at(
                    run_info.start_time, test.timeout, watchdog_attempts, step_deadline
                )
                if now <= kill_at:
                    continue
                if run_info.watchdog_reason is None:
                    # Re-poll immediately before committing: a child that exits
                    # between the check above and here has genuinely finished, and
                    # recording a reason would force its result to FAILED and
                    # overwrite the JUnit report it just wrote.
                    if run_info.proc.poll() is not None:
                        continue
                    elapsed = now - run_info.start_time
                    # Ask the same function what the deadline would have been
                    # with no job wall: if the real one is earlier, the job
                    # budget is what bound it. Recomputing the window inline
                    # here would be a second copy of that arithmetic to keep in
                    # step with the first.
                    own_kill_at = _watchdog_kill_at(
                        run_info.start_time, test.timeout, watchdog_attempts, None
                    )
                    if step_deadline is not None and kill_at < own_kill_at:
                        limit_note = (
                            f"ran {elapsed:.0f}s and the job step runs out of time in "
                            f"{step_deadline - now:.0f}s"
                        )
                        run_info.watchdog_reason = (
                            f"killed by the orchestrator: the job step ran out of time "
                            f"({limit_note}). Its {test.timeout:.0f}s timeout does not "
                            f"fit the remaining budget — raise gpu_test_timeout_minutes "
                            f"or lower the test's timeout"
                        )
                    else:
                        limit_note = (
                            f"ran {elapsed:.0f}s against a "
                            f"{kill_at - run_info.start_time:.0f}s limit, from a "
                            f"{test.timeout:.0f}s test timeout x{watchdog_attempts} "
                            f"attempts"
                        )
                        run_info.watchdog_reason = (
                            f"killed by the orchestrator: hit its time limit "
                            f"({limit_note}) and its own "
                            f"{test.timeout:.0f}s timeout did not stop it"
                        )
                    _print(
                        f"[watchdog] w{w_id} {limit_note} — killing the test process "
                        f"and everything it started."
                    )
                # Escalate on a clock, not a retry count. Re-sending a signal to
                # the same process is a no-op -- SIGKILL cannot be caught and the
                # kernel already holds it pending -- so what these stages buy is
                # a teardown window, and then a bound on how long to wait.
                if run_info.kill_started_at is None:
                    run_info.kill_started_at = now
                    _signal_process_tree(
                        run_info.proc.pid, run_info.pgid, signal.SIGTERM
                    )
                    continue
                since_kill = now - run_info.kill_started_at
                if since_kill >= _WATCHDOG_GIVE_UP_S:
                    # SIGKILL cannot be caught, so a tree still alive now is in
                    # uninterruptible sleep and is never going to exit. Its VRAM
                    # is gone for the rest of the job and this stage runs on a
                    # single GPU, so there is nothing left to run anything on.
                    # Stop, and say why -- spinning here is the stall this
                    # watchdog exists to prevent.
                    _print(
                        f"[watchdog] w{w_id} still alive {since_kill:.0f}s after "
                        f"SIGKILL, so it is stuck where signals cannot reach it. "
                        f"Its GPU memory is not coming back — stopping the run."
                    )
                    _record_unreaped(run_info, now)
                    del running[w_id]
                    abort_reason = (
                        f"w{w_id} could not be killed and still holds the GPU"
                    )
                    break
                elif since_kill >= _WATCHDOG_TERM_GRACE_S:
                    # Past the teardown window. Re-signalled every pass, which
                    # costs nothing and re-walks the tree: _signal_process_tree
                    # snapshots descendants before signalling, so a grandchild
                    # spawned or reparented since the last pass is only caught by
                    # taking a fresh snapshot.
                    _signal_process_tree(
                        run_info.proc.pid, run_info.pgid, signal.SIGKILL
                    )

            if abort_reason:
                break

            # Check for completed subprocesses
            for w_id in list(running.keys()):
                run_info = running[w_id]
                rc = run_info.proc.poll()
                if rc is not None:
                    _close_child_pipe(run_info)
                    # Re-sampled rather than reusing the loop-top `now`: the kills
                    # above and the pipe join here each take real time, so by the
                    # later entries in this pass that value is already stale -- and
                    # it feeds the reported duration, the sequential-time summary,
                    # the JUnit time= attribute and the status-line schedule.
                    now = time.monotonic()
                    duration = now - run_info.start_time
                    # Do not simplify to `rc == 0`. A watchdog kill can still leave
                    # rc at 0 -- anything that reaps the child behind Popen's back
                    # makes poll() report a false success -- so without the second
                    # term every killed test would be logged as PASSED, which is the
                    # exact failure this watchdog exists to make visible.
                    passed = rc == 0 and run_info.watchdog_reason is None
                    test = run_info.test
                    gi = test.assigned_gpu

                    # Detect retryable init errors (profiling race, OOM at startup)
                    # A stuck test is not a transient startup failure, so never
                    # relaunch one even if its output carries a retryable marker.
                    #
                    # The elapsed check bounds the retries as a whole: each relaunch
                    # takes a fresh start_time, so a test that dies just under its
                    # deadline every time would otherwise get a brand-new full
                    # budget on each of its attempts and could burn the entire step
                    # -- the stall this watchdog exists to prevent -- while the
                    # watchdog itself never fires, because its clock keeps
                    # restarting.
                    first_start = test.first_start_time or run_info.start_time
                    total_elapsed = now - first_start
                    retry_budget = test.timeout * watchdog_attempts + _WATCHDOG_GRACE_S
                    room_in_step = (
                        step_deadline is None
                        or now + test.timeout
                        < step_deadline - _WATCHDOG_REPORT_RESERVE_S
                    )
                    if (
                        not passed
                        and run_info.watchdog_reason is None
                        and test.retries < _MAX_RETRIES
                        and total_elapsed < retry_budget
                        and room_in_step
                    ):
                        matched_marker = None
                        for line in run_info.captured:
                            for marker in _RETRYABLE_INIT_MARKERS:
                                if marker in line:
                                    matched_marker = marker
                                    break
                            if matched_marker:
                                break
                        if matched_marker:
                            test.retries += 1
                            _print(
                                f"[w{w_id}] retrying ({test.retries}/{_MAX_RETRIES})"
                                f" — {matched_marker}"
                            )
                            if gi is not None:
                                gpu_states[gi].budget_used -= test.profiled_gib
                                gpu_states[gi].running_count -= 1
                            del running[w_id]
                            test.assigned_gpu = None
                            pending.insert(0, test)
                            continue

                    # Detect runtime skips via JUnit XML (subprocess exit 0
                    # covers both "all passed" and "all skipped").
                    skipped = False
                    skip_reason: str | None = None
                    if passed:
                        junit_path = _junit_path(test.name)
                        skip_reason = _parse_junit_skipped(junit_path)
                        if skip_reason is not None:
                            passed = False
                            skipped = True

                    # Dump buffered output on failure only (matches pytest behavior).
                    # With -s, output was already streamed live.
                    fail_reason = ""
                    if not passed and not skipped:
                        if not stream:
                            prefix = f"[w{w_id}]"
                            for line in run_info.captured:
                                _print(f"{prefix} {line}")
                        for line in reversed(run_info.captured):
                            stripped = line.strip()
                            if stripped and not stripped.startswith("="):
                                fail_reason = stripped
                                break
                        # Stated outright; after a SIGKILL the child's output is
                        # only whatever happened to flush.
                        if run_info.watchdog_reason is not None:
                            fail_reason = run_info.watchdog_reason
                            # Only synthesize what the child never wrote. pytest
                            # emits its --junitxml from pytest_sessionfinish, so a
                            # wedge in post-session teardown -- atexit handlers,
                            # CUDA/NCCL teardown, a lingering non-daemon thread,
                            # exactly what pytest-timeout cannot see -- leaves a
                            # real report on disk that this would otherwise clobber.
                            if not os.path.exists(_junit_path(test.name)):
                                _write_watchdog_junit(test, duration, fail_reason)

                    if skipped:
                        status = "SKIPPED"
                    elif passed:
                        status = "PASSED"
                    else:
                        status = "FAILED"

                    if skipped:
                        _print(f"[w{w_id}] {test.name} SKIPPED" f" - {skip_reason}")
                    else:
                        _print(f"[w{w_id}] {test.name} {status} [{duration:.0f}s]")

                    if gi is not None:
                        gpu_states[gi].budget_used -= test.profiled_gib
                        gpu_states[gi].running_count -= 1
                    completed.append(
                        _CompletedTest(
                            test=test,
                            duration=duration,
                            passed=passed,
                            skipped=skipped,
                            skip_reason=skip_reason,
                            fail_reason=fail_reason,
                        )
                    )
                    del running[w_id]

                    # Print status immediately after completion
                    lines = _build_status_lines(now)
                    if pending:
                        queued_str = ", ".join(f"w{t.w_id}" for t in pending)
                        lines[-1] += f" [queued: {queued_str}]"
                    for ln in lines:
                        _print(ln)
                    next_status = now + 10

            # --- Launch pending tests ---
            # _select_launches packs VRAM tests up to budget (pairing a big test
            # with smaller ones), backfills spare slots with zero-VRAM fillers, and
            # reserves space for a blocked high-priority test so it can't be starved
            # onto the tail. The vLLM stagger below is per-GPU only — tests on
            # different GPUs launch simultaneously.
            # Only offer the scheduler tests the job still has time to finish.
            # Launching one it does not would put the child past its watchdog
            # deadline on the first pass: killed within seconds and reported
            # FAILED without having run.
            latest_start = _latest_start(now)
            launchable = (
                pending
                if latest_start is None
                else [t for t in pending if now + t.timeout <= latest_start]
            )
            if not launchable and not running:
                # Nothing left that can be started and nothing running to change
                # that, so the remaining tests are reported below rather than
                # waited on.
                break

            if launchable and len(running) < num_slots:
                actual_free = {
                    gi: gs.total_gib - _get_gpu_used_gib(gi)
                    for gi, gs in gpu_states.items()
                }
                to_launch = _select_launches(
                    pending=launchable,
                    gpu_states=gpu_states,
                    actual_free=actual_free,
                    num_slots=num_slots,
                    running_count=len(running),
                )

                # Indices address `launchable`, so take the entries by identity
                # and drop them from `pending` separately. Highest priority
                # first, which is the order _select_launches returns.
                batch: list[_TestEntry] = []
                for launch_idx, assigned_gpu in to_launch:
                    entry = launchable[launch_idx]
                    entry.assigned_gpu = assigned_gpu
                    pending.remove(entry)
                    batch.append(entry)

                for entry in batch:
                    w_id = entry.w_id
                    gi = entry.assigned_gpu
                    assert gi is not None
                    is_vllm = (
                        entry.requested_vllm_kv_cache_bytes is not None
                        and entry.profiled_gib > 0
                    )

                    # Per-GPU vLLM stagger — only between vLLM tests on the
                    # same GPU.  Tests on different GPUs launch simultaneously.
                    if is_vllm:
                        last_t = last_vllm_launch.get(gi, 0)
                        wait = _VLLM_LAUNCH_STAGGER_S - (time.monotonic() - last_t)
                        if wait > 0:
                            time.sleep(wait)

                    gpu_states[gi].budget_used += entry.profiled_gib
                    gpu_states[gi].running_count += 1
                    run_info = _launch_test(entry, env_base)
                    running[w_id] = run_info

                    if is_vllm:
                        last_vllm_launch[gi] = time.monotonic()

                    retry_str = f" (retry {entry.retries})" if entry.retries else ""
                    _print(
                        f"[w{w_id}] {entry.name} "
                        f"(GPU{gi}, profiled={entry.profiled_gib:.1f} GiB, "
                        f"{_fmt_req(entry)}) RUNNING{retry_str}"
                    )

                    now = time.monotonic()
                    if now >= next_status and (running or pending):
                        lines = _build_status_lines(now)
                        if pending:
                            queued_str = ", ".join(f"w{t.w_id}" for t in pending)
                            lines[-1] += f" [queued: {queued_str}]"
                        for ln in lines:
                            _print(ln)
                        next_status = now + 10

            # Periodic status (print even when waiting for VRAM to free up)
            if now >= next_status and (running or pending):
                lines = _build_status_lines(now)
                if pending:
                    queued_str = ", ".join(f"w{t.w_id}" for t in pending)
                    if not running:
                        next_needed = pending[0].profiled_gib
                        lines[-1] += f" [waiting for {next_needed:.1f} GiB free]"
                    lines[-1] += f" [queued: {queued_str}]"
                for ln in lines:
                    _print(ln)
                next_status = now + 10

            if running or pending:
                time.sleep(1.0)

        # Anything still queued when the loop ends could not be started: the
        # job ran out of time for it, or a child would not die and took the GPU
        # with it. Reported apart from failures, since these did not fail, but
        # still counted against the run -- a suite that did not finish must not
        # come out green.
        for entry in pending:
            reason = abort_reason or (
                f"the job step had less time left than its own "
                f"{entry.timeout:.0f}s timeout needs"
            )
            _print(f"[w{entry.w_id}] {entry.name} NOT RUN - {reason}")
            completed.append(
                _CompletedTest(
                    test=entry,
                    duration=0,
                    passed=False,
                    not_run=True,
                    fail_reason=reason,
                )
            )
        pending.clear()
    finally:
        # Nothing above is allowed to leave a child behind holding VRAM. This
        # covers an unhandled error escaping the loop and, because
        # start_new_session detaches children from the orchestrator's terminal,
        # it is also what stops a Ctrl-C from stranding a live engine on the GPU.
        for w_id, run_info in running.items():
            if run_info.proc.poll() is None or _process_group_alive(run_info.pgid):
                _print(f"[watchdog] cleaning up w{w_id} before exit")
                _signal_process_tree(run_info.proc.pid, run_info.pgid, signal.SIGKILL)

    # Summary
    wall_time = time.monotonic() - t0
    sequential_time = sum(c.duration for c in completed if not c.skipped)
    n_passed = sum(1 for c in completed if c.passed)
    n_skipped = sum(1 for c in completed if c.skipped)
    n_not_run = sum(1 for c in completed if c.not_run)
    n_failed = sum(
        1 for c in completed if not c.passed and not c.skipped and not c.not_run
    )

    completed.sort(key=lambda c: c.test.w_id)

    _print()
    _print(f"{'=' * 27} short test summary info {'=' * 27}")
    for c in completed:
        test = c.test
        w_id = test.w_id
        if c.not_run:
            _print(f"NOT RUN [w{w_id}] {test.name} - {c.fail_reason}")
        elif c.skipped:
            reason = c.skip_reason or "skipped"
            _print(f"SKIPPED [w{w_id}] {test.name} - {reason}")
        elif c.passed:
            duration = int(c.duration)
            timeout = int(test.timeout)
            retries = test.retries
            retry_str = f" ({retries} retries)" if retries else ""
            _print(
                f"PASSED [w{w_id}] {test.name} " f"[{duration}s/{timeout}s]{retry_str}"
            )
        else:
            duration = int(c.duration)
            timeout = int(test.timeout)
            retries = test.retries
            retry_str = f" ({retries} retries)" if retries else ""
            fail_str = f" - {c.fail_reason}" if c.fail_reason else ""
            _print(
                f"FAILED [w{w_id}] {test.name} "
                f"[{duration}s/{timeout}s]{retry_str}{fail_str}"
            )

    n_summary_parts = []
    if n_failed:
        n_summary_parts.append(f"{n_failed} failed")
    if n_not_run:
        n_summary_parts.append(f"{n_not_run} not run")
    n_summary_parts.append(f"{n_passed} passed")
    if n_skipped:
        n_summary_parts.append(f"{n_skipped} skipped")

    wall_int = int(wall_time)
    h, remainder = divmod(wall_int, 3600)
    m, s = divmod(remainder, 60)
    time_str = f"{wall_time:.2f}s"
    if h:
        time_str += f" ({h}:{m:02d}:{s:02d})"
    elif m:
        time_str += f" ({m:01d}:{s:02d})"

    summary = ", ".join(n_summary_parts) + f" in {time_str}"
    if n_passed > 1 and sequential_time > 0:
        speedup = sequential_time / wall_time
        summary += f" (vs {sequential_time:.0f}s seq, {speedup:.1f}x)"

    pad = max(0, (78 - len(summary) - 2) // 2)
    _print(f"{'=' * pad} {summary} {'=' * pad}")

    if abort_reason:
        _print(f"WARNING: run stopped early — {abort_reason}.")

    combined = _aggregate_junit_xml(_JUNIT_DIR)
    if combined:
        _print(f"JUnit XML: {combined}")

    # Tests the job never got to are counted against the run as well: a suite
    # that did not finish must not report green just because nothing it managed
    # to run happened to fail.
    return 0 if n_failed == 0 and n_not_run == 0 else 1


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GPU tests in parallel with VRAM-aware scheduling.",
        usage="%(prog)s --max-vram-gib=N [-n SLOTS] [pytest-args...]",
    )
    parser.add_argument(
        "--max-vram-gib",
        type=float,
        required=True,
        help="Only run tests with profiled_vram_gib <= N.",
    )
    parser.add_argument(
        "-n",
        type=str,
        default="auto",
        help="Number of concurrent slots. 'auto' = gpu_usable / max_vram_gib.",
    )

    raw = sys.argv[1:]
    if "--" in raw:
        split = raw.index("--")
        args = parser.parse_args(raw[:split])
        pytest_args = raw[split + 1 :]
    else:
        args, pytest_args = parser.parse_known_args(raw)

    if not pytest_args:
        parser.error("No pytest arguments provided")

    is_stream = any(a in ("-s", "--capture=no") or "-s" in a for a in pytest_args)

    gpus = detect_gpus()
    if not gpus:
        _print("ERROR: No GPUs detected")
        return 1

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_indices = _parse_cuda_visible(cvd, gpus)
    if not gpu_indices:
        _print("ERROR: CUDA_VISIBLE_DEVICES hides all GPUs")
        return 1

    _print(f"Collecting tests with --max-vram-gib={args.max_vram_gib}...")
    test_ids = _collect_tests(pytest_args, args.max_vram_gib)
    if not test_ids:
        _print("No tests collected.")
        return 0

    meta = load_test_meta()

    if args.n == "auto":
        profiled_gibs = [
            meta.get(tid, {}).get("profiled_vram_gib", args.max_vram_gib)
            for tid in test_ids
        ]
        selected_gpus = [g for g in gpus if g["index"] in gpu_indices]
        num_slots = auto_worker_count(selected_gpus, args.max_vram_gib, profiled_gibs)
    else:
        num_slots = int(args.n)

    return run_parallel(
        test_ids=test_ids,
        meta=meta,
        max_vram_gib=args.max_vram_gib,
        num_slots=num_slots,
        gpu_indices=gpu_indices,
        stream=is_stream,
    )


if __name__ == "__main__":
    sys.exit(main())
