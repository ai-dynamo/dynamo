#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-parallel test runner (used by conftest.py, not invoked directly).

Runs pytest tests as independent subprocesses with VRAM-aware scheduling.
Each test gets CUDA_VISIBLE_DEVICES and KV cache overrides
(_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES / _PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS)
so the engine allocates only its declared VRAM budget.

A test needing several GPUs (``gpu_2`` and friends) is scheduled as a gang: it
is given that many distinct devices atomically or none at all, and
``profiled_vram_gib`` -- the maximum per-device peak -- is reserved on every one
of them. CUDA_VISIBLE_DEVICES then lists the whole gang in ascending order.

Usage (always via pytest):
    pytest --max-vram-gib=6 -n auto -m "gpu_1 and vllm" tests/serve/
    pytest --max-vram-gib=6 -n 4 -sv -m "gpu_1 and vllm" tests/serve/
    pytest --max-vram-gib=80 -n auto -m "vllm and (gpu_1 or gpu_2)" tests/serve/

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
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from tests.utils.vram_utils import (  # noqa: E402
    DEFAULT_GPU_COUNT,
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
    gpu_count: int = 1
    w_id: int = 0
    # The GPUs this test currently holds, ascending. Empty == holds none.
    # A multi-GPU test owns either all `gpu_count` of them or none: the
    # scheduler never assigns a partial gang.
    assigned_gpus: tuple[int, ...] = ()
    retries: int = 0

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
    captured: list[str] = field(default_factory=list)
    reader_thread: threading.Thread | None = None


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


def _fmt_gpus(gpus: tuple[int, ...]) -> str:
    """Render an assigned gang for the log: ``GPU3`` or ``GPUs 0,1``."""
    if len(gpus) == 1:
        return f"GPU{gpus[0]}"
    return "GPUs " + ",".join(str(gi) for gi in gpus)


_JUNIT_DIR = os.path.join(tempfile.gettempdir(), "gpu_parallel_junit")
_JUNIT_COMBINED = os.path.join(_JUNIT_DIR, "combined.xml")


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

# vLLM launches are staggered for two independent reasons.
#
# The primary one is scheduler-level and holds no matter how the engine is
# configured: profiled_vram_gib is a *peak*, so during a launch's allocation
# ramp the live NVML reading this module admits against has not settled yet.
# Launching two engines concurrently lets an admission decision be taken
# against a card that is still filling. Capping the KV pool does not remove
# this; it is a property of observing an allocation in flight.
#
# The secondary one is engine-internal: when --gpu-memory-utilization drives
# vLLM's own memory-profiling step, concurrent launches corrupt each other's
# snapshots (bug #10643). Pinning --kv-cache-memory-bytes removes that second
# effect, and only that one.
#
# SGLang uses --max-total-tokens which is deterministic, so no stagger is
# needed.
_VLLM_LAUNCH_STAGGER_S = 5.0


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


def _status_lines(
    now: float,
    t0: float,
    gpu_states: dict[int, _GpuState],
    running: dict[int, _RunningTest],
    gpu_used_gib: Callable[[int], float],
) -> list[str]:
    """Per-GPU status lines for periodic output.

    A test is listed under every GPU of its gang, so a multi-GPU worker shows
    up on each device it actually occupies instead of appearing to be a
    single-GPU test parked on one of them.

    Pure apart from ``gpu_used_gib``, which is injected so the caller decides
    whether that means NVML or a stub.
    """
    elapsed = int(now - t0)
    lines = []
    for gi in sorted(gpu_states):
        gs = gpu_states[gi]
        actual = gpu_used_gib(gi)
        workers = sorted(
            w for w, run_info in running.items() if gi in run_info.test.assigned_gpus
        )
        wstr = ", ".join(f"w{w}({int(now - running[w].start_time)}s)" for w in workers)
        part = f"GPU{gi}: {actual:.1f}/{gs.total_gib:.0f} GiB"
        if wstr:
            part += f" [{wstr}]"
        lines.append(f"[elapsed {elapsed}s] {part}")
    return lines


def _reserve_gpus(
    test: _TestEntry, gpus: tuple[int, ...], gpu_states: dict[int, _GpuState]
) -> None:
    """Commit a test onto its gang, charging every device it will occupy.

    ``profiled_vram_gib`` is the maximum per-device peak, so the same figure is
    reserved on each member, and each member counts one more resident process
    (the multi-process margin is about CUDA contexts, and a gang puts one
    context on every GPU it touches).
    """
    test.assigned_gpus = tuple(sorted(gpus))
    for gi in test.assigned_gpus:
        gpu_states[gi].budget_used += test.profiled_gib
        gpu_states[gi].running_count += 1


def _release_gpus(test: _TestEntry, gpu_states: dict[int, _GpuState]) -> None:
    """Return every device a test holds, and record that it holds none.

    Idempotent by construction: the second call sees an empty gang and does
    nothing. Every terminal path -- pass, fail, runtime skip, retry -- goes
    through here, so a reservation is returned exactly once and no count can be
    driven negative by a double release.
    """
    for gi in test.assigned_gpus:
        gs = gpu_states[gi]
        gs.budget_used -= test.profiled_gib
        gs.running_count -= 1
    test.assigned_gpus = ()


def _unschedulable_reason(
    test: _TestEntry, gpu_states: dict[int, _GpuState]
) -> str | None:
    """Why this test can never run here, or None if some future state fits it.

    Pure, and deliberately independent of live usage: it asks only whether the
    *hardware* could ever satisfy the request once every other test has
    finished. That is the state the scheduler reaches whenever nothing is
    running, so a test that fails this check would otherwise sit in ``pending``
    forever while the run loop waits for memory that is never coming.
    """
    n_gpus = len(gpu_states)
    if test.gpu_count > n_gpus:
        return f"needs {test.gpu_count} GPUs, only {n_gpus} available"
    if test.profiled_gib > 0:
        big_enough = [
            gi for gi, gs in gpu_states.items() if gs.total_gib >= test.profiled_gib
        ]
        if len(big_enough) < test.gpu_count:
            return (
                f"needs {test.gpu_count} GPU(s) of >= {test.profiled_gib:.1f} GiB, "
                f"only {len(big_enough)} of {n_gpus} that large"
            )
    return None


def _select_launches(
    pending: list[_TestEntry],
    gpu_states: dict[int, _GpuState],
    actual_free: dict[int, float],
    num_slots: int,
    running_count: int,
) -> list[tuple[int, tuple[int, ...]]]:
    """Pick which pending tests to launch this pass, and on which GPUs.

    Pure (no NVML / no subprocesses): the caller passes the live per-GPU budget
    state and the actual free VRAM (from nvidia-smi). ``pending`` must already be
    in scheduling-priority order (VRAM tests by longest est_duration / largest
    VRAM first, zero-VRAM fillers last -- see the sort in ``run_parallel``).

    Returns a list of ``(pending_index, assigned_gpus)`` to launch now, where
    ``assigned_gpus`` is the complete ascending device set for that test -- one
    GPU for a ``gpu_1`` test, ``gpu_count`` distinct GPUs for a gang. Honors:

      * ``num_slots`` -- global cap on concurrently running subprocesses. A gang
        is one subprocess and so costs one slot however many GPUs it holds.
      * Per-GPU VRAM budget with two independent gates (same as before): a test
        fits only if BOTH the reserved-budget sum AND the actual nvidia-smi
        usage leave room under the cap. The cap is the full card for the first
        test on an idle GPU, then ``budget_multi`` once it hosts 2+ (reserving
        the multi-process margin for CUDA context overhead). For a gang, every
        member must clear both gates on its own -- ``profiled_vram_gib`` is the
        maximum per-device peak, so the same figure is reserved on each device.
      * Atomicity -- a gang is placed only when all ``gpu_count`` devices clear
        every gate, and then all of them are committed together. There is no
        partial allocation.
      * Pairing -- best-fit places each VRAM test on the GPU(s) with the most
        free budget, so a large test that anchored an empty GPU gets backfilled
        with smaller tests up to the budget instead of running alone.
      * Anti-starvation -- when a VRAM test does not fit, the GPUs where it is
        closest to fitting are *reserved* for it, and lower-priority tests are
        limited in how far they may backfill those GPUs. The limit differs by
        kind, because the two kinds need different guarantees:

          - ``gpu_1`` (unchanged from upstream): backfill added *this pass* is
            capped at ``cap - required``. Rebuilt on every call, so it bounds
            backfill within a pass rather than across passes. The honest
            property is finite-workload conditional progress, not starvation
            freedom -- a single-GPU test only ever needs one card to come free
            and any card will do, so the creep is survivable in practice.
          - gangs: the *absolute* committed budget is capped at
            ``budget_multi - required``. A gang needs headroom on ``gpu_count``
            cards at the same instant, and a per-pass cap measured against
            ``_cap`` cannot deliver that -- see the gate itself for why each
            half matters. This makes per-card headroom monotone, which is what
            buys the gang a finite wait.

        A gang reserves all ``gpu_count`` devices or none -- headroom held on
        fewer devices than it needs could never assemble into a launch.
        Zero-VRAM fillers bypass the budget gates entirely (they allocate no
        memory) so transient memory pressure can't strand an otherwise-free
        slot; a zero-VRAM gang still takes its full count of distinct devices,
        because the count is a visibility requirement, not a memory one. Note
        that a filler still raises ``running_count``, which drops a card's cap
        to ``budget_multi``; a test profiled above ``budget_multi`` can
        therefore still be held off by fillers. That is pre-existing upstream
        behaviour for ``gpu_1`` and is not addressed here.

    Gang progress, conditional. Clearing ``_unschedulable_reason`` is NOT
    sufficient -- that check sees only hardware, and an earlier revision of this
    code claimed finite-time progress from it alone, which was false. A gang
    ``J`` needing ``r`` GiB on each of ``k`` devices launches in finite time when
    all of the following hold for as long as it is queued:

      1. at least ``k`` devices have ``r <= budget_multi``, so ``J`` can share a
         card rather than needing one to itself;
      2. at least ``k`` of those are not blocked by memory held outside this
         run; holds prefer the cards that are, so foreign memory delays ``J``
         only while it leaves fewer than ``k`` usable, and never merely because
         it sits on a card ``J`` had no need of;
      3. the competing work is VRAM-bearing -- a zero-VRAM filler bypasses the
         budget gates yet still raises ``running_count``, which drops ``_cap``
         from ``total_gib`` to ``budget_multi`` and can hold off a gang whose
         ``r`` exceeds ``budget_multi - foreign``. That threshold, not
         ``budget_multi`` alone, is the real one: a gang profiled strictly
         *below* ``budget_multi`` is blocked too once foreign memory takes up
         the difference. Nor is the block bounded by one filler's lifetime --
         fillers are placed by least load and a drained held card is
         systematically the least loaded, so the occupancy renews. This
         limitation is pre-existing and shared with the single-GPU path, which
         the paragraph below declines to address for the same reason;
      4. ``J`` outranks that competing work under ``_priority_key``; a test that
         is simply lowest-priority against an unbounded stream of higher-priority
         arrivals is not being starved by backfill, and no reservation applies;
      5. every test this run launches terminates.

    Given those: each held member's committed budget can only fall to
    ``budget_multi - foreign - required`` and never rise back above it -- net of
    memory held outside the run, because that is the quantity ``J``'s own gate
    is denominated in, and a hold bounding anything else bounds nothing. So any
    device admitting backfill under a hold is left able to take a member
    immediately.

    That statement is exact only once every launched test has allocated what it
    was profiled at. ``profiled_vram_gib`` is a *peak*, so during a launch's
    allocation window the live free reading is higher than the settled one,
    ``_foreign_held`` reads correspondingly lower, and the line sits above its
    settled value -- admitting a filler the settled line would have refused. The
    ramp is self-limiting rather than unsound: it loosens ``J``'s own
    actual-usage gate by the same amount at the same instant, and searches
    across ramp models found no starvation attributable to it. Every observer in
    this repo models allocation as instantaneous, so none of them can see the
    window; ``_VLLM_LAUNCH_STAGGER_S`` exists because it is real.

    The foreign term is subtracted only on devices ``J`` could occupy *today*.
    On a device it could not, subtracting foreign as well would make the line
    ``budget_multi - foreign - required < budget_multi - total_gib < 0`` for
    every test of every size -- B1, a whole-node freeze at ``n == gpu_count``.
    The unusable-card line therefore drops the foreign term but still charges
    the multi-process margin ``total_gib - budget_multi``, so admitted backfill
    cannot forfeit the cap drop that the first tenant causes. Residual foreign
    above that margin is a reachable state; the progress claim does not cover it.

    That widening is scoped to condition 1, ``r <= budget_multi``. For a gang
    profiled above ``budget_multi`` the line is ``budget_multi - r < 0`` even on
    a device it could take today, and every backfill test is still refused --
    correctly, and necessarily: admitting one drives ``running_count`` to 1,
    which drops ``_cap`` from ``total_gib`` to ``budget_multi``, and a member
    needing more than ``budget_multi`` can then never fit on that device at all.
    Such a gang runs only by having a device to itself. The pre-flight admits
    it (it checks physical size, not the multi-process budget), so the state is
    reachable; it is outside the progress theorem by condition 1, and outside
    the work-conservation claim for the same reason.

    Holds are placed only on devices ``J`` could actually occupy,
    the pre-hold occupants all finish, and ``J`` is scanned before every
    lower-priority test -- so the ``k`` members eventually satisfy the gate
    simultaneously.

    Conditions 1-4 are properties of the workload and the hardware, not of this
    scheduler. Nothing here assumes memory held outside the run ever frees: if it
    leaves fewer than ``k`` devices usable, condition 2 fails and ``J`` waits,
    which is correct because it cannot run. It waits holding the devices that
    were usable, so it launches on the pass the neighbour releases rather than
    re-entering the queue behind the backfill that accumulated meanwhile.
    """
    tentative = {
        gi: _TentativeGpu(
            budget=gs.budget_used,
            free=actual_free[gi],
            count=gs.running_count,
        )
        for gi, gs in gpu_states.items()
    }
    # Rank position of each GPU, used as the deterministic tie-break between
    # equally-good candidates. It is the caller's `gpu_states` ordering rather
    # than the device number so that a run restricted to, say, GPUs "2,0,1"
    # keeps picking them in the order it was given -- which is what the
    # single-GPU best-fit scan did when it kept the first strict maximum.
    rank = {gi: pos for pos, gi in enumerate(gpu_states)}
    # GPU -> required GiB of a blocked higher-priority VRAM test, and the budget
    # we have since added to that GPU via lower-priority backfill this pass.
    # `backfill_added` is only consulted for gpu_1 reservations; gang holds are
    # gated on absolute committed budget instead (see the gate below).
    reserved_req: dict[int, float] = {}
    backfill_added: dict[int, float] = {}
    # Subset of `reserved_req` held for a blocked *gang*. Gangs need a stronger
    # gate than gpu_1 tests do -- see the reservation block below.
    reserved_gang: set[int] = set()
    to_launch: list[tuple[int, tuple[int, ...]]] = []

    def _cap(gi: int) -> float:
        # First test on an idle GPU may use the whole card; once it hosts 2+,
        # reserve the multi-process margin for CUDA context overhead.
        gs = gpu_states[gi]
        return gs.total_gib if tentative[gi].count < 1 else gs.budget_multi

    def _can_ever_host(gi: int, required: float) -> bool:
        """Could a test needing ``required`` GiB per device ever run on ``gi``?

        Deliberately asks only about the card's physical size, which is exact and
        permanent: a card smaller than the requirement cannot host the test in
        any future state, so a hold placed on it is protection that can never be
        collected. The comparison matches ``_unschedulable_reason`` exactly, so a
        test that clears pre-flight is never rejected here by a rounding
        difference.

        Live free memory is deliberately not consulted HERE. It would have to
        distinguish memory held by a neighbour on the box, which may never free,
        from our own test overshooting its profile or still ramping, which always
        does -- and ``(total - free) - budget`` cannot tell those apart in either
        direction. A filter built on it would reject a card that recovers, and a
        rejection here is permanent by construction.

        That reasoning bars it from the FILTER, not from the ordering. See
        ``_usable_now``, which consults exactly that quantity to rank candidates
        the filter has already admitted: a misjudgement there costs a card its
        place in a list that is rebuilt next pass, never its candidacy.
        """
        return gpu_states[gi].total_gib >= required

    def _foreign_held(gi: int) -> float:
        """GiB on ``gi`` held by a process outside this run.

        Budget we committed ourselves is excluded because our own tests end --
        what remains, ``total - free - budget``, is somebody else's and may not.
        The two terms cancel exactly against this pass's tentative launches, so
        the answer does not drift as the pass fills.

        Defined once because both users of it must agree: the hold's ranking
        (which card to protect) and the hold's admission gate (how much to let
        in beside it). They were denominated differently once, and a hold that
        bounds a different quantity from the one its gang is gated on bounds
        nothing.
        """
        ts = tentative[gi]
        return max(0.0, gpu_states[gi].total_gib - ts.free - ts.budget)

    def _usable_now(gi: int, required: float) -> bool:
        """Could ``gi`` host a member *today*, discounting only what we cannot free?

        ``_can_ever_host`` asks about the card; this asks about the card as it
        stands.

        A card that fails this is never rejected, only outranked. That is the
        whole distinction from the filter above: foreign memory is a fact about
        right now, and the ordering is rebuilt from scratch on every pass.
        """
        return gpu_states[gi].total_gib - _foreign_held(gi) >= required

    for idx, test in enumerate(pending):
        if running_count + len(to_launch) >= num_slots:
            break

        need = test.gpu_count
        if need > len(gpu_states):
            # Impossible on this node. run_parallel rejects these up front, so
            # reaching here means the caller chose to keep going; skip rather
            # than let the test block anything.
            continue

        # Zero-VRAM filler: no budget impact, just needs a free slot and its
        # devices. Place on the least-loaded GPUs for balance; never reserves
        # and is never blocked.
        if test.profiled_gib <= 0:
            chosen = sorted(gpu_states, key=lambda g: (tentative[g].count, rank[g]))[
                :need
            ]
            to_launch.append((idx, tuple(sorted(chosen))))
            for gi in chosen:
                tentative[gi].count += 1
            continue

        # VRAM test: collect every GPU that passes all gates on its own, then
        # best-fit onto the `need` with the most free budget.
        eligible: list[tuple[int, float]] = []
        for gi, gs in gpu_states.items():
            ts = tentative[gi]
            cap = _cap(gi)
            avail = cap - ts.budget
            if avail < test.profiled_gib:
                continue  # reserved-budget gate
            actual_used = gs.total_gib - ts.free
            if actual_used + test.profiled_gib > cap:
                continue  # actual-usage gate (catches init-time spikes)
            if gi in reserved_req:
                if gi in reserved_gang:
                    # Gang reservations are gated on the ABSOLUTE committed
                    # budget, in budget_multi units. Both halves are load
                    # bearing, and the gpu_1 rule below is wrong for a gang in
                    # both of them:
                    #   * absolute, not per-pass: `backfill_added` is rebuilt on
                    #     every call, so a per-pass cap lets committed budget
                    #     creep upward one pass at a time. A gpu_1 test rides
                    #     that out -- it needs one card to come free, and any
                    #     card will do -- but a gang needs headroom on `k` cards
                    #     at the same instant, and creep means the k never
                    #     coincide. Gating on `ts.budget` (which starts from
                    #     gpu_states, i.e. already-committed budget) makes the
                    #     headroom monotone: once a member drops below the line
                    #     it cannot be pushed back over it.
                    #   * budget_multi, not `_cap(gi)`: an IDLE reserved card
                    #     reports the whole-card cap, which is the cap for
                    #     whoever lands first -- but the instant a filler lands,
                    #     the gang's own cap on that card drops to budget_multi.
                    #     Gating on `_cap` therefore over-grants by exactly
                    #     VRAM_MULTI_PROC_MARGIN * total_gib and admits a filler
                    #     that immediately re-blocks the gang; on a card that
                    #     keeps draining and refilling, forever.
                    # Together these give the gang a finite wait: every member
                    # reaches `budget_multi - required` and stays there, so the
                    # k members eventually hold simultaneously and the gang --
                    # scanned before every lower-priority test -- launches.
                    #   * net of foreign memory, not gross: the gate the GANG
                    #     must pass counts observed usage (`total - free`),
                    #     which includes memory held outside this run; this gate
                    #     counts only what WE committed. The two are the same
                    #     number only when that foreign hold is zero, and for
                    #     any F > 0 the difference is a band exactly `required`
                    #     wide in which a filler clears this gate and then
                    #     leaves the reserved card unable to host a member. The
                    #     hold then bounds nothing: every reserved card carries
                    #     a filler the gang cannot fit beside, and the gang
                    #     waits for all `k` of them to fall clean on the same
                    #     pass -- a coincidence whose odds drop geometrically in
                    #     `gpu_count`. Subtracting it here is safe in a way it
                    #     would not be in `_can_ever_host`: this is a per-pass
                    #     admission decision, not a permanent candidacy one, so
                    #     a card whose neighbour leaves is admitted again on the
                    #     very next pass.
                    #   * the foreign term only where it can buy the gang
                    #     something. On a card the gang could take today the
                    #     bound above is exactly tight. On one it could not
                    #     (`_usable_now` false) we have `foreign > total_gib -
                    #     required`, so the line is `budget_multi - foreign -
                    #     required < budget_multi - total_gib < 0` for every
                    #     test of every size -- the card refuses ALL backfill
                    #     for as long as the gang is queued. Nothing is bought
                    #     by that: what gates the gang on such a card is the
                    #     foreign hold retiring, which this run neither owns nor
                    #     influences, and holding the card idle does not hasten
                    #     it. At `n == gpu_count` -- the only topology gangs
                    #     ship to -- every card is held and the node makes zero
                    #     progress until the CI wall-clock kills it. Keeping
                    #     `required` in the line unconditionally is what still
                    #     lets the gang fit the moment the neighbour leaves.
                    #   * on a device it could NOT take today, the foreign term is
                    #     dropped -- but the widened line still has to reserve the
                    #     multi-process margin, because admitting anything at all is
                    #     what costs the device that margin. With nothing resident,
                    #     ``_cap`` is the whole card and the device becomes able to
                    #     host a member as soon as ``foreign <= total_gib - required``.
                    #     The first admission takes ``running_count`` to 1, ``_cap``
                    #     falls to ``budget_multi``, and the threshold collapses to
                    #     ``foreign <= budget_multi - committed - required``. Reserving
                    #     exactly the forfeited margin floors that at ``total_gib -
                    #     budget_multi``: the gang can still take the device at any
                    #     residual foreign hold up to the multi-process margin, which
                    #     is the guarantee -- and the limit -- of this branch. Dropping
                    #     the term entirely bought backfill that outlived the device's
                    #     usefulness; keeping it whole refused every test of every size
                    #     and stopped the node.
                    committed = ts.budget + test.profiled_gib
                    line = gs.budget_multi - reserved_req[gi]
                    if _usable_now(gi, reserved_req[gi]):
                        line -= _foreign_held(gi)
                    else:
                        line -= gs.total_gib - gs.budget_multi
                    if committed > line:
                        continue  # would crowd out the reserved gang
                elif backfill_added[gi] + test.profiled_gib > cap - reserved_req[gi]:
                    # Single-GPU reservation: unchanged per-pass semantics, so
                    # gpu_1-only scheduling stays bit-identical to upstream.
                    continue  # would crowd out the reserved higher-priority test
            eligible.append((gi, avail))

        if len(eligible) >= need:
            eligible.sort(key=lambda e: (-e[1], rank[e[0]]))
            chosen = [gi for gi, _ in eligible[:need]]
            to_launch.append((idx, tuple(sorted(chosen))))
            for gi in chosen:
                tentative[gi].budget += test.profiled_gib
                tentative[gi].free -= test.profiled_gib
                tentative[gi].count += 1
                if gi in reserved_req:
                    backfill_added[gi] += test.profiled_gib
            continue

        # Blocked: reserve the GPUs where this test is closest to fitting (most
        # free budget), unless they are already held for an even-higher-priority
        # test. All-or-none -- holding headroom on fewer devices than the test
        # needs would starve backfill without ever letting the test launch.
        # Keep scanning -- smaller tests may still fit elsewhere or backfill
        # under the reservation, and fillers keep filling slots.
        unreserved = [gi for gi in gpu_states if gi not in reserved_req]
        if need > 1:
            # A hold is only protection if the gang could actually run there.
            # Ranked by headroom alone, an IDLE card scores its whole capacity --
            # often the best score on the node -- even when that capacity is
            # below what the gang needs, or is already spoken for by a process
            # outside this run. Such a card wins one of the `need` slots, offers
            # no path to a launch, and costs the gang a hold on a card that
            # would have been one; the unprotected card then saturates under the
            # very backfill this reservation exists to bound.
            #
            # There is no fallback to the unfiltered list on purpose. The test
            # is on physical size, so a card that fails it fails permanently --
            # holding it could never turn into a launch, which is the whole
            # defect. Reserving nothing beats reserving something useless, and
            # `_unschedulable_reason` has already established that the node has
            # `need` cards large enough; a shortfall here means a
            # higher-priority test is holding them, which is correct.
            #
            # gpu_1 keeps the unfiltered rule: its reservation is a hint, not a
            # liveness mechanism -- it needs one card to come free and any card
            # will do -- so filtering there would change upstream behaviour for
            # no correctness gain.
            unreserved = [
                gi for gi in unreserved if _can_ever_host(gi, test.profiled_gib)
            ]
        if len(unreserved) >= need:
            if need > 1:
                # Headroom alone is the wrong key, for the same reason it was the
                # wrong key before the size filter: an IDLE card scores its whole
                # capacity, so a card sitting under somebody else's 4.5 GiB still
                # scores a perfect 10.0 and takes a protector slot off a card
                # that is genuinely free. Size is permanent and belongs in the
                # filter; foreign memory is a fact about right now and belongs
                # here, as a preference, where being wrong costs a card its place
                # in a list rebuilt next pass rather than its candidacy. Cards
                # that could host a member today therefore sort ahead of cards
                # that could not, and headroom breaks ties within each group.
                #
                # It cannot manufacture capacity: when fewer than `need` cards
                # are usable the gang waits, which is correct -- but it holds the
                # ones that came back, so it launches on the pass they do.
                req = test.profiled_gib

                def _hold_key(gi: int, req: float = req) -> tuple[bool, float, int]:
                    return (
                        not _usable_now(gi, req),
                        -(_cap(gi) - tentative[gi].budget),
                        rank[gi],
                    )

                unreserved.sort(key=_hold_key)
            else:
                # gpu_1 keeps the upstream ordering, bit for bit.
                unreserved.sort(
                    key=lambda gi: (-(_cap(gi) - tentative[gi].budget), rank[gi])
                )
            for gi in unreserved[:need]:
                reserved_req[gi] = test.profiled_gib
                backfill_added[gi] = 0.0
                if need > 1:
                    reserved_gang.add(gi)

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
    # container's true CPU budget.
    cpu_budget = effective_cpu_budget()
    if num_slots > cpu_budget:
        _print(
            f"Capping concurrency: {num_slots} -> {cpu_budget} slots "
            "(detected CPU budget or NUM_CPUS ceiling)"
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
                gpu_count=m.get("gpu_count", DEFAULT_GPU_COUNT),
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

    # Reject tests this node can never satisfy, before anything launches.
    # Without this the run loop would sit in `while pending or running` with
    # nothing running and nothing launchable, printing "waiting for N GiB free"
    # until the CI job's timeout killed it. A gang asking for more GPUs than
    # exist is the new way to reach that state; a test profiled larger than any
    # card was always able to.
    impossible = []
    for t in tests:
        reason = _unschedulable_reason(t, gpu_states)
        if reason is not None:
            impossible.append((t, reason))
    if impossible:
        _print(f"\nERROR: {len(impossible)} test(s) cannot be scheduled on this node:")
        for t, reason in impossible:
            _print(f"  {t.name}  ({reason})")
        _print(
            "\nRun them on a node with enough GPUs, or narrow the selection "
            "with -m / --max-vram-gib."
        )
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
    last_vllm_launch: dict[int, float] = {}  # gpu_index -> monotonic timestamp

    def _build_status_lines(now: float) -> list[str]:
        return _status_lines(now, t0, gpu_states, running, _get_gpu_used_gib)

    def _launch_test(test: _TestEntry, env_base: dict) -> _RunningTest:
        """Build env, spawn subprocess, start output streamer thread."""
        env = env_base.copy()
        # The complete gang, ascending, so the child sees the same device
        # order on every run and logical device i is a stable physical GPU.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gi) for gi in test.assigned_gpus)
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
        junit_path = os.path.join(_JUNIT_DIR, f"{safe_name}.xml")
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

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        run_info = _RunningTest(proc=proc, test=test, start_time=time.monotonic())
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

    env_base = os.environ.copy()

    while pending or running:
        now = time.monotonic()

        # Check for completed subprocesses
        for w_id in list(running.keys()):
            run_info = running[w_id]
            rc = run_info.proc.poll()
            if rc is not None:
                if run_info.reader_thread is not None:
                    run_info.reader_thread.join(timeout=5)
                duration = now - run_info.start_time
                passed = rc == 0
                test = run_info.test

                # Detect retryable init errors (profiling race, OOM at startup)
                if not passed and test.retries < _MAX_RETRIES:
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
                        _release_gpus(test, gpu_states)
                        del running[w_id]
                        pending.insert(0, test)
                        continue

                # Detect runtime skips via JUnit XML (subprocess exit 0
                # covers both "all passed" and "all skipped").
                skipped = False
                skip_reason: str | None = None
                if passed:
                    safe_name = test.name.replace("/", "_").replace("::", "__")
                    junit_path = os.path.join(_JUNIT_DIR, f"{safe_name}.xml")
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

                _release_gpus(test, gpu_states)
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
        if pending and len(running) < num_slots:
            actual_free = {
                gi: gs.total_gib - _get_gpu_used_gib(gi)
                for gi, gs in gpu_states.items()
            }
            to_launch = _select_launches(
                pending=pending,
                gpu_states=gpu_states,
                actual_free=actual_free,
                num_slots=num_slots,
                running_count=len(running),
            )

            # Pop from pending in reverse to preserve indices, then reverse
            # back so highest-priority tests launch first.
            batch: list[tuple[_TestEntry, tuple[int, ...]]] = []
            for pending_idx, assigned_gpus in reversed(to_launch):
                batch.append((pending.pop(pending_idx), assigned_gpus))
            batch.reverse()

            for entry, assigned_gpus in batch:
                w_id = entry.w_id
                is_vllm = (
                    entry.requested_vllm_kv_cache_bytes is not None
                    and entry.profiled_gib > 0
                )

                # Per-GPU vLLM stagger — only between vLLM tests sharing a
                # GPU. Tests on disjoint GPUs launch simultaneously. A gang
                # has to respect the most recent launch on *any* device it is
                # about to touch, or it would race a test that just started on
                # one of them.
                if is_vllm:
                    last_t = max(
                        (last_vllm_launch.get(gi, 0.0) for gi in assigned_gpus),
                        default=0.0,
                    )
                    wait = _VLLM_LAUNCH_STAGGER_S - (time.monotonic() - last_t)
                    if wait > 0:
                        time.sleep(wait)

                _reserve_gpus(entry, assigned_gpus, gpu_states)
                run_info = _launch_test(entry, env_base)
                running[w_id] = run_info

                if is_vllm:
                    # Stamp every member, so the next launch on any of them
                    # waits out this test's profiling window too.
                    stamp = time.monotonic()
                    for gi in entry.assigned_gpus:
                        last_vllm_launch[gi] = stamp

                retry_str = f" (retry {entry.retries})" if entry.retries else ""
                _print(
                    f"[w{w_id}] {entry.name} "
                    f"({_fmt_gpus(entry.assigned_gpus)}, "
                    f"profiled={entry.profiled_gib:.1f} GiB"
                    f"{' per GPU' if len(entry.assigned_gpus) > 1 else ''}, "
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
                    # Nothing of ours is holding memory, and pre-flight proved
                    # the budget fits, so the block is live VRAM held outside
                    # this run rather than anything the scheduler can resolve.
                    head = pending[0]
                    need = head.profiled_gib
                    devices = (
                        f" on each of {head.gpu_count} GPUs"
                        if head.gpu_count > 1
                        else ""
                    )
                    lines[-1] += (
                        f" [stalled: nothing running; w{head.w_id} needs "
                        f"{need:.1f} GiB{devices} but live GPU memory is held "
                        f"outside this run]"
                    )
                lines[-1] += f" [queued: {queued_str}]"
            for ln in lines:
                _print(ln)
            next_status = now + 10

        if running or pending:
            time.sleep(1.0)

    # Summary
    wall_time = time.monotonic() - t0
    sequential_time = sum(c.duration for c in completed if not c.skipped)
    n_passed = sum(1 for c in completed if c.passed)
    n_skipped = sum(1 for c in completed if c.skipped)
    n_failed = sum(1 for c in completed if not c.passed and not c.skipped)

    completed.sort(key=lambda c: c.test.w_id)

    _print()
    _print(f"{'=' * 27} short test summary info {'=' * 27}")
    for c in completed:
        test = c.test
        w_id = test.w_id
        if c.skipped:
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

    combined = _aggregate_junit_xml(_JUNIT_DIR)
    if combined:
        _print(f"JUnit XML: {combined}")

    return 0 if n_failed == 0 else 1


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
