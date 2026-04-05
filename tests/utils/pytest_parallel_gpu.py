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
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pynvml

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from tests.utils.vram_utils import (  # noqa: E402
    VRAM_MULTI_PROC_MARGIN,
    auto_worker_count,
    detect_gpus,
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
    skip_reason: str | None = None
    w_id: int = 0
    assigned_gpu: int | None = None
    retries: int = 0


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
    if test.requested_vllm_kv_cache_bytes is not None:
        gib = int(test.requested_vllm_kv_cache_bytes) / (1024**3)
        return f"req_kv={gib:.2f} GiB"
    return "req_kv=None"


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


def run_parallel(
    test_ids: list[str],
    meta: dict[str, dict],
    max_vram_gib: float,
    num_slots: int,
    gpu_indices: list[int] | None = None,
    extra_pytest_args: list[str] | None = None,
    stream: bool = False,
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
                skip_reason=m.get("skip_reason"),
            )
        )

    # Separate skip-marked tests — they won't actually run, so don't
    # validate KV markers or consume GPU budget.
    skipped_tests = [t for t in tests if t.skip_reason is not None]
    tests = [t for t in tests if t.skip_reason is None]

    # Sort by timeout descending (longest first to minimize tail latency)
    tests.sort(key=lambda t: t.timeout, reverse=True)

    # Reject tests without a KV marker — without explicit memory control
    # they'd each grab the engine's default (e.g. vLLM 90%) and OOM when
    # run concurrently. Tests with profiled_gib=0 are exempt (mock/CPU-only).
    no_kv = [
        t
        for t in tests
        if t.requested_vllm_kv_cache_bytes is None
        and t.requested_sglang_kv_tokens is None
        and t.profiled_gib > 0
    ]
    if no_kv:
        _print(
            f"\nERROR: {len(no_kv)} test(s) lack a requested_vllm_kv_cache_bytes "
            f"or requested_sglang_kv_tokens marker and cannot run in parallel:"
        )
        for t in no_kv:
            _print(f"  {t.name}")
        _print(
            "\nAdd the appropriate marker via profile_pytest.py --kv-bytes, "
            "then rerun."
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
        elif test.requested_vllm_kv_cache_bytes is not None:
            env["_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES"] = str(
                int(test.requested_vllm_kv_cache_bytes)
            )

        safe_name = test.name.replace("/", "_").replace("::", "__")
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
                gi = test.assigned_gpu

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
        # For each pending test, find the GPU with most available budget.
        # Gate on BOTH budget tracking AND actual GPU free memory.
        # vLLM stagger is per-GPU only — tests on different GPUs launch
        # simultaneously.
        if pending and len(running) < num_slots:
            actual_free = {
                gi: gs.total_gib - _get_gpu_used_gib(gi)
                for gi, gs in gpu_states.items()
            }
            tentative = {
                gi: _TentativeGpu(
                    budget=gs.budget_used,
                    free=actual_free[gi],
                    count=gs.running_count,
                )
                for gi, gs in gpu_states.items()
            }

            to_launch: list[tuple[int, int]] = []  # (pending_idx, gpu_idx)
            n_total = len(running)
            for i, test in enumerate(pending):
                if n_total + len(to_launch) >= num_slots:
                    break
                best_gi: int | None = None
                best_avail = -1.0
                for gi, gs in gpu_states.items():
                    ts = tentative[gi]
                    will_be_multi = ts.count >= 1
                    cap = gs.budget_multi if will_be_multi else gs.total_gib
                    avail = cap - ts.budget
                    if avail < test.profiled_gib:
                        continue
                    if ts.free < test.profiled_gib:
                        continue
                    if avail > best_avail:
                        best_gi = gi
                        best_avail = avail
                if best_gi is not None:
                    to_launch.append((i, best_gi))
                    tentative[best_gi].budget += test.profiled_gib
                    tentative[best_gi].free -= test.profiled_gib
                    tentative[best_gi].count += 1

            # Pop from pending in reverse to preserve indices, then reverse
            # back so longest-timeout tests launch first.
            batch: list[_TestEntry] = []
            for pending_idx, assigned_gpu in reversed(to_launch):
                entry = pending.pop(pending_idx)
                entry.assigned_gpu = assigned_gpu
                batch.append(entry)
            batch.reverse()

            for entry in batch:
                w_id = entry.w_id
                gi = entry.assigned_gpu
                assert gi is not None
                is_vllm = (
                    entry.requested_sglang_kv_tokens is None and entry.profiled_gib > 0
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
