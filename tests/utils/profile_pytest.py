#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile GPU VRAM usage during a pytest run.

How it works
~~~~~~~~~~~~
A background thread queries NVML (via ``pynvml``) every 100 ms (configurable
with ``--interval``) to record GPU memory usage while the test runs as a
subprocess.  This captures *all* GPU memory (model weights, KV cache, CUDA
contexts, NCCL buffers — not just PyTorch allocations) without requiring any
in-process instrumentation.  Using NVML directly (the same C library that
``nvidia-smi`` wraps) avoids the overhead of forking a subprocess each sample
and allows high-frequency sampling.

In **binary-search mode** (the default), the profiler bisects the KV cache
allocation — ``_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES`` for vLLM (bytes) or
``_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS`` for SGLang (tokens).
If the test passes, the allocation is lowered; if it OOMs, it is raised —
standard bisection to find the minimum the test needs.  A safety factor
is applied and the peak ``memory.used`` from the last passing run becomes
the ``@pytest.mark.profiled_vram_gib`` recommendation.

**IMPORTANT**: The test under profile **MUST** read the appropriate KV cache
override — either directly (see ``test_mock_gpu_alloc.py``) or via launch
scripts that call ``build_gpu_mem_args`` (e.g. ``agg.sh``).  If the test
ignores the override, every probe will pass at the same peak and the profiler
will warn that the binary search is unreliable.

Usage::

    python tests/utils/profile_pytest.py [options] pytest-args...

Examples (``-xvs`` is optional: stop on first failure, verbose, no capture)::

    python tests/utils/profile_pytest.py tests/frontend/test_vllm.py::test_tool_calling
    python tests/utils/profile_pytest.py tests/frontend/test_vllm.py::test_reasoning_effort -xvs

Single-pass profiling (no binary search, just measure one run using default RAM)::

    python tests/utils/profile_pytest.py --no-find-min-vram tests/frontend/test_vllm.py::test_tool_calling

The report is written to stdout after the test finishes.
The raw CSV samples are saved to ``--csv`` if specified.
Use ``--no-recommend`` to suppress the marker recommendation section.
"""

import argparse
import atexit
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field

import pynvml

logger = logging.getLogger(__name__)

# Safety margin for VRAM tier recommendations.  Peak VRAM is multiplied by
# this factor before comparing against tier thresholds, so the recommended
# tier has headroom for variance across runs.
_VRAM_SAFETY_FACTOR = 1.1

# Safety margin for KV cache recommendations (both SGLang tokens and vLLM bytes).
# The minimum passing value is multiplied by this factor to provide headroom for
# prompt length variation, scheduling jitter, and multi-turn conversations.
_KV_SAFETY_FACTOR = 2.0

# Phase detection: a memory jump exceeding this threshold (MiB) between
# consecutive samples marks a phase boundary.
_PHASE_JUMP_MIB = 200

# How long memory must be stable (within this tolerance) to consider it
# a plateau, in consecutive samples.
_PLATEAU_TOLERANCE_MIB = 50
_PLATEAU_MIN_SAMPLES = 3

# Early-stop threshold for binary search: if the last 3 probes have peak
# VRAM within this range, the bisection is in the noise floor (model weights
# dominate) and further probes won't yield meaningful data.
_EARLY_STOP_RANGE_MIB = 768  # 0.75 GiB


def _extract_model_from_markers(pytest_args: list[str]) -> str | None:
    """Extract the model name from @pytest.mark.model(...) via pytest-json-report.

    Runs ``pytest --collect-only`` with the json-report plugin to inspect markers
    without executing the test.  Returns None if the plugin is missing or the
    test has no ``model`` marker.
    """
    fd, json_path = tempfile.mkstemp(prefix="_profile_collect_", suffix=".json")
    os.close(fd)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "--rootdir=.",
                "--override-ini=testpaths=tests",
                f"--json-report-file={json_path}",
            ]
            + list(pytest_args),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode not in (0, 5):
            return None
        with open(json_path) as f:
            data = json.load(f)
        for collector in data.get("collectors", []):
            for marker in collector.get("markers", []):
                if marker.get("name") == "model" and marker.get("args"):
                    return marker["args"][0]
        for test in data.get("tests", []):
            for marker in test.get("markers", []):
                if marker.get("name") == "model" and marker.get("args"):
                    return marker["args"][0]
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("model marker extraction failed: %s", exc)
        return None
    finally:
        try:
            os.remove(json_path)
        except OSError:
            pass
    return None


@dataclass
class GpuSample:
    timestamp: float  # time.monotonic() offset from start
    gpu_idx: int
    mem_used_mib: int
    mem_total_mib: int
    gpu_util_pct: int


@dataclass
class PhaseInfo:
    name: str
    start_sec: float
    end_sec: float
    mem_start_mib: int
    mem_peak_mib: int
    mem_end_mib: int
    description: str = ""


@dataclass
class GpuReport:
    gpu_idx: int
    mem_total_mib: int
    baseline_mib: int
    peak_mib: int
    peak_timestamp: float
    final_mib: int
    leaked_mib: int  # final - baseline
    phases: list[PhaseInfo] = field(default_factory=list)


_nvml_initialized = False
_nvml_handles: list = []


def _nvml_init() -> None:
    """Lazily initialize NVML and cache device handles."""
    global _nvml_initialized, _nvml_handles
    if _nvml_initialized:
        return
    pynvml.nvmlInit()
    _nvml_initialized = True
    count = pynvml.nvmlDeviceGetCount()
    _nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    atexit.register(_nvml_shutdown)


def _nvml_shutdown() -> None:
    global _nvml_initialized, _nvml_handles
    if _nvml_initialized:
        _nvml_handles = []
        pynvml.nvmlShutdown()
        _nvml_initialized = False


def _query_gpu_stats() -> list[tuple[int, int, int, int]]:
    """Return [(gpu_idx, mem_used_mib, mem_total_mib, util_pct), ...] via NVML."""
    _nvml_init()
    results = []
    for idx, handle in enumerate(_nvml_handles):
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        used_mib = int(mem.used) // (1024 * 1024)
        total_mib = int(mem.total) // (1024 * 1024)
        results.append((idx, used_mib, total_mib, int(util.gpu)))
    return results


class _Sampler:
    """Background thread that queries NVML at a fixed interval."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t0 = time.monotonic()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval * 3)

    def _run(self):
        while not self._stop.is_set():
            ts = time.monotonic() - self._t0
            try:
                for gpu_idx, mem_used, mem_total, util_pct in _query_gpu_stats():
                    self.samples.append(
                        GpuSample(ts, gpu_idx, mem_used, mem_total, util_pct)
                    )
            except pynvml.NVMLError:
                pass  # transient NVML error; skip this sample
            self._stop.wait(self.interval)


def _detect_phases(
    samples: list[GpuSample], baseline_end: float, test_end: float
) -> list[PhaseInfo]:
    """Heuristic phase detection from a single GPU's memory timeline.

    Looks for large jumps (model load, KV cache alloc) and identifies
    the inference peak and teardown regions.
    """
    if not samples:
        return []

    phases: list[PhaseInfo] = []
    baseline_samples = [s for s in samples if s.timestamp <= baseline_end]
    test_samples = [s for s in samples if baseline_end < s.timestamp <= test_end]
    teardown_samples = [s for s in samples if s.timestamp > test_end]

    if baseline_samples:
        bl = baseline_samples[-1].mem_used_mib
        phases.append(
            PhaseInfo(
                name="Baseline",
                start_sec=samples[0].timestamp,
                end_sec=baseline_end,
                mem_start_mib=baseline_samples[0].mem_used_mib,
                mem_peak_mib=max(s.mem_used_mib for s in baseline_samples),
                mem_end_mib=bl,
                description="Idle GPU before test starts",
            )
        )

    if not test_samples:
        return phases

    # Walk test samples and detect jumps
    prev_mem = baseline_samples[-1].mem_used_mib if baseline_samples else 0
    phase_start = test_samples[0].timestamp
    phase_start_mem = prev_mem
    phase_peak = prev_mem
    jump_count = 0
    phase_names = ["Model load", "KV cache alloc", "Inference"]

    for s in test_samples:
        delta = s.mem_used_mib - prev_mem
        phase_peak = max(phase_peak, s.mem_used_mib)

        if delta > _PHASE_JUMP_MIB and jump_count < len(phase_names) - 1:
            # Close current phase, start new one
            if phase_start < s.timestamp:
                name = phase_names[min(jump_count, len(phase_names) - 1)]
                phases.append(
                    PhaseInfo(
                        name=name,
                        start_sec=phase_start,
                        end_sec=s.timestamp,
                        mem_start_mib=phase_start_mem,
                        mem_peak_mib=phase_peak,
                        mem_end_mib=prev_mem,
                    )
                )
            jump_count += 1
            phase_start = s.timestamp
            phase_start_mem = s.mem_used_mib
            phase_peak = s.mem_used_mib

        prev_mem = s.mem_used_mib

    # Close final test phase
    name = phase_names[min(jump_count, len(phase_names) - 1)]
    phases.append(
        PhaseInfo(
            name=name,
            start_sec=phase_start,
            end_sec=test_end,
            mem_start_mib=phase_start_mem,
            mem_peak_mib=phase_peak,
            mem_end_mib=test_samples[-1].mem_used_mib,
        )
    )

    if teardown_samples:
        phases.append(
            PhaseInfo(
                name="Teardown",
                start_sec=test_end,
                end_sec=teardown_samples[-1].timestamp,
                mem_start_mib=teardown_samples[0].mem_used_mib,
                mem_peak_mib=max(s.mem_used_mib for s in teardown_samples),
                mem_end_mib=teardown_samples[-1].mem_used_mib,
                description="After pytest exits; should return to baseline",
            )
        )

    return phases


def _build_reports(
    samples: list[GpuSample], baseline_end: float, test_end: float
) -> list[GpuReport]:
    """Build per-GPU reports from collected samples."""
    gpu_indices = sorted({s.gpu_idx for s in samples})
    reports = []

    for idx in gpu_indices:
        gpu_samples = [s for s in samples if s.gpu_idx == idx]
        if not gpu_samples:
            continue

        baseline_samples = [s for s in gpu_samples if s.timestamp <= baseline_end]
        baseline_mib = baseline_samples[-1].mem_used_mib if baseline_samples else 0
        peak_sample = max(gpu_samples, key=lambda s: s.mem_used_mib)
        final_mib = gpu_samples[-1].mem_used_mib

        reports.append(
            GpuReport(
                gpu_idx=idx,
                mem_total_mib=gpu_samples[0].mem_total_mib,
                baseline_mib=baseline_mib,
                peak_mib=peak_sample.mem_used_mib,
                peak_timestamp=peak_sample.timestamp,
                final_mib=final_mib,
                leaked_mib=final_mib - baseline_mib,
                phases=_detect_phases(gpu_samples, baseline_end, test_end),
            )
        )

    return reports


def _format_mib(mib: int) -> str:
    if mib >= 1024:
        return f"{mib / 1024:.1f} GiB"
    return f"{mib} MiB"


def _print_report(
    reports: list[GpuReport],
    pytest_rc: int,
    wall_secs: float,
    model_name: str | None = None,
):
    """Print a human-readable profiling report."""
    print("\n--- GPU MEMORY PROFILE ---")
    print(f"  pytest exit code : {pytest_rc}")
    print(f"  wall time        : {wall_secs:.1f}s")
    print(f"  GPUs sampled     : {len(reports)}")
    if model_name:
        print(f"  model            : {model_name}")

    for r in reports:
        print(f"\n{'─' * 72}")
        print(f"  GPU {r.gpu_idx}  ({_format_mib(r.mem_total_mib)} total)")
        print(f"{'─' * 72}")
        print(f"  Baseline         : {_format_mib(r.baseline_mib)}")
        print(
            f"  Peak             : {_format_mib(r.peak_mib)}  "
            f"({r.peak_mib * 100 // r.mem_total_mib}% of total)  "
            f"@ t={r.peak_timestamp:.1f}s"
        )
        print(f"  Final            : {_format_mib(r.final_mib)}")
        delta = r.leaked_mib
        tag = "OK" if abs(delta) < _PLATEAU_TOLERANCE_MIB else "LEAKED"
        sign = "+" if delta > 0 else ""
        print(f"  Delta (final-bl) : {sign}{_format_mib(delta)}  [{tag}]")

        if r.phases:
            print()
            print(
                f"  {'Phase':<16} {'Time':>12}  {'Start':>10} {'Peak':>10} {'End':>10}"
            )
            print(f"  {'─' * 16} {'─' * 12}  {'─' * 10} {'─' * 10} {'─' * 10}")
            for p in r.phases:
                dur = p.end_sec - p.start_sec
                time_range = (
                    f"{p.start_sec:.0f}s-{p.end_sec:.0f}s"
                    if dur > 0
                    else f"{p.start_sec:.0f}s"
                )
                print(
                    f"  {p.name:<16} {time_range:>12}  "
                    f"{_format_mib(p.mem_start_mib):>10} "
                    f"{_format_mib(p.mem_peak_mib):>10} "
                    f"{_format_mib(p.mem_end_mib):>10}"
                )

    print()


def _write_csv(samples: list[GpuSample], path: str):
    with open(path, "w") as f:
        f.write("timestamp_s,gpu,mem_used_mib,mem_total_mib,gpu_util_pct\n")
        for s in samples:
            f.write(
                f"{s.timestamp:.2f},{s.gpu_idx},{s.mem_used_mib},"
                f"{s.mem_total_mib},{s.gpu_util_pct}\n"
            )


_GPU_REFERENCE_CARDS: list[tuple[int, str]] = [
    (4, "edge/embedded"),
    (8, "RTX 3060/4060"),
    (16, "T4"),
    (24, "L4"),
    (32, "V100-32GB"),
    (48, "A6000/A40"),
    (80, "A100/H100"),
]


@dataclass
class MarkerRecommendation:
    marker: str
    reason: str


def _recommend_markers(
    reports: list[GpuReport],
    wall_secs: float,
    model_name: str | None = None,
    num_runs: int = 1,
    requested_sglang_kv_tokens: int | None = None,
    requested_vllm_kv_cache_bytes: int | None = None,
    min_kv_value: int | None = None,
) -> tuple[list[MarkerRecommendation], list[str]]:
    """Generate marker recommendations from profiling data.

    Returns (recommendations, warnings).
    """
    recs: list[MarkerRecommendation] = []
    warnings: list[str] = []

    if model_name:
        recs.append(
            MarkerRecommendation(
                f'model("{model_name}")',
                "detected from test source",
            )
        )

    max_peak_mib = max((r.peak_mib for r in reports), default=0)
    max_baseline_mib = max((r.baseline_mib for r in reports), default=0)
    used_vram = max_peak_mib - max_baseline_mib
    gpus_with_vram = sum(
        1 for r in reports if (r.peak_mib - r.baseline_mib) > _PLATEAU_TOLERANCE_MIB
    )
    has_model_load = any(
        p.name == "Model load"
        for r in reports
        for p in r.phases
        if p.mem_peak_mib - p.mem_start_mib > _PHASE_JUMP_MIB
    )
    any_leaked = any(abs(r.leaked_mib) >= _PLATEAU_TOLERANCE_MIB for r in reports)

    # -- Test Type --
    if wall_secs < 1.0 and used_vram < _PLATEAU_TOLERANCE_MIB:
        recs.append(
            MarkerRecommendation("unit", f"wall time {wall_secs:.1f}s, no GPU usage")
        )
    elif wall_secs < 30.0 and not has_model_load:
        recs.append(
            MarkerRecommendation(
                "integration", f"wall time {wall_secs:.1f}s, no model load detected"
            )
        )
    else:
        reason = f"wall time avg {wall_secs:.1f}s based on {num_runs} run{'s' if num_runs != 1 else ''}"
        if has_model_load:
            reason += ", loads a real model"
        recs.append(MarkerRecommendation("e2e", reason))

    # -- Lifecycle --
    if wall_secs < 20.0:
        recs.append(
            MarkerRecommendation(
                "pre_merge", f"wall time {wall_secs:.1f}s (< 20s, fast enough per PR)"
            )
        )
    elif wall_secs < 300.0:
        warnings.append(
            f"Wall time {wall_secs:.1f}s is too slow for pre_merge (> 20s). "
            f"Consider post_merge or nightly instead."
        )
    else:
        warnings.append(
            f"Wall time {wall_secs:.1f}s is very slow (> 300s). "
            f"Consider nightly instead."
        )

    # -- Hardware: GPU count --
    if gpus_with_vram == 0:
        recs.append(MarkerRecommendation("gpu_0", "no GPU VRAM used"))
    else:
        marker = f"gpu_{gpus_with_vram}"
        recs.append(
            MarkerRecommendation(
                marker,
                f"{gpus_with_vram} GPU(s) used, peak {_format_mib(max_peak_mib)}",
            )
        )

    # -- Hardware: VRAM requirements (two markers) --
    if used_vram > _PLATEAU_TOLERANCE_MIB:
        max_peak_gib = round(max_peak_mib / 1024, 1)
        padded_peak_mib = int(max_peak_mib * _VRAM_SAFETY_FACTOR)
        padded_peak_gib = round(padded_peak_mib / 1024, 1)

        # profiled_vram_gib: actual nvidia-smi peak (for scheduling/filtering)
        recs.append(
            MarkerRecommendation(
                f"profiled_vram_gib({max_peak_gib})",
                f"actual nvidia-smi peak {_format_mib(max_peak_mib)}",
            )
        )
        if requested_sglang_kv_tokens is not None:
            min_label = f" over min={min_kv_value}" if min_kv_value is not None else ""
            recs.append(
                MarkerRecommendation(
                    f"requested_sglang_kv_tokens({requested_sglang_kv_tokens})",
                    f"KV cache cap ({_KV_SAFETY_FACTOR:.0f}x safety{min_label})",
                )
            )
        if requested_vllm_kv_cache_bytes is not None:
            min_label = (
                f" over min={min_kv_value:_}" if min_kv_value is not None else ""
            )
            recs.append(
                MarkerRecommendation(
                    f"requested_vllm_kv_cache_bytes({requested_vllm_kv_cache_bytes:_})",
                    f"KV cache cap ({_KV_SAFETY_FACTOR:.0f}x safety{min_label})",
                )
            )

        # Warn about GPU cards that would OOM
        for card_gib, card_name in _GPU_REFERENCE_CARDS:
            if padded_peak_gib > card_gib:
                warnings.append(f"Will OOM on {card_name} ({card_gib} GiB).")

    # -- Timeout --
    timeout_val = int(math.ceil(wall_secs * 6.0))
    timeout_val = max(timeout_val, 10)
    recs.append(
        MarkerRecommendation(
            f"timeout({timeout_val})",
            f"wall time {wall_secs:.1f}s, based on {num_runs} run{'s' if num_runs != 1 else ''}",
        )
    )

    # -- Memory leak warning --
    if any_leaked:
        leaked_reports = [
            r for r in reports if abs(r.leaked_mib) >= _PLATEAU_TOLERANCE_MIB
        ]
        for r in leaked_reports:
            warnings.append(
                f"GPU {r.gpu_idx}: VRAM not fully released "
                f"(baseline {_format_mib(r.baseline_mib)} -> "
                f"final {_format_mib(r.final_mib)}, "
                f"delta {_format_mib(r.leaked_mib)}). "
                f"Possible leak or teardown issue."
            )

    return recs, warnings


def _print_recommendations(
    recs: list[MarkerRecommendation],
    warnings: list[str],
    pytest_args: list[str] | None = None,
):
    print("--- Recommended markers (copy-paste into your test) ---")
    if pytest_args:
        print(
            f"# Measured using: tests/utils/profile_pytest.py {' '.join(pytest_args)}"
        )
    else:
        print("# Measured using: tests/utils/profile_pytest.py")
    for r in recs:
        print(f"@pytest.mark.{r.marker}  # {r.reason}")

    # Show example so user knows where to place the markers
    test_name = None
    if pytest_args:
        test_name = next(
            (a.rsplit("::", 1)[-1] for a in pytest_args if "::" in a), None
        )
    print(f"def {test_name or 'test_something'}(...):")
    print("    ...")

    if warnings:
        print()
        for w in warnings:
            print(f"  WARNING: {w}")
    print()


_SGLANG_NODEID_MARKERS = ["test_sglang", "sglang"]


def _is_sglang_test(pytest_args: list[str]) -> bool:
    """Check if any pytest arg looks like a SGLang test node ID."""
    return any(
        marker in arg for arg in pytest_args for marker in _SGLANG_NODEID_MARKERS
    )


_OOM_PATTERNS = [
    "OutOfMemoryError",
    "CUDA out of memory",
    "CUDA error: out of memory",
    "not enough memory",
    "Cannot allocate",
    "oom-kill",
]


def _looks_like_oom(stdout: str) -> bool:
    """Check if captured output contains OOM-like errors."""
    stdout_lower = stdout.lower()
    return any(pat.lower() in stdout_lower for pat in _OOM_PATTERNS)


_SGLANG_MAX_TOKENS_RE = re.compile(r"max_total_tokens=(\d+)")


def _extract_requested_sglang_kv_tokens(stdout: str) -> int | None:
    """Extract max_total_tokens from SGLang engine output.

    SGLang logs: "Got total KV blocks from scheduler: N (max_total_tokens=M, page_size=P)"
    """
    match = _SGLANG_MAX_TOKENS_RE.search(stdout)
    if match:
        return int(match.group(1))
    return None


_DEFAULT_PROBE_TIMEOUT = 300  # 5 minutes max per profile run


def _run_once(
    pytest_args: list[str],
    interval: float = 0.1,
    baseline_seconds: float = 3.0,
    teardown_seconds: float = 5.0,
    extra_env: dict[str, str] | None = None,
    quiet: bool = False,
    run_label: str | None = None,
    timeout: float = _DEFAULT_PROBE_TIMEOUT,
) -> tuple[int, float, list[GpuReport], list[GpuSample], str]:
    """Run pytest once with GPU sampling.

    When *run_label* is set, each line of pytest stdout/stderr is prefixed
    with ``[run_label]`` so multi-run output is easy to follow.

    Returns (exit_code, wall_secs, reports, raw_samples, captured_stdout).
    """
    sampler = _Sampler(interval=interval)
    sampler.start()

    if not quiet:
        print(f"Sampling baseline for {baseline_seconds}s ...")
    time.sleep(baseline_seconds)
    baseline_end = time.monotonic() - sampler._t0

    pytest_cmd = [sys.executable, "-m", "pytest"] + list(pytest_args)
    if not quiet:
        print(f"Running: {' '.join(pytest_cmd)}")
    sys.stdout.flush()

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    if extra_env:
        env.update(extra_env)

    capture = run_label is not None
    t_start = time.monotonic()
    timed_out = False
    captured_stdout = ""
    try:
        result = subprocess.run(
            pytest_cmd,
            env=env,
            capture_output=capture,
            text=capture or None,
            timeout=timeout,
        )
        rc = result.returncode
        if capture:
            captured_stdout = result.stdout or ""
    except subprocess.TimeoutExpired:
        timed_out = True
        rc = 1
        if not quiet or run_label:
            print(
                f"  [TIMEOUT] pytest exceeded {timeout:.0f}s limit "
                f"(teardown likely hung)"
            )
    if not timed_out and capture:
        prefix = f"[{run_label}] "
        for line in captured_stdout.splitlines():
            print(f"{prefix}{line}")
        for line in (result.stderr or "").splitlines():
            print(f"{prefix}{line}", file=sys.stderr)
    sys.stdout.flush()
    wall_secs = time.monotonic() - t_start
    test_end = time.monotonic() - sampler._t0

    if not quiet:
        print(f"Sampling teardown for {teardown_seconds}s ...")
    time.sleep(teardown_seconds)

    sampler.stop()
    reports = _build_reports(sampler.samples, baseline_end, test_end)
    return rc, wall_secs, reports, sampler.samples, captured_stdout


def _find_min_vram(
    pytest_args: list[str],
    interval: float = 0.1,
    baseline_seconds: float = 2.0,
    teardown_seconds: float = 2.0,
    recommend: bool = True,
    csv_path: str | None = None,
    kv_bytes_mode: bool = False,
    gpu_index: int = 0,
) -> int:
    """Binary search to find the minimum VRAM a test needs.

    Three modes, two patterns:

    KV bisection (deterministic, no profiling race):
      vLLM:   bisects _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES (bytes)
      SGLang: bisects _PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS (tokens)
      Both use the same _KV_SAFETY_FACTOR (2x) and the same bisect loop.
      The only differences are env var name, units, display, and bounds.
    """
    is_sglang = _is_sglang_test(pytest_args)

    gpu_info = _query_gpu_stats()
    if not gpu_info:
        raise RuntimeError("NVML returned no GPU data")
    if gpu_index >= len(gpu_info):
        raise RuntimeError(
            f"GPU {gpu_index} not found (available: 0..{len(gpu_info) - 1})"
        )
    used_mib = gpu_info[gpu_index][1]
    total_mib = gpu_info[gpu_index][2]
    free_mib = total_mib - used_mib
    total_gib = total_mib / 1024

    # Base env: pin subprocess to the selected GPU
    _gpu_env = {"CUDA_VISIBLE_DEVICES": str(gpu_index)}

    model_name = _extract_model_from_markers(pytest_args)

    if not is_sglang:
        kv_bytes_mode = True

    if kv_bytes_mode:
        mode_label = "KV CACHE BYTES (vLLM, deterministic)"
    else:
        mode_label = "KV TOKENS (SGLang)"
    print(f"\n--- FIND MINIMUM {mode_label} (binary search) ---")
    print(f"  GPU total : {total_gib:.1f} GiB")
    print(
        f"  GPU free  : {free_mib / 1024:.1f} GiB  "
        f"(in use: {used_mib / 1024:.1f} GiB)"
    )
    print(f"  Test      : {' '.join(pytest_args)}")
    if model_name:
        print(f"  Model     : {model_name}")

    hogged_pct = used_mib / total_mib * 100
    if hogged_pct > 10:
        print(f"\n  {'!' * 72}")
        print(
            f"  WARNING: {used_mib / 1024:.1f} GiB ({hogged_pct:.0f}%) of GPU memory "
            f"is already in use!"
        )
        print("  Another process is hogging the GPU. Free memory is reduced,")
        print("  which limits KV cache headroom. Kill other GPU processes first.")
        print(f"  {'!' * 72}")
    print()

    # -- Validation run --
    validation_env: dict[str, str] = dict(_gpu_env)
    if kv_bytes_mode:
        # Start at 50% of free GPU. If it passes, that's the upper bound and we
        # search downward. If it fails (model weights too large), halve again
        # until we find a passing point, then search downward from there.
        max_kv_bytes = int(max(free_mib // 2, 1024) * 1024 * 1024)
        validation_env["_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES"] = str(max_kv_bytes)
        validation_desc = f"kv_cache={max_kv_bytes // (1024**2)} MiB (50% of free)"
    else:
        validation_desc = "no token cap, default fraction"

    print(f"  [probe 1] Validation run ({validation_desc})")
    sys.stdout.flush()
    t_iter_start = time.monotonic()
    rc, wall, reports, raw_samples, stdout = _run_once(
        pytest_args,
        interval=interval,
        baseline_seconds=baseline_seconds,
        teardown_seconds=teardown_seconds,
        extra_env=validation_env or None,
        quiet=True,
        run_label="probe 1",
    )
    iter_elapsed = time.monotonic() - t_iter_start

    # kv-bytes mode: if validation fails, check whether it's OOM (over-allocated)
    # or a genuine test failure (unrelated to KV cache). Only retry with less KV
    # if the output looks like OOM; otherwise the test is broken and retrying won't help.
    if rc != 0 and kv_bytes_mode:
        if _looks_like_oom(stdout):
            for attempt in range(4):
                max_kv_bytes //= 2
                if max_kv_bytes < 64 * 1024 * 1024:
                    break
                validation_env["_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES"] = str(
                    max_kv_bytes
                )
                print(
                    f"  [OOM] Reducing KV cache to {max_kv_bytes // (1024**2)} MiB "
                    f"(retry {attempt + 1}/4)"
                )
                sys.stdout.flush()
                t_iter_start = time.monotonic()
                rc, wall, reports, raw_samples, stdout = _run_once(
                    pytest_args,
                    interval=interval,
                    baseline_seconds=baseline_seconds,
                    teardown_seconds=teardown_seconds,
                    extra_env=validation_env,
                    quiet=True,
                    run_label=f"probe 1 (retry {attempt + 1})",
                )
                iter_elapsed = time.monotonic() - t_iter_start
                if rc == 0:
                    break
        else:
            print(
                "  [FAIL] Test failed but NOT from OOM — the test appears genuinely broken."
            )
            print(
                "  Hint: check the test output above for the root cause "
                "(EngineDeadError, timeout, assertion, etc.)."
            )

    if rc != 0:
        reason = (
            "OOM at all KV sizes"
            if _looks_like_oom(stdout)
            else "test broken (not OOM)"
        )
        print(f"  [FAIL] Cannot determine minimum KV cache: {reason}.")
        return rc

    peak_mib = max((r.peak_mib for r in reports), default=0)

    if kv_bytes_mode:
        # Search range: 64 MiB to 40 GiB in bytes.
        # Lower bound at 64 MiB to skip probes that always fail (no model
        # can serve even 1 request with < 64 MiB KV cache).
        lo: float | int = 64 * 1024 * 1024  # 64 MiB minimum
        hi: float | int = max_kv_bytes
        tolerance: float | int = 16 * 1024 * 1024  # 16 MiB tolerance
        print(
            f"  [PASS] peak {_format_mib(peak_mib)}, wall {wall:.0f}s, "
            f"iter took {iter_elapsed:.0f}s"
        )
    else:
        max_tokens = _extract_requested_sglang_kv_tokens(stdout)
        if max_tokens is None:
            print(
                "  [ERROR] Could not extract max_total_tokens from SGLang output.\n"
                "  The launch script must log 'max_total_tokens=N' (SGLang does this by default)."
            )
            return 4
        page_size = 16
        lo = page_size
        hi = max_tokens
        tolerance = page_size * 2
        print(
            f"  [PASS] peak {_format_mib(peak_mib)}, wall {wall:.0f}s, "
            f"max_total_tokens={max_tokens}, iter took {iter_elapsed:.0f}s"
        )

    baseline_time = iter_elapsed
    probe_timeout = max(baseline_time * 2, 60)
    print(f"  Profile timeout: {probe_timeout:.0f}s (2x first probe)")

    max_iterations = (
        max(1, math.ceil(math.log2((hi - lo) / tolerance))) if hi > lo else 0
    )
    last_pass_value: float | int = hi
    last_pass_peak_mib: int = peak_mib
    last_pass_reports = reports
    last_pass_samples = raw_samples
    elapsed_times: list[float] = [iter_elapsed]
    pass_wall_times: list[float] = [wall]
    all_peak_mibs: list[int] = [peak_mib]

    if kv_bytes_mode:
        print(
            f"\n  Range   : {int(lo) // (1024**2)} - {int(hi) // (1024**2)} MiB  (tolerance {int(tolerance) // (1024**2)} MiB)"
        )
    else:
        print(f"\n  Range   : {lo} - {hi} tokens  (tolerance {tolerance} tokens)")
    print(
        f"  Max iter: {max_iterations + 1} (1 validation + {max_iterations} bisections)"
    )
    print()

    # -- Binary search loop --
    iteration = 0
    while (hi - lo) > tolerance:
        iteration += 1
        probe_num = iteration + 1
        remaining = max_iterations + 1 - probe_num
        avg_iter = sum(elapsed_times) / len(elapsed_times)
        eta_s = remaining * avg_iter

        if kv_bytes_mode:
            mid_int = (int(lo) + int(hi)) // 2
            mid_int = max(mid_int, 1024 * 1024)  # minimum 1 MiB
            probe_env = {
                **_gpu_env,
                "_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES": str(mid_int),
            }
            probe_desc = f"kv_cache={mid_int // (1024**2)} MiB ({mid_int:,} bytes)"
        else:
            mid_int = ((int(lo) + int(hi)) // 2 // page_size) * page_size
            mid_int = max(mid_int, page_size)
            probe_env = {
                **_gpu_env,
                "_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS": str(mid_int),
            }
            probe_desc = f"tokens={mid_int}"

        label = f"probe {probe_num}/{max_iterations + 1}"
        print(f"  [{label}] {probe_desc}  [~{remaining} left, ETA ~{eta_s:.0f}s]")
        sys.stdout.flush()

        stop_progress = threading.Event()
        t_iter_start = time.monotonic()
        is_tty = sys.stderr.isatty()

        def _print_progress(t0: float, expected: float, stop: threading.Event) -> None:
            if not is_tty:
                return
            term_width = shutil.get_terminal_size((80, 24)).columns
            bar_total = max(term_width - 40, 10)
            while not stop.wait(2):
                elapsed = time.monotonic() - t0
                frac = min(elapsed / expected, 1.0) if expected > 0 else 0
                filled = int(frac * bar_total)
                bar = "\u2588" * filled + "\u2591" * (bar_total - filled)
                pct = frac * 100
                line = f"    [{bar}] {elapsed:5.0f}s / ~{expected:.0f}s ({pct:3.0f}%)"
                sys.stderr.write(f"\r{line}")
                sys.stderr.flush()

        progress_thread = threading.Thread(
            target=_print_progress,
            args=(t_iter_start, baseline_time, stop_progress),
            daemon=True,
        )
        progress_thread.start()

        rc, wall, reports, raw_samples, stdout = _run_once(
            pytest_args,
            interval=interval,
            baseline_seconds=baseline_seconds,
            teardown_seconds=teardown_seconds,
            extra_env=probe_env,
            quiet=True,
            run_label=label,
            timeout=probe_timeout,
        )

        stop_progress.set()
        progress_thread.join(timeout=2)
        if is_tty:
            sys.stderr.write(
                "\r" + " " * shutil.get_terminal_size((80, 24)).columns + "\r"
            )
            sys.stderr.flush()

        iter_elapsed = time.monotonic() - t_iter_start
        elapsed_times.append(iter_elapsed)
        peak_mib = max((r.peak_mib for r in reports), default=0)
        all_peak_mibs.append(peak_mib)

        mid_value = mid_int
        if rc == 0:
            last_pass_value = mid_value
            last_pass_peak_mib = peak_mib
            last_pass_reports = reports
            last_pass_samples = raw_samples
            pass_wall_times.append(wall)
            hi = mid_value
            print(
                f"  [PASS] {probe_desc}, peak {_format_mib(peak_mib)}, "
                f"wall {wall:.0f}s, iter took {iter_elapsed:.0f}s"
            )
        else:
            lo = mid_value
            print(f"  [FAIL] {probe_desc}, iter took {iter_elapsed:.0f}s")

        # Early termination: if last 3 probes have peak VRAM within
        # _EARLY_STOP_RANGE_MIB, further bisection is in the noise floor.
        if len(all_peak_mibs) >= 4:
            recent = all_peak_mibs[-3:]
            peak_range = max(recent) - min(recent)
            if peak_range < _EARLY_STOP_RANGE_MIB:
                print(
                    f"  [EARLY STOP] Peak VRAM stable at ~{_format_mib(recent[-1])} "
                    f"for last 3 probes (range {peak_range} MiB < "
                    f"{_EARLY_STOP_RANGE_MIB} MiB threshold) "
                    f"-- stopping bisection early"
                )
                break

    # -- Results --
    test_name = next(
        (a for a in pytest_args if "::" in a or a.endswith(".py")),
        " ".join(pytest_args),
    )
    test_short = test_name.rsplit("::", 1)[-1] if "::" in test_name else test_name
    peak_gib = round(last_pass_peak_mib / 1024, 1)

    print(f"\n{'=' * 72}")
    if kv_bytes_mode:
        min_kv_bytes = int(last_pass_value)
        safe_kv_bytes = int(min_kv_bytes * _KV_SAFETY_FACTOR)
        # Round up to nearest 1000 for clean marker values
        safe_kv_bytes = ((safe_kv_bytes + 999) // 1000) * 1000
        safe_kv_mib = safe_kv_bytes // (1024 * 1024)
        min_kv_mib = min_kv_bytes // (1024 * 1024)

        print(f"  Minimum KV cache : {min_kv_mib} MiB ({min_kv_bytes:,} bytes)")
        print(
            f"  Safe KV cache    : {safe_kv_mib} MiB ({safe_kv_bytes:,} bytes) ({_KV_SAFETY_FACTOR:.0f}x safety)"
        )
        print(f"  Peak VRAM        : {_format_mib(last_pass_peak_mib)}")
        print()
        print("  Recommended markers:")
        print(f"    @pytest.mark.profiled_vram_gib({peak_gib})")
        print(
            f"    @pytest.mark.requested_vllm_kv_cache_bytes({safe_kv_bytes:_}),  # KV cache cap ({_KV_SAFETY_FACTOR:.0f}x safety over min={min_kv_bytes:_})"
        )
        print(f"{'=' * 72}")

    else:
        min_tokens = int(last_pass_value)
        safe_tokens = int(min_tokens * _KV_SAFETY_FACTOR)
        page_size = 16
        safe_tokens = ((safe_tokens + page_size - 1) // page_size) * page_size

        # Final validation probe at safe_tokens to get accurate profiled_vram_gib.
        # The bisection's last pass was at min_tokens; the recommended marker uses
        # safe_tokens which allocates more KV cache and thus more VRAM.
        print(f"  [final probe] Measuring VRAM at safe_tokens={safe_tokens}")
        sys.stdout.flush()
        rc_final, wall_final, reports_final, samples_final, stdout_final = _run_once(
            pytest_args,
            interval=interval,
            baseline_seconds=baseline_seconds,
            teardown_seconds=teardown_seconds,
            extra_env={
                **_gpu_env,
                "_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS": str(safe_tokens),
            },
            quiet=True,
            run_label="final",
            timeout=probe_timeout,
        )
        if rc_final == 0:
            last_pass_peak_mib = max((r.peak_mib for r in reports_final), default=0)
            last_pass_reports = reports_final
            last_pass_samples = samples_final
            pass_wall_times.append(wall_final)
            peak_gib = round(last_pass_peak_mib / 1024, 1)
            print(
                f"  [PASS] tokens={safe_tokens}, peak {_format_mib(last_pass_peak_mib)}, "
                f"wall {wall_final:.0f}s"
            )
        else:
            print(
                f"  [FAIL] tokens={safe_tokens} failed unexpectedly, "
                f"using VRAM from min_tokens={min_tokens} instead"
            )

        print(f"\n{'=' * 72}")
        print("MINIMUM KV TOKENS RESULT")
        print(f"{'=' * 72}")
        print(f"  Minimum tokens  : {min_tokens} (raw bisection result)")
        print(f"  Recommended     : {safe_tokens} ({_KV_SAFETY_FACTOR:.0f}x safety)")
        print(
            f"  Peak VRAM       : {_format_mib(last_pass_peak_mib)} (at {safe_tokens} tokens)"
        )
        print(f"  {test_short}: @pytest.mark.profiled_vram_gib({peak_gib})")
        print(
            f"  {test_short}: @pytest.mark.requested_sglang_kv_tokens({safe_tokens}),  # KV cache cap ({_KV_SAFETY_FACTOR:.0f}x safety over min={min_tokens})"
        )
    print(f"{'=' * 72}")

    # Marker recommendations
    requested_sglang_kv_tokens = safe_tokens if is_sglang else None
    requested_vllm_kv_cache_bytes = safe_kv_bytes if kv_bytes_mode else None
    min_kv_value = int(last_pass_value)
    if recommend:
        avg_pass_wall = sum(pass_wall_times) / len(pass_wall_times)
        recs, warnings = _recommend_markers(
            last_pass_reports,
            avg_pass_wall,
            model_name,
            num_runs=len(pass_wall_times),
            requested_sglang_kv_tokens=requested_sglang_kv_tokens,
            requested_vllm_kv_cache_bytes=requested_vllm_kv_cache_bytes,
            min_kv_value=min_kv_value,
        )
        _print_recommendations(recs, warnings, pytest_args=pytest_args)

    if csv_path and last_pass_samples:
        _write_csv(last_pass_samples, csv_path)
        print(f"Raw samples (last passing run) written to {csv_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Profile GPU memory during a pytest run.",
        usage="%(prog)s [options] [-- ] pytest-args...",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Sampling interval in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=3.0,
        help="Seconds to sample baseline before launching pytest (default: 3.0)",
    )
    parser.add_argument(
        "--teardown-seconds",
        type=float,
        default=5.0,
        help="Seconds to sample after pytest exits to measure teardown (default: 5.0)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Write raw samples to this CSV file",
    )
    parser.add_argument(
        "--no-recommend",
        action="store_true",
        default=False,
        help="Suppress marker recommendations",
    )
    parser.add_argument(
        "--no-find-min-vram",
        action="store_true",
        default=False,
        help="Disable the default binary-search mode that finds minimum VRAM. "
        "When set, runs a single profiling pass instead.",
    )
    parser.add_argument(
        "--kv-bytes",
        action="store_true",
        default=False,
        help="(No-op, kept for backward compat.) vLLM always uses KV byte "
        "bisection via _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES. "
        "Outputs @pytest.mark.requested_vllm_kv_cache_bytes(N).",
    )
    parser.add_argument(
        "--gpu",
        "--gpus",
        type=int,
        default=0,
        help="GPU index to profile on (default: 0). "
        "Sets CUDA_VISIBLE_DEVICES for the subprocess.",
    )

    raw = argv if argv is not None else sys.argv[1:]

    if "--" in raw:
        split_idx = raw.index("--")
        args = parser.parse_args(raw[:split_idx])
        pytest_args = raw[split_idx + 1 :]
    else:
        args, pytest_args = parser.parse_known_args(raw)

    if not pytest_args:
        parser.error("No pytest arguments provided")

    # Validate that test file paths actually exist
    for arg in pytest_args:
        if arg.startswith("-"):
            continue
        test_path = arg.split("::")[0]
        looks_like_test_path = test_path.endswith(".py") or (os.path.sep in test_path)
        if looks_like_test_path and not os.path.exists(test_path):
            parser.error(f"Test path does not exist: {test_path}")

    gpu_idx = args.gpu
    gpu_info = _query_gpu_stats()
    if not gpu_info:
        raise RuntimeError("NVML returned no GPU data")
    if gpu_idx >= len(gpu_info):
        raise RuntimeError(
            f"GPU {gpu_idx} not found (available: 0..{len(gpu_info) - 1})"
        )

    used_mib = gpu_info[gpu_idx][1]
    total_mib = gpu_info[gpu_idx][2]
    hogged_pct = used_mib / total_mib * 100
    if hogged_pct > 10:
        print(
            f"\nWARNING: GPU {gpu_idx}: {used_mib / 1024:.1f} GiB ({hogged_pct:.0f}%) "
            f"of GPU memory is already in use! Results may be inaccurate.\n"
        )

    gpu_env = {"CUDA_VISIBLE_DEVICES": str(gpu_idx)}

    if not args.no_find_min_vram:
        return _find_min_vram(
            pytest_args,
            interval=args.interval,
            baseline_seconds=args.baseline_seconds,
            teardown_seconds=args.teardown_seconds,
            recommend=not args.no_recommend,
            csv_path=args.csv,
            kv_bytes_mode=args.kv_bytes,
            gpu_index=gpu_idx,
        )

    model_name = _extract_model_from_markers(pytest_args)
    is_sglang = _is_sglang_test(pytest_args)

    rc, wall_secs, reports, samples, stdout = _run_once(
        pytest_args,
        interval=args.interval,
        baseline_seconds=args.baseline_seconds,
        teardown_seconds=args.teardown_seconds,
        extra_env=gpu_env,
        run_label="profile" if is_sglang else None,
    )

    _print_report(reports, rc, wall_secs, model_name=model_name)

    if not args.no_recommend and reports:
        requested_sglang_kv_tokens = None
        if is_sglang:
            requested_sglang_kv_tokens = _extract_requested_sglang_kv_tokens(stdout)
        recs, warnings = _recommend_markers(
            reports,
            wall_secs,
            model_name=model_name,
            requested_sglang_kv_tokens=requested_sglang_kv_tokens,
        )
        _print_recommendations(recs, warnings, pytest_args=pytest_args)

    if args.csv:
        _write_csv(samples, args.csv)
        print(f"Raw samples written to {args.csv}")

    return rc


if __name__ == "__main__":
    if (
        os.environ.get("CI")
        or os.environ.get("GITHUB_ACTIONS")
        or os.environ.get("GITLAB_CI")
    ):
        print("ERROR: profile_pytest.py must not run in CI.", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(main())
