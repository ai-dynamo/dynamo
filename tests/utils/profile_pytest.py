#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile GPU VRAM usage during a pytest run.

By default, runs a binary search on ``DYN_GPU_MEMORY_FRACTION_OVERRIDE`` to find the
minimum VRAM the test needs, then prints recommended pytest markers.

**IMPORTANT**: The binary search works by setting the ``DYN_GPU_MEMORY_FRACTION_OVERRIDE``
environment variable to a fraction between 0.0 and 1.0 (fraction of total GPU
RAM).  The test under profile **MUST** honor this env var — either directly
(see ``test_mock_gpu_alloc.py`` for an example) or via launch scripts that
pass it through to the engine (e.g. ``agg.sh`` passes it as
``--gpu-memory-utilization`` to vLLM).  If the test ignores this variable,
the binary search cannot find a boundary and every profile will pass.

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
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Safety margin for VRAM tier recommendations.  Peak VRAM is multiplied by
# this factor before comparing against tier thresholds, so the recommended
# tier has headroom for variance across runs.
_VRAM_SAFETY_FACTOR = 1.1

# Phase detection: a memory jump exceeding this threshold (MiB) between
# consecutive samples marks a phase boundary.
_PHASE_JUMP_MIB = 200

# How long memory must be stable (within this tolerance) to consider it
# a plateau, in consecutive samples.
_PLATEAU_TOLERANCE_MIB = 50
_PLATEAU_MIN_SAMPLES = 3


def _extract_model_from_markers(pytest_args: list[str]) -> str | None:
    """Extract the model name from @pytest.mark.model(...) via pytest-json-report.

    Runs ``pytest --collect-only`` with the json-report plugin to inspect markers
    without executing the test.  Returns None if the plugin is missing or the
    test has no ``model`` marker.
    """
    json_path = "/tmp/_profile_collect.json"
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


def _query_nvidia_smi() -> list[tuple[int, int, int, int]]:
    """Return [(gpu_idx, mem_used_mib, mem_total_mib, util_pct), ...] for all GPUs."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.error("nvidia-smi query failed: %s", exc)
        raise

    results = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                results.append(
                    (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
                )
            except ValueError:
                logger.warning("skipping malformed nvidia-smi line: %r", line)
    return results


class _Sampler:
    """Background thread that polls nvidia-smi at a fixed interval."""

    def __init__(self, interval: float = 1.0):
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
                for gpu_idx, mem_used, mem_total, util_pct in _query_nvidia_smi():
                    self.samples.append(
                        GpuSample(ts, gpu_idx, mem_used, mem_total, util_pct)
                    )
            except (subprocess.SubprocessError, OSError):
                pass  # GPU busy; skip this sample and retry next interval
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
    sep = "=" * 72

    print(f"\n{sep}")
    print("GPU MEMORY PROFILE")
    print(sep)
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

    print(f"\n{sep}\n")


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

    # -- Hardware: VRAM requirement --
    if used_vram > _PLATEAU_TOLERANCE_MIB:
        padded_peak_mib = int(max_peak_mib * _VRAM_SAFETY_FACTOR)
        padded_peak_gib = int(math.ceil(padded_peak_mib / 1024))
        recs.append(
            MarkerRecommendation(
                f"max_vram({padded_peak_gib})",
                f"peak {_format_mib(max_peak_mib)} GPU RAM used "
                f"(+10% safety: {_format_mib(padded_peak_mib)})",
            )
        )

        # Warn about GPU cards that would OOM
        for card_gib, card_name in _GPU_REFERENCE_CARDS:
            if padded_peak_gib > card_gib:
                warnings.append(f"Will OOM on {card_name} ({card_gib} GiB).")

    # -- Timeout --
    timeout_val = int(math.ceil(wall_secs * 3.0))
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
    sep = "=" * 72
    print(sep)
    print("Recommended markers to add to your pytest. You can copy-paste this:")
    print(sep)
    if pytest_args:
        print(
            f"# Measured using: tests/utils/profile_pytest.py {' '.join(pytest_args)}"
        )
    else:
        print("# Measured using: tests/utils/profile_pytest.py")
    for r in recs:
        print(f"@pytest.mark.{r.marker}  # {r.reason}")
    if warnings:
        print()
        for w in warnings:
            print(f"  WARNING: {w}")
    print(f"\n{sep}\n")


_DEFAULT_PROBE_TIMEOUT = 300  # 5 minutes max per profile run


def _run_once(
    pytest_args: list[str],
    interval: float = 1.0,
    baseline_seconds: float = 3.0,
    teardown_seconds: float = 5.0,
    extra_env: dict[str, str] | None = None,
    quiet: bool = False,
    run_label: str | None = None,
    timeout: float = _DEFAULT_PROBE_TIMEOUT,
) -> tuple[int, float, list[GpuReport], list[GpuSample]]:
    """Run pytest once with GPU sampling.

    When *run_label* is set, each line of pytest stdout/stderr is prefixed
    with ``[run_label]`` so multi-run output is easy to follow.

    Returns (exit_code, wall_secs, reports, raw_samples).
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
    try:
        result = subprocess.run(
            pytest_cmd,
            env=env,
            capture_output=capture,
            text=capture or None,
            timeout=timeout,
        )
        rc = result.returncode
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
        for line in result.stdout.splitlines():
            print(f"{prefix}{line}")
        for line in result.stderr.splitlines():
            print(f"{prefix}{line}", file=sys.stderr)
    sys.stdout.flush()
    wall_secs = time.monotonic() - t_start
    test_end = time.monotonic() - sampler._t0

    if not quiet:
        print(f"Sampling teardown for {teardown_seconds}s ...")
    time.sleep(teardown_seconds)

    sampler.stop()
    reports = _build_reports(sampler.samples, baseline_end, test_end)
    return rc, wall_secs, reports, sampler.samples


def _find_min_vram(
    pytest_args: list[str],
    interval: float = 1.0,
    baseline_seconds: float = 2.0,
    teardown_seconds: float = 2.0,
) -> int:
    """Binary search DYN_GPU_MEMORY_FRACTION_OVERRIDE to find the minimum VRAM a test needs.

    Sets DYN_GPU_MEMORY_FRACTION_OVERRIDE env var (honored by agg.sh and similar scripts),
    runs the test at each profile point, and bisects until the boundary is found.
    """
    gpu_info = _query_nvidia_smi()
    if not gpu_info:
        raise RuntimeError("nvidia-smi returned no GPU data")
    used_mib = gpu_info[0][1]
    total_mib = gpu_info[0][2]
    free_mib = total_mib - used_mib
    total_gib = total_mib / 1024

    model_name = _extract_model_from_markers(pytest_args)

    sep = "=" * 72
    print(f"\n{sep}")
    print("FIND MINIMUM VRAM (binary search)")
    print(sep)
    print(f"  GPU total : {total_gib:.1f} GiB")
    print(
        f"  GPU free  : {free_mib / 1024:.1f} GiB  "
        f"(in use: {used_mib / 1024:.1f} GiB)"
    )
    print(f"  Test      : {' '.join(pytest_args)}")
    if model_name:
        print(f"  Model     : {model_name}")

    # Warn if something is already consuming significant GPU memory
    hogged_pct = used_mib / total_mib * 100
    if hogged_pct > 10:
        print(f"\n  {'!' * 72}")
        print(
            f"  WARNING: {used_mib / 1024:.1f} GiB ({hogged_pct:.0f}%) of GPU memory "
            f"is already in use!"
        )
        print("  Another process is hogging the GPU. Results will be inaccurate")
        print(
            "  because DYN_GPU_MEMORY_FRACTION_OVERRIDE is a fraction of TOTAL memory,"
        )
        print("  not FREE memory. Kill other GPU processes first.")
        print(f"  {'!' * 72}")
    print()

    lo = 0.05
    hi = 0.95
    tolerance = 0.05
    max_iterations = math.ceil(math.log2((hi - lo) / tolerance))
    last_pass_util: float | None = None
    last_pass_peak_mib: int = 0
    elapsed_times: list[float] = []
    all_peak_mibs: list[int] = []
    pass_wall_times: list[float] = []

    print(f"  Range   : {lo:.0%} - {hi:.0%}  (tolerance {tolerance:.0%})")
    print(
        f"  Max iter: {max_iterations + 1} (1 validation + {max_iterations} bisections)"
    )
    print()

    # First, verify the test passes at hi (0.95)
    print(
        f"  [profile 1/{max_iterations + 1}] DYN_GPU_MEMORY_FRACTION_OVERRIDE={hi:.2f} "
        f"(allowed max GPU {hi * total_gib:.1f} GiB)  [validation run]"
    )
    sys.stdout.flush()
    t_iter_start = time.monotonic()
    label = f"profile 1/{max_iterations + 1}"
    rc, wall, reports, _ = _run_once(
        pytest_args,
        interval=interval,
        baseline_seconds=baseline_seconds,
        teardown_seconds=teardown_seconds,
        extra_env={"DYN_GPU_MEMORY_FRACTION_OVERRIDE": f"{hi:.2f}"},
        quiet=True,
        run_label=label,
    )
    iter_elapsed = time.monotonic() - t_iter_start
    elapsed_times.append(iter_elapsed)
    if rc != 0:
        print(
            f"  [FAIL] allowed GPU = {hi * total_gib:.1f} GiB ({hi:.0%}), "
            f"test fails even at max utilization. Cannot determine minimum."
        )
        return rc

    peak_mib = max((r.peak_mib for r in reports), default=0)
    all_peak_mibs.append(peak_mib)
    last_pass_util = hi
    last_pass_peak_mib = peak_mib
    last_pass_reports = reports
    pass_wall_times.append(wall)
    print(
        f"  [PASS] allowed GPU = {hi * total_gib:.1f} GiB ({hi:.0%}), "
        f"peak GPU used = {_format_mib(peak_mib)}, wall {wall:.0f}s, "
        f"iter took {iter_elapsed:.0f}s"
    )

    # Use 2x the first profile's time as the timeout for subsequent profiles.
    # If a profile takes longer than this, it's likely stuck in teardown.
    baseline_time = iter_elapsed
    probe_timeout = max(baseline_time * 2, 60)
    print(f"  Profile timeout: {probe_timeout:.0f}s (2x first profile)")

    iteration = 0
    while (hi - lo) > tolerance:
        iteration += 1
        probe_num = iteration + 1
        mid = (lo + hi) / 2
        remaining = max_iterations + 1 - probe_num
        avg_iter = sum(elapsed_times) / len(elapsed_times)
        eta_s = remaining * avg_iter

        label = f"profile {probe_num}/{max_iterations + 1}"
        print(
            f"\n  [{label}] "
            f"DYN_GPU_MEMORY_FRACTION_OVERRIDE={mid:.2f} "
            f"(allowed max GPU {mid * total_gib:.1f} GiB)  "
            f"[~{remaining} left, ETA ~{eta_s:.0f}s]"
        )
        sys.stdout.flush()

        stop_progress = threading.Event()
        t_iter_start = time.monotonic()

        def _print_progress(t0: float, expected: float, stop: threading.Event) -> None:
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

        rc, wall, reports, _ = _run_once(
            pytest_args,
            interval=interval,
            baseline_seconds=baseline_seconds,
            teardown_seconds=teardown_seconds,
            extra_env={"DYN_GPU_MEMORY_FRACTION_OVERRIDE": f"{mid:.2f}"},
            quiet=True,
            run_label=label,
            timeout=probe_timeout,
        )

        stop_progress.set()
        progress_thread.join(timeout=2)
        sys.stderr.write("\r" + " " * shutil.get_terminal_size((80, 24)).columns + "\r")
        sys.stderr.flush()

        iter_elapsed = time.monotonic() - t_iter_start
        elapsed_times.append(iter_elapsed)
        peak_mib = max((r.peak_mib for r in reports), default=0)
        all_peak_mibs.append(peak_mib)

        if rc == 0:
            last_pass_util = mid
            last_pass_peak_mib = peak_mib
            last_pass_reports = reports
            pass_wall_times.append(wall)
            hi = mid
            print(
                f"  [PASS] allowed GPU = {mid * total_gib:.1f} GiB ({mid:.0%}), "
                f"peak GPU used = {_format_mib(peak_mib)}, wall {wall:.0f}s, "
                f"iter took {iter_elapsed:.0f}s"
            )
        else:
            lo = mid
            print(
                f"  [FAIL] allowed GPU = {mid * total_gib:.1f} GiB ({mid:.0%}), "
                f"OOM or error, iter took {iter_elapsed:.0f}s"
            )

    # Detect if DYN_GPU_MEMORY_FRACTION_OVERRIDE is being ignored: all peaks are nearly
    # identical despite wildly different utilization caps.
    if len(all_peak_mibs) >= 3:
        peak_range = max(all_peak_mibs) - min(all_peak_mibs)
        if peak_range < _PLATEAU_TOLERANCE_MIB:
            print(f"\n  {'!' * 72}")
            print(
                f"  WARNING: Peak VRAM was ~{_format_mib(all_peak_mibs[0])} across ALL "
                f"{len(all_peak_mibs)} probes (range: {peak_range} MiB)."
            )
            print(
                "  This strongly suggests the test IGNORES the DYN_GPU_MEMORY_FRACTION_OVERRIDE"
            )
            print("  env var.  Binary search results are UNRELIABLE — no marker")
            print("  recommendation will be provided.")
            print("  ")
            print(
                "  FIX: The test (or its launch script) must read DYN_GPU_MEMORY_FRACTION_OVERRIDE"
            )
            print("  and pass --gpu-memory-utilization to vLLM / the engine.")
            print("  See tests/README.md 'GPU VRAM Profiler' for details.")
            print(f"  {'!' * 72}")
            return 4

    # Results
    assert last_pass_util is not None
    min_vram_gib = last_pass_util * total_gib

    padded_peak_mib = int(last_pass_peak_mib * _VRAM_SAFETY_FACTOR)
    padded_peak_gib = int(math.ceil(padded_peak_mib / 1024))

    # Extract a short test name from pytest args for the summary
    test_name = next(
        (a for a in pytest_args if "::" in a or a.endswith(".py")),
        " ".join(pytest_args),
    )
    test_short = test_name.rsplit("::", 1)[-1] if "::" in test_name else test_name

    fits_on = [(g, n) for g, n in _GPU_REFERENCE_CARDS if padded_peak_gib <= g]
    oom_on = [(g, n) for g, n in _GPU_REFERENCE_CARDS if padded_peak_gib > g]

    print(f"\n{sep}")
    print("MINIMUM VRAM RESULT")
    print(sep)
    print(f"  Lowest passing utilization : {last_pass_util:.0%}")
    print(
        f"  Minimum VRAM needed        : ~{min_vram_gib:.1f} GiB "
        f"(peak observed: {_format_mib(last_pass_peak_mib)}, "
        f"+10% safety: {_format_mib(padded_peak_mib)})"
    )
    print()
    print(f"  # {test_short}: @pytest.mark.max_vram({padded_peak_gib})")
    if fits_on:
        print(f"  # Fits on: {', '.join(f'{n} ({g} GiB)' for g, n in fits_on)}")
    if oom_on:
        print(f"  # Will OOM on: {', '.join(f'{n} ({g} GiB)' for g, n in oom_on)}")
    print(f"{sep}\n")

    # Full marker recommendations using average wall time across all passing runs
    avg_pass_wall = sum(pass_wall_times) / len(pass_wall_times)
    recs, warnings = _recommend_markers(
        last_pass_reports, avg_pass_wall, model_name, num_runs=len(pass_wall_times)
    )
    _print_recommendations(recs, warnings, pytest_args=pytest_args)
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
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
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
        if not os.path.exists(test_path):
            parser.error(f"Test path does not exist: {test_path}")

    gpu_info = _query_nvidia_smi()
    if not gpu_info:
        raise RuntimeError("nvidia-smi returned no GPU data")

    used_mib = gpu_info[0][1]
    total_mib = gpu_info[0][2]
    hogged_pct = used_mib / total_mib * 100
    if hogged_pct > 10:
        print(
            f"\nWARNING: {used_mib / 1024:.1f} GiB ({hogged_pct:.0f}%) of GPU memory "
            f"is already in use! Results may be inaccurate.\n"
        )

    if not args.no_find_min_vram:
        return _find_min_vram(
            pytest_args,
            interval=args.interval,
            baseline_seconds=args.baseline_seconds,
            teardown_seconds=args.teardown_seconds,
        )

    model_name = _extract_model_from_markers(pytest_args)

    rc, wall_secs, reports, samples = _run_once(
        pytest_args,
        interval=args.interval,
        baseline_seconds=args.baseline_seconds,
        teardown_seconds=args.teardown_seconds,
    )

    _print_report(reports, rc, wall_secs, model_name=model_name)

    if not args.no_recommend and reports:
        recs, warnings = _recommend_markers(reports, wall_secs, model_name=model_name)
        _print_recommendations(recs, warnings, pytest_args=pytest_args)

    if args.csv:
        _write_csv(samples, args.csv)
        print(f"Raw samples written to {args.csv}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
