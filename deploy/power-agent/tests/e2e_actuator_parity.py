#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E parity harness: NvmlActuator vs DcgmActuator on real hardware.

Verifies the v1.4 design's central claim: that against real GPUs,
both actuator paths produce observably identical results. Specifically:

  1. device_count returns the same number of GPUs.
  2. get_uuid returns the same UUID per GPU (hardware-level identifier,
     not library-specific).
  3. constraints_w returns the same [min, max] range.
  4. apply_cap produces the same effective power limit (verified by
     `nvidia-smi --query-gpu=power.limit` between calls).
  5. restore_default returns the GPU to its factory default (also
     verified via nvidia-smi).
  6. list_running_pids — both actuators call NVML on this method
     (DcgmActuator deliberately uses NVML),
     so this is more of a smoke test than a parity test.

NOT covered (intentionally — those are unit-test territory or live-node
operational tests):

  * Multi-pod conflict resolution / annotation parsing — needs a K8s
    API, out of scope for actuator-only parity.
  * dcgmConfigEnforce re-assertion semantics — needs a clobbering
    test (e.g. `nvidia-smi -pl` mid-test) plus a wait loop; left for
    a follow-up live-node test.
  * Stale-handle recovery — restarting the hostengine mid-test is a
    multi-pod orchestration concern, covered by unit tests.

Run:
    python3 e2e_actuator_parity.py --test-watts 250 --hostengine-host localhost

The script expects nv-hostengine running on `--hostengine-host:--hostengine-port`
(default localhost:5555). Start it locally with:
    nv-hostengine -b 127.0.0.1 -p 5555

Exit codes:
    0 — all parity checks passed
    1 — a parity check failed (diff printed)
    2 — pre-flight failed (no GPUs, hostengine unreachable, etc.)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Optional


def _add_actuator_path():
    """Make `actuator` + `power_agent` importable when run as a script."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))  # deploy/power-agent


_add_actuator_path()


# ---------------------------------------------------------------------------
# Stub metrics — the actuators require something with .applied_limit_watts /
# .apply_failures_total / .cap_clamped_total. A real prometheus_client
# CollectorRegistry would work too, but we don't need that overhead here.
# ---------------------------------------------------------------------------


class _StubLabels:
    def set(self, *args, **kwargs):
        pass

    def inc(self, *args, **kwargs):
        pass


class _StubMetricFamily:
    def labels(self, *args, **kwargs):
        return _StubLabels()

    def inc(self, *args, **kwargs):
        pass


class StubMetrics:
    """Bare minimum surface to satisfy NvmlActuator/DcgmActuator."""

    applied_limit_watts = _StubMetricFamily()
    apply_failures_total = _StubMetricFamily()
    cap_clamped_total = _StubMetricFamily()


# ---------------------------------------------------------------------------
# Ground-truth read via nvidia-smi. Independent of both NVML and DCGM
# Python bindings, so it catches the case where both libraries agree on
# the same wrong value.
# ---------------------------------------------------------------------------


def nvidia_smi_power_limit(gpu_idx: int) -> float:
    """Return the current power limit on GPU `gpu_idx`, in watts.

    Calls `nvidia-smi --query-gpu=power.limit` rather than going through
    the actuators — this is the third-party ground truth.
    """
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu_idx),
            "--query-gpu=power.limit",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return float(out)


def nvidia_smi_default_limit(gpu_idx: int) -> float:
    """Factory default power limit, in watts."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu_idx),
            "--query-gpu=power.default_limit",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return float(out)


# ---------------------------------------------------------------------------
# Per-actuator probe — collects everything we want to compare in one pass.
# ---------------------------------------------------------------------------


def probe(
    actuator,
    test_watts: int,
    *,
    verify_writes: bool,
    sleep_s: float,
    skip_busy_gpus: bool = False,
):
    """Run a full lifecycle against `actuator`, return a result dict.

    Steps per GPU:
      1. Snapshot UUID, constraints, current pids.
      2. Read cap via nvidia-smi (ground truth, before apply).
      3. apply_cap(gpu, test_watts) — measure return value.
      4. Read cap via nvidia-smi (ground truth, after apply).
      5. restore_default(gpu).
      6. Read cap via nvidia-smi (ground truth, after restore).

    `verify_writes=False` skips the apply/restore steps entirely
    (read-only mode).

    `skip_busy_gpus=True` keeps the read path on every GPU but skips
    the write path on any GPU with active compute processes — safe
    pattern for shared clusters where touching caps on a busy GPU
    would perturb someone else's workload.
    """
    actuator.init()
    try:
        n = actuator.device_count()
        gpus = []
        for i in range(n):
            uuid = actuator.get_uuid(i)
            min_w, max_w = actuator.constraints_w(i)
            pids = actuator.list_running_pids(i)
            ns_before = nvidia_smi_power_limit(i)
            ns_default = nvidia_smi_default_limit(i)

            write_skipped_reason = None
            if not verify_writes:
                write_skipped_reason = "read-only mode"
            elif skip_busy_gpus and len(pids) > 0:
                write_skipped_reason = f"busy ({len(pids)} compute pid(s))"

            if write_skipped_reason is None:
                eff = actuator.apply_cap(i, test_watts)
                time.sleep(sleep_s)
                ns_applied = nvidia_smi_power_limit(i)
                actuator.restore_default(i)
                time.sleep(sleep_s)
                ns_restored = nvidia_smi_power_limit(i)
            else:
                eff = ns_applied = ns_restored = None

            gpus.append(
                {
                    "idx": i,
                    "uuid": uuid,
                    "min_w": min_w,
                    "max_w": max_w,
                    "pid_count": len(pids),
                    "ns_before": ns_before,
                    "ns_default": ns_default,
                    "apply_cap_returned": eff,
                    "ns_after_apply": ns_applied,
                    "ns_after_restore": ns_restored,
                    "write_skipped_reason": write_skipped_reason,
                }
            )
        return {"device_count": n, "gpus": gpus}
    finally:
        actuator.shutdown()


# ---------------------------------------------------------------------------
# Parity diff — compares two probe results, prints any discrepancies,
# returns the count of failures.
# ---------------------------------------------------------------------------


def parity_check(nvml_result, dcgm_result, *, tolerance_w: float):
    """Return number of failed parity checks (0 = pass)."""
    failures = []

    if nvml_result["device_count"] != dcgm_result["device_count"]:
        failures.append(
            f"device_count diff: nvml={nvml_result['device_count']} "
            f"dcgm={dcgm_result['device_count']}"
        )
        return len(failures)  # bail — per-GPU comparison meaningless

    for nvml_gpu, dcgm_gpu in zip(nvml_result["gpus"], dcgm_result["gpus"]):
        idx = nvml_gpu["idx"]

        if nvml_gpu["uuid"] != dcgm_gpu["uuid"]:
            failures.append(
                f"GPU {idx}: uuid diff — nvml={nvml_gpu['uuid']!r} "
                f"dcgm={dcgm_gpu['uuid']!r}"
            )

        # Constraints: allow ±1 W slop (DCGM returns floats; NVML
        # returns ints; both come from the same firmware register but
        # the conversion path differs slightly).
        if abs(nvml_gpu["min_w"] - dcgm_gpu["min_w"]) > tolerance_w:
            failures.append(
                f"GPU {idx}: min_w diff — nvml={nvml_gpu['min_w']} "
                f"dcgm={dcgm_gpu['min_w']}"
            )
        if abs(nvml_gpu["max_w"] - dcgm_gpu["max_w"]) > tolerance_w:
            failures.append(
                f"GPU {idx}: max_w diff — nvml={nvml_gpu['max_w']} "
                f"dcgm={dcgm_gpu['max_w']}"
            )

        # apply_cap return value
        if (
            nvml_gpu["apply_cap_returned"] is not None
            and dcgm_gpu["apply_cap_returned"] is not None
        ):
            if (
                abs(nvml_gpu["apply_cap_returned"] - dcgm_gpu["apply_cap_returned"])
                > tolerance_w
            ):
                failures.append(
                    f"GPU {idx}: apply_cap return diff — "
                    f"nvml={nvml_gpu['apply_cap_returned']} "
                    f"dcgm={dcgm_gpu['apply_cap_returned']}"
                )

        # Ground truth: nvidia-smi reading after apply must agree
        # between paths (this is the critical end-to-end check)
        if (
            nvml_gpu["ns_after_apply"] is not None
            and dcgm_gpu["ns_after_apply"] is not None
        ):
            if (
                abs(nvml_gpu["ns_after_apply"] - dcgm_gpu["ns_after_apply"])
                > tolerance_w
            ):
                failures.append(
                    f"GPU {idx}: nvidia-smi after-apply diff — "
                    f"nvml={nvml_gpu['ns_after_apply']:.1f} W "
                    f"dcgm={dcgm_gpu['ns_after_apply']:.1f} W"
                )

        # Restore must return to default (per actuator, vs ground truth)
        if nvml_gpu["ns_after_restore"] is not None:
            if abs(nvml_gpu["ns_after_restore"] - nvml_gpu["ns_default"]) > tolerance_w:
                failures.append(
                    f"GPU {idx}: nvml restore_default — "
                    f"got {nvml_gpu['ns_after_restore']:.1f} W, "
                    f"expected {nvml_gpu['ns_default']:.1f} W (factory default)"
                )
        if dcgm_gpu["ns_after_restore"] is not None:
            if abs(dcgm_gpu["ns_after_restore"] - dcgm_gpu["ns_default"]) > tolerance_w:
                failures.append(
                    f"GPU {idx}: dcgm restore_default — "
                    f"got {dcgm_gpu['ns_after_restore']:.1f} W, "
                    f"expected {dcgm_gpu['ns_default']:.1f} W (factory default)"
                )

    for f in failures:
        print(f"  FAIL: {f}", file=sys.stderr)
    return len(failures)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def print_probe(label, result):
    print(f"\n=== {label} ===")
    print(f"device_count = {result['device_count']}")
    for gpu in result["gpus"]:
        print(f"  GPU {gpu['idx']}:")
        print(f"    uuid              = {gpu['uuid']}")
        print(f"    constraints       = [{gpu['min_w']}, {gpu['max_w']}] W")
        print(f"    pid_count         = {gpu['pid_count']}")
        print(f"    nvidia-smi before = {gpu['ns_before']:.1f} W")
        print(f"    nvidia-smi default= {gpu['ns_default']:.1f} W")
        if gpu["apply_cap_returned"] is not None:
            print(f"    apply_cap return  = {gpu['apply_cap_returned']} W")
            print(f"    nvidia-smi applied= {gpu['ns_after_apply']:.1f} W")
            print(f"    nvidia-smi restore= {gpu['ns_after_restore']:.1f} W")
        elif gpu.get("write_skipped_reason"):
            print(f"    write path        = SKIPPED ({gpu['write_skipped_reason']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="E2E parity test: NvmlActuator vs DcgmActuator"
    )
    parser.add_argument(
        "--test-watts",
        type=int,
        default=250,
        help="Watts to apply via apply_cap. Must be inside the SKU "
        "[min, max] range; clamping behaviour is unit-tested elsewhere.",
    )
    parser.add_argument(
        "--hostengine-host",
        default="127.0.0.1",
        help="nv-hostengine address for the DCGM path.",
    )
    parser.add_argument(
        "--hostengine-port",
        type=int,
        default=5555,
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Pass enforce=True to DcgmActuator (test the dcgmConfigEnforce path).",
    )
    parser.add_argument(
        "--tolerance-watts",
        type=float,
        default=2.0,
        help="Max diff between actuator paths to still count as parity. "
        "DCGM returns floats from firmware, NVML returns ints in mW; "
        "small rounding differences (<2 W) are expected.",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Skip apply_cap/restore_default — read-only parity check "
        "(uuid, constraints, pids only). Use when you don't want to "
        "touch caps on a shared node.",
    )
    parser.add_argument(
        "--skip-busy-gpus",
        action="store_true",
        help="Apply/restore only on GPUs with zero compute pids. "
        "Reads stay on every GPU. Safe pattern for shared clusters: "
        "exercises the write path where possible without disturbing "
        "anyone else's running workload.",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=0.5,
        help="Sleep between cap write and nvidia-smi readback. The "
        "driver propagates the cap inside ~100ms but DCGM's RPC adds "
        "a few hundred ms more.",
    )
    parser.add_argument(
        "--skip-nvml",
        action="store_true",
        help="DCGM-only run. Useful for catching DCGM-specific bugs "
        "without an NVML reference.",
    )
    parser.add_argument(
        "--skip-dcgm",
        action="store_true",
        help="NVML-only run. Useful when nv-hostengine isn't available.",
    )
    args = parser.parse_args()

    if args.skip_nvml and args.skip_dcgm:
        print(
            "ERROR: --skip-nvml and --skip-dcgm together is a no-op.", file=sys.stderr
        )
        return 2

    # Pre-flight: nvidia-smi must work, otherwise everything downstream lies.
    try:
        subprocess.check_call(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"PRE-FLIGHT FAIL: nvidia-smi unavailable: {e}", file=sys.stderr)
        return 2

    # Import the actuators only after sys.path is set up (above).
    # Initialize NVML once. NvmlActuator.init() is a no-op by design
    # (PR A contract: PowerAgent.__init__ owns the nvmlInit call); when
    # we drive the actuator directly here, we have to do it ourselves.
    # nvmlInit is idempotent, so even if DcgmActuator also touches NVML
    # under the hood (via list_running_pids), this stays safe.
    import pynvml
    from actuator import DcgmActuator, NvmlActuator

    pynvml.nvmlInit()

    metrics = StubMetrics()
    nvml_result: Optional[dict] = None
    dcgm_result: Optional[dict] = None

    try:
        if not args.skip_nvml:
            print("Running NvmlActuator probe...")
            nvml = NvmlActuator(metrics=metrics)
            nvml_result = probe(
                nvml,
                args.test_watts,
                verify_writes=not args.read_only,
                sleep_s=args.sleep_s,
                skip_busy_gpus=args.skip_busy_gpus,
            )
            print_probe("NvmlActuator", nvml_result)

        if not args.skip_dcgm:
            print(
                f"\nRunning DcgmActuator probe ("
                f"hostengine={args.hostengine_host}:{args.hostengine_port}, "
                f"enforce={args.enforce})..."
            )
            dcgm = DcgmActuator(
                host=args.hostengine_host,
                port=args.hostengine_port,
                enforce=args.enforce,
                metrics=metrics,
            )
            try:
                dcgm_result = probe(
                    dcgm,
                    args.test_watts,
                    verify_writes=not args.read_only,
                    sleep_s=args.sleep_s,
                    skip_busy_gpus=args.skip_busy_gpus,
                )
            except Exception as e:
                print(
                    f"PRE-FLIGHT FAIL: DcgmActuator could not init "
                    f"({type(e).__name__}: {e}). Is nv-hostengine running?",
                    file=sys.stderr,
                )
                return 2
            print_probe("DcgmActuator", dcgm_result)

        # Parity comparison
        if nvml_result is not None and dcgm_result is not None:
            print("\n=== Parity check ===")
            n_failures = parity_check(
                nvml_result, dcgm_result, tolerance_w=args.tolerance_watts
            )
            if n_failures == 0:
                print(
                    f"  PASS: NvmlActuator and DcgmActuator agree on all probes "
                    f"(tolerance ±{args.tolerance_watts} W)."
                )
                return 0
            print(f"  {n_failures} parity check(s) failed.")
            return 1

        print("\n(Skipped parity comparison — one actuator was skipped.)")
        return 0
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
