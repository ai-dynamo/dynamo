#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E parity harness: NvmlActuator vs DcgmActuator on real hardware.

Verifies the actuator abstraction's central claim: that against real GPUs,
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
  * Stale-handle recovery — restarting the hostengine mid-test is a
    multi-pod orchestration concern. Unit tests cover the state machine, but
    the chart README still requires a live lifecycle qualification row.

The Power Agent chart README defines the required live lifecycle and advertised
runtime-image qualification record. This actuator-only fixture cannot replace
those rows.

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
# Stub metrics — the actuators require something with .configured_cap_watts /
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

    configured_cap_watts = _StubMetricFamily()
    apply_failures_total = _StubMetricFamily()
    cap_clamped_total = _StubMetricFamily()


# ---------------------------------------------------------------------------
# Ground-truth read via nvidia-smi. Independent of both NVML and DCGM
# Python bindings, so it catches the case where both libraries agree on
# the same wrong value.
# ---------------------------------------------------------------------------


def nvidia_smi_power_limit(gpu_uuid: str) -> float:
    """Return the current power limit on the GPU carrying `gpu_uuid`, in watts.

    Queries by UUID (``nvidia-smi -i`` accepts a GPU UUID), NOT by an
    actuator-native index: DCGM and NVML index spaces can differ, so an
    index-based read could observe a DIFFERENT physical GPU than the actuator
    just probed and report false parity failures. Binding ground truth to the
    UUID the actuator reported keeps this read on the same physical GPU on both
    paths. This is the third-party ground truth (independent of both bindings).
    """
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            gpu_uuid,
            "--query-gpu=power.limit",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return float(out)


def nvidia_smi_default_limit(gpu_uuid: str) -> float:
    """Factory default power limit for the GPU carrying `gpu_uuid`, in watts.

    Queried by UUID for the same reason as `nvidia_smi_power_limit`.
    """
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            gpu_uuid,
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
    require_default_before_write: bool = False,
):
    """Run a full lifecycle against `actuator`, return a result dict.

    Steps per GPU:
      1. Snapshot UUID, constraints, current pids.
      2. Read cap via nvidia-smi BY UUID (ground truth, before apply).
      3. apply_cap(gpu, test_watts) — measure return value.
      4. Read cap via nvidia-smi BY UUID (ground truth, after apply).
      5. restore_default(gpu).
      6. Read cap via nvidia-smi BY UUID (ground truth, after restore).

    All nvidia-smi reads are keyed on the UUID the actuator reported for the
    index, never the index itself, so a DCGM/NVML index-space mismatch cannot
    make ground truth read a different physical GPU than the actuator probed.

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
        snapshots = []
        for i in range(n):
            uuid = actuator.get_uuid(i)
            min_w, max_w = actuator.constraints_w(i)
            pids = actuator.list_running_pids(i, expected_uuid=uuid)
            ns_before = nvidia_smi_power_limit(uuid)
            ns_default = nvidia_smi_default_limit(uuid)
            snapshots.append(
                {
                    "idx": i,
                    "uuid": uuid,
                    "min_w": min_w,
                    "max_w": max_w,
                    "pids": pids,
                    "ns_before": ns_before,
                    "ns_default": ns_default,
                }
            )

        if verify_writes and require_default_before_write:
            nondefault = [
                f"{item['uuid']}={item['ns_before']:.1f}W "
                f"(default {item['ns_default']:.1f}W)"
                for item in snapshots
                if abs(item["ns_before"] - item["ns_default"]) > 1.0
            ]
            if nondefault:
                raise RuntimeError(
                    "write preflight refused non-default GPU cap(s): "
                    + ", ".join(nondefault)
                )

        gpus = []
        for snapshot in snapshots:
            i = snapshot["idx"]
            uuid = snapshot["uuid"]
            min_w = snapshot["min_w"]
            max_w = snapshot["max_w"]
            pids = snapshot["pids"]
            ns_before = snapshot["ns_before"]
            ns_default = snapshot["ns_default"]

            write_skipped_reason = None
            if not verify_writes:
                write_skipped_reason = "read-only mode"
            elif skip_busy_gpus and len(pids) > 0:
                write_skipped_reason = f"busy ({len(pids)} compute pid(s))"

            ns_after_cleanup = None
            if write_skipped_reason is None:
                try:
                    apply_result = actuator.apply_cap(i, test_watts, expected_uuid=uuid)
                    time.sleep(sleep_s)
                    ns_applied = nvidia_smi_power_limit(uuid)
                    actuator.restore_default(i)
                    time.sleep(sleep_s)
                    ns_restored = nvidia_smi_power_limit(uuid)
                finally:
                    # The qualification harness must never strand its test cap,
                    # including when an independent read or parity assertion
                    # raises after a successful hardware write. Restore the
                    # exact entry cap, not merely the factory default, so a
                    # pre-existing operator cap is preserved. Reapply it
                    # unconditionally: using an independent read to decide
                    # whether cleanup is needed would let a transient
                    # nvidia-smi failure strand the test cap.
                    cleanup_result = actuator.apply_cap(
                        i, round(ns_before), expected_uuid=uuid
                    )
                    time.sleep(sleep_s)
                    ns_after_cleanup = nvidia_smi_power_limit(uuid)
                    if (
                        cleanup_result.write_outcome != "succeeded"
                        or cleanup_result.readback_outcome != "succeeded"
                        or abs(ns_after_cleanup - ns_before) > 1.0
                    ):
                        raise RuntimeError(
                            f"failed to restore original cap for {uuid}: "
                            f"before={ns_before:.1f}W "
                            f"after={ns_after_cleanup:.1f}W"
                        )
            else:
                apply_result = ns_applied = ns_restored = None

            gpus.append(
                {
                    "idx": i,
                    "uuid": uuid,
                    "min_w": min_w,
                    "max_w": max_w,
                    "pid_count": len(pids),
                    "ns_before": ns_before,
                    "ns_default": ns_default,
                    "apply_result": apply_result,
                    "ns_after_apply": ns_applied,
                    "ns_after_restore": ns_restored,
                    "ns_after_cleanup": ns_after_cleanup,
                    "write_skipped_reason": write_skipped_reason,
                }
            )
        return {"device_count": n, "gpus": gpus}
    finally:
        actuator.shutdown()


# ---------------------------------------------------------------------------
# Single-actuator qualification — a skipped peer must never weaken the typed
# write/readback contract into a probe-only false PASS.
# ---------------------------------------------------------------------------


def actuator_check(
    result,
    *,
    actuator_name: str,
    tolerance_w: float,
    require_writes: bool = False,
    expected_request_watts: int | None = None,
):
    """Return failed checks for one actuator (0 = independently qualified)."""
    failures = []
    gpus = result["gpus"]
    if result["device_count"] <= 0:
        failures.append(f"{actuator_name}: no GPUs discovered")
    if result["device_count"] != len(gpus):
        failures.append(
            f"{actuator_name}: device_count={result['device_count']} but "
            f"probe returned {len(gpus)} GPU rows"
        )
    uuids = [gpu["uuid"] for gpu in gpus]
    if len(set(uuids)) != len(uuids):
        failures.append(f"{actuator_name}: duplicate GPU UUIDs in probe result")

    successful_writes = 0
    for gpu in gpus:
        uuid = gpu["uuid"]
        where = f"UUID {uuid} ({actuator_name} idx {gpu['idx']})"
        if gpu["min_w"] > gpu["max_w"]:
            failures.append(
                f"{where}: invalid constraints [{gpu['min_w']}, {gpu['max_w']}]"
            )
        apply_result = gpu["apply_result"]
        if apply_result is None:
            if require_writes:
                failures.append(
                    f"{where}: required write was skipped "
                    f"({gpu.get('write_skipped_reason') or 'no reason'})"
                )
            continue

        valid = True
        expected_target = None
        if expected_request_watts is not None:
            expected_target = min(
                max(expected_request_watts, gpu["min_w"]), gpu["max_w"]
            )
        exact_fields = {
            "gpu_uuid": uuid,
            "policy_outcome": "annotated",
            "write_outcome": "succeeded",
            "readback_outcome": "succeeded",
            "actuator": actuator_name,
        }
        if expected_request_watts is not None:
            exact_fields.update(
                {
                    "requested_watts": expected_request_watts,
                    "constraint_min_watts": gpu["min_w"],
                    "constraint_max_watts": gpu["max_w"],
                    "target_watts": expected_target,
                }
            )
        for field, expected in exact_fields.items():
            actual = getattr(apply_result, field)
            if actual != expected:
                failures.append(f"{where}: apply {field}={actual!r}, want {expected!r}")
                valid = False

        enforced = apply_result.enforced_cap_watts
        independent = gpu["ns_after_apply"]
        if enforced is None:
            failures.append(f"{where}: successful apply lacks enforced cap")
            valid = False
        elif expected_target is not None and enforced != expected_target:
            failures.append(
                f"{where}: enforced_cap_watts={enforced!r}, "
                f"want clamped target {expected_target!r}"
            )
            valid = False
        if independent is None:
            failures.append(f"{where}: apply lacks independent nvidia-smi readback")
            valid = False
        else:
            independent_target = (
                expected_target
                if expected_target is not None
                else apply_result.target_watts
            )
            if abs(independent_target - independent) > tolerance_w:
                failures.append(
                    f"{where}: expected/independent diff — "
                    f"target={independent_target} nvidia-smi={independent:.1f} W"
                )
                valid = False
            if enforced is not None and abs(enforced - independent) > tolerance_w:
                failures.append(
                    f"{where}: enforced/independent diff — "
                    f"enforced={enforced} nvidia-smi={independent:.1f} W"
                )
                valid = False

        restored = gpu["ns_after_restore"]
        if restored is None:
            failures.append(f"{where}: restore lacks independent readback")
            valid = False
        elif abs(restored - gpu["ns_default"]) > tolerance_w:
            failures.append(
                f"{where}: restore_default got {restored:.1f} W, "
                f"expected {gpu['ns_default']:.1f} W"
            )
            valid = False
        cleanup = gpu["ns_after_cleanup"]
        if cleanup is None or abs(cleanup - gpu["ns_default"]) > tolerance_w:
            failures.append(
                f"{where}: final cleanup cap {cleanup!r}, "
                f"expected {gpu['ns_default']:.1f} W"
            )
            valid = False
        if valid:
            successful_writes += 1

    if require_writes:
        if expected_request_watts is None:
            failures.append(
                "write qualification lacks an explicit requested-watts contract"
            )
        if successful_writes != result["device_count"]:
            failures.append(
                f"{actuator_name}: independently verified {successful_writes}/"
                f"{result['device_count']} required writes"
            )

    for failure in failures:
        print(f"  FAIL: {failure}", file=sys.stderr)
    return len(failures)


# ---------------------------------------------------------------------------
# Parity diff — compares two probe results, prints any discrepancies,
# returns the count of failures.
# ---------------------------------------------------------------------------


def parity_check(
    nvml_result,
    dcgm_result,
    *,
    tolerance_w: float,
    require_writes: bool = False,
    expected_request_watts: int | None = None,
):
    """Return number of failed parity checks (0 = pass)."""
    failures = []
    successful_writes = {"nvml": 0, "dcgm": 0}

    if nvml_result["device_count"] != dcgm_result["device_count"]:
        failures.append(
            f"device_count diff: nvml={nvml_result['device_count']} "
            f"dcgm={dcgm_result['device_count']}"
        )
        return len(failures)  # bail — per-GPU comparison meaningless

    # Join by UUID, NOT positional zip: the two actuators can enumerate the same
    # physical GPUs under DIFFERENT indices (NVML index order need not match
    # DCGM's), so `zip(nvml, dcgm)` would compare mismatched GPUs and report
    # phantom diffs. The UUID is the hardware-stable identity shared by both
    # paths, so it is the correct join key.
    nvml_by_uuid = {g["uuid"]: g for g in nvml_result["gpus"]}
    dcgm_by_uuid = {g["uuid"]: g for g in dcgm_result["gpus"]}

    for missing in sorted(set(nvml_by_uuid) - set(dcgm_by_uuid)):
        failures.append(f"UUID {missing}: seen by NVML but NOT DCGM")
    for missing in sorted(set(dcgm_by_uuid) - set(nvml_by_uuid)):
        failures.append(f"UUID {missing}: seen by DCGM but NOT NVML")

    for uuid in sorted(set(nvml_by_uuid) & set(dcgm_by_uuid)):
        nvml_gpu = nvml_by_uuid[uuid]
        dcgm_gpu = dcgm_by_uuid[uuid]
        where = f"UUID {uuid} (nvml idx {nvml_gpu['idx']}, dcgm idx {dcgm_gpu['idx']})"

        # The operator treats any live constraint drift as unqualified, so the
        # two hardware paths must report the exact same range. Tolerance is
        # reserved for the independent nvidia-smi float readback below.
        if nvml_gpu["min_w"] != dcgm_gpu["min_w"]:
            failures.append(
                f"{where}: min_w diff — nvml={nvml_gpu['min_w']} "
                f"dcgm={dcgm_gpu['min_w']}"
            )
        if nvml_gpu["max_w"] != dcgm_gpu["max_w"]:
            failures.append(
                f"{where}: max_w diff — nvml={nvml_gpu['max_w']} "
                f"dcgm={dcgm_gpu['max_w']}"
            )

        # Typed apply evidence. First validate each actuator independently;
        # matching failures or matching incorrect values are not parity.
        for actuator_name, gpu in (("nvml", nvml_gpu), ("dcgm", dcgm_gpu)):
            result = gpu["apply_result"]
            if result is None:
                continue
            valid = True
            expected_target = None
            if expected_request_watts is not None:
                expected_target = min(
                    max(expected_request_watts, gpu["min_w"]), gpu["max_w"]
                )
            exact_fields = {
                "gpu_uuid": uuid,
                "policy_outcome": "annotated",
                "write_outcome": "succeeded",
                "readback_outcome": "succeeded",
                "actuator": actuator_name,
            }
            if expected_request_watts is not None:
                exact_fields.update(
                    {
                        "requested_watts": expected_request_watts,
                        "constraint_min_watts": gpu["min_w"],
                        "constraint_max_watts": gpu["max_w"],
                        "target_watts": expected_target,
                    }
                )
            for field, expected in exact_fields.items():
                actual = getattr(result, field)
                if actual != expected:
                    failures.append(
                        f"{where}: {actuator_name} apply {field}={actual!r}, "
                        f"want {expected!r}"
                    )
                    valid = False
            enforced = result.enforced_cap_watts
            independent = gpu["ns_after_apply"]
            if enforced is None:
                failures.append(
                    f"{where}: {actuator_name} successful apply lacks enforced cap"
                )
                valid = False
            elif expected_target is not None and enforced != expected_target:
                failures.append(
                    f"{where}: {actuator_name} enforced_cap_watts={enforced!r}, "
                    f"want clamped target {expected_target!r}"
                )
                valid = False
            if independent is None:
                failures.append(
                    f"{where}: {actuator_name} apply lacks independent readback"
                )
                valid = False
            else:
                independent_target = (
                    expected_target
                    if expected_target is not None
                    else result.target_watts
                )
                if abs(independent_target - independent) > tolerance_w:
                    failures.append(
                        f"{where}: {actuator_name} expected/independent diff — "
                        f"target={independent_target} "
                        f"nvidia-smi={independent:.1f} W"
                    )
                    valid = False
                if enforced is not None and abs(enforced - independent) > tolerance_w:
                    failures.append(
                        f"{where}: {actuator_name} enforced/independent diff — "
                        f"enforced={enforced} "
                        f"nvidia-smi={independent:.1f} W"
                    )
                    valid = False
            if valid:
                successful_writes[actuator_name] += 1

        # Then compare the two independently accepted paths. Outcomes and exact
        # identity are categorical; targets and enforced readbacks allow the
        # same small cross-actuator wattage tolerance as firmware values.
        if (nvml_gpu["apply_result"] is None) != (dcgm_gpu["apply_result"] is None):
            failures.append(
                f"{where}: apply evidence presence diff — "
                f"nvml={nvml_gpu['apply_result'] is not None} "
                f"dcgm={dcgm_gpu['apply_result'] is not None}"
            )
        elif nvml_gpu["apply_result"] is not None:
            nvml_apply = nvml_gpu["apply_result"]
            dcgm_apply = dcgm_gpu["apply_result"]
            for field in (
                "gpu_uuid",
                "policy_outcome",
                "write_outcome",
                "readback_outcome",
            ):
                if getattr(nvml_apply, field) != getattr(dcgm_apply, field):
                    failures.append(
                        f"{where}: apply {field} diff — "
                        f"nvml={getattr(nvml_apply, field)!r} "
                        f"dcgm={getattr(dcgm_apply, field)!r}"
                    )
            if abs(nvml_apply.target_watts - dcgm_apply.target_watts) > tolerance_w:
                failures.append(
                    f"{where}: apply target diff — "
                    f"nvml={nvml_apply.target_watts} "
                    f"dcgm={dcgm_apply.target_watts}"
                )
            if (nvml_apply.enforced_cap_watts is None) != (
                dcgm_apply.enforced_cap_watts is None
            ):
                failures.append(
                    f"{where}: apply enforced presence diff — "
                    f"nvml={nvml_apply.enforced_cap_watts} "
                    f"dcgm={dcgm_apply.enforced_cap_watts}"
                )
            elif (
                nvml_apply.enforced_cap_watts is not None
                and abs(nvml_apply.enforced_cap_watts - dcgm_apply.enforced_cap_watts)
                > tolerance_w
            ):
                failures.append(
                    f"{where}: apply enforced-cap diff — "
                    f"nvml={nvml_apply.enforced_cap_watts} "
                    f"dcgm={dcgm_apply.enforced_cap_watts}"
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
                    f"{where}: nvidia-smi after-apply diff — "
                    f"nvml={nvml_gpu['ns_after_apply']:.1f} W "
                    f"dcgm={dcgm_gpu['ns_after_apply']:.1f} W"
                )

        # Restore must return to default (per actuator, vs ground truth)
        if nvml_gpu["ns_after_restore"] is not None:
            if abs(nvml_gpu["ns_after_restore"] - nvml_gpu["ns_default"]) > tolerance_w:
                failures.append(
                    f"{where}: nvml restore_default — "
                    f"got {nvml_gpu['ns_after_restore']:.1f} W, "
                    f"expected {nvml_gpu['ns_default']:.1f} W (factory default)"
                )
        if dcgm_gpu["ns_after_restore"] is not None:
            if abs(dcgm_gpu["ns_after_restore"] - dcgm_gpu["ns_default"]) > tolerance_w:
                failures.append(
                    f"{where}: dcgm restore_default — "
                    f"got {dcgm_gpu['ns_after_restore']:.1f} W, "
                    f"expected {dcgm_gpu['ns_default']:.1f} W (factory default)"
                )

    if require_writes:
        if expected_request_watts is None:
            failures.append(
                "write qualification lacks an explicit requested-watts contract"
            )
        for actuator_name, count in successful_writes.items():
            if count == 0:
                failures.append(
                    f"{actuator_name}: no independently verified successful write"
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
        if gpu["apply_result"] is not None:
            apply_result = gpu["apply_result"]
            print(f"    apply target      = {apply_result.target_watts} W")
            print(
                "    apply outcomes    = "
                f"write:{apply_result.write_outcome}, "
                f"readback:{apply_result.readback_outcome}"
            )
            print(f"    enforced readback = {apply_result.enforced_cap_watts} W")
            print(f"    nvidia-smi applied= {gpu['ns_after_apply']:.1f} W")
            print(f"    nvidia-smi restore= {gpu['ns_after_restore']:.1f} W")
            print(f"    final cleanup cap = {gpu['ns_after_cleanup']:.1f} W")
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
        help="Watts to apply via apply_cap. Values outside the live range "
        "qualify the actuator's clamp behavior.",
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
        "--require-default-before-write",
        action="store_true",
        help="Refuse the entire write phase before the first cap change unless "
        "every visible GPU is at its factory default. Recommended for shared "
        "qualification clusters.",
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
    # This probe drives the actuators directly (via probe()) rather than
    # through their init()/shutdown() lifecycle, and probe() needs NVML for
    # list_running_pids on BOTH paths, so we init NVML once here ourselves.
    # nvmlInit is idempotent, so this stays safe even though NvmlActuator.init()
    # / DcgmActuator.init() would also init NVML if they were called.
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
                require_default_before_write=args.require_default_before_write,
            )
            print_probe("NvmlActuator", nvml_result)

        if not args.skip_dcgm:
            print(
                f"\nRunning DcgmActuator probe ("
                f"hostengine={args.hostengine_host}:{args.hostengine_port})..."
            )
            dcgm = DcgmActuator(
                host=args.hostengine_host,
                port=args.hostengine_port,
                metrics=metrics,
            )
            try:
                dcgm_result = probe(
                    dcgm,
                    args.test_watts,
                    verify_writes=not args.read_only,
                    sleep_s=args.sleep_s,
                    skip_busy_gpus=args.skip_busy_gpus,
                    require_default_before_write=args.require_default_before_write,
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
                nvml_result,
                dcgm_result,
                tolerance_w=args.tolerance_watts,
                require_writes=not args.read_only,
                expected_request_watts=(None if args.read_only else args.test_watts),
            )
            if n_failures == 0:
                print(
                    f"  PASS: NvmlActuator and DcgmActuator agree on all probes "
                    f"(tolerance ±{args.tolerance_watts} W)."
                )
                return 0
            print(f"  {n_failures} parity check(s) failed.")
            return 1

        actuator_name, result = (
            ("nvml", nvml_result) if nvml_result is not None else ("dcgm", dcgm_result)
        )
        print(f"\n=== {actuator_name.upper()} qualification check ===")
        n_failures = actuator_check(
            result,
            actuator_name=actuator_name,
            tolerance_w=args.tolerance_watts,
            require_writes=not args.read_only,
            expected_request_watts=(None if args.read_only else args.test_watts),
        )
        if n_failures == 0:
            print(f"  PASS: {actuator_name} independently qualified.")
            return 0
        print(f"  {n_failures} {actuator_name} qualification check(s) failed.")
        return 1
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
