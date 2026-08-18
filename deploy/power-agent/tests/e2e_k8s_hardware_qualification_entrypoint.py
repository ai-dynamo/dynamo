#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the bounded P2.8 actuator qualification inside an exclusive GPU Job.

The outer Kubernetes fixture requests every GPU on the selected node. This
entrypoint refuses writes unless every visible GPU is idle and at its factory
default, exercises the selected Power Agent actuator for in-range and both
clamp directions, and restores the exact entry limits in a final safety
boundary. NVML qualification does not require a DCGM hostengine; the optional
parity mode additionally proves that NVML and DCGM agree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pynvml

DCGM_LOCAL_BUILD_MARKER = "Local build info:"
DCGM_HOSTENGINE_BUILD_MARKER = "Hostengine build info:"


@dataclass(frozen=True)
class GPUSnapshot:
    uuid: str
    name: str
    min_watts: int
    default_watts: int
    max_watts: int
    current_watts: int
    compute_pids: tuple[int, ...]


def _watts(milliwatts: int) -> int:
    return int(round(milliwatts / 1000))


def _text(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _snapshot() -> list[GPUSnapshot]:
    snapshots: list[GPUSnapshot] = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        minimum, maximum = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        snapshots.append(
            GPUSnapshot(
                uuid=_text(pynvml.nvmlDeviceGetUUID(handle)),
                name=_text(pynvml.nvmlDeviceGetName(handle)),
                min_watts=_watts(minimum),
                default_watts=_watts(
                    pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
                ),
                max_watts=_watts(maximum),
                current_watts=_watts(pynvml.nvmlDeviceGetPowerManagementLimit(handle)),
                compute_pids=tuple(sorted(process.pid for process in processes)),
            )
        )
    return snapshots


def _restore_entry_caps(entry: list[GPUSnapshot]) -> None:
    failures: list[str] = []
    for gpu in entry:
        try:
            handle = pynvml.nvmlDeviceGetHandleByUUID(gpu.uuid)
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, gpu.current_watts * 1000)
            restored = _watts(pynvml.nvmlDeviceGetPowerManagementLimit(handle))
            if abs(restored - gpu.current_watts) > 1:
                failures.append(
                    f"{gpu.uuid}: restored {restored}W, want {gpu.current_watts}W"
                )
        except Exception as exc:  # pragma: no cover - real-hardware boundary
            failures.append(f"{gpu.uuid}: {exc}")
    if failures:
        raise RuntimeError("exact entry-cap restoration failed: " + "; ".join(failures))


def _run_parity(
    parity_script: str,
    actuator_mode: str,
    timeout_seconds: float,
    *extra_args: str,
    host: str | None = None,
    port: int = 5555,
) -> None:
    command = [sys.executable, parity_script]
    if actuator_mode == "nvml":
        command.append("--skip-dcgm")
    elif actuator_mode == "nvml-dcgm-parity":
        if not host:
            raise ValueError("parity mode requires a DCGM hostengine host")
        command.extend(
            [
                "--hostengine-host",
                host,
                "--hostengine-port",
                str(port),
            ]
        )
    else:
        raise ValueError(f"unsupported actuator mode: {actuator_mode}")
    command.extend(extra_args)
    subprocess.run(
        command,
        check=True,
        timeout=timeout_seconds,
    )


def _remaining_seconds(deadline: float, maximum: float = 120) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("hardware qualification internal deadline expired")
    return max(1, min(maximum, remaining))


def _parse_dcgm_versions(output: str) -> dict[str, str]:
    def version_after(marker: str) -> str:
        if marker not in output:
            raise RuntimeError(f"dcgmi version output lacks {marker.rstrip(':')}")
        section = output.split(marker, 1)[1]
        match = re.search(r"^Version\s*:\s*(\S+)\s*$", section, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"dcgmi {marker.rstrip(':')} lacks Version")
        return match.group(1)

    return {
        "client": version_after(DCGM_LOCAL_BUILD_MARKER),
        "hostengine": version_after(DCGM_HOSTENGINE_BUILD_MARKER),
    }


def _run_gate_qualification(
    entry: list[GPUSnapshot],
    effective_watts: int,
    deadline: float,
) -> dict[str, object]:
    gate_dir = Path("/var/run/dynamo/power-gate")
    gate_dir.mkdir(parents=True, exist_ok=True)
    pod_uid = "p2-8-qualification-pod"
    (gate_dir / "pod-uid").write_text(pod_uid + "\n", encoding="utf-8")
    report_path = gate_dir / "report"
    report_path.unlink(missing_ok=True)
    marker = Path("/tmp/p2-8-gated-backend.json")
    marker.unlink(missing_ok=True)

    environment = os.environ.copy()
    environment.update(
        {
            "DYNAMO_POWER_DGD_UID": "p2-8-qualification-dgd",
            "DYNAMO_POWER_COMPONENT": "decode",
            "DYNAMO_POWER_EXPECTED_GPU_COUNT": str(len(entry)),
            "DYNAMO_POWER_IN_GATE_BOUND_WATTS_PER_GPU": str(effective_watts),
        }
    )
    gate_command = [
        sys.executable,
        "/app/e2e_gate_entrypoint.py",
        "--",
        sys.executable,
        "/app/e2e_gate_backend_probe.py",
        "--expected-max-watts",
        str(effective_watts),
        "--marker",
        str(marker),
    ]
    gate = subprocess.Popen(gate_command, env=environment)
    try:
        time.sleep(1)
        if gate.poll() is not None:
            raise RuntimeError(f"power gate exited before evidence: {gate.returncode}")
        if marker.exists():
            raise RuntimeError("backend marker appeared before enforcement evidence")

        observed_at = datetime.now(timezone.utc)
        gpu_reports: list[dict[str, object]] = []
        for gpu in entry:
            handle = pynvml.nvmlDeviceGetHandleByUUID(gpu.uuid)
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, effective_watts * 1000)
            enforced = _watts(pynvml.nvmlDeviceGetPowerManagementLimit(handle))
            independent = float(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        gpu.uuid,
                        "--query-gpu=power.limit",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=_remaining_seconds(deadline, 30),
                ).strip()
            )
            if enforced != effective_watts or abs(independent - effective_watts) > 1:
                raise RuntimeError(
                    f"gate setup readback mismatch for {gpu.uuid}: "
                    f"NVML={enforced}W nvidia-smi={independent}W"
                )
            gpu_reports.append(
                {
                    "uuid": gpu.uuid,
                    "requestedWatts": effective_watts,
                    "targetWatts": effective_watts,
                    "constraintMinWatts": gpu.min_watts,
                    "constraintMaxWatts": gpu.max_watts,
                    "policyOutcome": "annotated",
                    "writeOutcome": "succeeded",
                    "readbackOutcome": "succeeded",
                    "enforcedCapWatts": enforced,
                    "actuator": "nvml",
                    "observedAt": observed_at.isoformat(),
                }
            )

        uuids = sorted(gpu.uuid for gpu in entry)
        report = {
            "version": 1,
            "dgdUID": "p2-8-qualification-dgd",
            "component": "decode",
            "podUID": pod_uid,
            "node": "qualification-node",
            "allocationID": f"{pod_uid}/main/{','.join(uuids)}",
            "gpus": gpu_reports,
        }
        temporary = gate_dir / "report.tmp"
        temporary.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        os.replace(temporary, report_path)
        report_published_at = datetime.now(timezone.utc)

        returncode = gate.wait(timeout=_remaining_seconds(deadline, 30))
        if returncode != 0:
            raise RuntimeError(f"power gate/backend returned {returncode}")
        backend = json.loads(marker.read_text(encoding="utf-8"))
        backend_started_at = datetime.fromisoformat(backend["startedAt"])
        if backend_started_at < report_published_at:
            raise RuntimeError(
                "backend initialized before the valid report was published"
            )
        if backend.get("cuInitReturn") != 0:
            raise RuntimeError(f"backend CUDA initialization failed: {backend}")

        missing_marker = Path("/tmp/p2-8-missing-gate-backend.json")
        missing_marker.unlink(missing_ok=True)
        missing_gate_blocked = False
        try:
            subprocess.run(
                [
                    "/missing/dynamo-power-gate",
                    "--",
                    sys.executable,
                    "/app/e2e_gate_backend_probe.py",
                    "--expected-max-watts",
                    str(effective_watts),
                    "--marker",
                    str(missing_marker),
                ],
                check=False,
                timeout=_remaining_seconds(deadline, 30),
            )
        except FileNotFoundError:
            missing_gate_blocked = not missing_marker.exists()
        if not missing_gate_blocked:
            raise RuntimeError(
                "missing power-gate executable did not block the backend"
            )

        return {
            "backendMarkerAbsentBeforeReport": True,
            "reportPublishedAt": report_published_at.isoformat(),
            "backendStartedAt": backend["startedAt"],
            "observedCapsWatts": backend["observedCapsWatts"],
            "cudaInitializedAfterEvidence": True,
            "missingGateBlocksBackend": True,
        }
    finally:
        if gate.poll() is None:
            gate.terminate()
            try:
                gate.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gate.kill()
                gate.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpu-count", type=int, required=True)
    parser.add_argument("--gpu-product", required=True)
    parser.add_argument(
        "--actuator-mode",
        choices=("nvml", "nvml-dcgm-parity"),
        default="nvml-dcgm-parity",
    )
    parser.add_argument("--hostengine-host")
    parser.add_argument("--hostengine-port", type=int, default=5555)
    parser.add_argument("--verified-hostengine-version")
    parser.add_argument("--dcgm-version-file", default="/opt/dcgm/dcgmi-version.txt")
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--parity-script", default="/app/e2e_actuator_parity.py")
    args = parser.parse_args()
    if args.timeout_seconds < 60:
        parser.error("--timeout-seconds must be at least 60")
    if args.actuator_mode == "nvml-dcgm-parity":
        if not args.hostengine_host:
            parser.error("parity mode requires --hostengine-host")
        if not args.verified_hostengine_version:
            parser.error("parity mode requires --verified-hostengine-version")
    deadline = time.monotonic() + args.timeout_seconds

    def terminate_for_cleanup(signum, frame):  # noqa: ARG001
        raise RuntimeError("hardware qualification received SIGTERM")

    previous_sigterm = signal.signal(signal.SIGTERM, terminate_for_cleanup)

    entry: list[GPUSnapshot] = []
    nvml_initialized = False
    primary_error: BaseException | None = None
    dcgm_versions: dict[str, str] | None = None
    try:
        if args.actuator_mode == "nvml-dcgm-parity":
            dcgm_versions = _parse_dcgm_versions(
                Path(args.dcgm_version_file).read_text(encoding="utf-8")
            )
            if dcgm_versions["hostengine"] != args.verified_hostengine_version:
                raise RuntimeError(
                    "scheduled-node dcgmi hostengine version "
                    f"{dcgm_versions['hostengine']!r} != verified "
                    f"{args.verified_hostengine_version!r}"
                )
            print(
                "P2_8_DCGM_VERSIONS=" + json.dumps(dcgm_versions, sort_keys=True),
                flush=True,
            )
        pynvml.nvmlInit()
        nvml_initialized = True
        entry = _snapshot()
        print(
            "P2_8_PREFLIGHT="
            + json.dumps([asdict(gpu) for gpu in entry], sort_keys=True),
            flush=True,
        )
        if len(entry) != args.expected_gpu_count:
            raise RuntimeError(
                f"visible GPU count {len(entry)} != requested {args.expected_gpu_count}"
            )

        bounds = {(gpu.min_watts, gpu.default_watts, gpu.max_watts) for gpu in entry}
        if len(bounds) != 1:
            raise RuntimeError(
                f"selected exact product has nonuniform bounds: {bounds}"
            )
        minimum, default, maximum = next(iter(bounds))
        if not (1 <= minimum <= default <= maximum):
            raise RuntimeError(
                f"invalid live min/default/max tuple: {minimum}/{default}/{maximum}"
            )

        busy = [gpu.uuid for gpu in entry if gpu.compute_pids]
        if busy:
            raise RuntimeError(f"exclusive qualification GPUs are busy: {busy}")
        nondefault = [
            f"{gpu.uuid}={gpu.current_watts}W(default={gpu.default_watts}W)"
            for gpu in entry
            if abs(gpu.current_watts - gpu.default_watts) > 1
        ]
        if nondefault:
            raise RuntimeError(
                "write qualification requires factory-default entry caps: "
                + ", ".join(nondefault)
            )

        _run_parity(
            args.parity_script,
            args.actuator_mode,
            _remaining_seconds(deadline),
            "--read-only",
            host=args.hostengine_host,
            port=args.hostengine_port,
        )

        in_range = minimum + ((maximum - minimum) // 2)
        cases = {
            "in_range": in_range,
            "below_min": minimum - 1,
            "above_max": maximum + 1,
        }
        for case, requested in cases.items():
            print(f"P2_8_CASE={case} requested_watts={requested}", flush=True)
            _run_parity(
                args.parity_script,
                args.actuator_mode,
                _remaining_seconds(deadline),
                "--test-watts",
                str(requested),
                "--require-default-before-write",
                "--sleep-s",
                "1.0",
                host=args.hostengine_host,
                port=args.hostengine_port,
            )

        gate_result = _run_gate_qualification(entry, in_range, deadline)
        _restore_entry_caps(entry)

        final = _snapshot()
        by_uuid = {gpu.uuid: gpu for gpu in final}
        for original in entry:
            current = by_uuid.get(original.uuid)
            if (
                current is None
                or abs(current.current_watts - original.current_watts) > 1
            ):
                raise RuntimeError(
                    f"postflight cap mismatch for {original.uuid}: {current}"
                )

        result = {
            "status": "PASS",
            "actuatorMode": args.actuator_mode,
            "gpuProduct": args.gpu_product,
            "gpuCount": len(entry),
            "driverVersion": _text(pynvml.nvmlSystemGetDriverVersion()),
            "minWatts": minimum,
            "defaultWatts": default,
            "maxWatts": maximum,
            "requests": cases,
            "entryCapsRestored": True,
            "nvmlQualified": True,
            "gate": gate_result,
        }
        if args.actuator_mode == "nvml-dcgm-parity":
            if dcgm_versions is None:
                raise RuntimeError("DCGM client version evidence is missing")
            result.update(
                {
                    "dcgmClientVersion": dcgm_versions["client"],
                    "dcgmHostengineVersion": args.verified_hostengine_version,
                    "nvmlDcgmParity": True,
                }
            )
        print("P2_8_QUALIFICATION_RESULT=" + json.dumps(result, sort_keys=True))
        return 0
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if entry:
            try:
                _restore_entry_caps(entry)
            except BaseException as cleanup_error:
                print(f"P2_8_CLEANUP_ERROR={cleanup_error}", file=sys.stderr)
                if primary_error is None:
                    raise
        if nvml_initialized:
            pynvml.nvmlShutdown()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
