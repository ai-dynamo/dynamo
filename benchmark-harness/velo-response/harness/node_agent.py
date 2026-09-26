#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-node Slurm agent for the Tyche AgentX frontend on-CPU profile.

The module intentionally uses only the Python standard library.  Rank 0 owns
the frontend and experiment order; rank 1 owns the request plane, mockers, and
load generator.  The agents communicate only through a job-scoped directory
on the shared filesystem.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT_DEFAULT = Path(
    "/lustre/fsw/coreai_comparch_trtllm/jothomson/"
    "dynamo-agentx-c2048-oncpu-20260716"
)
EXPECTED_RUSTFLAGS = (
    "-C target-cpu=native -C force-frame-pointers=yes --cfg tokio_unstable"
)
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MODEL_NAME_DEFAULT = "Qwen/Qwen3-0.6B"


class HarnessError(RuntimeError):
    """An actionable benchmark or environment failure."""


@dataclasses.dataclass(frozen=True)
class RunSpec:
    workload: str
    arm: str
    ordinal: int
    calibration: bool = False
    interleave: bool = False

    @property
    def arm_slug(self) -> str:
        return self.arm.replace("+", "p").replace("/", "-")

    @property
    def run_id(self) -> str:
        if self.calibration:
            return f"calibration-{self.workload}-{self.arm_slug}"
        suffix = "-interleave" if self.interleave else ""
        return f"{self.workload}-{self.ordinal:02d}-{self.arm_slug}{suffix}"

    @property
    def relative_dir(self) -> Path:
        if self.calibration:
            return Path("calibration") / self.workload
        if self.interleave:
            return Path("diagnostics") / self.workload / self.run_id
        return Path("runs") / self.workload / f"{self.ordinal:02d}-{self.arm_slug}"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {
            "run_id": self.run_id,
            "relative_dir": str(self.relative_dir),
        }


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def job_paths(root: str | Path, job_id: str) -> dict[str, Path]:
    """Return job-scoped paths without creating them."""

    if not job_id or not JOB_ID_RE.fullmatch(job_id) or job_id in {".", ".."}:
        raise ValueError(f"unsafe job id: {job_id!r}")
    root_path = Path(root).expanduser()
    return {
        "state_dir": root_path / "state" / job_id,
        "result_dir": root_path / "results" / job_id,
    }


def parse_cpu_set(value: str) -> set[int]:
    cpus: set[int] = set()
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid CPU list: {value!r}")
    for item in value.split(","):
        item = item.strip()
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", item)
        if not match:
            raise ValueError(f"invalid CPU list element: {item!r}")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if last < first:
            raise ValueError(f"descending CPU range: {item!r}")
        cpus.update(range(first, last + 1))
    return cpus


def private_anonymous_residency(lines: Iterable[str]) -> dict[int, int]:
    """Count private anonymous pages by NUMA node from /proc/PID/numa_maps."""

    totals: dict[int, int] = {}
    for line in lines:
        # Exclude mapped files and shared memory.  Anonymous heap/stack pages
        # are the allocation placement controlled by the process mempolicy.
        if "anon=" not in line or "file=" in line or "shmem=" in line:
            continue
        for node, pages in re.findall(r"\bN(\d+)=(\d+)\b", line):
            totals[int(node)] = totals.get(int(node), 0) + int(pages)
    return totals


def build_run_sequence(config: Mapping[str, Any]) -> list[RunSpec]:
    sequence: list[RunSpec] = []
    calibration_arm = config["matrix"].get("calibration_arm", "8+0")
    arms = list(config["matrix"]["sequence"])
    workload_names = config["matrix"].get("workloads", list(config["workloads"]))
    for workload in workload_names:
        if config["matrix"].get("calibration_enabled", True):
            sequence.append(
                RunSpec(workload=workload, arm=calibration_arm, ordinal=0, calibration=True)
            )
        sequence.extend(
            RunSpec(workload=workload, arm=arm, ordinal=index)
            for index, arm in enumerate(arms, start=1)
        )
    return sequence


def validate_nonoverlap(config: Mapping[str, Any]) -> None:
    topology = config["topology"]
    control = parse_cpu_set(topology["frontend_control_cpus"])
    placements = topology.get("frontend_arms", topology.get("frontend_placements", {}))
    for name, arm in placements.items():
        overlap = control & parse_cpu_set(arm["cpus"])
        if overlap:
            raise ValueError(f"frontend arm {name} overlaps control CPUs: {sorted(overlap)}")

    remote = topology["remote"]
    if "mocker_cpus" in remote:
        pools = {
            "mocker": parse_cpu_set(remote["mocker_cpus"]),
            "aiperf": parse_cpu_set(remote["aiperf_cpus"]),
            "control": parse_cpu_set(remote["control_cpus"]),
        }
    else:
        pools = {role: parse_cpu_set(remote[role]["cpus"]) for role in ("mocker", "aiperf", "control")}
    for left, right in (("mocker", "aiperf"), ("mocker", "control"), ("aiperf", "control")):
        overlap = pools[left] & pools[right]
        if overlap:
            raise ValueError(f"remote {left}/{right} CPU pools overlap: {sorted(overlap)}")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _embedded_config(root: Path = ROOT_DEFAULT) -> dict[str, Any]:
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    return {
        "root": str(root),
        "revisions": {
            "dynamo": "ccc835a80cf8e5ccc022b30db7df1f9828930333",
            "aiperf": "8473e1545476c1d91932aa2402b642b416a23df6",
            "transformers": "18693fc6771c948cf762fde6327d3961c48f28f1",
            "model": revision,
        },
        "paths": {
            "dynamo_checkout": str(root / "src/dynamo"),
            "dynamo_python": str(root / "src/dynamo/.venv/bin/python"),
            "aiperf": str(root / "env/aiperf/bin/aiperf"),
            "model": str(
                root
                / "cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
                / revision
            ),
            "huggingface_cache": str(root / "cache/huggingface"),
            "jemalloc": str(root / "env/services/jemalloc-5.3.0/lib/libjemalloc.so.2"),
        },
        "model": {
            "name": MODEL_NAME_DEFAULT,
            "agentx_dataset": "semianalysisai/cc-traces-weka-with-subagents-060526",
        },
        "network": {
            "interface": "auto",
            "frontend_http_port": 8000,
            "etcd_client_port": 2379,
            "etcd_peer_port": 2380,
            "nats_port": 4222,
        },
        "topology": {
            "expected_logical_cpus": 144,
            "expected_numa_nodes": 2,
            "smt_offset": 0,
            "frontend_control_cpus": "8-15",
            "frontend_arms": {
                "8+0": {
                    "cpus": "0-7",
                    "memory_policy": "bind",
                    "memory_nodes": [0],
                },
                "4+4": {
                    "cpus": "0-3,72-75",
                    "memory_policy": "first-touch",
                    "memory_nodes": [0, 1],
                },
            },
            "remote": {
                "mocker_cpus": "0-63",
                "mocker_memory_node": 0,
                "aiperf_cpus": "72-135",
                "aiperf_memory_node": 1,
                "control_cpus": "136-143",
                "control_memory_node": 1,
            },
        },
        "runtime": {
            "num_mockers": 16,
            "mocker_speedup_ratio": 1_000_000,
            "mocker_num_gpu_blocks": 1_000_000,
            "tokenizer": "fastokens",
            "tokenizer_cache": True,
            "tokenizer_cache_bytes": 4_294_967_296,
            "request_plane_codec": "msgpack",
            "nats_max_payload_bytes": 8_388_608,
        },
        "workloads": {
            "agentx": {
                "dataset": "weka",
                "block_size": 16,
                "concurrency": 64,
                "duration_seconds": 120,
                "grace_period_seconds": 90,
                "warmup_duration_seconds": 120,
                "dataset_entries": 336,
            },
        },
        "matrix": {
            "calibration_arm": "8+0",
            "calibration_enabled": False,
            "calibration_duration_seconds": 30,
            "sequence": ["8+0", "4+4"],
            "workloads": ["agentx"],
        },
        "acceptance": {
            "bound_private_anon_local_fraction_min": 0.99,
            "split_private_anon_node_fraction_min": 0.4,
            "split_private_anon_node_fraction_max": 0.6,
            "remote_pool_average_busy_max": 0.85,
            "nic_link_utilization_max": 0.7,
        },
    }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    # config.py is the canonical schema and validation implementation shipped
    # beside this agent.  The fallback keeps this file independently testable
    # while the shared environment is first being staged.
    try:
        import config as canonical_config  # type: ignore

        loaded = canonical_config.load_config(path)
        loaded["_config_path"] = str(Path(path).resolve()) if path else "<defaults>"
        return validate_config(loaded)
    except ImportError:
        pass

    defaults = _embedded_config()
    if path is None:
        candidate = Path(__file__).with_name("config.json")
        path = candidate if candidate.exists() else Path(__file__).with_name("config.example.json")
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    config = _deep_merge(defaults, loaded)
    config["_config_path"] = str(config_path.resolve())
    return validate_config(config)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    validated = copy.deepcopy(dict(config))
    if "schema_version" in validated and "pins" in validated:
        try:
            import config as canonical_config  # type: ignore

            canonical_config.validate_config(validated)
        except ImportError:
            validate_nonoverlap(validated)
        return validated

    root = Path(validated["root"])
    if not root.is_absolute():
        raise ValueError("config root must be absolute")

    topology = validated["topology"]
    expected_cpus = int(topology["expected_logical_cpus"])
    smt_offset = int(topology["smt_offset"])
    all_cpus = set(range(expected_cpus))
    if smt_offset and smt_offset * 2 != expected_cpus:
        raise ValueError("smt_offset must divide the expected CPU space into sibling halves")

    arms = topology["frontend_arms"]
    for required in ("8+0", "4+4"):
        if required not in arms:
            raise ValueError(f"missing frontend arm {required}")
    for name, arm in arms.items():
        cpus = parse_cpu_set(arm["cpus"])
        if not cpus <= all_cpus:
            raise ValueError(f"frontend arm {name} references out-of-range CPUs")
        if smt_offset:
            if len(cpus) != 16:
                raise ValueError(f"frontend arm {name} must contain 16 logical CPUs")
            physical = {cpu % smt_offset for cpu in cpus}
            if len(physical) != 8 or any({core, core + smt_offset} - cpus for core in physical):
                raise ValueError(f"frontend arm {name} must contain eight complete SMT pairs")
        elif len(cpus) != 8:
            raise ValueError(f"frontend arm {name} must contain eight no-SMT CPUs")
        policy = arm["memory_policy"]
        nodes = arm["memory_nodes"]
        if policy not in {"bind", "first-touch", "interleave"}:
            raise ValueError(f"unsupported memory policy for {name}: {policy}")
        if policy == "bind" and len(nodes) != 1:
            raise ValueError(f"bound frontend arm {name} needs exactly one memory node")

    for key in ("mocker_cpus", "aiperf_cpus", "control_cpus"):
        cpus = parse_cpu_set(topology["remote"][key])
        if not cpus <= all_cpus:
            raise ValueError(f"remote {key} references out-of-range CPUs")
    validate_nonoverlap(validated)

    known_arms = set(arms)
    matrix = validated["matrix"]
    if matrix["calibration_arm"] not in known_arms:
        raise ValueError("matrix calibration_arm is unknown")
    if not matrix["sequence"] or any(arm not in known_arms for arm in matrix["sequence"]):
        raise ValueError("matrix sequence contains an unknown frontend arm")
    expected_sequence = ["8+0", "4+4"]
    if list(matrix["sequence"]) != expected_sequence:
        raise ValueError(f"matrix sequence must remain {expected_sequence}")
    if matrix.get("calibration_enabled") is not False:
        raise ValueError("matrix calibration must remain disabled")

    for name, workload in validated["workloads"].items():
        if workload["dataset"] not in {"synthetic", "weka"}:
            raise ValueError(f"unsupported dataset for {name}")
        for key in ("block_size", "concurrency", "duration_seconds", "grace_period_seconds", "warmup_duration_seconds"):
            if int(workload[key]) <= 0:
                raise ValueError(f"workload {name}.{key} must be positive")

    ports = [
        int(validated["network"][key])
        for key in ("frontend_http_port", "etcd_client_port", "etcd_peer_port", "nats_port")
    ]
    if len(set(ports)) != len(ports) or any(port < 1024 or port > 65535 for port in ports):
        raise ValueError("network ports must be distinct unprivileged TCP ports")
    return validated


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def harness_content_hash() -> str:
    root = Path(__file__).resolve().parent
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix in {".py", ".sh", ".sbatch"}
    )
    if not files:
        raise HarnessError(f"no harness source files found beneath {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def wait_for_path(path: Path, timeout: float, shutdown: threading.Event) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if shutdown.is_set():
            raise HarnessError(f"interrupted while waiting for {path}")
        if time.monotonic() >= deadline:
            raise HarnessError(f"timed out waiting for {path}")
        time.sleep(0.1)


class ManagedProcess:
    def __init__(
        self,
        name: str,
        command: Sequence[str],
        log_path: Path,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.name = name
        self.command = [str(item) for item in command]
        self.log_path = log_path
        self.env = dict(env) if env else None
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None

    def start(self) -> "ManagedProcess":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("wb")
        self.process = subprocess.Popen(
            self.command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=self.env,
            start_new_session=True,
        )
        return self

    @property
    def pid(self) -> int:
        if self.process is None:
            raise HarnessError(f"process {self.name} has not started")
        return self.process.pid

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def require_alive(self) -> None:
        rc = self.poll()
        if rc is not None:
            raise HarnessError(f"{self.name} exited early with status {rc}; see {self.log_path}")

    def wait(self, timeout: float | None = None) -> int:
        if self.process is None:
            raise HarnessError(f"process {self.name} has not started")
        try:
            return self.process.wait(timeout=timeout)
        finally:
            if self.process.poll() is not None:
                self._close_log()

    def stop(self, grace: float = 10.0) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
        self._close_log()

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def _start_system_telemetry(
    run_dir: Path,
    *,
    role: str,
    tracked: Mapping[str, int],
    metrics_url: str | None = None,
) -> list[ManagedProcess]:
    """Start the requested one-second system and process telemetry streams."""

    commands: list[tuple[str, list[str], Path]] = [
        (
            f"{role}-mpstat",
            ["mpstat", "-P", "ALL", "1"],
            run_dir / f"{role}-mpstat.log",
        ),
        (
            f"{role}-pidstat",
            ["pidstat", "-h", "-u", "-r", "-w", "-t", "-p", "ALL", "1"],
            run_dir / f"{role}-pidstat.log",
        ),
    ]
    collector = [
        sys.executable,
        str(Path(__file__).with_name("telemetry_capture.py")),
        "--role",
        role,
        "--output",
        str(run_dir / f"{role}-system-telemetry.jsonl"),
    ]
    for name, pgid in tracked.items():
        collector.extend(["--tracked-pgid", f"{name}:{pgid}"])
    if metrics_url is not None:
        collector.extend(["--metrics-url", metrics_url])
    commands.append((f"{role}-collector", collector, run_dir / f"{role}-collector.log"))

    processes: list[ManagedProcess] = []
    try:
        for name, command, log_path in commands:
            processes.append(ManagedProcess(name, command, log_path).start())
    except BaseException:
        for process in reversed(processes):
            process.stop()
        raise
    return processes


def _stop_processes(processes: Iterable[ManagedProcess]) -> None:
    errors: list[str] = []
    for process in reversed(list(processes)):
        try:
            process.stop()
        except Exception as error:
            errors.append(f"{process.name}: {type(error).__name__}: {error}")
    if errors:
        raise HarnessError("telemetry cleanup failed: " + "; ".join(errors))


def _process_group_snapshot(pgid: int) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for path in sorted(Path("/proc").glob("[0-9]*"), key=lambda item: int(item.name)):
        pid = int(path.name)
        try:
            if os.getpgid(pid) != pgid:
                continue
            cmdline = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            ).strip()
            processes.append(
                {
                    "pid": pid,
                    "ppid": int((path / "stat").read_text().split(") ", 1)[1].split()[1]),
                    "pgid": pgid,
                    "comm": (path / "comm").read_text().strip(),
                    "cmdline": cmdline,
                    "thread_count": sum(1 for _ in (path / "task").iterdir()),
                    "cpus_allowed": sorted(os.sched_getaffinity(pid)),
                }
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    return processes


def _wait_for_aiperf_process_tree(
    process: ManagedProcess,
    run_dir: Path,
    *,
    expected_workers: int,
    expected_record_processors: int,
    timeout: float = 180,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        process.require_alive()
        latest = _process_group_snapshot(os.getpgid(process.pid))
        workers = [
            item
            for item in latest
            if re.search(r"(?:^| )aiperf worker_[0-9a-f]{8}(?: |$)", item["cmdline"])
        ]
        record_processors = [
            item
            for item in latest
            if re.search(
                r"(?:^| )aiperf record_processor_[0-9a-f]{8}(?: |$)",
                item["cmdline"],
            )
        ]
        if (
            len(workers) == expected_workers
            and len(record_processors) == expected_record_processors
        ):
            result = {
                "validated_at": now(),
                "pgid": os.getpgid(process.pid),
                "expected_workers": expected_workers,
                "observed_workers": len(workers),
                "expected_record_processors": expected_record_processors,
                "observed_record_processors": len(record_processors),
                "processes": latest,
            }
            atomic_json(run_dir / "aiperf_process_tree.json", result)
            return result
        time.sleep(1)
    atomic_json(
        run_dir / "aiperf_process_tree.json",
        {
            "validated_at": now(),
            "accepted": False,
            "expected_workers": expected_workers,
            "expected_record_processors": expected_record_processors,
            "processes": latest,
        },
    )
    raise HarnessError(
        "AIPerf did not expose exactly "
        f"{expected_workers} worker and {expected_record_processors} record-processor processes"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture_frontend_dso_manifest(pid: int, run_dir: Path) -> dict[str, Any]:
    """Resolve the live ARM extension from proc maps and validate its exact ABI."""

    maps_text = Path(f"/proc/{pid}/maps").read_text()
    (run_dir / "frontend.maps").write_text(maps_text, encoding="utf-8")
    paths = sorted(
        {
            line.split(maxsplit=5)[5].removesuffix(" (deleted)")
            for line in maps_text.splitlines()
            if len(line.split(maxsplit=5)) == 6
            and line.split(maxsplit=5)[5].startswith("/")
        }
    )
    core_paths = [Path(path) for path in paths if path.endswith(("/_core.abi3.so", "/_core.so"))]
    if len(core_paths) != 1 or not core_paths[0].is_file():
        raise HarnessError(f"expected one live _core.abi3.so mapping, found {core_paths}")
    core = core_paths[0].resolve()
    file_output = subprocess.check_output(["file", "-L", str(core)], text=True).strip()
    if "ELF 64-bit LSB" not in file_output or not re.search(
        r"AArch64|ARM aarch64", file_output, flags=re.IGNORECASE
    ):
        raise HarnessError(f"mapped Dynamo extension is not AArch64: {file_output}")
    notes = subprocess.check_output(["readelf", "-n", str(core)], text=True)
    build_match = re.search(r"Build ID:\s*([0-9a-f]+)", notes, flags=re.IGNORECASE)
    if build_match is None:
        raise HarnessError(f"mapped Dynamo extension lacks a build ID: {core}")
    result = {
        "captured_at": now(),
        "pid": pid,
        "architecture": os.uname().machine,
        "core_path_from_proc_maps": str(core),
        "core_file": file_output,
        "core_size": core.stat().st_size,
        "core_sha256": _sha256_file(core),
        "core_build_id": build_match.group(1).lower(),
        "addr2line": subprocess.check_output(["addr2line", "--version"], text=True).splitlines()[0],
        "mapped_files": paths,
    }
    atomic_json(run_dir / "frontend-dso-manifest.json", result)
    return result


def _capture_mapped_dso_checksums(run_dir: Path) -> None:
    paths = sorted(
        {
            line.split(maxsplit=5)[5].removesuffix(" (deleted)")
            for line in (run_dir / "frontend.maps").read_text().splitlines()
            if len(line.split(maxsplit=5)) == 6
            and line.split(maxsplit=5)[5].startswith("/")
            and ".so" in Path(line.split(maxsplit=5)[5]).name
        }
    )
    checksums = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            checksums.append(
                {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": _sha256_file(path)}
            )
    atomic_json(run_dir / "mapped-dso-checksums.json", {"files": checksums})


class CommandChannel:
    def __init__(self, state_dir: Path, shutdown: threading.Event) -> None:
        self.commands = state_dir / "commands"
        self.acks = state_dir / "acks"
        self.shutdown = shutdown
        self.sequence = 0

    def send(self, action: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        self.sequence += 1
        name = f"{self.sequence:04d}"
        command = {
            "sequence": self.sequence,
            "action": action,
            "payload": dict(payload),
            "issued_at": now(),
        }
        atomic_json(self.commands / f"{name}.json", command)
        ack_path = self.acks / f"{name}.json"
        wait_for_path(ack_path, timeout, self.shutdown)
        ack = read_json(ack_path)
        if not ack.get("ok"):
            raise HarnessError(
                f"remote action {action} failed: {ack.get('error', 'unknown remote error')}"
            )
        return ack

    def receive(self, expected_sequence: int) -> dict[str, Any]:
        path = self.commands / f"{expected_sequence:04d}.json"
        wait_for_path(path, 24 * 60 * 60, self.shutdown)
        command = read_json(path)
        if command.get("sequence") != expected_sequence:
            raise HarnessError(f"malformed command at {path}")
        return command

    def acknowledge(
        self,
        sequence: int,
        action: str,
        *,
        ok: bool,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        atomic_json(
            self.acks / f"{sequence:04d}.json",
            {
                "sequence": sequence,
                "action": action,
                "ok": ok,
                "result": dict(result or {}),
                "error": error,
                "acknowledged_at": now(),
            },
        )


def completed_run_keys(
    root: str | Path,
    old_job_id: str,
    *,
    config_hash: str | None = None,
    harness_hash: str | None = None,
) -> set[str]:
    """Return only previously accepted, complete, matching measured run IDs."""

    result_root = job_paths(root, old_job_id)["result_dir"]
    completed: set[str] = set()
    runs_root = result_root / "runs"
    if not runs_root.is_dir():
        return completed
    for marker in runs_root.rglob("COMPLETE"):
        run_dir = marker.parent
        run_file = run_dir / "run.json"
        acceptance_file = run_dir / "acceptance.json"
        summary_file = run_dir / "summary.json"
        if (
            not run_file.is_file()
            or not acceptance_file.is_file()
            or not summary_file.is_file()
        ):
            continue
        try:
            run = read_json(run_file)
            acceptance = read_json(acceptance_file)
            summary = read_json(summary_file)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            not acceptance.get("accepted")
            or not summary.get("accepted")
            or run.get("calibration")
        ):
            continue
        if config_hash is not None and run.get("config_hash") != config_hash:
            continue
        if harness_hash is not None and run.get("harness_hash") != harness_hash:
            continue
        if isinstance(run.get("run_id"), str):
            completed.add(run["run_id"])
    return completed


def _frontend_placements(config: Mapping[str, Any]) -> Mapping[str, Any]:
    topology = config["topology"]
    return topology.get("frontend_placements", topology.get("frontend_arms", {}))


def _placement(config: Mapping[str, Any], arm: str, interleave: bool = False) -> dict[str, Any]:
    placements = _frontend_placements(config)
    key = "4+4-interleave" if interleave and "4+4-interleave" in placements else arm
    raw = dict(placements[key])
    if "memory" in raw:
        memory = raw["memory"]
    elif raw["memory_policy"] == "bind":
        memory = f"bind:{raw['memory_nodes'][0]}"
    elif raw["memory_policy"] == "interleave":
        memory = "interleave:" + ",".join(str(node) for node in raw["memory_nodes"])
    else:
        memory = "default"
    return {"cpus": raw["cpus"], "memory": memory}


def _remote_role(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    remote = config["topology"]["remote"]
    if role in remote and isinstance(remote[role], Mapping):
        return dict(remote[role])
    return {
        "cpus": remote[f"{role}_cpus"],
        "memory": f"bind:{remote[f'{role}_memory_node']}",
    }


def _port(config: Mapping[str, Any], name: str) -> int:
    if "ports" in config:
        return int(config["ports"][name])
    legacy = {
        "frontend_http": "frontend_http_port",
        "etcd_client": "etcd_client_port",
        "etcd_peer": "etcd_peer_port",
        "nats": "nats_port",
    }
    return int(config["network"][legacy[name]])


def _path(config: Mapping[str, Any], name: str) -> str:
    paths = config["paths"]
    aliases = {
        "dynamo_repo": ("dynamo_repo", "dynamo_checkout"),
        "dynamo_python": ("dynamo_python",),
        "aiperf": ("aiperf",),
        "model": ("model",),
        "hf_home": ("hf_home", "huggingface_cache"),
        "frontend_jemalloc": ("frontend_jemalloc", "jemalloc"),
    }
    for key in aliases[name]:
        if key in paths:
            return str(paths[key])
    raise KeyError(name)


def _pin(config: Mapping[str, Any], name: str) -> str:
    pins = config.get("pins", config.get("revisions", {}))
    aliases = {
        "model_revision": ("model_revision", "model"),
        "dynamo": ("dynamo",),
        "aiperf": ("aiperf",),
        "transformers": ("transformers",),
    }
    for key in aliases[name]:
        if key in pins:
            return str(pins[key])
    raise KeyError(name)


def _model_name(config: Mapping[str, Any]) -> str:
    model = config.get("model", {})
    return str(model.get("name", MODEL_NAME_DEFAULT)) if isinstance(model, Mapping) else MODEL_NAME_DEFAULT


def _workload_kind(workload: Mapping[str, Any]) -> str:
    kind = str(workload.get("kind", workload.get("dataset", "")))
    return "agentx" if kind in {"weka", "agentx"} else kind


def process_prefix(cpus: str, memory: str) -> list[str]:
    """Apply only the explicitly controlled placements used by this campaign."""

    if cpus == "0-143" and memory == "default":
        return []
    if cpus == "0-71" and memory == "bind:0":
        return ["numactl", "--physcpubind=0-71", "--membind=0"]
    if cpus == "72-143" and memory == "bind:1":
        return ["numactl", "--physcpubind=72-143", "--membind=1"]
    if cpus == "0-35" and memory == "bind:0":
        return ["numactl", "--physcpubind=0-35", "--membind=0"]
    if cpus == "36-71" and memory == "bind:0":
        return ["numactl", "--physcpubind=36-71", "--membind=0"]
    if cpus == "108-143" and memory == "bind:1":
        return ["numactl", "--physcpubind=108-143", "--membind=1"]
    if cpus == "72-107" and memory == "bind:1":
        return ["numactl", "--physcpubind=72-107", "--membind=1"]
    if cpus == "0-35,72-107" and memory == "local":
        return [
            "numactl",
            "--physcpubind=0-35,72-107",
            "--localalloc",
        ]
    raise HarnessError(f"unsupported Tyche placement: cpus={cpus}, memory={memory}")


def _read_cpu_snapshots(cpus: set[int]) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    with Path("/proc/stat").open() as handle:
        for line in handle:
            match = re.match(r"cpu(\d+)\s+(.+)", line)
            if not match or int(match.group(1)) not in cpus:
                continue
            values = [int(value) for value in match.group(2).split()]
            result[int(match.group(1))] = (
                sum(values),
                values[3] + (values[4] if len(values) > 4 else 0),
            )
    return result


def _read_cpu_snapshot(cpus: set[int]) -> tuple[int, int]:
    per_cpu = _read_cpu_snapshots(cpus)
    return sum(item[0] for item in per_cpu.values()), sum(item[1] for item in per_cpu.values())


def _read_nic_counters(interface: str) -> dict[str, int]:
    base = Path("/sys/class/net") / interface / "statistics"
    names = ("rx_bytes", "tx_bytes", "rx_dropped", "tx_dropped", "rx_errors", "tx_errors")
    result: dict[str, int] = {}
    for name in names:
        try:
            result[name] = int((base / name).read_text().strip())
        except (OSError, ValueError):
            result[name] = -1
    return result


def _tcp_stats() -> dict[str, int]:
    lines = Path("/proc/net/snmp").read_text().splitlines()
    for index in range(len(lines) - 1):
        if lines[index].startswith("Tcp:") and lines[index + 1].startswith("Tcp:"):
            headers = lines[index].split()[1:]
            values = lines[index + 1].split()[1:]
            return {
                name: int(values[headers.index(name)]) if name in headers else -1
                for name in ("OutSegs", "RetransSegs")
            }
    return {"OutSegs": -1, "RetransSegs": -1}


class Sampler:
    def __init__(
        self,
        output: Path,
        *,
        pools: Mapping[str, str],
        interface: str | Sequence[str],
        process: ManagedProcess | None = None,
        interval: float = 5.0,
        gate_path: Path | None = None,
        end_gate_path: Path | None = None,
    ) -> None:
        self.output = output
        self.pools = {name: parse_cpu_set(value) for name, value in pools.items()}
        self.interfaces = (
            [interface] if isinstance(interface, str) else list(dict.fromkeys(interface))
        )
        if not self.interfaces:
            raise HarnessError("sampler requires at least one network interface")
        # Keep the original single-interface field and sample shape for the
        # existing analysis scripts. Multi-NIC campaigns additionally record
        # every selected interface under `nics`.
        self.interface = self.interfaces[0]
        self.process = process
        self.interval = interval
        self.gate_path = gate_path
        self.end_gate_path = end_gate_path
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run_guarded, name=f"sampler-{output.name}", daemon=True
        )

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def stop(self, *, raise_on_error: bool = True) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 5)
        if self.thread.is_alive():
            if raise_on_error:
                raise HarnessError(f"sampler thread did not stop: {self.output}")
            return
        if self.error is not None and raise_on_error:
            raise HarnessError(f"sampler failed for {self.output}: {self.error}") from self.error

    def _run_guarded(self) -> None:
        try:
            self._run()
        except BaseException as error:
            self.error = error

    def _run(self) -> None:
        while self.gate_path is not None and not self.gate_path.exists():
            if self.stop_event.wait(0.1):
                return
        previous = {name: _read_cpu_snapshots(cpus) for name, cpus in self.pools.items()}
        with self.output.open("a", encoding="utf-8", buffering=1) as handle:
            self._write_sample(handle, previous)
            while True:
                deadline = time.monotonic() + self.interval
                while True:
                    if self.end_gate_path is not None and self.end_gate_path.exists():
                        self._write_sample(handle, previous)
                        return
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    if self.stop_event.wait(min(0.1, remaining)):
                        self._write_sample(handle, previous)
                        return
                self._write_sample(handle, previous)

    def _write_sample(
        self, handle: Any, previous: dict[str, dict[int, tuple[int, int]]]
    ) -> None:
        sample: dict[str, Any] = {
            "timestamp": now(),
            "monotonic_seconds": time.monotonic(),
            "nic": _read_nic_counters(self.interface),
            "link_speed_bits_per_second": _link_speed_bits(self.interface),
            "nics": {
                interface: {
                    "counters": _read_nic_counters(interface),
                    "link_speed_bits_per_second": _link_speed_bits(interface),
                }
                for interface in self.interfaces
            },
            "pools": {},
        }
        tcp = _tcp_stats()
        sample["tcp"] = tcp
        # Kept for analyze.py's compact-sampler adapter.
        sample["tcp_retransmits"] = tcp["RetransSegs"]
        for name, cpus in self.pools.items():
            current = _read_cpu_snapshots(cpus)
            previous_pool = previous[name]
            deltas = {
                cpu: (
                    current[cpu][0] - previous_pool[cpu][0],
                    current[cpu][1] - previous_pool[cpu][1],
                )
                for cpu in current.keys() & previous_pool.keys()
            }
            delta_total = sum(item[0] for item in deltas.values())
            delta_idle = sum(item[1] for item in deltas.values())
            sample["pools"][name] = {
                "busy_percent": (
                    100.0 * (delta_total - delta_idle) / delta_total
                    if delta_total > 0
                    else None
                ),
                "per_cpu_busy_percent": {
                    str(cpu): (
                        100.0 * (delta[0] - delta[1]) / delta[0]
                        if delta[0] > 0
                        else None
                    )
                    for cpu, delta in sorted(deltas.items())
                },
            }
            previous[name] = current
        if self.process is not None and self.process.poll() is None:
            pid = self.process.pid
            try:
                with Path(f"/proc/{pid}/numa_maps").open() as maps:
                    sample["private_anon_pages"] = private_anonymous_residency(maps)
                sample["affinity"] = sorted(os.sched_getaffinity(pid))
                if len(self.pools) == 1:
                    sample["task_affinity"] = _task_affinity_summary(
                        pid, next(iter(self.pools.values()))
                    )
                sample["process_stat"] = Path(f"/proc/{pid}/stat").read_text()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                sample["process_disappeared"] = True
        handle.write(json.dumps(sample, sort_keys=True) + "\n")


def _route_to(peer: str, configured_interface: str = "auto") -> tuple[str, str]:
    peer_ip = socket.gethostbyname(peer)
    command = ["ip", "-j", "route", "get", peer_ip]
    route = json.loads(subprocess.check_output(command, text=True))[0]
    interface = configured_interface if configured_interface != "auto" else route["dev"]
    if configured_interface != "auto" and route["dev"] != configured_interface:
        raise HarnessError(
            f"route to {peer} uses {route['dev']}, not configured interface {configured_interface}"
        )
    source = route.get("prefsrc") or route.get("src")
    if not source:
        address = json.loads(
            subprocess.check_output(["ip", "-j", "address", "show", "dev", interface], text=True)
        )
        candidates = [
            item["local"]
            for item in address[0].get("addr_info", [])
            if item.get("family") == "inet" and item.get("scope") == "global"
        ]
        if not candidates:
            raise HarnessError(f"no global IPv4 address found on {interface}")
        source = candidates[0]
    return interface, source


def _slurm_hosts() -> list[str]:
    node_list = os.environ.get("SLURM_STEP_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    if not node_list:
        raise HarnessError("SLURM_JOB_NODELIST is missing")
    hosts = subprocess.check_output(["scontrol", "show", "hostnames", node_list], text=True).split()
    if len(hosts) != 2 or len(set(hosts)) != 2:
        raise HarnessError(f"expected two distinct Slurm hosts, found {hosts}")
    return hosts


def _wait_http(url: str, model: str, process: ManagedProcess, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        process.require_alive()
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                body = response.read().decode()
            if model in body:
                return
            last_error = f"model absent from response: {body[:200]}"
        except (OSError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(1)
    raise HarnessError(f"frontend did not become ready: {last_error}")


def _port_is_available(address: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((address, port))
        except OSError:
            return False
    return True


def _runtime(config: Mapping[str, Any], key: str, default: Any) -> Any:
    return config.get("runtime", {}).get(key, default)


def _base_runtime_env(
    config: Mapping[str, Any], advertised_ip: str, remote_ip: str, _interface: str
) -> dict[str, str]:
    env = os.environ.copy()
    for key in ('RAYON_NUM_THREADS','RAYON_RS_NUM_THREADS','TOKENIZERS_PARALLELISM','FASTOKENS_BPE_THREADS'):
        env.pop(key, None)
    for key in list(env):
        if key.startswith('OTEL_'): env.pop(key, None)
    for key in list(env):
        if key.startswith(('DYN_QUIC_', 'DYN_VELO_', 'UCX_')): env.pop(key, None)
    env.update({'DYN_RESPONSE_PLANE': config['runtime']['response_plane'],
                'DYN_LOG':'warn', 'OTEL_SDK_DISABLED':'true',
                'DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB':'1', 'DYN_TOKENIZER_FALLBACK':'0'})
    if config['runtime']['response_plane'] == 'velo':
        transport = config['runtime']['velo_response_transport']
        env['DYN_VELO_RESPONSE_TRANSPORT'] = transport
        if transport == 'ucx':
            env.update(config['network']['ucx_env'])
    if config['runtime']['response_plane'] == 'quic':
        env.update({'DYN_QUIC_RESPONSE_BATCH_INTERVAL_US':'5000', 'DYN_QUIC_RESPONSE_BUFFER_CAPACITY':'16384'})
    env.update(
        {
            "ETCD_ENDPOINTS": f"http://{remote_ip}:{_port(config, 'etcd_client')}",
            "DYN_TCP_RPC_HOST": advertised_ip,
            "DYN_TCP_RESPONSE_STREAM_HOST": advertised_ip,
            "DYN_REQUEST_PLANE": str(_runtime(config, "request_plane", "tcp")),
            "DYN_REQUEST_PLANE_CODEC": str(_runtime(config, "request_plane_codec", "json")),
            "DYN_EVENT_PLANE": str(_runtime(config, "event_plane", "zmq")),
            "DYN_EVENT_PLANE_CODEC": str(_runtime(config, "event_plane_codec", "msgpack")),
            "DYN_ZMQ_BROKER_ENABLED": (
                "true" if _runtime(config, "zmq_broker_enabled", False) else "false"
            ),
            "DYN_TOKENIZER": str(_runtime(config, "tokenizer", "fastokens")),
            "DYN_TOKENIZER_CACHE": "1" if _runtime(config, "tokenizer_cache", True) else "0",
            "DYN_TOKENIZER_CACHE_BYTES": str(
                _runtime(config, "tokenizer_cache_bytes", 4_294_967_296)
            ),
            "HF_HOME": _path(config, "hf_home"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    if bool(config.get("network", {}).get("event_plane_host_override", False)):
        env["DYN_EVENT_PLANE_HOST"] = advertised_ip
    if bool(config.get("network", {}).get("response_stream_host_override", False)):
        # Pin the call-home listener to the same explicit NUMA-local address as
        # request-plane and event-plane advertisement. The runtime also accepts
        # an interface name, but the literal keeps benchmark provenance exact.
        env["DYN_RESPONSE_STREAM_HOST"] = advertised_ip
        env["DYN_TCP_RESPONSE_STREAM_HOST"] = advertised_ip
    return env


def _process_affinity(pid: int) -> list[int]:
    return sorted(os.sched_getaffinity(pid))


def _task_affinity_summary(pid: int, allowed: set[int]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    outside: dict[str, list[int]] = {}
    for path in Path(f"/proc/{pid}/task").iterdir():
        try:
            affinity = set(os.sched_getaffinity(int(path.name)))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
        key = ",".join(str(cpu) for cpu in sorted(affinity))
        counts[key] = counts.get(key, 0) + 1
        unexpected = affinity - allowed
        if unexpected:
            outside[path.name] = sorted(unexpected)
    return {
        "affinity_sets": counts,
        "outside_allowed": outside,
        "task_count": sum(counts.values()),
    }


def _validate_process_affinity(process: ManagedProcess, cpu_spec: str) -> dict[str, Any]:
    allowed = parse_cpu_set(cpu_spec)
    root = set(os.sched_getaffinity(process.pid))
    tasks = _task_affinity_summary(process.pid, allowed)
    valid = root == allowed and not tasks["outside_allowed"] and tasks["task_count"] > 0
    return {
        "valid": valid,
        "pid": process.pid,
        "expected": sorted(allowed),
        "root": sorted(root),
        "tasks": tasks,
    }


def _validate_process_memory(
    process: ManagedProcess, expected_node: int, minimum_fraction: float = 0.99
) -> dict[str, Any]:
    with Path(f"/proc/{process.pid}/numa_maps").open() as maps:
        pages = private_anonymous_residency(maps)
    total = sum(pages.values())
    local_fraction = pages.get(expected_node, 0) / total if total else 0.0
    return {
        "valid": total > 0 and local_fraction >= minimum_fraction,
        "pid": process.pid,
        "expected_node": expected_node,
        "private_anon_pages": {str(node): count for node, count in sorted(pages.items())},
        "private_anon_total_pages": total,
        "local_fraction": local_fraction,
        "minimum_fraction": minimum_fraction,
    }


def _read_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if not path.exists():
        return samples
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def _link_speed_bits(interface: str) -> int | None:
    try:
        megabits = int((Path("/sys/class/net") / interface / "speed").read_text().strip())
    except (OSError, ValueError):
        return None
    return megabits * 1_000_000 if megabits > 0 else None


def _nic_summary(samples: Sequence[Mapping[str, Any]], interface: str) -> dict[str, Any]:
    usable = [sample for sample in samples if sample.get("nic")]
    if len(usable) < 2:
        return {"accepted": False, "reason": "fewer than two NIC samples"}
    first, last = usable[0], usable[-1]
    elapsed = float(last["monotonic_seconds"]) - float(first["monotonic_seconds"])
    if elapsed <= 0:
        return {"accepted": False, "reason": "non-positive NIC sample interval"}
    deltas = {
        key: int(last["nic"].get(key, -1)) - int(first["nic"].get(key, -1))
        for key in ("rx_bytes", "tx_bytes", "rx_dropped", "tx_dropped", "rx_errors", "tx_errors")
    }
    speed = _link_speed_bits(interface)
    utilization = None
    if speed:
        utilization = 100.0 * max(deltas["rx_bytes"], deltas["tx_bytes"]) * 8 / elapsed / speed
    retransmit_delta = int(last.get("tcp", {}).get("RetransSegs", -1)) - int(
        first.get("tcp", {}).get("RetransSegs", -1)
    )
    out_segments_delta = int(last.get("tcp", {}).get("OutSegs", -1)) - int(
        first.get("tcp", {}).get("OutSegs", -1)
    )
    retransmit_fraction = (
        retransmit_delta / out_segments_delta if out_segments_delta > 0 else 0.0
    )
    counters_available = all(
        int(first["nic"].get(key, -1)) >= 0 and int(last["nic"].get(key, -1)) >= 0
        for key in deltas
    )
    tcp_available = all(
        int(sample.get("tcp", {}).get(key, -1)) >= 0
        for sample in (first, last)
        for key in ("RetransSegs", "OutSegs")
    )
    return {
        "accepted": counters_available
        and tcp_available
        and all(
            deltas[key] == 0
            for key in ("rx_dropped", "tx_dropped", "rx_errors", "tx_errors")
        ),
        "counters_available": counters_available,
        "tcp_counters_available": tcp_available,
        "elapsed_seconds": elapsed,
        "deltas": deltas,
        "link_speed_bits_per_second": speed,
        "peak_direction_utilization_percent": utilization,
        "tcp_retransmit_delta": retransmit_delta,
        "tcp_out_segments_delta": out_segments_delta,
        "tcp_retransmit_fraction": retransmit_fraction,
    }


def _sampling_coverage(
    samples: Sequence[Mapping[str, Any]],
    profiling_started_at: str,
    profiling_ended_at: str,
    *,
    minimum_fraction: float,
    maximum_fraction: float,
    interval: float = 5.0,
) -> dict[str, Any]:
    expected = (
        dt.datetime.fromisoformat(profiling_ended_at)
        - dt.datetime.fromisoformat(profiling_started_at)
    ).total_seconds()
    monotonic = [
        float(sample["monotonic_seconds"])
        for sample in samples
        if isinstance(sample.get("monotonic_seconds"), (int, float))
    ]
    observed = max(monotonic) - min(monotonic) if len(monotonic) >= 2 else 0.0
    fraction = observed / expected if expected > 0 else 0.0
    minimum_samples = max(3, int(expected / interval * 0.8))
    return {
        "accepted": len(monotonic) >= minimum_samples
        and minimum_fraction <= fraction <= maximum_fraction,
        "sample_count": len(monotonic),
        "minimum_sample_count": minimum_samples,
        "expected_seconds": expected,
        "observed_seconds": observed,
        "coverage_fraction": fraction,
        "minimum_coverage_fraction": minimum_fraction,
        "maximum_coverage_fraction": maximum_fraction,
    }


def _memory_summary(
    samples: Sequence[Mapping[str, Any]],
    placement: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    *,
    profiling_started_at: str | None = None,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for sample in samples:
        pages = {
            int(key): int(value)
            for key, value in sample.get("private_anon_pages", {}).items()
        }
        total = sum(pages.values())
        if total:
            observations.append(
                {
                    "timestamp": sample["timestamp"],
                    "pages": pages,
                    "fractions": {
                        str(node): count / total for node, count in pages.items()
                    },
                }
            )
    if not observations:
        return {
            "accepted": False,
            "reason": "no private-anonymous residency observations",
        }

    policy = placement["memory"]
    if policy.startswith("bind:"):
        node = int(policy.split(":", 1)[1])
        minimum = float(
            acceptance.get(
                "bound_memory_min_fraction",
                acceptance.get("bound_private_anon_local_fraction_min", 0.99),
            )
        )
        fractions = [
            item["fractions"].get(str(node), 0.0) for item in observations
        ]
        accepted = min(fractions) >= minimum
        return {
            "accepted": accepted,
            "policy": policy,
            "minimum_observed_local_fraction": min(fractions),
            "required_local_fraction": minimum,
            "observations": observations,
        }

    if policy == "default":
        return {
            "accepted": True,
            "policy": "inherited-default",
            "observations": observations,
        }

    minimum = float(
        acceptance.get(
            "split_memory_min_fraction",
            acceptance.get("split_private_anon_node_fraction_min", 0.4),
        )
    )
    maximum = float(
        acceptance.get(
            "split_memory_max_fraction",
            acceptance.get("split_private_anon_node_fraction_max", 0.6),
        )
    )
    grace_seconds = float(
        acceptance.get("split_memory_startup_grace_seconds", 0.0)
    )
    evaluated = observations
    if grace_seconds > 0.0 and profiling_started_at is not None:
        started = dt.datetime.fromisoformat(profiling_started_at)
        evaluated = [
            item
            for item in observations
            if (
                dt.datetime.fromisoformat(item["timestamp"]) - started
            ).total_seconds()
            >= grace_seconds
        ]
    if not evaluated:
        return {
            "accepted": False,
            "policy": policy,
            "reason": (
                "no private-anonymous residency observations after startup grace"
            ),
            "startup_grace_seconds": grace_seconds,
            "ignored_initial_observation_count": len(observations),
            "evaluated_observation_count": 0,
            "observations": observations,
        }

    all_fractions = [
        item["fractions"].get("0", 0.0) for item in observations
    ]
    fractions = [item["fractions"].get("0", 0.0) for item in evaluated]
    accepted = all(minimum <= value <= maximum for value in fractions)
    return {
        "accepted": accepted,
        "policy": policy,
        "node0_fraction_range": [min(fractions), max(fractions)],
        "all_observations_node0_fraction_range": [
            min(all_fractions),
            max(all_fractions),
        ],
        "required_range": [minimum, maximum],
        "startup_grace_seconds": grace_seconds,
        "ignored_initial_observation_count": len(observations) - len(evaluated),
        "evaluated_observation_count": len(evaluated),
        "observations": observations,
    }


def _performance_similarity(
    arm_mean_throughputs: Mapping[str, float], threshold_fraction: float
) -> dict[str, Any]:
    expected = {"8+0", "4+4", "0+8"}
    if set(arm_mean_throughputs) != expected:
        return {
            "similar": False,
            "reason": "all three primary arms need throughput observations",
            "arm_mean_throughputs_rps": dict(arm_mean_throughputs),
            "relative_span": None,
            "threshold_fraction": threshold_fraction,
        }
    center = sum(arm_mean_throughputs.values()) / len(arm_mean_throughputs)
    relative_span = (
        (
            max(arm_mean_throughputs.values())
            - min(arm_mean_throughputs.values())
        )
        / center
        if center
        else float("inf")
    )
    similar = relative_span <= threshold_fraction
    return {
        "similar": similar,
        "reason": (
            "arm-mean throughput span is within threshold"
            if similar
            else "arm-mean throughput span exceeds threshold"
        ),
        "arm_mean_throughputs_rps": dict(arm_mean_throughputs),
        "relative_span": relative_span,
        "threshold_fraction": threshold_fraction,
    }


def _execution_summary(
    samples: Sequence[Mapping[str, Any]],
    placement: Mapping[str, Any],
    topology: Mapping[str, Any],
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    assigned = parse_cpu_set(str(placement["cpus"]))
    node_cpus = {
        int(node): parse_cpu_set(cpus)
        for node, cpus in topology.get(
            "node_cpus", {"0": "0-71", "1": "72-143"}
        ).items()
    }
    by_node: dict[int, list[float]] = {
        node: [] for node, cpus in node_cpus.items() if assigned & cpus
    }
    per_cpu: dict[int, list[float]] = {cpu: [] for cpu in assigned}
    for sample in samples:
        values = sample.get("pools", {}).get("frontend", {}).get(
            "per_cpu_busy_percent", {}
        )
        if not isinstance(values, Mapping):
            continue
        numeric = {
            int(cpu): float(value)
            for cpu, value in values.items()
            if isinstance(value, (int, float)) and int(cpu) in assigned
        }
        for cpu, value in numeric.items():
            per_cpu[cpu].append(value)
        for node, cpus in node_cpus.items():
            selected = [value for cpu, value in numeric.items() if cpu in cpus]
            if selected and node in by_node:
                by_node[node].append(sum(selected) / len(selected))

    node_means = {
        str(node): sum(values) / len(values) if values else None
        for node, values in by_node.items()
    }
    minimum = float(acceptance.get("split_cpu_min_busy_percent", 5.0))
    spans_nodes = len(by_node) > 1
    accepted = bool(node_means) and all(value is not None for value in node_means.values())
    if spans_nodes:
        accepted = accepted and all(
            float(value) >= minimum for value in node_means.values() if value is not None
        )
    return {
        "accepted": accepted,
        "spans_nodes": spans_nodes,
        "node_mean_busy_percent": node_means,
        "split_node_minimum_busy_percent": minimum if spans_nodes else None,
        "per_cpu_mean_busy_percent": {
            str(cpu): sum(values) / len(values) if values else None
            for cpu, values in sorted(per_cpu.items())
        },
        "sample_count": len(samples),
    }


def _validate_profile_artifact(path: Path, workload: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise HarnessError(f"missing AIPerf record export: {path}")
    records = cancelled = mismatched_tokens = wrong_synthetic_length = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise HarnessError(f"invalid JSONL in {path}: {error}") from error
            metadata = record.get("metadata", {})
            if metadata.get("benchmark_phase") != "profiling":
                continue
            records += 1
            cancelled += int(bool(metadata.get("was_cancelled")))
            metrics = record.get("metrics", {})
            output = metrics.get("output_token_count", {}).get("value")
            usage = metrics.get("usage_completion_tokens", {}).get("value")
            if output is None or usage is None or int(output) != int(usage):
                mismatched_tokens += 1
            if _workload_kind(workload) == "synthetic" and output is not None:
                expected = int(workload.get("osl", workload.get("output_tokens", 1024)))
                wrong_synthetic_length += int(int(output) != expected)
    if records == 0 or cancelled or mismatched_tokens or wrong_synthetic_length:
        raise HarnessError(
            "AIPerf record validation failed: "
            f"records={records}, cancelled={cancelled}, token_mismatches={mismatched_tokens}, "
            f"wrong_synthetic_length={wrong_synthetic_length}"
        )
    return {
        "profiling_records": records,
        "cancelled": cancelled,
        "token_count_mismatches": mismatched_tokens,
        "wrong_synthetic_length": wrong_synthetic_length,
    }


class RemoteAgent:
    def __init__(
        self,
        config: Mapping[str, Any],
        channel: CommandChannel,
        shutdown: threading.Event,
        *,
        job_id: str,
        advertised_ip: str,
        frontend_ip: str,
        interface: str,
        simulate: bool,
    ) -> None:
        self.config = config
        self.channel = channel
        self.shutdown = shutdown
        self.job_id = job_id
        self.advertised_ip = advertised_ip
        self.frontend_ip = frontend_ip
        self.interface = interface
        self.simulate = simulate
        self.processes: dict[str, ManagedProcess] = {}
        self.telemetry_processes: list[ManagedProcess] = []
        self.current_run_dir: Path | None = None
        self.current_scratch_dir: Path | None = None

    def serve(self) -> None:
        sequence = 1
        try:
            while not self.shutdown.is_set():
                command = self.channel.receive(sequence)
                action = command["action"]
                try:
                    if action == "prepare":
                        result = self.prepare(command["payload"])
                    elif action == "run_load":
                        result = self.run_load(command["payload"])
                    elif action == "stop":
                        self.cleanup()
                        result = {"stopped_at": now()}
                    elif action == "terminate":
                        self.cleanup()
                        self.channel.acknowledge(sequence, action, ok=True, result={})
                        return
                    else:
                        raise HarnessError(f"unknown remote action: {action}")
                    self.channel.acknowledge(sequence, action, ok=True, result=result)
                except Exception as error:  # preserve the failure on the shared filesystem
                    cleanup_error = None
                    try:
                        self.cleanup()
                    except Exception as secondary:
                        cleanup_error = f"{type(secondary).__name__}: {secondary}"
                    detail = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
                    if cleanup_error is not None:
                        detail += f"\nsecondary cleanup failure: {cleanup_error}"
                    self.channel.acknowledge(
                        sequence,
                        action,
                        ok=False,
                        error=detail,
                    )
                sequence += 1
        finally:
            self.cleanup()

    def prepare(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.cleanup()
        self.current_run_dir = Path(payload["run_dir"])
        if self.simulate:
            atomic_json(
                self.current_run_dir / "remote_simulation.json",
                {"prepared_at": now(), "advertised_ip": self.advertised_ip},
            )
            return {"simulated": True}

        run_dir = self.current_run_dir
        scratch_root = Path(
            os.environ.get(
                "SLURM_TMPDIR",
                f"/var/tmp/dynamo-numa-{os.environ.get('SLURM_JOB_ID', 'local')}",
            )
        )
        self.current_scratch_dir = (
            scratch_root
            / "dynamo-numa"
            / self.job_id
            / "rank1"
            / str(payload["run"]["run_id"])
        )
        self.current_scratch_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(
            run_dir / "remote_scratch.json",
            {"path": str(self.current_scratch_dir), "created_at": now()},
        )
        service_data = self.current_scratch_dir / "service-data"
        etcd_data = service_data / "etcd"
        etcd_data.mkdir(parents=True, exist_ok=False)

        for port_name in ("etcd_client", "etcd_peer"):
            port = _port(self.config, port_name)
            if not _port_is_available(self.advertised_ip, port):
                raise HarnessError(f"remote port {port} is already in use")

        control = _remote_role(self.config, "control")
        etcd_client = _port(self.config, "etcd_client")
        etcd_peer = _port(self.config, "etcd_peer")
        etcd_env = os.environ.copy()
        etcd_env["ETCD_UNSUPPORTED_ARCH"] = "arm64"
        etcd_command = process_prefix(control["cpus"], control["memory"]) + [
            "etcd",
            "--name",
            "default",
            "--data-dir",
            str(etcd_data),
            "--listen-client-urls",
            f"http://{self.advertised_ip}:{etcd_client}",
            "--advertise-client-urls",
            f"http://{self.advertised_ip}:{etcd_client}",
            "--listen-peer-urls",
            f"http://{self.advertised_ip}:{etcd_peer}",
            "--initial-advertise-peer-urls",
            f"http://{self.advertised_ip}:{etcd_peer}",
            "--initial-cluster",
            f"default=http://{self.advertised_ip}:{etcd_peer}",
        ]
        self.processes["etcd"] = ManagedProcess(
            "etcd", etcd_command, self.current_scratch_dir / "etcd.log", env=etcd_env
        ).start()
        self._wait_remote_ports((etcd_client,), timeout=45)
        endpoint = f"http://{self.advertised_ip}:{etcd_client}"
        subprocess.run(
            ["etcdctl", f"--endpoints={endpoint}", "endpoint", "health"],
            check=True,
            stdout=(run_dir / "etcd_health.txt").open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            timeout=10,
        )

        workload = self.config["workloads"][payload["workload"]]
        mocker = _remote_role(self.config, "mocker")
        env = _base_runtime_env(
            self.config, self.advertised_ip, self.advertised_ip, self.interface
        )
        atomic_json(
            run_dir / "mocker_environment.json",
            {
                key: env[key]
                for key in (
                    "ETCD_ENDPOINTS",
                    "DYN_TCP_RPC_HOST",
                    "DYN_REQUEST_PLANE",
                    "DYN_REQUEST_PLANE_CODEC",
                    "DYN_EVENT_PLANE",
                    "DYN_EVENT_PLANE_CODEC",
                    "DYN_ZMQ_BROKER_ENABLED",
                    "HF_HOME",
                )
            },
        )
        mocker_command = process_prefix(mocker["cpus"], mocker["memory"]) + [
            _path(self.config, "dynamo_python"),
            "-m",
            "dynamo.mocker",
            "--model-path",
            _path(self.config, "model"),
            "--model-name",
            _model_name(self.config),
            "--endpoint",
            "dyn://dynamo.backend.generate",
            "--num-workers",
            str(_runtime(self.config, "num_mockers", 16)),
            "--num-gpu-blocks-override",
            str(_runtime(self.config, "mocker_num_gpu_blocks", 1_000_000)),
            "--speedup-ratio",
            str(_runtime(self.config, "mocker_speedup_ratio", 1_000_000)),
            "--max-num-seqs",
            "100000",
            "--max-num-batched-tokens",
            "10000000",
            "--block-size",
            str(workload["block_size"]),
        ]
        self.processes["mocker"] = ManagedProcess(
            "mocker", mocker_command, self.current_scratch_dir / "mocker.log", env=env
        ).start()
        self._wait_for_mockers(endpoint, int(_runtime(self.config, "num_mockers", 16)))
        affinities = {
            "etcd": _validate_process_affinity(self.processes["etcd"], control["cpus"]),
            "mocker": _validate_process_affinity(self.processes["mocker"], mocker["cpus"]),
        }
        if not all(item["valid"] for item in affinities.values()):
            raise HarnessError(f"remote service affinity validation failed: {affinities}")
        atomic_json(run_dir / "remote_affinity.json", affinities)
        memory = {
            "etcd": {"valid": True, "policy": "inherited-default"},
            "mocker": {"valid": True, "policy": "inherited-default"},
        }
        if not all(item["valid"] for item in memory.values()):
            raise HarnessError(f"remote service memory validation failed: {memory}")
        atomic_json(run_dir / "remote_memory.json", memory)
        self._write_pids()
        return {
            "remote_ready_at": now(),
            "mocker_pid": self.processes["mocker"].pid,
            "mocker_affinity": _process_affinity(self.processes["mocker"].pid),
        }

    def _wait_remote_ports(self, ports: Sequence[int], timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for process in self.processes.values():
                process.require_alive()
            if all(self._can_connect(port) for port in ports):
                return
            time.sleep(0.5)
        raise HarnessError(f"remote services did not listen on ports {ports}")

    def _can_connect(self, port: int) -> bool:
        try:
            with socket.create_connection((self.advertised_ip, port), timeout=0.5):
                return True
        except OSError:
            return False

    def _wait_for_mockers(self, endpoint: str, expected: int) -> None:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            self.processes["mocker"].require_alive()
            completed = subprocess.run(
                [
                    "etcdctl",
                    f"--endpoints={endpoint}",
                    "get",
                    "--prefix",
                    "v1/instances/dynamo/backend/generate/",
                    "--keys-only",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            count = completed.stdout.count("generate/") if completed.returncode == 0 else 0
            if count == expected:
                return
            time.sleep(1)
        raise HarnessError(f"expected {expected} mocker registrations")

    def run_load(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.current_run_dir is None:
            raise HarnessError("run_load received before prepare")
        if self.simulate:
            atomic_json(
                self.current_run_dir / "remote_load_simulation.json",
                {"completed_at": now(), "payload": dict(payload)},
            )
            return {
                "simulated": True,
                "accepted": True,
                "profiling_records": 1,
                "remote_pool_average_busy_percent": {"mocker": 0.0, "aiperf": 0.0},
            }

        run_dir = self.current_run_dir
        workload = self.config["workloads"][payload["workload"]]
        if self.current_scratch_dir is None:
            raise HarnessError("remote scratch directory is unavailable")
        artifact_dir = self.current_scratch_dir / "load_artifacts"
        artifact_dir.mkdir(exist_ok=False)
        duration = int(payload["duration_seconds"])
        aiperf = _remote_role(self.config, "aiperf")
        command = self._aiperf_command(workload, duration, artifact_dir)
        env = _base_runtime_env(
            self.config, self.advertised_ip, self.advertised_ip, self.interface
        )
        load = ManagedProcess(
            "aiperf",
            process_prefix(aiperf["cpus"], aiperf["memory"]) + command,
            self.current_scratch_dir / "aiperf.log",
            env=env,
        ).start()
        self.processes["aiperf"] = load
        self._write_pids()
        atomic_json(run_dir / "aiperf_command.json", {"argv": load.command})

        process_tree = _wait_for_aiperf_process_tree(
            load,
            run_dir,
            expected_workers=int(_runtime(self.config, "aiperf_workers_max", 64)),
            expected_record_processors=int(
                _runtime(self.config, "aiperf_record_processors", 8)
            ),
        )
        self.telemetry_processes = _start_system_telemetry(
            run_dir,
            role="load",
            tracked={
                "mocker": os.getpgid(self.processes["mocker"].pid),
                "aiperf": os.getpgid(load.pid),
            },
        )

        profiling_started_at = self._wait_for_profiling(load, load.log_path)
        (run_dir / "PROFILING_STARTED").write_text(profiling_started_at + "\n", encoding="utf-8")
        remote_samples_path = self.current_scratch_dir / "remote_samples.jsonl"
        sampler = Sampler(
            remote_samples_path,
            pools={"load_node": "0-143"},
            interface=self.interface,
            end_gate_path=run_dir / "PROFILING_ENDED",
        )
        sampler.start()
        try:
            affinities = read_json(run_dir / "remote_affinity.json")
            affinities["mocker_under_load"] = _validate_process_affinity(
                self.processes["mocker"], _remote_role(self.config, "mocker")["cpus"]
            )
            affinities["aiperf"] = _validate_process_affinity(load, aiperf["cpus"])
            atomic_json(run_dir / "remote_affinity.json", affinities)
            affinity_accepted = all(item["valid"] for item in affinities.values())
            memory = read_json(run_dir / "remote_memory.json")
            memory["mocker_under_load"] = {
                "valid": True,
                "policy": "inherited-default",
            }
            memory["aiperf"] = {"valid": True, "policy": "inherited-default"}
            atomic_json(run_dir / "remote_memory.json", memory)
            memory_accepted = all(item["valid"] for item in memory.values())
        except BaseException:
            sampler.stop(raise_on_error=False)
            raise
        profiling_ended = False
        finalizer_terminated = False
        finalization_deadline: float | None = None
        try:
            deadline = time.monotonic() + duration + int(workload["grace_period_seconds"]) + 300
            while load.poll() is None:
                if self.shutdown.is_set():
                    raise HarnessError("interrupted while AIPerf was running")
                if time.monotonic() > deadline:
                    raise HarnessError("AIPerf exceeded duration, grace period, and finalization timeout")
                if finalization_deadline is not None and time.monotonic() > finalization_deadline:
                    load.stop(grace=10)
                    finalizer_terminated = True
                    break
                self.processes["mocker"].require_alive()
                if not profiling_ended and self._profiling_has_ended(load.log_path):
                    profiling_ended_at = now()
                    (run_dir / "PROFILING_ENDED").write_text(
                        profiling_ended_at + "\n", encoding="utf-8"
                    )
                    profiling_ended = True
                    finalization_deadline = time.monotonic() + 90
                    sampler.stop()
                time.sleep(1)
            rc = load.wait()
            if not profiling_ended and self._profiling_has_ended(load.log_path):
                profiling_ended_at = now()
                (run_dir / "PROFILING_ENDED").write_text(
                    profiling_ended_at + "\n", encoding="utf-8"
                )
                profiling_ended = True
        finally:
            sampler.stop(raise_on_error=sys.exc_info()[0] is None)
        if rc != 0 and not finalizer_terminated:
            raise HarnessError(f"AIPerf exited with status {rc}; see {load.log_path}")
        if not profiling_ended:
            raise HarnessError("AIPerf exited without a detectable profiling-complete marker")

        profile_summary = _validate_profile_artifact(
            artifact_dir / "profile_export.jsonl", workload
        )
        shutil.copytree(artifact_dir, run_dir / "load_artifacts")
        shutil.copy2(remote_samples_path, run_dir / "remote_samples.jsonl")
        log_text = load.log_path.read_text(errors="replace")
        if not finalizer_terminated and not re.search(
            r"Processed [0-9,]+ valid requests and 0 errors|completed=[0-9,]+, cancelled=0, errors=0",
            log_text,
        ):
            raise HarnessError("AIPerf log does not contain a zero-error completion summary")
        context_overflow_lines = [
            line
            for line in log_text.splitlines()
            if re.search(
                r"(?:context|token).*(?:overflow|exceed).*(?:skip|drop)|"
                r"(?:skip|drop).*(?:context|token).*(?:overflow|exceed)",
                line,
                flags=re.IGNORECASE,
            )
        ]
        if context_overflow_lines:
            raise HarnessError(
                f"AIPerf reported context-overflow skips: {context_overflow_lines[:10]}"
            )

        samples = _read_samples(remote_samples_path)
        sampling = _sampling_coverage(
            samples,
            profiling_started_at,
            profiling_ended_at,
            minimum_fraction=float(
                self.config["acceptance"].get("telemetry_min_coverage_fraction", 0.98)
            ),
            maximum_fraction=float(
                self.config["acceptance"].get("telemetry_max_coverage_fraction", 1.02)
            ),
        )
        busy_values = [
            float(sample["pools"]["load_node"]["busy_percent"])
            for sample in samples
            if sample.get("pools", {}).get("load_node", {}).get("busy_percent")
            is not None
        ]
        average_busy = sum(busy_values) / len(busy_values) if busy_values else 100.0
        ordered_busy = sorted(busy_values)
        p95_busy = (
            ordered_busy[min(len(ordered_busy) - 1, int(0.95 * len(ordered_busy)))]
            if ordered_busy
            else 100.0
        )
        average_limit = float(
            self.config["acceptance"].get("load_node_average_busy_max_percent", 85.0)
        )
        p95_limit = float(
            self.config["acceptance"].get("load_node_p95_busy_max_percent", 95.0)
        )
        load_health_accepted = average_busy < average_limit and p95_busy < p95_limit
        nic = _nic_summary(samples, self.interface)
        nic_limit = float(
            self.config["acceptance"].get(
                "nic_max_utilization_percent",
                100.0 * self.config["acceptance"].get("nic_link_utilization_max", 0.7),
            )
        )
        nic_accepted = bool(nic.get("accepted")) and (
            nic.get("peak_direction_utilization_percent") is None
            or float(nic["peak_direction_utilization_percent"]) <= nic_limit
        ) and float(nic.get("tcp_retransmit_fraction", 0.0)) <= float(
            self.config["acceptance"].get("tcp_retransmit_max_fraction", 0.05)
        )
        result = {
            **profile_summary,
            "profiling_started_at": profiling_started_at,
            "sampling": sampling,
            "aiperf_process_tree": process_tree,
            "aiperf_finalizer_terminated_after_records": finalizer_terminated,
            "context_overflow_skips": 0,
            "load_node_busy_percent": {
                "average": average_busy,
                "p95": p95_busy,
                "average_limit": average_limit,
                "p95_limit": p95_limit,
                "accepted": load_health_accepted,
            },
            "client_mocker_contaminated": not load_health_accepted,
            "nic": nic,
            "affinity": {"accepted": affinity_accepted, "roles": affinities},
            "memory": {"accepted": memory_accepted, "roles": memory},
            "accepted": affinity_accepted
            and memory_accepted
            and bool(sampling["accepted"]),
        }
        atomic_json(run_dir / "remote_acceptance.json", result)
        return result

    def _wait_for_profiling(self, process: ManagedProcess, log_path: Path) -> str:
        deadline = time.monotonic() + 420
        while time.monotonic() < deadline:
            process.require_alive()
            try:
                text = log_path.read_text(errors="replace")
            except OSError:
                text = ""
            if "Phase profiling started" in text:
                return now()
            time.sleep(1)
        raise HarnessError("AIPerf did not enter the profiling phase")

    @staticmethod
    def _profiling_has_ended(log_path: Path) -> bool:
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            return False
        return bool(
            re.search(
                r"Phase profiling (?:complete|completed|ended|finished)",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _aiperf_command(
        self, workload: Mapping[str, Any], duration: int, artifact_dir: Path
    ) -> list[str]:
        if bool(_runtime(self.config, "frontend_dual_numa", False)):
            urls = [
                f"http://{self.frontend_ip}:8001",
                f"http://{self.frontend_ip}:8002",
            ]
        else:
            urls = [f"http://{self.frontend_ip}:{_port(self.config, 'frontend_http')}"]
        command = [
            _path(self.config, "aiperf"),
            "profile",
            "--model",
            _model_name(self.config),
            "--tokenizer",
            _path(self.config, "model"),
            "--url",
            urls[0],
            "--url-strategy",
            "round-robin",
            "--endpoint-type",
            "chat",
            "--streaming",
            "--concurrency",
            str(workload["concurrency"]),
            "--benchmark-duration",
            str(duration),
            "--benchmark-grace-period",
            str(workload["grace_period_seconds"]),
            "--warmup-request-count",
            str(workload["warmup_request_count"]),
            "--workers-max",
            str(_runtime(self.config, "aiperf_workers_max", 64)),
            "--record-processors",
            str(_runtime(self.config, "aiperf_record_processors", 8)),
            "--random-seed",
            str(self.config["matrix"].get("random_seed", 12345)),
            "--output-artifact-dir",
            str(artifact_dir),
            "--export-level",
            "records",
            "--server-metrics-formats",
            "json",
            "csv",
            "jsonl",
            "--ui",
            "none",
            "--extra-inputs",
            "ignore_eos:true",
            "--use-server-token-count",
        ]
        for url in urls[1:]:
            command.extend(["--url", url])
        if _workload_kind(workload) == "synthetic":
            command.extend(
                [
                    "--isl",
                    str(workload.get("isl", workload.get("input_tokens", 1024))),
                    "--isl-stddev",
                    "0",
                    "--osl",
                    str(workload.get("osl", workload.get("output_tokens", 1024))),
                    "--osl-stddev",
                    "0",
                    "--conversation-turn-mean",
                    "1",
                    "--num-dataset-entries",
                    str(workload.get("dataset_entries", 1024)),
                ]
            )
        else:
            repo = workload.get(
                "weka_repo",
                self.config.get("model", {}).get(
                    "agentx_dataset", "semianalysisai/cc-traces-weka-with-subagents-060526"
                ),
            )
            command.extend(
                [
                    "--no-fixed-schedule",
                    "--ignore-trace-delays",
                    "--public-dataset",
                    "weka_hf",
                    "--hf-weka-repo",
                    str(repo),
                    "--num-dataset-entries",
                    str(workload.get("dataset_entries", 336)),
                ]
            )
        return command

    def _write_pids(self) -> None:
        if self.current_run_dir is None:
            return
        atomic_json(
            self.current_run_dir / "remote_pids.json",
            {
                name: {
                    "pid": process.pid,
                    "pgid": os.getpgid(process.pid),
                    "argv": process.command,
                }
                for name, process in self.processes.items()
                if process.poll() is None
            },
        )

    def cleanup(self) -> None:
        errors: list[str] = []
        run_dir = self.current_run_dir
        scratch_dir = self.current_scratch_dir
        process_order = ["aiperf", "mocker", "nats", "etcd"] + sorted(
            set(self.processes) - {"aiperf", "mocker", "nats", "etcd"}
        )
        try:
            try:
                _stop_processes(self.telemetry_processes)
            except Exception as error:
                errors.append(f"stop telemetry: {type(error).__name__}: {error}")
            self.telemetry_processes = []
            for name in process_order:
                process = self.processes.pop(name, None)
                if process is None:
                    continue
                try:
                    process.stop()
                except Exception as error:
                    errors.append(f"stop {name}: {type(error).__name__}: {error}")
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except Exception as kill_error:
                        errors.append(
                            f"kill {name}: {type(kill_error).__name__}: {kill_error}"
                        )
                if run_dir is not None and process.log_path.is_file():
                    try:
                        shutil.copy2(process.log_path, run_dir / f"{name}.log")
                    except Exception as error:
                        errors.append(f"copy {name}.log: {type(error).__name__}: {error}")
            if run_dir is not None and scratch_dir is not None:
                source = scratch_dir / "load_artifacts"
                destination = run_dir / "load_artifacts.partial"
                if source.is_dir() and not (run_dir / "load_artifacts").exists():
                    try:
                        shutil.copytree(source, destination, dirs_exist_ok=True)
                    except Exception as error:
                        errors.append(
                            f"copy partial load artifacts: {type(error).__name__}: {error}"
                        )
                samples = scratch_dir / "remote_samples.jsonl"
                if samples.is_file() and not (run_dir / "remote_samples.jsonl").exists():
                    try:
                        shutil.copy2(samples, run_dir / "remote_samples.partial.jsonl")
                    except Exception as error:
                        errors.append(
                            f"copy partial remote samples: {type(error).__name__}: {error}"
                        )
            if run_dir is not None:
                try:
                    (run_dir / "REMOTE_STOPPED").write_text(
                        now() + "\n", encoding="utf-8"
                    )
                except Exception as error:
                    errors.append(f"write REMOTE_STOPPED: {type(error).__name__}: {error}")
        finally:
            if scratch_dir is not None:
                try:
                    shutil.rmtree(scratch_dir)
                except FileNotFoundError:
                    pass
                except Exception as error:
                    errors.append(f"remove remote scratch: {type(error).__name__}: {error}")
            self.current_scratch_dir = None
            self.current_run_dir = None
        if errors:
            raise HarnessError("remote cleanup failures: " + "; ".join(errors))


class Coordinator:
    def __init__(
        self,
        config: Mapping[str, Any],
        channel: CommandChannel,
        shutdown: threading.Event,
        *,
        job_id: str,
        result_dir: Path,
        advertised_ip: str,
        remote_ip: str,
        interface: str,
        resume_from_job: str | None,
        simulate: bool,
    ) -> None:
        self.config = config
        self.channel = channel
        self.shutdown = shutdown
        self.job_id = job_id
        self.result_dir = result_dir
        self.advertised_ip = advertised_ip
        self.remote_ip = remote_ip
        self.interface = interface
        self.resume_from_job = resume_from_job
        self.simulate = simulate
        hashable_config = {key: value for key, value in config.items() if not key.startswith("_")}
        self.config_hash = canonical_hash(hashable_config)
        self.harness_hash = harness_content_hash()
        self.frontend: ManagedProcess | None = None
        self.frontend_secondary: ManagedProcess | None = None
        self.frontend_proxy: ManagedProcess | None = None
        self.frontend_scratch_dir: Path | None = None
        self.frontend_run_dir: Path | None = None
        self.frontend_telemetry_processes: list[ManagedProcess] = []
        self.statuses: list[dict[str, Any]] = []

    def run(self, profile_request: Mapping[str, Any] | None = None) -> None:
        try:
            if profile_request is None:
                self._run_matrix()
            else:
                self._run_profile_comparison(profile_request)
        finally:
            primary_active = sys.exc_info()[0] is not None
            cleanup_errors: list[str] = []
            try:
                self._stop_frontend()
            except Exception as error:
                cleanup_errors.append(
                    f"frontend cleanup: {type(error).__name__}: {error}"
                )
            try:
                self.channel.send("terminate", {}, timeout=30)
            except Exception as error:
                if not self.shutdown.is_set():
                    cleanup_errors.append(
                        f"remote terminate: {type(error).__name__}: {error}"
                    )
            if cleanup_errors:
                try:
                    atomic_json(
                        self.result_dir / "cleanup_errors.json",
                        {"recorded_at": now(), "errors": cleanup_errors},
                    )
                except Exception:
                    pass
                if not primary_active:
                    raise HarnessError("; ".join(cleanup_errors))

    def _run_matrix(self) -> None:
        sequence = build_run_sequence(self.config)
        atomic_json(
            self.result_dir / "matrix_manifest.json",
            {
                "schema_version": 1,
                "job_id": self.job_id,
                "created_at": now(),
                "config_hash": self.config_hash,
                "harness_hash": self.harness_hash,
                "config": {
                    key: value
                    for key, value in self.config.items()
                    if not key.startswith("_")
                },
                "sequence": [spec.as_dict() for spec in sequence],
                "frontend_ip": self.advertised_ip,
                "remote_ip": self.remote_ip,
                "interface": self.interface,
                "resume_from_job": self.resume_from_job,
                "simulated": self.simulate,
            },
        )
        resumed: set[str] = set()
        if self.resume_from_job:
            resumed = completed_run_keys(
                self.config["root"],
                self.resume_from_job,
                config_hash=self.config_hash,
                harness_hash=self.harness_hash,
            )

        by_workload: dict[str, list[RunSpec]] = {}
        for spec in sequence:
            by_workload.setdefault(spec.workload, []).append(spec)

        for workload, specs in by_workload.items():
            split_needs_diagnostic = False
            for spec in specs:
                if not spec.calibration and spec.run_id in resumed:
                    self._record_resumed(spec)
                    continue
                acceptance = self._run_one(spec)
                if (
                    not self.simulate
                    and not spec.calibration
                    and spec.arm == "4+4"
                    and not acceptance.get("memory", {}).get("accepted", False)
                ):
                    split_needs_diagnostic = True

            performance = self._primary_performance_summary(workload)
            performance["split_memory_violation_observed"] = (
                split_needs_diagnostic
            )
            performance["in_allocation_diagnostics_enabled"] = False
            performance["followup_recommended"] = False
            performance["suggested_followup"] = []
            atomic_json(
                self.result_dir / f"{workload}_diagnostic_recommendation.json",
                performance,
            )

        matrix_accepted = all(
            item["status"]
            in {"accepted", "calibration", "resumed", "simulated"}
            or item.get("diagnostic")
            for item in self.statuses
        )
        summary = {
            "completed_at": now(),
            "accepted": matrix_accepted,
            "runs": self.statuses,
        }
        atomic_json(self.result_dir / "matrix_status.json", summary)
        marker = (
            "SIMULATED"
            if self.simulate
            else ("COMPLETE" if matrix_accepted else "COMPLETE_WITH_INVALID_RUNS")
        )
        (self.result_dir / marker).write_text(now() + "\n", encoding="utf-8")

    def _run_profile_comparison(
        self, profile_request: Mapping[str, Any]
    ) -> None:
        workload = str(profile_request["workload"])
        arms = [str(arm) for arm in profile_request["arms"]]
        reference_job = profile_request.get("reference_job")
        specs = [
            RunSpec(workload=workload, arm=arm, ordinal=index)
            for index, arm in enumerate(arms, start=1)
        ]
        duration = int(self.config["workloads"][workload]["duration_seconds"])
        atomic_json(
            self.result_dir / "profile_manifest.json",
            {
                "schema_version": 1,
                "mode": "profile-only",
                "job_id": self.job_id,
                "created_at": now(),
                "config_hash": self.config_hash,
                "harness_hash": self.harness_hash,
                "config": {
                    key: value
                    for key, value in self.config.items()
                    if not key.startswith("_")
                },
                "workload": workload,
                "arms": arms,
                "sequence": [
                    spec.as_dict()
                    | {
                        "profile_capture": True,
                        "relative_dir": (
                            f"profiles/{workload}/{spec.arm_slug}"
                        ),
                    }
                    for spec in specs
                ],
                "load_duration_seconds": duration,
                "warmup_request_count": int(
                    self.config["workloads"][workload]["warmup_request_count"]
                ),
                "perf_frequency_hz": 99,
                "perf_record_seconds": int(
                    self.config["matrix"].get(
                        "profile_record_seconds", 40
                    )
                ),
                "perf_call_graph": "dwarf",
                "reference_job": reference_job,
                "frontend_ip": self.advertised_ip,
                "remote_ip": self.remote_ip,
                "interface": self.interface,
                "simulated": self.simulate,
            },
        )
        for spec in specs:
            self._run_one(spec, profile_capture=True)

        if not self.simulate:
            analysis_log = self.result_dir / "analysis.log"
            with analysis_log.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("analyze_profile.py")),
                        "--job-dir",
                        str(self.result_dir),
                    ],
                    check=True,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                )

        accepted = all(
            item["status"] in {"accepted", "simulated"}
            for item in self.statuses
        )
        summary = {
            "completed_at": now(),
            "accepted": accepted,
            "reference_job": reference_job,
            "runs": self.statuses,
        }
        atomic_json(self.result_dir / "profile_status.json", summary)
        marker = (
            "PROFILE_SIMULATED"
            if self.simulate
            else (
                "PROFILE_COMPLETE"
                if accepted
                else "PROFILE_COMPLETE_WITH_INVALID_RUNS"
            )
        )
        (self.result_dir / marker).write_text(
            now() + "\n", encoding="utf-8"
        )

    def _record_resumed(self, spec: RunSpec) -> None:
        run_dir = self.result_dir / spec.relative_dir
        run_dir.mkdir(parents=True, exist_ok=False)
        source = job_paths(self.config["root"], str(self.resume_from_job))["result_dir"] / spec.relative_dir
        atomic_json(
            run_dir / "RESUMED_FROM.json",
            {
                "old_job_id": self.resume_from_job,
                "source": str(source),
                "run_id": spec.run_id,
                "validated_at": now(),
                "config_hash": self.config_hash,
                "harness_hash": self.harness_hash,
            },
        )
        for name in ("run.json", "run_manifest.json", "acceptance.json", "summary.json"):
            source_file = source / name
            if not source_file.is_file():
                raise HarnessError(f"validated resume source lost required artifact: {source_file}")
            document = read_json(source_file)
            if name == "summary.json":
                document["resumed_from"] = str(source)
                document["resumed_into_job_id"] = self.job_id
            atomic_json(run_dir / name, document)
        (run_dir / "RESUMED").write_text(now() + "\n", encoding="utf-8")
        self.statuses.append(
            {"run_id": spec.run_id, "status": "resumed", "source": str(source)}
        )

    def _run_one(self, spec: RunSpec, *, profile_capture: bool = False) -> dict[str, Any]:
        run_dir = (
            self.result_dir / "profiles" / spec.workload / spec.arm_slug
            if profile_capture
            else self.result_dir / spec.relative_dir
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        workload = self.config["workloads"][spec.workload]
        duration = int(workload["duration_seconds"]) if profile_capture else (
            int(
                self.config["matrix"].get(
                    "calibration_seconds",
                    self.config["matrix"].get("calibration_duration_seconds", 30),
                )
            )
            if spec.calibration
            else int(workload["duration_seconds"])
        )
        run_record = spec.as_dict() | {
            "config_hash": self.config_hash,
            "harness_hash": self.harness_hash,
            "job_id": self.job_id,
            "duration_seconds": duration,
            "started_at": now(),
            "placement": _placement(self.config, spec.arm, spec.interleave),
            "profile_capture": profile_capture,
        }
        node_cpus = self.config["topology"].get(
            "node_cpus", {"0": "0-71", "1": "72-143"}
        )
        run_record["node_cpus"] = node_cpus
        if _workload_kind(workload) == "synthetic":
            run_record["expected_output_tokens"] = int(
                workload.get("osl", workload.get("output_tokens", 1024))
            )
        atomic_json(run_dir / "run.json", run_record)
        atomic_json(run_dir / "run_manifest.json", run_record)

        if self.simulate:
            self.channel.send(
                "prepare",
                {"run_dir": str(run_dir), "workload": spec.workload, "run": spec.as_dict()},
                timeout=30,
            )
            self.channel.send(
                "run_load",
                {
                    "run_dir": str(run_dir),
                    "workload": spec.workload,
                    "duration_seconds": duration,
                },
                timeout=30,
            )
            self.channel.send("stop", {"run_id": spec.run_id}, timeout=30)
            acceptance = {"accepted": True, "simulated": True}
            atomic_json(run_dir / "acceptance.json", acceptance)
            (run_dir / "SIMULATED").write_text(now() + "\n", encoding="utf-8")
            self.statuses.append({"run_id": spec.run_id, "status": "simulated"})
            return acceptance

        frontend_sampler: Sampler | None = None
        remote_stopped = False
        perf_capture_summary: dict[str, Any] | None = None
        try:
            self.channel.send(
                "prepare",
                {"run_dir": str(run_dir), "workload": spec.workload, "run": spec.as_dict()},
                timeout=300,
            )
            self.frontend = self._start_frontend(spec, run_dir)
            sse = self._smoke(run_dir)
            self.frontend_telemetry_processes = _start_system_telemetry(
                run_dir,
                role="frontend",
                tracked={"frontend": os.getpgid(self.frontend.pid)},
                metrics_url=(
                    f"http://{self.advertised_ip}:"
                    f"{8001 if self.frontend_secondary is not None else _port(self.config, 'frontend_http')}/metrics"
                ),
            )
            frontend_sampler = Sampler(
                self.frontend_scratch_dir / "frontend_samples.jsonl",
                pools={"frontend": _placement(self.config, spec.arm, spec.interleave)["cpus"]},
                interface=self.interface,
                process=self.frontend,
                gate_path=run_dir / "PROFILING_STARTED",
                end_gate_path=run_dir / "PROFILING_ENDED",
            )
            frontend_sampler.start()
            load_payload = {
                "run_dir": str(run_dir),
                "workload": spec.workload,
                "duration_seconds": duration,
            }
            if profile_capture:
                outcome: dict[str, Any] = {}

                def run_remote_load() -> None:
                    try:
                        outcome["ack"] = self.channel.send(
                            "run_load",
                            load_payload,
                            timeout=duration + int(workload["grace_period_seconds"]) + 600,
                        )
                    except BaseException as error:
                        outcome["error"] = error

                load_thread = threading.Thread(target=run_remote_load, daemon=True)
                load_thread.start()
                wait_for_path(run_dir / "PROFILING_STARTED", 420, self.shutdown)
                assert self.frontend is not None
                assert self.frontend_scratch_dir is not None
                delay = int(
                    self.config["matrix"].get("profile_start_delay_seconds", 40)
                )
                delay_deadline = time.monotonic() + delay
                while time.monotonic() < delay_deadline:
                    self.frontend.require_alive()
                    if not load_thread.is_alive():
                        raise HarnessError(
                            "AIPerf ended before the delayed perf capture began"
                        )
                    if self.shutdown.wait(min(1.0, delay_deadline - time.monotonic())):
                        raise HarnessError("interrupted before perf capture")
                dso_manifest = _capture_frontend_dso_manifest(
                    self.frontend.pid, run_dir
                )
                perf_data = self.frontend_scratch_dir / "oncpu.data"
                perf_error_path = self.frontend_scratch_dir / "perf.err"
                with perf_error_path.open("w", encoding="utf-8") as perf_error:
                    subprocess.run(
                        [
                            "perf",
                            "record",
                            "-e",
                            "cycles:u",
                            "-F",
                            "99",
                            "-m",
                            str(self.config["matrix"].get("perf_mmap_pages", 32768)),
                            "--call-graph",
                            "dwarf,16384",
                            "-p",
                            str(self.frontend.pid),
                            "-o",
                            str(perf_data),
                            "--",
                            "sleep",
                            str(
                                int(
                                    self.config["matrix"].get(
                                        "profile_record_seconds", 40
                                    )
                                )
                            ),
                        ],
                        check=True,
                        stderr=perf_error,
                        timeout=int(
                            self.config["matrix"].get("profile_record_seconds", 40)
                        )
                        + int(
                            self.config["matrix"].get(
                                "perf_flush_timeout_seconds", 240
                            )
                        ),
                        )
                _capture_mapped_dso_checksums(run_dir)
                load_thread.join(
                    timeout=duration + int(workload["grace_period_seconds"]) + 600
                )
                if load_thread.is_alive():
                    raise HarnessError("remote load thread did not finish after perf capture")
                if "error" in outcome:
                    raise outcome["error"]
                load_ack = outcome["ack"]
                if not perf_data.is_file() or perf_data.stat().st_size == 0:
                    raise HarnessError("perf capture did not produce oncpu.data")
                atomic_json(
                    run_dir / "perf-live-dso.json",
                    {"mapped_core": dso_manifest, "captured_at": now()},
                )
            else:
                load_ack = self.channel.send(
                    "run_load",
                    load_payload,
                    timeout=duration + int(workload["grace_period_seconds"]) + 600,
                )
            frontend_sampler.stop()
            frontend_sampler = None
            self._stop_frontend()
            if profile_capture:
                perf_capture_summary = self._process_perf_capture(run_dir)
            self.channel.send("stop", {"run_id": spec.run_id}, timeout=60)
            remote_stopped = True

            profiling_started_at = (run_dir / "PROFILING_STARTED").read_text().strip()
            profiling_ended_at = (run_dir / "PROFILING_ENDED").read_text().strip()
            samples = _read_samples(run_dir / "frontend_samples.jsonl")
            sampling = _sampling_coverage(
                samples,
                profiling_started_at,
                profiling_ended_at,
                minimum_fraction=float(
                    self.config["acceptance"].get(
                        "telemetry_min_coverage_fraction", 0.98
                    )
                ),
                maximum_fraction=float(
                    self.config["acceptance"].get(
                        "telemetry_max_coverage_fraction", 1.02
                    )
                ),
            )
            placement = _placement(self.config, spec.arm, spec.interleave)
            memory = _memory_summary(
                samples,
                placement,
                self.config["acceptance"],
                profiling_started_at=profiling_started_at,
            )
            execution = _execution_summary(
                samples,
                placement,
                self.config["topology"],
                self.config["acceptance"],
            )
            expected_affinity = sorted(parse_cpu_set(placement["cpus"]))
            affinity_ok = bool(samples) and all(
                sample.get("affinity") == expected_affinity
                for sample in samples
                if sample.get("affinity") is not None
            ) and all(
                not sample.get("task_affinity", {}).get("outside_allowed")
                and int(sample.get("task_affinity", {}).get("task_count", 0)) > 0
                for sample in samples
                if sample.get("task_affinity") is not None
            )
            nic = _nic_summary(samples, self.interface)
            nic_limit = float(
                self.config["acceptance"].get(
                    "nic_max_utilization_percent",
                    100.0 * self.config["acceptance"].get("nic_link_utilization_max", 0.7),
                )
            )
            nic_ok = bool(nic.get("accepted")) and (
                nic.get("peak_direction_utilization_percent") is None
                or float(nic["peak_direction_utilization_percent"]) <= nic_limit
            ) and float(nic.get("tcp_retransmit_fraction", 0.0)) <= float(
                self.config["acceptance"].get("tcp_retransmit_max_fraction", 0.05)
            )
            remote = dict(load_ack.get("result", {}))
            accepted = (
                bool(sse.get("accepted"))
                and memory["accepted"]
                and sampling["accepted"]
                and affinity_ok
                and bool(remote.get("accepted"))
            )
            acceptance = {
                "accepted": accepted,
                "sse": sse,
                "memory": memory,
                "execution": execution,
                "sampling": sampling,
                "affinity": {"accepted": affinity_ok, "expected": expected_affinity},
                "nic": nic,
                "remote": remote,
                "perf_capture": perf_capture_summary,
                "profiling_interval": {
                    "started_at": profiling_started_at,
                    "ended_at": profiling_ended_at,
                },
                "evaluated_at": now(),
            }
            atomic_json(
                run_dir / "placement_validation.json",
                {
                    "valid": affinity_ok,
                    "affinity": {"valid": affinity_ok, "expected": expected_affinity},
                    "memory": memory,
                    "execution": execution,
                    "placement": placement,
                },
            )
            from analyze import analyze_run  # type: ignore

            node_cpu_map = {
                int(node): cpus for node, cpus in run_record["node_cpus"].items()
            }
            summary = analyze_run(
                run_dir,
                arm=spec.arm,
                workload=spec.workload,
                manifest_path=run_dir / "run_manifest.json",
                telemetry_paths=(
                    run_dir / "frontend_samples.jsonl",
                    run_dir / "remote_samples.jsonl",
                ),
                remote_roles=("load_node",),
                node_cpus=node_cpu_map,
                # Load saturation is preserved as contamination metadata rather
                # than invalidating the otherwise usable frontend profile.
                max_remote_busy_percent=101.0,
                max_nic_utilization_percent=nic_limit,
                # The core acceptance above uses the configured retransmit
                # fraction; avoid replacing it with analyze.py's absolute-count
                # compatibility check.
                max_retransmits=sys.maxsize,
                max_retransmit_fraction=float(
                    self.config["acceptance"].get(
                        "tcp_retransmit_max_fraction", 0.05
                    )
                ),
            )
            analysis_accepted = bool(summary["accepted"])
            acceptance["accepted"] = accepted
            acceptance["analysis_accepted"] = analysis_accepted
            summary["analysis_accepted"] = analysis_accepted
            summary["accepted"] = accepted
            summary["harness_acceptance"] = acceptance
            atomic_json(run_dir / "summary.json", summary)
            atomic_json(run_dir / "acceptance.json", acceptance)
            if spec.calibration:
                if not accepted:
                    raise HarnessError(f"calibration {spec.run_id} failed acceptance")
                (run_dir / "CALIBRATION_COMPLETE").write_text(now() + "\n", encoding="utf-8")
                status = "calibration"
            elif accepted:
                (run_dir / "COMPLETE").write_text(now() + "\n", encoding="utf-8")
                status = "accepted"
            else:
                (run_dir / "INVALID").write_text(now() + "\n", encoding="utf-8")
                status = "invalid"
            self.statuses.append(
                {
                    "run_id": spec.run_id,
                    "status": status,
                    "diagnostic": spec.interleave or profile_capture,
                    "profile_capture": profile_capture,
                    "relative_dir": str(run_dir.relative_to(self.result_dir)),
                }
            )
            return acceptance
        except Exception as error:
            atomic_json(
                run_dir / "FAILED.json",
                {"failed_at": now(), "error": f"{type(error).__name__}: {error}"},
            )
            raise
        finally:
            if frontend_sampler is not None:
                frontend_sampler.stop(raise_on_error=False)
            cleanup_errors: list[str] = []
            try:
                self._stop_frontend()
            except Exception as error:
                cleanup_errors.append(f"frontend cleanup: {type(error).__name__}: {error}")
            if not remote_stopped:
                try:
                    self.channel.send("stop", {"run_id": spec.run_id}, timeout=60)
                except Exception as error:
                    cleanup_errors.append(f"remote stop: {type(error).__name__}: {error}")
            if cleanup_errors:
                try:
                    atomic_json(
                        run_dir / "cleanup_errors.json",
                        {"recorded_at": now(), "errors": cleanup_errors},
                    )
                except Exception:
                    pass
                if sys.exc_info()[0] is None:
                    raise HarnessError("; ".join(cleanup_errors))

    def _process_perf_capture(self, run_dir: Path) -> dict[str, Any]:
        data_path = run_dir / "oncpu.data"
        if not data_path.is_file() or data_path.stat().st_size == 0:
            raise HarnessError("perf capture did not produce a nonempty oncpu.data")

        commands = [
            (
                "perf-script",
                [
                    "perf",
                    "script",
                    "--no-inline",
                    "--show-lost-events",
                    "-i",
                    str(data_path),
                ],
                run_dir / "perf-script.txt",
                run_dir / "perf-script.err",
            ),
            (
                "perf-buildids",
                ["perf", "buildid-list", "-i", str(data_path)],
                run_dir / "perf-buildids.txt",
                run_dir / "perf-buildids.err",
            ),
            (
                "perf-header",
                [
                    "perf",
                    "report",
                    "--stdio",
                    "--header-only",
                    "-i",
                    str(data_path),
                ],
                run_dir / "perf-header.txt",
                run_dir / "perf-header.err",
            ),
        ]
        results: list[dict[str, Any]] = []
        for name, command, output_path, error_path in commands:
            with output_path.open("w", encoding="utf-8") as output:
                with error_path.open("w", encoding="utf-8") as error:
                    completed = subprocess.run(
                        command,
                        stdout=output,
                        stderr=error,
                        text=True,
                        timeout=int(
                            self.config["matrix"].get(
                                "perf_processing_timeout_seconds", 1800
                            )
                        ),
                    )
            result = {
                "name": name,
                "argv": command,
                "returncode": completed.returncode,
                "stdout": str(output_path),
                "stdout_bytes": output_path.stat().st_size,
                "stderr": str(error_path),
                "stderr_bytes": error_path.stat().st_size,
            }
            results.append(result)
            if completed.returncode != 0:
                atomic_json(
                    run_dir / "perf-processing-commands.json",
                    {"commands": results},
                )
                raise HarnessError(
                    f"{name} failed with status {completed.returncode}; "
                    f"see {error_path}"
                )

        script_path = run_dir / "perf-script.txt"
        frame_pattern = re.compile(
            r"^\s*[0-9a-fA-F]+\s+.+\s+\([^)]*\)\s*$",
        )
        frame_count = 0
        unresolved_frame_count = 0
        script_nonempty = False
        warning_lines: list[str] = []
        lost_event_lines: list[str] = []
        for diagnostic_path in (
            script_path,
            run_dir / "perf-script.err",
            run_dir / "perf.err",
            run_dir / "perf-header.err",
        ):
            with diagnostic_path.open(errors="replace") as diagnostic:
                for line in diagnostic:
                    if diagnostic_path == script_path:
                        script_nonempty = script_nonempty or bool(line.strip())
                        frame_count += bool(frame_pattern.match(line))
                        unresolved_frame_count += line.count("[unknown]")
                    if re.search(
                        r"\b(?:lost|unwind|truncat|failed)\b",
                        line,
                        flags=re.IGNORECASE,
                    ):
                        warning_lines.append(line.rstrip())
                    if re.search(r"\blost\b", line, flags=re.IGNORECASE):
                        lost_event_lines.append(line.rstrip())
        lost_diagnostics = "\n".join(lost_event_lines)
        lost_sample_values = [
            int(match.group(1))
            for line in lost_event_lines
            if (
                match := re.search(
                    r"PERF_RECORD_LOST.*\blost\s+(\d+)\b",
                    line,
                    flags=re.IGNORECASE,
                )
            )
        ]
        if lost_sample_values:
            lost_sample_count = sum(lost_sample_values)
        else:
            fallback_patterns = (
                r"\blost\s*[:=]\s*(\d+)\b",
                r"\blost\s+(\d+)\s+(?:samples?|events?)\b",
                r"\b(\d+)\s+(?:samples?|events?)\s+lost\b",
            )
            lost_sample_count = sum(
                max(
                    (
                        int(value)
                        for pattern in fallback_patterns
                        for value in re.findall(
                            pattern, line, flags=re.IGNORECASE
                        )
                    ),
                    default=0,
                )
                for line in lost_event_lines
            )
        lost_percentages = [
            float(value)
            for value in re.findall(
                r"\blost\s+(\d+(?:\.\d+)?)%",
                lost_diagnostics,
                flags=re.IGNORECASE,
            )
        ]
        lost_chunk_counts = [
            int(value)
            for value in re.findall(
                r"\blost\s+(\d+)\s+chunks?\b",
                lost_diagnostics,
                flags=re.IGNORECASE,
            )
        ]
        dso_manifest = read_json(run_dir / "frontend-dso-manifest.json")
        buildid_text = (run_dir / "perf-buildids.txt").read_text(errors="replace")
        expected_build_id = str(dso_manifest["core_build_id"]).lower()
        matching_buildid_lines = [
            line
            for line in buildid_text.splitlines()
            if "_core.abi3.so" in line and expected_build_id in line.lower()
        ]
        summary = {
            "accepted": (
                script_nonempty
                and frame_count > 0
                and lost_sample_count == 0
                and max(lost_chunk_counts, default=0) == 0
                and len(matching_buildid_lines) == 1
            ),
            "processed_at": now(),
            "processor_architecture": os.uname().machine,
            "data_path": str(data_path),
            "data_bytes": data_path.stat().st_size,
            "perf_script_path": str(script_path),
            "perf_script_bytes": script_path.stat().st_size,
            "frame_count": frame_count,
            "unresolved_frame_count": unresolved_frame_count,
            "unresolved_frame_fraction": (
                unresolved_frame_count / frame_count if frame_count else None
            ),
            "lost_sample_count": lost_sample_count,
            "lost_sample_percent_reported": (
                max(lost_percentages) if lost_percentages else None
            ),
            "lost_chunk_count_reported": (
                max(lost_chunk_counts) if lost_chunk_counts else 0
            ),
            "mapped_core_path": dso_manifest["core_path_from_proc_maps"],
            "mapped_core_sha256": dso_manifest["core_sha256"],
            "mapped_core_build_id": expected_build_id,
            "matching_perf_buildid_lines": matching_buildid_lines,
            "lost_event_lines": lost_event_lines,
            "truncated_callchain_warning_count": sum(
                bool(re.search(r"truncat", line, flags=re.IGNORECASE))
                for line in warning_lines
            ),
            "unwind_warning_count": sum(
                bool(re.search(r"unwind", line, flags=re.IGNORECASE))
                for line in warning_lines
            ),
            "warning_lines": warning_lines,
            "perf_record_stderr_path": str(run_dir / "perf.err"),
            "commands": results,
        }
        atomic_json(run_dir / "perf-capture.json", summary)
        atomic_json(
            run_dir / "perf-processing-commands.json",
            {"commands": results},
        )
        if not summary["accepted"]:
            raise HarnessError(
                "ARM perf script did not produce symbolized callchain frames"
            )
        return summary

    def _start_frontend(self, spec: RunSpec, run_dir: Path) -> ManagedProcess:
        workload = self.config["workloads"][spec.workload]
        placement = _placement(self.config, spec.arm, spec.interleave)
        env = _base_runtime_env(
            self.config, self.advertised_ip, self.remote_ip, self.interface
        )
        if bool(_runtime(self.config, "frontend_jemalloc_enabled", True)):
            env["LD_PRELOAD"] = _path(self.config, "frontend_jemalloc")
        else:
            env.pop("LD_PRELOAD", None)
        frontend_env = _runtime(self.config, "frontend_env", {})
        if not isinstance(frontend_env, Mapping):
            raise HarnessError("runtime.frontend_env must be a mapping")
        env.update({str(key): str(value) for key, value in frontend_env.items()})
        environment_keys = (
            "ETCD_ENDPOINTS",
            "DYN_TCP_RPC_HOST",
            "DYN_REQUEST_PLANE",
            "DYN_REQUEST_PLANE_CODEC",
            "DYN_EVENT_PLANE",
            "DYN_EVENT_PLANE_CODEC",
            "DYN_ZMQ_BROKER_ENABLED",
            "DYN_TOKENIZER",
            "DYN_TOKENIZER_CACHE",
            "DYN_TOKENIZER_CACHE_BYTES",
            "HF_HOME",
            "LD_PRELOAD",
            *sorted(frontend_env),
        )
        atomic_json(
            run_dir / "frontend_environment.json",
            {
                key: env[key]
                for key in environment_keys
            },
        )
        frontend_argv = [
            _path(self.config, "dynamo_python"),
            "-m",
            "dynamo.frontend",
            "--router-mode",
            "kv",
            "--kv-cache-block-size",
            str(workload["block_size"]),
            "--http-host",
            self.advertised_ip,
            "--http-port",
            str(_port(self.config, "frontend_http")),
            "--admission-control",
            "none",
            "--active-decode-blocks-threshold",
            "None",
            "--active-prefill-tokens-threshold",
            "None",
            "--active-prefill-tokens-threshold-frac",
            "None",
        ]
        scratch_root = Path(
            os.environ.get(
                "SLURM_TMPDIR",
                f"/var/tmp/dynamo-numa-{os.environ.get('SLURM_JOB_ID', 'local')}",
            )
        )
        self.frontend_scratch_dir = (
            scratch_root / "dynamo-numa" / self.job_id / "rank0" / spec.run_id
        )
        self.frontend_scratch_dir.mkdir(parents=True, exist_ok=False)
        self.frontend_run_dir = run_dir
        atomic_json(
            run_dir / "frontend_scratch.json",
            {"path": str(self.frontend_scratch_dir), "created_at": now()},
        )
        if bool(_runtime(self.config, "frontend_dual_numa", False)):
            # Build each command from the unprefixed frontend argv.  Reusing
            # the arm-prefixed command here would nest numactl: the NUMA-1
            # process would first restrict itself to CPUs 72-143 and then
            # ask the nested command for CPUs 0-71, which numactl correctly
            # rejects as outside its allowed set.
            primary_placement = self.config["topology"]["frontend"]
            secondary_placement = self.config["topology"]["frontend_control"]
            primary_command = process_prefix(
                primary_placement["cpus"], primary_placement["memory"]
            ) + frontend_argv
            primary_command[primary_command.index("--http-port") + 1] = "8001"
            secondary_command = process_prefix(
                secondary_placement["cpus"], secondary_placement["memory"]
            ) + frontend_argv
            secondary_command[secondary_command.index("--http-port") + 1] = "8002"
            atomic_json(
                run_dir / "frontend_commands.json",
                {
                    "numa0": primary_command,
                    "numa1": secondary_command,
                },
            )
            process = ManagedProcess(
                "frontend-numa0",
                primary_command,
                self.frontend_scratch_dir / "frontend-numa0.log",
                env=env,
            ).start()
            secondary = ManagedProcess(
                "frontend-numa1",
                secondary_command,
                self.frontend_scratch_dir / "frontend-numa1.log",
                env=env,
            ).start()
            self.frontend_secondary = secondary
            _wait_http(
                f"http://{self.advertised_ip}:8001/v1/models",
                _model_name(self.config),
                process,
            )
            _wait_http(
                f"http://{self.advertised_ip}:8002/v1/models",
                _model_name(self.config),
                secondary,
            )
        else:
            command = process_prefix(placement["cpus"], placement["memory"]) + frontend_argv
            process = ManagedProcess(
                "frontend", command, self.frontend_scratch_dir / "frontend.log", env=env
            ).start()
        self.frontend = process
        atomic_json(
            run_dir / "frontend_pid.json",
            {
                "pid": process.pid,
                "pgid": os.getpgid(process.pid),
                "argv": process.command,
                "secondary": (
                    {
                        "pid": self.frontend_secondary.pid,
                        "pgid": os.getpgid(self.frontend_secondary.pid),
                        "argv": self.frontend_secondary.command,
                    }
                    if self.frontend_secondary is not None
                    else None
                ),
            },
        )
        if self.frontend_secondary is None:
            _wait_http(
                f"http://{self.advertised_ip}:{_port(self.config, 'frontend_http')}/v1/models",
                _model_name(self.config),
                process,
            )
        actual = _process_affinity(process.pid)
        expected = sorted(
            parse_cpu_set(
                self.config["topology"]["frontend"]["cpus"]
                if self.frontend_secondary is not None
                else placement["cpus"]
            )
        )
        if actual != expected:
            raise HarnessError(f"frontend affinity is {actual}, expected {expected}")
        if self.frontend_secondary is not None:
            secondary_actual = _process_affinity(self.frontend_secondary.pid)
            secondary_expected = sorted(
                parse_cpu_set(self.config["topology"]["frontend_control"]["cpus"])
            )
            if secondary_actual != secondary_expected:
                raise HarnessError(
                    f"secondary frontend affinity is {secondary_actual}, expected {secondary_expected}"
                )
        return process

    def _smoke(self, run_dir: Path) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": _model_name(self.config),
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Say hello."},
                ],
                "stream": True,
                "max_tokens": 8,
                "ignore_eos": True,
            }
        ).encode()
        frontend_port = 8001 if self.frontend_secondary is not None else _port(
            self.config, "frontend_http"
        )
        request = urllib.request.Request(
            f"http://{self.advertised_ip}:{frontend_port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode()
        (run_dir / "sse_witness.txt").write_text(raw, encoding="utf-8")
        try:
            from validate_sse import validate as validate_sse  # type: ignore

            summary = dict(validate_sse(raw.encode(), 8))
            summary["accepted"] = bool(summary["valid"])
        except ImportError:
            # Keep the agent executable in isolation, while the installed
            # harness always takes the stricter independent-event path above.
            content_events = done_events = 0
            for line in raw.splitlines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done_events += 1
                    continue
                event = json.loads(payload)
                choices = event.get("choices", [])
                if choices and "content" in choices[0].get("delta", {}):
                    content_events += 1
            summary = {
                "accepted": content_events == 8 and done_events == 1,
                "content_events": content_events,
                "done_events": done_events,
                "expected_output_tokens": 8,
            }
        summary["streaming_interval_changed"] = False
        atomic_json(run_dir / "sse_witness.json", summary)
        if not summary["accepted"]:
            raise HarnessError(f"SSE smoke cardinality failed: {summary}")
        return summary

    def _stop_frontend(self) -> None:
        errors: list[str] = []
        try:
            _stop_processes(self.frontend_telemetry_processes)
        except Exception as error:
            errors.append(f"stop telemetry: {type(error).__name__}: {error}")
        self.frontend_telemetry_processes = []
        frontend = self.frontend
        secondary = self.frontend_secondary
        proxy = self.frontend_proxy
        scratch_dir = self.frontend_scratch_dir
        run_dir = self.frontend_run_dir
        self.frontend = None
        self.frontend_secondary = None
        self.frontend_proxy = None
        self.frontend_scratch_dir = None
        self.frontend_run_dir = None
        if frontend is not None:
            try:
                frontend.stop()
            except Exception as error:
                errors.append(f"stop frontend: {type(error).__name__}: {error}")
                try:
                    os.killpg(frontend.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception as kill_error:
                    errors.append(
                        f"kill frontend: {type(kill_error).__name__}: {kill_error}"
                    )
        for name, process in (("frontend proxy", proxy), ("secondary frontend", secondary)):
            if process is None:
                continue
            try:
                process.stop()
            except Exception as error:
                errors.append(f"stop {name}: {type(error).__name__}: {error}")
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception as kill_error:
                    errors.append(f"kill {name}: {type(kill_error).__name__}: {kill_error}")
        if scratch_dir is not None:
            try:
                for name in (
                    "frontend.log",
                    "frontend-numa0.log",
                    "frontend-numa1.log",
                    "frontend_samples.jsonl",
                    "oncpu.data",
                    "perf.err",
                ):
                    source = scratch_dir / name
                    if run_dir is not None and source.is_file():
                        try:
                            shutil.copy2(source, run_dir / name)
                        except Exception as error:
                            errors.append(f"copy {name}: {type(error).__name__}: {error}")
            finally:
                try:
                    shutil.rmtree(scratch_dir)
                except FileNotFoundError:
                    pass
                except Exception as error:
                    errors.append(f"remove frontend scratch: {type(error).__name__}: {error}")
        if errors:
            raise HarnessError("frontend cleanup failures: " + "; ".join(errors))

    def _primary_performance_summary(self, workload: str) -> dict[str, Any]:
        summaries: dict[str, list[Mapping[str, Any]]] = {
            arm: [] for arm in self.config["matrix"]["sequence"]
        }
        for run_dir in sorted((self.result_dir / "runs" / workload).glob("*")):
            manifest_path = run_dir / "run.json"
            summary_path = run_dir / "summary.json"
            if (
                not summary_path.exists()
                and (run_dir / "RESUMED_FROM.json").exists()
            ):
                source = Path(
                    read_json(run_dir / "RESUMED_FROM.json")["source"]
                )
                manifest_path = source / "run.json"
                summary_path = source / "summary.json"
            if not manifest_path.exists() or not summary_path.exists():
                continue
            manifest = read_json(manifest_path)
            arm = manifest.get("arm")
            if arm in summaries:
                summaries[arm].append(read_json(summary_path))

        metrics = ("request_throughput_rps", "frontend_cpu_ms_per_request")
        metric_means: dict[str, dict[str, float]] = {}
        changes: dict[str, float] = {}
        for metric in metrics:
            means: dict[str, float] = {}
            for arm, values in summaries.items():
                observed = [
                    float(value["metrics"][metric])
                    for value in values
                    if isinstance(
                        value.get("metrics", {}).get(metric), (int, float)
                    )
                ]
                if observed:
                    means[arm] = sum(observed) / len(observed)
            metric_means[metric] = means
            if set(means) == {"8+0", "4+4", "0+8"}:
                endpoint = (means["8+0"] + means["0+8"]) / 2.0
                if endpoint:
                    changes[metric] = (
                        means["4+4"] - endpoint
                    ) / endpoint

        threshold = float(
            self.config["acceptance"].get(
                "profile_similarity_fraction", 0.05
            )
        )
        similarity = _performance_similarity(
            metric_means["request_throughput_rps"], threshold
        )
        return {
            "workload": workload,
            **similarity,
            "metric_arm_means": metric_means,
            "changes_from_endpoint_mean": changes,
            "evaluated_at": now(),
        }



def _host_preflight(config: Mapping[str, Any], rank: int, interface: str) -> dict[str, Any]:
    topology = config["topology"]
    expected_cpus = int(topology.get("logical_cpus", topology.get("expected_logical_cpus", 144)))
    online = parse_cpu_set(Path("/sys/devices/system/cpu/online").read_text().strip())
    if online != set(range(expected_cpus)):
        raise HarnessError(
            f"online CPU set does not match 0-{expected_cpus - 1}: {sorted(online)}"
        )
    cpu_nodes: dict[int, set[int]] = {}
    for path in Path("/sys/devices/system/node").glob("node[0-9]*"):
        cpulist = (path / "cpulist").read_text().strip()
        if cpulist:
            cpu_nodes[int(path.name[4:])] = parse_cpu_set(cpulist)
    configured_nodes = {
        int(node): parse_cpu_set(cpus)
        for node, cpus in topology.get(
            "node_cpus", {"0": "0-71", "1": "72-143"}
        ).items()
    }
    if cpu_nodes != configured_nodes:
        raise HarnessError(
            "CPU-bearing NUMA topology differs from the configured node map: "
            f"observed={cpu_nodes}, expected={configured_nodes}"
        )

    threads_per_core = int(topology.get("threads_per_core", 1))
    for cpu in range(expected_cpus):
        siblings = parse_cpu_set(
            Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
            .read_text()
            .strip()
        )
        if threads_per_core == 1 and siblings != {cpu}:
            raise HarnessError(
                f"Tyche requires no SMT, but CPU {cpu} has siblings {sorted(siblings)}"
            )

    required = ["git", "ip", "mpstat", "pidstat"]
    if rank == 0:
        required.extend([_path(config, "dynamo_python"), "perf"])
    else:
        required.extend(
            [
                _path(config, "dynamo_python"),
                _path(config, "aiperf"),
                "etcd",
                "etcdctl",
            ]
        )
    missing = [item for item in required if shutil.which(item) is None]
    if missing:
        raise HarnessError(f"missing executables: {', '.join(missing)}")
    version_commands = {
        "git": ["git", "--version"],
        "ip": ["ip", "-V"],
    }
    if rank == 0:
        version_commands["perf"] = ["perf", "--version"]
    else:
        version_commands.update(
            {
                "etcd": ["etcd", "--version"],
                "etcdctl": ["etcdctl", "version"],
            }
        )
    tool_versions: dict[str, str] = {}
    for name, command in version_commands.items():
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=os.environ | {"ETCD_UNSUPPORTED_ARCH": "arm64"},
            timeout=15,
        )
        if completed.returncode != 0:
            raise HarnessError(
                f"could not record {name} version: "
                f"{completed.stdout.strip()} {completed.stderr.strip()}"
            )
        tool_versions[name] = (completed.stdout + completed.stderr).strip()
    for name in ("model", "hf_home", "frontend_jemalloc"):
        if not Path(_path(config, name)).exists():
            raise HarnessError(f"configured {name} path does not exist: {_path(config, name)}")

    repo = Path(_path(config, "dynamo_repo"))
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != _pin(config, "dynamo"):
        raise HarnessError(f"Dynamo HEAD is {head}, expected {_pin(config, 'dynamo')}")
    if subprocess.run(["git", "-C", str(repo), "diff", "--quiet"]).returncode != 0:
        raise HarnessError("Dynamo checkout has tracked modifications")
    marker = repo / ".dynamo-native-rustflags"
    expected_flags = str(config.get("pins", {}).get("rustflags", EXPECTED_RUSTFLAGS))
    if not marker.is_file() or marker.read_text().strip() != expected_flags:
        raise HarnessError(f"native build marker is missing or does not match {expected_flags!r}")

    subprocess.run(
        [
            _path(config, "dynamo_python"),
            "-c",
            "import dynamo._core, dynamo.frontend, dynamo.mocker",
        ],
        check=True,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        timeout=60,
    )
    verifier = Path(__file__).with_name("verify_environment.py")
    verification: dict[str, Any] | None = None
    if verifier.is_file():
        verify_command = [sys.executable, str(verifier), "--rank", str(rank)]
        config_path = str(config.get("_config_path", ""))
        if config_path and config_path != "<defaults>":
            verify_command.extend(["--config", config_path])
        completed = subprocess.run(
            verify_command,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if completed.returncode != 0:
            raise HarnessError(
                "staged environment verification failed: "
                f"{completed.stdout.strip()} {completed.stderr.strip()}"
            )
        verification = json.loads(completed.stdout)
    governors = {
        path.read_text().strip()
        for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")
    }
    if governors and governors != {"performance"}:
        raise HarnessError(f"CPU governors are not uniformly performance: {sorted(governors)}")
    boost_path = Path("/sys/devices/system/cpu/cpufreq/boost")
    boost = boost_path.read_text().strip() if boost_path.exists() else "unavailable"
    if boost not in {"0", "unavailable"}:
        raise HarnessError(f"CPU boost must be disabled, found {boost}")
    if not (Path("/sys/class/net") / interface).exists():
        raise HarnessError(f"network interface does not exist: {interface}")
    scratch_root = Path(
        os.environ.get("SLURM_TMPDIR", f"/var/tmp/dynamo-numa-{os.environ.get('SLURM_JOB_ID', 'local')}")
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch_free = shutil.disk_usage(scratch_root).free
    scratch_minimum = int(_runtime(config, "scratch_min_free_bytes", 20 * 1024**3))
    if scratch_free < scratch_minimum:
        raise HarnessError(
            f"node-local scratch has {scratch_free} free bytes, requires {scratch_minimum}"
        )

    return {
        "ok": True,
        "checked_at": now(),
        "hostname": socket.gethostname(),
        "rank": rank,
        "online_cpus": sorted(online),
        "numa_nodes": sorted(cpu_nodes),
        "numa_node_cpus": {
            str(node): sorted(cpus) for node, cpus in sorted(cpu_nodes.items())
        },
        "dynamo_head": head,
        "rustflags": expected_flags,
        "governors": sorted(governors),
        "boost": boost,
        "interface": interface,
        "affinity_policy": "inherited from one unbound 144-CPU Slurm task",
        "memory_policy": "inherited default",
        "system_tool_versions": tool_versions,
        "environment_verification": verification,
        "scratch_root": str(scratch_root),
        "scratch_free_bytes": scratch_free,
        "scratch_minimum_bytes": scratch_minimum,
    }


def _prepare_job_directories(root: Path, job_id: str, rank: int, shutdown: threading.Event) -> tuple[Path, Path]:
    paths = job_paths(root, job_id)
    state_dir = paths["state_dir"]
    result_dir = paths["result_dir"]
    if rank == 0:
        state_dir.parent.mkdir(parents=True, exist_ok=True)
        result_dir.parent.mkdir(parents=True, exist_ok=True)
        # Reusing coordination state can target stale PIDs or acknowledge the
        # wrong command, so even an empty pre-existing directory is rejected.
        state_dir.mkdir(exist_ok=False)
        try:
            result_dir.mkdir(exist_ok=False)
        except Exception:
            state_dir.rmdir()
            raise
        for child in ("agents", "preflight", "commands", "acks"):
            (state_dir / child).mkdir()
    else:
        wait_for_path(state_dir / "commands", 180, shutdown)
        wait_for_path(result_dir, 180, shutdown)
    return state_dir, result_dir


def _resolve_profile_request(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> dict[str, Any] | None:
    profile_values = (
        args.profile_workload,
        args.profile_arms,
    )
    if not args.profile_only:
        if any(value is not None for value in profile_values):
            raise HarnessError(
                "--profile-workload and --profile-arms require --profile-only"
            )
        return None
    if args.resume_from_job:
        raise HarnessError(
            "--profile-only cannot be combined with --resume-from-job"
        )
    if any(value is None for value in profile_values):
        raise HarnessError(
            "--profile-only requires --profile-workload and --profile-arms"
        )

    workload = str(args.profile_workload)
    arms = [
        arm.strip()
        for arm in str(args.profile_arms).split(",")
        if arm.strip()
    ]
    if workload != "agentx":
        raise HarnessError(
            "this matched profile mode requires --profile-workload agentx"
        )
    expected_arms = [str(arm) for arm in config["matrix"]["sequence"]]
    if arms != expected_arms:
        raise HarnessError(
            f"this Tyche profile requires --profile-arms {','.join(expected_arms)}"
        )
    return {
        "workload": workload,
        "arms": arms,
        "reference_job": args.profile_reference_job,
    }


def _write_dry_run(
    config: Mapping[str, Any],
    job_id: str | None,
    profile_request: Mapping[str, Any] | None = None,
) -> int:
    if profile_request is None:
        sequence = [spec.as_dict() for spec in build_run_sequence(config)]
        mode = "dry-run"
    else:
        workload = str(profile_request["workload"])
        duration = int(config["workloads"][workload]["duration_seconds"])
        warmup = int(config["workloads"][workload]["warmup_request_count"])
        sequence = []
        for index, arm in enumerate(profile_request["arms"], start=1):
            spec = RunSpec(
                workload=workload,
                arm=str(arm),
                ordinal=index,
            )
            sequence.append(
                spec.as_dict()
                | {
                    "profile_capture": True,
                    "relative_dir": (
                        f"profiles/{workload}/{spec.arm_slug}"
                    ),
                    "duration_seconds": duration,
                    "warmup_request_count": warmup,
                    "perf_record_seconds": int(
                        config["matrix"].get(
                            "profile_record_seconds", 40
                        )
                    ),
                }
            )
        mode = "profile-only-dry-run"

    rendered_paths = None
    if job_id:
        rendered_paths = {
            key: str(path)
            for key, path in job_paths(config["root"], job_id).items()
        }
    value = {
        "schema_version": 1,
        "mode": mode,
        "job_id": job_id,
        "paths": rendered_paths,
        "config_hash": canonical_hash(
            {
                key: item
                for key, item in config.items()
                if not key.startswith("_")
            }
        ),
        "harness_hash": harness_content_hash(),
        "config": {
            key: item
            for key, item in config.items()
            if not key.startswith("_")
        },
        "profile_request": (
            dict(profile_request) if profile_request is not None else None
        ),
        "sequence": sequence,
        "service_ownership": {
            "rank0": ["coordinator", "frontend", "frontend-monitor"],
            "rank1": [
                "etcd",
                "mocker",
                "aiperf",
                "remote-monitor",
            ],
        },
        "builds_or_downloads_during_job": False,
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0

def _spawn_local_simulation(args: argparse.Namespace) -> int:
    job_id = args.job_id or f"sim-{os.getpid()}"
    base = [sys.executable, str(Path(__file__).resolve()), "--simulate", "--job-id", job_id]
    if args.config:
        base.extend(["--config", str(args.config)])
    if args.profile_only:
        base.extend(
            [
                "--profile-only",
                "--profile-workload",
                str(args.profile_workload),
                "--profile-arms",
                str(args.profile_arms),
            ]
        )
        if args.profile_reference_job:
            base.extend(["--profile-reference-job", str(args.profile_reference_job)])
    if args.resume_from_job:
        base.extend(["--resume-from-job", args.resume_from_job])
    processes = [
        subprocess.Popen(base + ["--rank", str(rank), "--world-size", "2"])
        for rank in (0, 1)
    ]
    statuses = [process.wait() for process in processes]
    return max(statuses)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides for the pinned default configuration")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the complete plan only")
    parser.add_argument("--simulate", action="store_true", help="exercise two-rank coordination without services")
    parser.add_argument("--resume-from-job", help="reuse only validated COMPLETE runs from this prior job ID")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="run only the explicitly selected matched perf captures",
    )
    parser.add_argument(
        "--profile-workload",
        help="workload selected by --profile-only",
    )
    parser.add_argument(
        "--profile-arms",
        help="comma-separated ordered arms selected by --profile-only",
    )
    parser.add_argument(
        "--profile-reference-job",
        help="matrix job whose exact configuration must match",
    )
    parser.add_argument("--job-id", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--world-size", type=int, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    profile_request = _resolve_profile_request(args, config)
    if args.dry_run:
        return _write_dry_run(
            config,
            args.job_id or slurm_job_id,
            profile_request,
        )

    rank_value = args.rank if args.rank is not None else os.environ.get("SLURM_PROCID")
    world_value = args.world_size if args.world_size is not None else os.environ.get("SLURM_NTASKS")
    if args.simulate and args.rank is None and int(world_value or 0) != 2:
        return _spawn_local_simulation(args)
    if not args.simulate and not slurm_job_id:
        raise HarnessError("real execution requires SLURM_JOB_ID")
    if not args.simulate and args.job_id and args.job_id != slurm_job_id:
        raise HarnessError("--job-id cannot override SLURM_JOB_ID for a real run")

    job_id = args.job_id or slurm_job_id
    if not job_id:
        raise HarnessError("job ID is required")
    if rank_value is None or world_value is None:
        raise HarnessError("two Slurm ranks (or hidden simulation rank arguments) are required")
    rank, world_size = int(rank_value), int(world_value)
    if world_size != 2 or rank not in {0, 1}:
        raise HarnessError(f"expected ranks 0/1 in a two-task step, got rank={rank}, size={world_size}")
    if (
        not args.simulate
        and rank == 1
        and os.environ.get("DYNAMO_NUMA_RANK_BOUND") != "1"
    ):
        # Re-exec the remote control agent itself under the fixed control pool.
        # Rank 0 deliberately retains the default memory policy so the 4+4
        # frontend can inherit genuine first-touch behavior.
        control = _remote_role(config, "control")
        environment = os.environ.copy()
        environment["DYNAMO_NUMA_RANK_BOUND"] = "1"
        command = process_prefix(control["cpus"], control["memory"]) + [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ]
        os.execvpe(command[0], command, environment)
    if args.resume_from_job:
        job_paths(config["root"], args.resume_from_job)

    shutdown = threading.Event()

    def handle_signal(signum: int, _frame: Any) -> None:
        print(f"rank {rank}: received signal {signum}; beginning owned-process cleanup", flush=True)
        shutdown.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    state_dir: Path | None = None
    try:
        state_dir, result_dir = _prepare_job_directories(
            Path(config["root"]), job_id, rank, shutdown
        )
        if args.simulate:
            hosts = ["simulated-rank0", "simulated-rank1"]
            interface = "lo"
            advertised_ip = "127.0.0.1"
        else:
            hosts = _slurm_hosts()
            configured_interface = str(
                config.get("network", {}).get(
                    "interface", os.environ.get("DYNAMO_NUMA_INTERFACE", "auto")
                )
            )
            interface, advertised_ip = _route_to(hosts[1 - rank], configured_interface)

        control_cpus = (
            config["topology"]["frontend_control_cpus"]
            if rank == 0
            else _remote_role(config, "control")["cpus"]
        )
        if not args.simulate:
            os.sched_setaffinity(0, parse_cpu_set(control_cpus))

        atomic_json(
            state_dir / "agents" / f"rank{rank}.json",
            {
                "rank": rank,
                "hostname": hosts[rank],
                "kernel_hostname": socket.gethostname(),
                "advertised_ip": advertised_ip,
                "interface": interface,
                "agent_pid": os.getpid(),
                "control_cpus": control_cpus,
                "registered_at": now(),
                "simulated": args.simulate,
            },
        )
        wait_for_path(state_dir / "agents" / f"rank{1 - rank}.json", 180, shutdown)
        peer = read_json(state_dir / "agents" / f"rank{1 - rank}.json")

        try:
            preflight = (
                {"ok": True, "simulated": True, "checked_at": now()}
                if args.simulate
                else _host_preflight(config, rank, interface)
            )
        except Exception as error:
            preflight = {
                "ok": False,
                "checked_at": now(),
                "error": f"{type(error).__name__}: {error}",
            }
        atomic_json(state_dir / "preflight" / f"rank{rank}.json", preflight)
        wait_for_path(state_dir / "preflight" / f"rank{1 - rank}.json", 180, shutdown)
        peer_preflight = read_json(state_dir / "preflight" / f"rank{1 - rank}.json")
        if not preflight["ok"] or not peer_preflight["ok"]:
            raise HarnessError(f"preflight failed: local={preflight}, peer={peer_preflight}")

        channel = CommandChannel(state_dir, shutdown)
        if rank == 0:
            Coordinator(
                config,
                channel,
                shutdown,
                job_id=job_id,
                result_dir=result_dir,
                advertised_ip=advertised_ip,
                remote_ip=peer["advertised_ip"],
                interface=interface,
                resume_from_job=args.resume_from_job,
                simulate=args.simulate,
            ).run(profile_request)
        else:
            RemoteAgent(
                config,
                channel,
                shutdown,
                job_id=job_id,
                advertised_ip=advertised_ip,
                frontend_ip=peer["advertised_ip"],
                interface=interface,
                simulate=args.simulate,
            ).serve()
        atomic_json(state_dir / f"rank{rank}.done.json", {"completed_at": now()})
        return 0
    except Exception as error:
        if state_dir is not None and state_dir.exists():
            atomic_json(
                state_dir / f"rank{rank}.failed.json",
                {
                    "failed_at": now(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
        print(f"rank {rank} failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
