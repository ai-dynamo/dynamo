# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark a combined custom-encoder service beside an encoder-only service."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from examples.custom_encoder.benchmark.safeguard_proxy_workload import (
    DECODER_MODEL,
    ENCODER_MODEL,
    TARGET_ISL,
    TARGET_OSL,
    generate_workload,
    validate_workload,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONCURRENCIES = (8, 16, 32, 64)
REQUESTS = 1000
WARMUP_REQUESTS = 20
CONFIRMATION_RUNS = 3
COMBINED_PORT = 8000
ENCODER_ONLY_PORT = 8001
QUEUE_DELAY_US = 1000
IMAGE_SIZE = 500
JPEG_MIN_BYTES = 50 * 1024
JPEG_MAX_BYTES = 60 * 1024
COMBINED_ROLE = "combined"
ENCODER_ONLY_ROLE = "encoder_only"
CONTROL_ARM = "control"
PARALLEL_ARM = "parallel"
ROLE_ORDER = (COMBINED_ROLE, ENCODER_ONLY_ROLE)
ARM_ORDER = (CONTROL_ARM, PARALLEL_ARM)


@dataclass(frozen=True)
class ProcessResult:
    role: str
    returncode: int
    released_ns: int
    finished_ns: int
    command: list[str]
    artifact_dir: Path


class ServiceProcess:
    """Launch one benchmark service in a dedicated process group."""

    def __init__(
        self,
        name: str,
        command: Sequence[str],
        env: dict[str, str],
        ready_url: str,
        log_path: Path,
        ready_text: str | None = None,
        timeout: int = 900,
    ) -> None:
        self.name = name
        self.command = list(command)
        self.env = env
        self.ready_url = ready_url
        self.log_path = log_path
        self.ready_text = ready_text
        self.timeout = timeout
        self._process: subprocess.Popen[str] | None = None
        self._log: Any = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self.running:
            raise RuntimeError(f"{self.name} is already running")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("a", encoding="utf-8")
        self._log.write(f"command={shlex.join(self.command)}\n")
        self._log.flush()
        self._process = subprocess.Popen(
            self.command,
            env=self.env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            self._wait_for_ready()
        except Exception:
            self.stop()
            raise

    def _wait_for_ready(self) -> None:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if not self.running:
                assert self._process is not None
                raise RuntimeError(
                    f"{self.name} exited during startup with "
                    f"code {self._process.returncode}; see {self.log_path}"
                )
            try:
                request = urllib.request.Request(self.ready_url)
                with urllib.request.urlopen(request, timeout=5) as response:
                    body = response.read().decode("utf-8")
                if self.ready_text is None or self.ready_text in body:
                    return
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(2)
        self.stop()
        raise TimeoutError(
            f"{self.name} was not ready after {self.timeout}s; see {self.log_path}"
        )

    def stop(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        self._process = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> ServiceProcess:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()


class GpuSampler:
    """Record one GPU utilization and memory sample per second."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._process: subprocess.Popen[str] | None = None
        self._output: Any = None

    def __enter__(self) -> GpuSampler:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output = self.output_path.open("a", encoding="utf-8")
        if self.output_path.stat().st_size == 0:
            self._output.write(
                "timestamp,utilization_gpu_pct,memory_used_mib,memory_total_mib\n"
            )
            self._output.flush()
        self._process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "--loop-ms=1000",
            ],
            stdout=self._output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        return self

    def __exit__(self, *_args: object) -> None:
        if self._process is not None:
            _terminate_process(self._process)
            self._process = None
        if self._output is not None:
            self._output.close()
            self._output = None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_output(command: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_concurrencies(values: Sequence[int]) -> tuple[int, ...]:
    concurrencies = tuple(int(value) for value in values)
    if not concurrencies or any(value < 1 for value in concurrencies):
        raise ValueError("concurrencies must be positive")
    if len(set(concurrencies)) != len(concurrencies):
        raise ValueError("concurrencies must be unique")
    return concurrencies


def _pool_dir(root: Path, concurrency: int, role: str) -> Path:
    return root / f"concurrency{concurrency}" / role


def _input_path(pool_dir: Path) -> Path:
    return pool_dir / f"image_custom_{REQUESTS}_isl{TARGET_ISL}.jsonl"


def _warmup_input_path(pool_dir: Path) -> Path:
    return pool_dir / f"image_custom_{WARMUP_REQUESTS}_isl{TARGET_ISL}.jsonl"


def _read_manifest(pool_dir: Path) -> dict[str, Any]:
    return json.loads((pool_dir / "workload_manifest.json").read_text(encoding="utf-8"))


def audit_pool_disjointness(pool_dirs: Sequence[Path]) -> dict[str, int]:
    encoded: set[str] = set()
    decoded: set[str] = set()
    image_count = 0
    for pool_dir in pool_dirs:
        manifest = _read_manifest(pool_dir)
        pool_encoded = {str(record["encoded_sha256"]) for record in manifest["images"]}
        pool_decoded = {
            str(record["decoded_rgb_sha256"]) for record in manifest["images"]
        }
        if encoded.intersection(pool_encoded) or decoded.intersection(pool_decoded):
            raise AssertionError(f"image hashes overlap with pool {pool_dir}")
        encoded.update(pool_encoded)
        decoded.update(pool_decoded)
        image_count += len(pool_encoded)
    if len(encoded) != image_count or len(decoded) != image_count:
        raise AssertionError("workload pools are not globally unique")
    return {
        "pools": len(pool_dirs),
        "images": image_count,
        "unique_encoded_sha256": len(encoded),
        "unique_decoded_rgb_sha256": len(decoded),
    }


def generate_parallel_workloads(
    root: Path, concurrencies: Sequence[int]
) -> dict[str, Any]:
    selected = _validate_concurrencies(concurrencies)
    root.mkdir(parents=True, exist_ok=True)
    pool_dirs: list[Path] = []
    for index, concurrency in enumerate(selected):
        for role_index, role in enumerate(ROLE_ORDER):
            pool_dir = _pool_dir(root, concurrency, role)
            seed = 42 + index * 100_000 + role_index * 50_000
            generate_workload(
                pool_dir,
                concurrencies=(concurrency,),
                requests=REQUESTS,
                unique_images=REQUESTS,
                target_isl=TARGET_ISL,
                seed=seed,
                image_size=IMAGE_SIZE,
            )
            validate_workload(
                pool_dir,
                expected_image_size=IMAGE_SIZE,
                expected_unique_images=REQUESTS,
            )
            pool_dirs.append(pool_dir)

    warmup_root = root / "warmup"
    for role_index, role in enumerate(ROLE_ORDER):
        pool_dir = warmup_root / role
        generate_workload(
            pool_dir,
            concurrencies=selected,
            requests=WARMUP_REQUESTS,
            unique_images=WARMUP_REQUESTS,
            target_isl=TARGET_ISL,
            seed=900_000 + role_index * 50_000,
            image_size=IMAGE_SIZE,
        )
        validate_workload(
            pool_dir,
            expected_image_size=IMAGE_SIZE,
            expected_unique_images=WARMUP_REQUESTS,
        )
        pool_dirs.append(pool_dir)

    uniqueness = audit_pool_disjointness(pool_dirs)
    manifest = {
        "axis": "concurrency",
        "concurrencies": list(selected),
        "requests_per_client": REQUESTS,
        "warmup_requests_per_service": WARMUP_REQUESTS,
        "parallel_clients": list(ROLE_ORDER),
        "target_isl": TARGET_ISL,
        "combined_target_osl": TARGET_OSL,
        "encoder_only_target_osl": 1,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "jpeg_bytes": [JPEG_MIN_BYTES, JPEG_MAX_BYTES],
        "uniqueness": uniqueness,
        "policy": (
            "control and parallel combined clients share the same per-concurrency "
            "pool; encoder-only and different concurrency values are disjoint"
        ),
    }
    manifest_path = root / "parallel_workload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"WORKLOAD_AUDIT=PASS pools={uniqueness['pools']} images={uniqueness['images']}"
    )
    return manifest


def validate_parallel_workloads(
    root: Path, concurrencies: Sequence[int]
) -> dict[str, Any]:
    selected = _validate_concurrencies(concurrencies)
    manifest_path = root / "parallel_workload_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest["concurrencies"]) != selected:
        raise AssertionError("workload concurrency selection does not match")
    pool_dirs = [
        _pool_dir(root, concurrency, role)
        for concurrency in selected
        for role in ROLE_ORDER
    ]
    pool_dirs.extend(root / "warmup" / role for role in ROLE_ORDER)
    for pool_dir in pool_dirs:
        expected = WARMUP_REQUESTS if "warmup" in pool_dir.parts else REQUESTS
        result = validate_workload(
            pool_dir,
            expected_image_size=IMAGE_SIZE,
            expected_unique_images=expected,
        )
        if result["target_isl"] != TARGET_ISL:
            raise AssertionError(f"unexpected ISL in {pool_dir}")
    uniqueness = audit_pool_disjointness(pool_dirs)
    if uniqueness != manifest["uniqueness"]:
        raise AssertionError("cross-pool uniqueness audit changed")
    print(
        f"WORKLOAD_AUDIT=PASS pools={uniqueness['pools']} images={uniqueness['images']}"
    )
    return manifest


def _aiperf_command(
    *,
    role: str,
    port: int,
    concurrency: int,
    requests: int,
    input_file: Path,
    artifact_dir: Path,
) -> list[str]:
    osl = TARGET_OSL if role == COMBINED_ROLE else 1
    zmq_dir = artifact_dir / "zmq"
    zmq_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "aiperf",
        "profile",
        "--model",
        DECODER_MODEL,
        "--url",
        f"http://localhost:{port}",
        "--endpoint-type",
        "chat",
        "--endpoint",
        "/v1/chat/completions",
        "--input-file",
        str(input_file.resolve()),
        "--custom-dataset-type",
        "single_turn",
        "--concurrency",
        str(concurrency),
        "--conversation-num",
        str(requests),
        "--extra-inputs",
        f"max_tokens:{osl}",
        "--extra-inputs",
        f"min_tokens:{osl}",
        "--extra-inputs",
        "ignore_eos:true",
        "--extra-inputs",
        "stream:true",
        "--streaming",
        "--random-seed",
        "42",
        "--workers-max",
        "20",
        "--record-processors",
        "32",
        "--request-timeout-seconds",
        "300",
        "--artifact-dir",
        str(artifact_dir),
        "--zmq-ipc-path",
        str(zmq_dir),
        "--ui",
        "none",
        "--no-server-metrics",
    ]
    if role == COMBINED_ROLE:
        command.append("--use-server-token-count")
    return command


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=5)


def run_barrier_pair(
    commands: dict[str, list[str]], artifact_dirs: dict[str, Path]
) -> dict[str, ProcessResult]:
    if tuple(commands) != ROLE_ORDER or tuple(artifact_dirs) != ROLE_ORDER:
        raise ValueError(f"paired commands must be ordered as {ROLE_ORDER}")

    processes: dict[str, subprocess.Popen[str]] = {}
    read_fds: dict[str, int] = {}
    write_fds: dict[str, int] = {}
    released: dict[str, int] = {}
    finished: dict[str, int] = {}
    with ExitStack() as stack:
        try:
            for role in ROLE_ORDER:
                artifact_dir = artifact_dirs[role]
                artifact_dir.mkdir(parents=True, exist_ok=True)
                command = commands[role]
                (artifact_dir / "command.txt").write_text(
                    shlex.join(command) + "\n", encoding="utf-8"
                )
                output = stack.enter_context(
                    (artifact_dir / "aiperf.log").open("w", encoding="utf-8")
                )
                read_fd, write_fd = os.pipe()
                read_fds[role] = read_fd
                write_fds[role] = write_fd
                wrapper = [
                    "bash",
                    "-c",
                    'IFS= read -r -n 1 <&"$1"; shift; exec "$@"',
                    "aiperf-barrier",
                    str(read_fd),
                    *command,
                ]
                processes[role] = subprocess.Popen(
                    wrapper,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    text=True,
                    pass_fds=(read_fd,),
                    start_new_session=True,
                )
                os.close(read_fd)

            for role in ROLE_ORDER:
                released[role] = time.monotonic_ns()
                os.write(write_fds[role], b"x")
                os.close(write_fds[role])

            pending = set(ROLE_ORDER)
            while pending:
                for role in tuple(pending):
                    returncode = processes[role].poll()
                    if returncode is None:
                        continue
                    finished[role] = time.monotonic_ns()
                    pending.remove(role)
                    if returncode != 0:
                        for other in pending:
                            _terminate_process(processes[other])
                            finished[other] = time.monotonic_ns()
                        pending.clear()
                        break
                if pending:
                    time.sleep(0.01)
        finally:
            for role, write_fd in write_fds.items():
                if role not in released:
                    os.close(write_fd)
            for process in processes.values():
                _terminate_process(process)

    results = {
        role: ProcessResult(
            role=role,
            returncode=int(processes[role].returncode),
            released_ns=released[role],
            finished_ns=finished[role],
            command=commands[role],
            artifact_dir=artifact_dirs[role],
        )
        for role in ROLE_ORDER
    }
    failures = [result for result in results.values() if result.returncode != 0]
    if failures:
        details = ", ".join(
            f"{result.role}=exit{result.returncode}" for result in failures
        )
        raise RuntimeError(f"paired AIPerf failed: {details}")
    return results


def _run_single(command: list[str], artifact_dir: Path) -> ProcessResult:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    start_ns = time.monotonic_ns()
    with (artifact_dir / "aiperf.log").open("w", encoding="utf-8") as output:
        process = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    finish_ns = time.monotonic_ns()
    if process.returncode != 0:
        raise RuntimeError(
            f"AIPerf failed with exit {process.returncode}; see "
            f"{artifact_dir / 'aiperf.log'}"
        )
    return ProcessResult(
        role=COMBINED_ROLE,
        returncode=process.returncode,
        released_ns=start_ns,
        finished_ns=finish_ns,
        command=command,
        artifact_dir=artifact_dir,
    )


def _run_warmup(
    arm: str, concurrency: int, workload_root: Path, cell_dir: Path
) -> None:
    combined_artifact = cell_dir / "warmup" / COMBINED_ROLE
    combined = _aiperf_command(
        role=COMBINED_ROLE,
        port=COMBINED_PORT,
        concurrency=min(concurrency, WARMUP_REQUESTS),
        requests=WARMUP_REQUESTS,
        input_file=_warmup_input_path(workload_root / "warmup" / COMBINED_ROLE),
        artifact_dir=combined_artifact,
    )
    if arm == CONTROL_ARM:
        _run_single(combined, combined_artifact)
        return
    encoder_artifact = cell_dir / "warmup" / ENCODER_ONLY_ROLE
    encoder = _aiperf_command(
        role=ENCODER_ONLY_ROLE,
        port=ENCODER_ONLY_PORT,
        concurrency=min(concurrency, WARMUP_REQUESTS),
        requests=WARMUP_REQUESTS,
        input_file=_warmup_input_path(workload_root / "warmup" / ENCODER_ONLY_ROLE),
        artifact_dir=encoder_artifact,
    )
    run_barrier_pair(
        {COMBINED_ROLE: combined, ENCODER_ONLY_ROLE: encoder},
        {COMBINED_ROLE: combined_artifact, ENCODER_ONLY_ROLE: encoder_artifact},
    )


def _write_timing(
    arm: str,
    concurrency: int,
    run_number: int,
    cell_dir: Path,
    results: dict[str, ProcessResult],
) -> dict[str, Any]:
    starts = [result.released_ns for result in results.values()]
    finishes = [result.finished_ns for result in results.values()]
    common_start = min(starts)
    joint_finish = max(finishes)
    requests = REQUESTS if arm == CONTROL_ARM else REQUESTS * 2
    timing = {
        "arm": arm,
        "concurrency_per_client": concurrency,
        "total_outstanding": concurrency if arm == CONTROL_ARM else concurrency * 2,
        "run": run_number,
        "requests_per_client": REQUESTS,
        "total_requests": requests,
        "common_start_ns": common_start,
        "joint_finish_ns": joint_finish,
        "joint_duration_s": (joint_finish - common_start) / 1_000_000_000,
        "joint_throughput_request_s": requests
        / ((joint_finish - common_start) / 1_000_000_000),
        "start_skew_ms": (max(starts) - min(starts)) / 1_000_000,
        "completion_skew_ms": (max(finishes) - min(finishes)) / 1_000_000,
        "clients": {
            role: {
                "released_ns": result.released_ns,
                "finished_ns": result.finished_ns,
                "duration_s": (result.finished_ns - result.released_ns) / 1_000_000_000,
                "wall_throughput_request_s": REQUESTS
                / ((result.finished_ns - result.released_ns) / 1_000_000_000),
                "artifact_dir": str(result.artifact_dir),
            }
            for role, result in results.items()
        },
    }
    (cell_dir / "cell_timing.json").write_text(
        json.dumps(timing, indent=2) + "\n", encoding="utf-8"
    )
    return timing


def _run_cell(
    arm: str,
    concurrency: int,
    run_number: int,
    workload_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    cell_dir = output_root / arm / f"concurrency{concurrency}" / f"run{run_number}"
    timing_path = cell_dir / "cell_timing.json"
    if timing_path.is_file():
        return json.loads(timing_path.read_text(encoding="utf-8"))

    _run_warmup(arm, concurrency, workload_root, cell_dir)
    combined_artifact = cell_dir / "measured" / COMBINED_ROLE
    combined = _aiperf_command(
        role=COMBINED_ROLE,
        port=COMBINED_PORT,
        concurrency=concurrency,
        requests=REQUESTS,
        input_file=_input_path(_pool_dir(workload_root, concurrency, COMBINED_ROLE)),
        artifact_dir=combined_artifact,
    )
    if arm == CONTROL_ARM:
        result = _run_single(combined, combined_artifact)
        return _write_timing(
            arm,
            concurrency,
            run_number,
            cell_dir,
            {COMBINED_ROLE: result},
        )

    encoder_artifact = cell_dir / "measured" / ENCODER_ONLY_ROLE
    encoder = _aiperf_command(
        role=ENCODER_ONLY_ROLE,
        port=ENCODER_ONLY_PORT,
        concurrency=concurrency,
        requests=REQUESTS,
        input_file=_input_path(
            _pool_dir(workload_root, concurrency, ENCODER_ONLY_ROLE)
        ),
        artifact_dir=encoder_artifact,
    )
    results = run_barrier_pair(
        {COMBINED_ROLE: combined, ENCODER_ONLY_ROLE: encoder},
        {COMBINED_ROLE: combined_artifact, ENCODER_ONLY_ROLE: encoder_artifact},
    )
    return _write_timing(arm, concurrency, run_number, cell_dir, results)


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DYN_MAX_MODEL_LEN": "2048",
            "DYN_MAX_NUM_SEQS": "64",
            "DYN_QWEN2_VL_ENCODER_MODEL": ENCODER_MODEL,
            "DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE": "1536",
            "DYN_QWEN2_VL_PREPROCESS_CONCURRENCY": "64",
            "DYN_QWEN2_VL_MAX_BATCH_COST": "64",
            "DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS": "1,2,4,8,16,32,64",
            "DYN_QWEN2_VL_GRAPH_IMAGE_SIZES": "500x500",
            "DYN_QWEN2_VL_PREPROCESS_CACHE_SIZE": "0",
            "DYN_CUSTOM_ENCODER_DISPATCH_LOG": "1",
        }
    )
    return env


def _combined_service(output_root: Path, arm: str) -> ServiceProcess:
    env = _base_env()
    env.update({"DYN_HTTP_PORT": str(COMBINED_PORT), "DYN_SYSTEM_PORT": "8081"})
    workflow = REPO_ROOT / "examples/custom_encoder/launch/agg_qwen2_5_vl_benchmark.sh"
    command = [
        "bash",
        str(workflow),
        "--custom-encoder-max-queue-delay-us",
        str(QUEUE_DELAY_US),
        "--gpu-memory-utilization",
        "0.4",
        "--no-enable-prefix-caching",
    ]
    return ServiceProcess(
        "combined",
        command,
        env,
        f"http://localhost:{COMBINED_PORT}/v1/models",
        output_root / arm / "combined_server.log",
        ready_text=DECODER_MODEL,
    )


def _encoder_service(output_root: Path) -> ServiceProcess:
    env = _base_env()
    command = [
        sys.executable,
        "-m",
        "examples.custom_encoder.benchmark.encoder_only_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(ENCODER_ONLY_PORT),
        "--model",
        DECODER_MODEL,
        "--max-queue-delay-us",
        str(QUEUE_DELAY_US),
    ]
    return ServiceProcess(
        "encoder-only",
        command,
        env,
        f"http://localhost:{ENCODER_ONLY_PORT}/health",
        output_root / PARALLEL_ARM / "encoder_only_server.log",
        ready_text='"ready"',
    )


def _metadata(concurrencies: Sequence[int]) -> dict[str, Any]:
    required_env = (
        "DYNAMO_BENCHMARK_COMMIT",
        "DYNAMO_BENCHMARK_BRANCH",
        "DYNAMO_BENCHMARK_IMAGE",
        "DYNAMO_BASE_IMAGE_COMMIT",
    )
    missing = [name for name in required_env if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing benchmark provenance: {', '.join(missing)}")
    return {
        "dynamo_commit": os.environ["DYNAMO_BENCHMARK_COMMIT"],
        "dynamo_branch": os.environ["DYNAMO_BENCHMARK_BRANCH"],
        "container_image": os.environ["DYNAMO_BENCHMARK_IMAGE"],
        "base_image_dynamo_commit": os.environ["DYNAMO_BASE_IMAGE_COMMIT"],
        "axis": "concurrency",
        "concurrencies_per_client": list(concurrencies),
        "requests_per_client": REQUESTS,
        "warmups_per_service": WARMUP_REQUESTS,
        "confirmation_runs": CONFIRMATION_RUNS,
        "models": {"decoder": DECODER_MODEL, "encoder": ENCODER_MODEL},
        "settings": {
            "target_isl": TARGET_ISL,
            "combined_target_osl": TARGET_OSL,
            "encoder_only_target_osl": 1,
            "queue_delay_us": QUEUE_DELAY_US,
            "preprocess_cache_size": 0,
            "preprocess_concurrency": 64,
            "max_batch_cost": 64,
            "graph_buckets": list(DEFAULT_CONCURRENCIES),
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "vllm_gpu_memory_utilization": 0.4,
            "vllm_prefix_cache": False,
        },
        "versions": {
            "python": platform.python_version(),
            "aiperf": _package_version("aiperf")
            or _command_output(["aiperf", "--version"]),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "vllm": _package_version("vllm"),
        },
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }


def _write_or_check_metadata(output_root: Path, metadata: dict[str, Any]) -> None:
    path = output_root / "benchmark_metadata.json"
    output_root.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError("refusing to resume with different benchmark provenance")
        return
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _select_confirmation_concurrencies(output_root: Path, arm: str) -> tuple[int, int]:
    path = output_root / arm / "confirmation_selection.json"
    if path.is_file():
        selected = json.loads(path.read_text(encoding="utf-8"))["concurrencies"]
        return int(selected[0]), int(selected[1])
    initial = []
    for timing_path in sorted(
        (output_root / arm).glob("concurrency*/run1/cell_timing.json")
    ):
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        initial.append(
            (
                float(timing["joint_throughput_request_s"]),
                int(timing["concurrency_per_client"]),
            )
        )
    if len(initial) < 2:
        raise RuntimeError(f"{arm} needs at least two initial cells")
    selected = tuple(
        concurrency for _throughput, concurrency in sorted(initial, reverse=True)[:2]
    )
    path.write_text(
        json.dumps({"concurrencies": list(selected)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return selected


def run_matrix(
    workload_root: Path,
    output_root: Path,
    concurrencies: Sequence[int],
    confirmation_runs: int = CONFIRMATION_RUNS,
) -> None:
    selected = _validate_concurrencies(concurrencies)
    if confirmation_runs < 1:
        raise ValueError("confirmation_runs must be positive")
    validate_parallel_workloads(workload_root, selected)
    metadata = _metadata(selected)
    metadata["confirmation_runs"] = confirmation_runs
    _write_or_check_metadata(output_root, metadata)

    with GpuSampler(output_root / "gpu_samples.csv"):
        with _combined_service(output_root, CONTROL_ARM):
            for concurrency in selected:
                _run_cell(CONTROL_ARM, concurrency, 1, workload_root, output_root)
            for concurrency in _select_confirmation_concurrencies(
                output_root, CONTROL_ARM
            ):
                for run_number in range(2, confirmation_runs + 1):
                    _run_cell(
                        CONTROL_ARM,
                        concurrency,
                        run_number,
                        workload_root,
                        output_root,
                    )

        with _combined_service(output_root, PARALLEL_ARM):
            with _encoder_service(output_root):
                for concurrency in selected:
                    _run_cell(PARALLEL_ARM, concurrency, 1, workload_root, output_root)
                for concurrency in _select_confirmation_concurrencies(
                    output_root, PARALLEL_ARM
                ):
                    for run_number in range(2, confirmation_runs + 1):
                        _run_cell(
                            PARALLEL_ARM,
                            concurrency,
                            run_number,
                            workload_root,
                            output_root,
                        )


def _metric(data: dict[str, Any], name: str, statistic: str = "avg") -> float | None:
    value = data.get(name)
    if not isinstance(value, dict) or statistic not in value:
        return None
    return float(value[statistic])


def validate_aiperf(path: Path, role: str, concurrency: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    command_path = path.parent / "command.txt"
    command = command_path.read_text(encoding="utf-8") if command_path.is_file() else ""
    loadgen = data.get("input_config", {}).get("loadgen", {})
    expected_osl = TARGET_OSL if role == COMBINED_ROLE else 1
    if _metric(data, "request_count") != float(REQUESTS):
        failures.append("request_count")
    if data.get("error_summary"):
        failures.append("errors")
    if data.get("was_cancelled"):
        failures.append("cancelled")
    if not data.get("input_config", {}).get("endpoint", {}).get("streaming", False):
        failures.append("streaming")
    if int(loadgen.get("concurrency", -1)) != concurrency:
        failures.append("concurrency")
    if "--request-rate" in command or f"--concurrency {concurrency}" not in command:
        failures.append("closed_loop_command")
    if "AIPERF_HTTP_CONNECTION_LIMIT" in command:
        failures.append("connection_limit")
    if (role == COMBINED_ROLE) != ("--use-server-token-count" in command):
        failures.append("server_token_count_policy")
    for metric_name, expected in (
        ("input_sequence_length", TARGET_ISL),
        ("output_sequence_length", expected_osl),
    ):
        for statistic_name in ("min", "avg", "max"):
            if _metric(data, metric_name, statistic_name) != float(expected):
                failures.append(f"{metric_name}_{statistic_name}")
    for metric_name in (
        "request_throughput",
        "request_latency",
        "time_to_first_token",
    ):
        if _metric(data, metric_name) is None:
            failures.append(metric_name)
    return {
        "path": str(path),
        "role": role,
        "concurrency": concurrency,
        "accepted": not failures,
        "failures": failures,
        "request_throughput": _metric(data, "request_throughput"),
        "ttft_p50_ms": _metric(data, "time_to_first_token", "p50"),
        "ttft_p95_ms": _metric(data, "time_to_first_token", "p95"),
        "ttft_p99_ms": _metric(data, "time_to_first_token", "p99"),
        "e2e_p50_ms": _metric(data, "request_latency", "p50"),
        "e2e_p95_ms": _metric(data, "request_latency", "p95"),
        "e2e_p99_ms": _metric(data, "request_latency", "p99"),
    }


def validate_matrix(output_root: Path) -> list[dict[str, Any]]:
    metadata = json.loads(
        (output_root / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    concurrencies = tuple(metadata["concurrencies_per_client"])
    confirmation_runs = int(metadata["confirmation_runs"])
    rows: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        confirmation = set(_select_confirmation_concurrencies(output_root, arm))
        for concurrency in concurrencies:
            expected_runs = confirmation_runs if concurrency in confirmation else 1
            for run_number in range(1, expected_runs + 1):
                cell_dir = (
                    output_root / arm / f"concurrency{concurrency}" / f"run{run_number}"
                )
                timing_path = cell_dir / "cell_timing.json"
                if not timing_path.is_file():
                    raise AssertionError(f"missing timing artifact: {timing_path}")
                timing = json.loads(timing_path.read_text(encoding="utf-8"))
                roles = (COMBINED_ROLE,) if arm == CONTROL_ARM else ROLE_ORDER
                expected_total = REQUESTS * len(roles)
                if int(timing["total_requests"]) != expected_total:
                    raise AssertionError(f"wrong total request count: {timing_path}")
                if arm == PARALLEL_ARM and float(timing["start_skew_ms"]) > 100.0:
                    raise AssertionError(
                        f"parallel start skew exceeded 100ms: {timing_path}"
                    )
                for role in roles:
                    result_path = (
                        cell_dir / "measured" / role / "profile_export_aiperf.json"
                    )
                    result = validate_aiperf(result_path, role, concurrency)
                    result.update(
                        {
                            "arm": arm,
                            "run": run_number,
                            "joint_duration_s": timing["joint_duration_s"],
                            "joint_throughput_request_s": timing[
                                "joint_throughput_request_s"
                            ],
                            "start_skew_ms": timing["start_skew_ms"],
                            "completion_skew_ms": timing["completion_skew_ms"],
                        }
                    )
                    rows.append(result)
    rejected = [row for row in rows if not row["accepted"]]
    if rejected:
        details = "; ".join(
            f"{row['arm']}/c{row['concurrency']}/r{row['run']}/{row['role']}="
            f"{','.join(row['failures'])}"
            for row in rejected
        )
        raise AssertionError(f"rejected benchmark artifacts: {details}")
    validation_path = output_root / "validation.json"
    validation_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"BENCHMARK_AUDIT=PASS aiperf_results={len(rows)}")
    return rows


def _timing_rows(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.glob("*/concurrency*/run*/cell_timing.json")):
        timing = json.loads(path.read_text(encoding="utf-8"))
        timing["path"] = str(path.relative_to(output_root))
        rows.append(timing)
    return rows


def _arm_summary(timings: Sequence[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    by_concurrency: dict[int, list[dict[str, Any]]] = {}
    for timing in timings:
        if timing["arm"] == arm:
            by_concurrency.setdefault(int(timing["concurrency_per_client"]), []).append(
                timing
            )
    rows = []
    for concurrency, samples in sorted(by_concurrency.items()):
        throughputs = [
            float(sample["joint_throughput_request_s"]) for sample in samples
        ]
        durations = [float(sample["joint_duration_s"]) for sample in samples]
        rows.append(
            {
                "arm": arm,
                "concurrency": concurrency,
                "runs": len(samples),
                "median_throughput": statistics.median(throughputs),
                "min_throughput": min(throughputs),
                "max_throughput": max(throughputs),
                "median_duration": statistics.median(durations),
            }
        )
    return rows


def summarize(output_root: Path, markdown_path: Path, csv_path: Path) -> None:
    validated = validate_matrix(output_root)
    timings = _timing_rows(output_root)
    metadata = json.loads(
        (output_root / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    summaries = {arm: _arm_summary(timings, arm) for arm in ARM_ORDER}
    confirmed = {
        arm: [
            row
            for row in summaries[arm]
            if row["runs"] == metadata["confirmation_runs"]
        ]
        for arm in ARM_ORDER
    }
    winners = {
        arm: max(rows, key=lambda row: row["median_throughput"])
        for arm, rows in confirmed.items()
    }

    combined_metrics = {
        (row["arm"], int(row["concurrency"]), int(row["run"])): row
        for row in validated
        if row["role"] == COMBINED_ROLE
    }
    lines = [
        "# Parallel custom-encoder throughput benchmark",
        "",
        f"Both arms use Dynamo `{metadata['dynamo_commit']}` in "
        f"`{metadata['container_image']}`. Each measured AIPerf client sends "
        f"{REQUESTS:,} streaming requests with exact ISL {TARGET_ISL}. The combined "
        f"service generates exactly {TARGET_OSL} tokens; the encoder-only service "
        "returns one dummy token.",
        "",
        "## Maximum observed throughput",
        "",
        "| Arm | Concurrency/client | Requests/joint run | Median req/s | "
        "Median makespan | Samples |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARM_ORDER:
        winner = winners[arm]
        requests = REQUESTS if arm == CONTROL_ARM else REQUESTS * 2
        lines.append(
            f"| {arm} | {winner['concurrency']} | {requests:,} | "
            f"**{winner['median_throughput']:.2f}** | "
            f"{winner['median_duration']:.2f} s | {winner['runs']} |"
        )

    lines.extend(
        [
            "",
            "## Sweep",
            "",
            "| Arm | Concurrency/client | Total outstanding | Runs | Median req/s | "
            "Range req/s | Median makespan |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ARM_ORDER:
        for row in summaries[arm]:
            outstanding = row["concurrency"] * (1 if arm == CONTROL_ARM else 2)
            lines.append(
                f"| {arm} | {row['concurrency']} | {outstanding} | {row['runs']} | "
                f"{row['median_throughput']:.2f} | {row['min_throughput']:.2f}–"
                f"{row['max_throughput']:.2f} | {row['median_duration']:.2f} s |"
            )

    lines.extend(
        [
            "",
            "## Combined-service contention",
            "",
            "| Concurrency | Control combined req/s | Parallel combined req/s | "
            "Change |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for concurrency in metadata["concurrencies_per_client"]:
        control = float(
            combined_metrics[(CONTROL_ARM, concurrency, 1)]["request_throughput"]
        )
        parallel = float(
            combined_metrics[(PARALLEL_ARM, concurrency, 1)]["request_throughput"]
        )
        lines.append(
            f"| {concurrency} | {control:.2f} | {parallel:.2f} | "
            f"{(parallel / control - 1.0) * 100:+.1f}% |"
        )

    lines.extend(
        [
            "",
            "## AIPerf client metrics (initial sweep)",
            "",
            "| Arm | Client | Concurrency | Req/s | TTFT p50/p95/p99 | "
            "E2E p50/p95/p99 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in validated:
        if int(row["run"]) != 1:
            continue
        lines.append(
            f"| {row['arm']} | {row['role']} | {row['concurrency']} | "
            f"{float(row['request_throughput']):.2f} | "
            f"{float(row['ttft_p50_ms']):.1f}/{float(row['ttft_p95_ms']):.1f}/"
            f"{float(row['ttft_p99_ms']):.1f} ms | "
            f"{float(row['e2e_p50_ms']):.1f}/{float(row['e2e_p95_ms']):.1f}/"
            f"{float(row['e2e_p99_ms']):.1f} ms |"
        )

    control_by_c = {row["concurrency"]: row for row in summaries[CONTROL_ARM]}
    parallel_by_c = {row["concurrency"]: row for row in summaries[PARALLEL_ARM]}
    saturation_notes = []
    for arm, by_c in ((CONTROL_ARM, control_by_c), (PARALLEL_ARM, parallel_by_c)):
        if 32 in by_c and 64 in by_c:
            gain = (
                by_c[64]["median_throughput"] / by_c[32]["median_throughput"] - 1.0
            ) * 100.0
            status = "not proven" if gain >= 5.0 else "observed"
            saturation_notes.append(
                f"- {arm}: 32→64 throughput change {gain:+.1f}%; saturation {status}."
            )
    lines.extend(
        [
            "",
            "## Saturation and artifacts",
            "",
            *saturation_notes,
            "",
            "- [Validation](validation.json)",
            "- [Metadata](benchmark_metadata.json)",
            "- [CSV](benchmark.csv)",
            "- [GPU samples](gpu_samples.csv)",
            "- Each cell contains `cell_timing.json`, AIPerf JSON, command, and log files.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "arm",
                "concurrency_per_client",
                "run",
                "total_requests",
                "joint_duration_s",
                "joint_throughput_request_s",
                "start_skew_ms",
                "completion_skew_ms",
                "path",
            ],
        )
        writer.writeheader()
        for timing in timings:
            writer.writerow({name: timing[name] for name in writer.fieldnames})
    print(f"report={markdown_path} csv={csv_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "validate-workload"):
        command = subparsers.add_parser(name)
        command.add_argument("--workload-dir", type=Path, required=True)
        command.add_argument(
            "--concurrencies",
            type=int,
            nargs="+",
            default=list(DEFAULT_CONCURRENCIES),
        )
    run = subparsers.add_parser("run")
    run.add_argument("--workload-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONCURRENCIES),
    )
    run.add_argument("--confirmation-runs", type=int, default=CONFIRMATION_RUNS)
    validate = subparsers.add_parser("validate")
    validate.add_argument("output_dir", type=Path)
    report = subparsers.add_parser("summarize")
    report.add_argument("output_dir", type=Path)
    report.add_argument("--markdown", type=Path)
    report.add_argument("--csv", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "generate":
        generate_parallel_workloads(args.workload_dir.resolve(), args.concurrencies)
    elif args.command == "validate-workload":
        validate_parallel_workloads(args.workload_dir.resolve(), args.concurrencies)
    elif args.command == "run":
        run_matrix(
            args.workload_dir.resolve(),
            args.output_dir.resolve(),
            args.concurrencies,
            confirmation_runs=args.confirmation_runs,
        )
    elif args.command == "validate":
        validate_matrix(args.output_dir.resolve())
    else:
        output_dir = args.output_dir.resolve()
        summarize(
            output_dir,
            args.markdown or output_dir / "benchmark.md",
            args.csv or output_dir / "benchmark.csv",
        )


if __name__ == "__main__":
    main()
