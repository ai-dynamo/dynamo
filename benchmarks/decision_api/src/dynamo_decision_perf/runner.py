# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded local AIPerf runs with immutable inputs and explicit GPU authorization."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from .lifecycle import termination_signal
from .telemetry import record_sample
from .workloads import MODEL_REVISION

AIPERF_COMMIT = "794f8bb75f8582f22e412d7e650fc71ca2a3d21a"
ENDPOINTS = {
    "oai": "decision_oai",
    "systemone": "decision_systemone",
    "native_score": "native_score",
}


@dataclass(frozen=True)
class RunSpec:
    dialect: str
    url: str
    model: str
    concurrency: int = 1
    requests: int | None = 4
    duration: float = 300
    warmup: float = 0
    rate: float | None = None
    timeout: float = 60
    gpu: bool = False
    approved_gpu_hours: float | None = None

    def __post_init__(self):
        parsed = urlparse(self.url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError("use a local HTTP base URL without credentials or path")
        if self.dialect not in ENDPOINTS or not self.model.strip():
            raise ValueError("unsupported dialect or empty model")
        if not 1 <= self.concurrency <= 8 or (
            self.requests is not None and self.requests < 1
        ):
            raise ValueError("concurrency must be 1..8 and request count positive")
        for name, value, low, high in (
            ("duration", self.duration, 0.01, 1800),
            ("warmup", self.warmup, 0, 30),
            ("timeout", self.timeout, 0.01, 120),
        ):
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} outside approved bounds")
        if self.rate is not None and (not math.isfinite(self.rate) or self.rate <= 0):
            raise ValueError("rate must be finite and positive")
        if self.gpu and (
            self.approved_gpu_hours is None or not 0 < self.approved_gpu_hours <= 4
        ):
            raise ValueError(
                "GPU execution requires explicit approval of at most four GPU-hours"
            )


def build_command(
    spec: RunSpec, dataset: Path, artifacts: Path, executable: str = "aiperf"
) -> list[str]:
    command = [
        executable,
        "profile",
        "--model",
        spec.model,
        "--url",
        spec.url,
        "--endpoint-type",
        ENDPOINTS[spec.dialect],
        "--transport",
        "decision_http",
        "--custom-dataset-type",
        "inputs_json",
        "--input-file",
        str(dataset),
        "--dataset-sampling-strategy",
        "sequential",
        "--random-seed",
        "17",
        "--export-level",
        "raw",
        "--artifact-dir",
        str(artifacts),
        "--no-auto-plot",
        "--ui",
        "none",
        "--no-server-metrics",
        "--request-timeout-seconds",
        str(spec.timeout),
        "--benchmark-duration",
        str(spec.duration),
        "--benchmark-grace-period",
        str(spec.timeout),
    ]
    command += ["--gpu-telemetry", "pynvml"] if spec.gpu else ["--no-gpu-telemetry"]
    command += (
        ["--request-rate", str(spec.rate)]
        if spec.rate
        else ["--concurrency", str(spec.concurrency)]
    )
    if spec.requests is not None:
        command += ["--request-count", str(spec.requests)]
    if spec.warmup:
        command += [
            "--warmup-duration",
            str(spec.warmup),
            "--warmup-grace-period",
            str(spec.timeout),
        ]
    if spec.dialect == "native_score":
        command += ["--streaming"]
    return command


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_hashes() -> dict:
    return {
        p.name: _hash(p.read_bytes())
        for p in sorted(Path(__file__).parent.iterdir())
        if p.suffix in (".py", ".yaml")
    }


def freeze_run(output: Path, spec: RunSpec, dataset: Path, profile: dict) -> dict:
    if spec.gpu:
        required = {
            "dynamo_commit",
            "frontend_commit",
            "model_revision",
            "tokenizer_sha256",
            "template_sha256",
            "engine_version",
            "precision",
            "gpu",
            "driver",
            "context_limit",
            "scheduler",
            "qualification_evidence",
            "series_id",
        }
        if missing := required - profile.keys():
            raise ValueError(f"GPU profile missing fields: {sorted(missing)}")
        if profile["engine_version"] != "0.5.19" or profile["context_limit"] != 2048:
            raise ValueError("profile is not the qualified SGLang configuration")
        if (
            profile["model_revision"] != MODEL_REVISION
            or spec.model != "Qwen/Qwen3.8-27B"
        ):
            raise ValueError("model must match the qualified Qwen revision")
        if profile["scheduler"] != {
            "tp": 1,
            "pp": 1,
            "max_running_requests": 1,
            "disable_overlap_schedule": True,
        }:
            raise ValueError("serialized scheduling restrictions must remain frozen")
    elif profile.get("kind") != "cpu_instrument":
        raise ValueError("non-GPU runs require an explicit cpu_instrument profile")
    payload = dataset.read_bytes()
    json.loads(payload)
    output.mkdir(parents=True, exist_ok=False)
    (output / "workload.json").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "spec": asdict(spec),
        "profile": profile,
        "workload_sha256": _hash(payload),
        "profile_sha256": _hash(json.dumps(profile, sort_keys=True).encode()),
        "aiperf_version": "0.13.0",
        "aiperf_commit": AIPERF_COMMIT,
        "plugin_version": importlib.metadata.version("dynamo-decision-perf"),
        "plugin_source_sha256": _source_hashes(),
        "cache_policy": "request-local; native fresh salt for every actual send",
        "automatic_http_retries": False,
        "retry_evidence": "pinned aiohttp transport single-send implementation; CPU 429 fixture observes exactly configured attempts",
    }
    metadata = {
        "run_id": output.name,
        "series_id": profile.get("series_id", "cpu-instrument"),
        "profile_id": manifest["profile_sha256"],
        "cache_phase": profile.get("cache_phase", "unspecified"),
        "dialect": spec.dialect,
        "aiperf_version": "0.13.0",
        "expected_requests": spec.requests,
        "warmup_expected_requests": None if spec.warmup else 0,
        "offered_requests": None,
        "cost_boundary": profile.get(
            "cost_boundary",
            "client through native scoring HTTP"
            if spec.dialect == "native_score"
            else "client through full decision HTTP frontend, renderer, routing and scoring",
        ),
    }
    manifest = {**manifest, "audit_metadata": metadata}
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output / "manifest.json").write_text(encoded)
    (output / "manifest.sha256").write_text(_hash(encoded.encode()) + "\n")
    (output / "audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return manifest


def campaign_points(capacity: float) -> tuple[dict, ...]:
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("screened capacity must be finite and positive")
    screening = tuple(
        {
            "campaign": "screening",
            "concurrency": c,
            "requests": 4 * c,
            "warmup": 0,
            "duration": 300,
        }
        for c in (1, 2, 4, 8)
    )
    sustained = tuple(
        {
            "campaign": "sustained",
            "rate": capacity * f,
            "requests": None,
            "warmup": 30,
            "duration": 300,
        }
        for f in (0.25, 0.5, 0.75, 1, 1.25)
    )
    return screening + sustained


def verify_frozen(output: Path, spec: RunSpec) -> None:
    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    source_hashes = _source_hashes()
    if (
        _hash(manifest_bytes) != (output / "manifest.sha256").read_text().strip()
        or manifest.get("spec") != asdict(spec)
        or manifest.get("workload_sha256")
        != _hash((output / "workload.json").read_bytes())
        or manifest.get("plugin_source_sha256") != source_hashes
    ):
        raise ValueError(
            "frozen run inputs or instrument source changed; create a new run"
        )
    if (output / "benchmark_execution.json").exists() or (output / "aiperf").exists():
        raise ValueError("frozen run was already executed; create a new run")


def _terminate_owned(process) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def execute(
    output: Path,
    spec: RunSpec,
    executable: str,
    budget_deadline: float | None = None,
    budget_started: float | None = None,
    monitor_pids: tuple[int, ...] = (),
    client_cpus: tuple[int, ...] = (),
) -> int:
    if importlib.metadata.version("aiperf") != "0.13.0":
        raise ValueError("this instrument requires AIPerf 0.13.0")
    command = build_command(
        spec, output / "workload.json", output / "aiperf", executable
    )
    if client_cpus:
        if not set(client_cpus) <= os.sched_getaffinity(0):
            raise ValueError(
                "client CPUs must be available in the current affinity set"
            )
        command = ["taskset", "--cpu-list", ",".join(map(str, client_cpus)), *command]
    if spec.gpu and not client_cpus:
        raise ValueError(
            "GPU measurements require explicit load-generator CPU affinity"
        )
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    wall_limit = spec.duration + spec.warmup + 2 * spec.timeout + 90
    if spec.gpu:
        if (
            budget_deadline is None
            or budget_started is None
            or not all(math.isfinite(v) for v in (budget_deadline, budget_started))
        ):
            raise ValueError(
                "GPU run requires an active serving-lifetime budget deadline"
            )
        if (
            not budget_started
            <= time.time()
            < budget_deadline
            <= budget_started + spec.approved_gpu_hours * 3600
        ):
            raise ValueError(
                "serving-lifetime deadline exceeds the approved GPU budget or has expired"
            )
        wall_limit = min(wall_limit, budget_deadline - time.time())
    verify_frozen(output, spec)
    started = time.time_ns()
    with termination_signal(), (output / "aiperf.log").open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "TZ": "UTC"},
        )
        try:
            deadline = time.monotonic() + wall_limit
            while process.poll() is None:
                record_sample(
                    output / "cpu_telemetry.jsonl", [process.pid, *monitor_pids]
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, wall_limit)
                try:
                    process.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    continue
            code = process.returncode
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            code = 124
        except SystemExit:
            code = 128 + signal.SIGTERM
        finally:
            if process.poll() is None:
                _terminate_owned(process)
    (output / "benchmark_execution.json").write_text(
        json.dumps(
            {
                "exit_code": code,
                "started_unix_ns": started,
                "ended_unix_ns": time.time_ns(),
                "wall_limit_seconds": wall_limit,
                "budget_started": budget_started,
                "budget_deadline": budget_deadline,
                "manifest_sha256": _hash((output / "manifest.json").read_bytes()),
                "export_timezone_offset_seconds": 0,
                "monitor_pids": monitor_pids,
                "client_cpus": client_cpus,
                "timing_note": "process lifetime is not the measurement window",
            },
            indent=2,
        )
        + "\n"
    )
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--dialect", choices=ENDPOINTS, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--warmup", type=float, default=0)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--approved-gpu-hours", type=float)
    parser.add_argument(
        "--budget-deadline",
        type=float,
        help="Unix deadline set when serving GPU reservation began",
    )
    parser.add_argument(
        "--budget-started",
        type=float,
        help="Unix time the serving GPU reservation began",
    )
    parser.add_argument("--aiperf", default="aiperf")
    parser.add_argument("--monitor-pid", type=int, action="append", default=[])
    parser.add_argument(
        "--client-cpus",
        help="comma-separated CPU numbers, disjoint from serving CPU affinity",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="otherwise freeze inputs and print command only",
    )
    args = parser.parse_args()
    spec = RunSpec(
        **{name: getattr(args, name) for name in RunSpec.__dataclass_fields__}
    )
    freeze_run(args.output, spec, args.dataset, json.loads(args.profile.read_text()))
    if args.execute:
        cpus = (
            tuple(int(c) for c in args.client_cpus.split(","))
            if args.client_cpus
            else ()
        )
        raise SystemExit(
            execute(
                args.output,
                spec,
                args.aiperf,
                args.budget_deadline,
                args.budget_started,
                tuple(args.monitor_pid),
                cpus,
            )
        )
    print(
        json.dumps(
            build_command(
                spec, args.output / "workload.json", args.output / "aiperf", args.aiperf
            )
        )
    )


if __name__ == "__main__":
    main()
