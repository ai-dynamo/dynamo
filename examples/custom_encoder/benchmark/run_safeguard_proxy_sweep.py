# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run, validate, and summarize the Qwen custom-encoder proxy sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Collection

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.multimodal.sweep.config import (  # noqa: E402
    BenchmarkConfig,
    SweepConfig,
)
from benchmarks.multimodal.sweep.orchestrator import run_sweep  # noqa: E402
from examples.custom_encoder.benchmark.safeguard_proxy_workload import (  # noqa: E402
    CONCURRENCIES,
    DECODER_MODEL,
    ENCODER_MODEL,
    INPUT_NAME,
    REQUESTS,
    TARGET_ISL,
    TARGET_OSL,
)

RUNTIME = "dynamo-custom-encoder"
WARMUP_REQUESTS = 20
TRITON_BASELINE: dict[int, tuple[float, float]] = {
    1: (16.61, 152.0),
    2: (18.33, 222.3),
    3: (19.16, 242.6),
    4: (20.55, 279.1),
    5: (22.10, 358.5),
    6: (26.49, 360.2),
    7: (27.38, 384.1),
    8: (27.54, 422.7),
    9: (31.09, 439.7),
    10: (35.97, 418.5),
}
AIPERF_EXTRA_ARGS = [
    "--endpoint-type",
    "chat",
    "--endpoint",
    "/v1/chat/completions",
    "--warmup-request-rate",
    "1000",
    "--warmup-arrival-pattern",
    "constant",
    "--random-seed",
    "42",
    "--workers-max",
    "20",
    "--record-processors",
    "32",
    "--use-server-token-count",
    "--request-timeout-seconds",
    "300",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_metadata() -> str | None:
    command = ["nvidia-smi"]
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        command.append(f"--id={visible_devices.split(',', 1)[0]}")
    command.extend(["--query-gpu=name,uuid,driver_version", "--format=csv,noheader"])
    return _command_output(command)


def _metadata(
    concurrencies: tuple[int, ...],
    smoke: bool,
    workload_dir: Path,
) -> dict[str, Any]:
    manifest_path = workload_dir / "workload_manifest.json"
    return {
        "axis": "concurrency",
        "concurrencies": list(concurrencies),
        "runtime": RUNTIME,
        "decoder_model": DECODER_MODEL,
        "encoder_model": ENCODER_MODEL,
        "requests_per_cell": 1 if smoke else REQUESTS,
        "warmup_requests": 1 if smoke else WARMUP_REQUESTS,
        "isl": TARGET_ISL,
        "osl": 1 if smoke else TARGET_OSL,
        "streaming": True,
        "performance_only_adapter": (
            "the complete native 2048-wide Qwen vision output is computed, then "
            "the first 1536 columns are passed to the decoder; no quality or "
            "same-model parity claim"
        ),
        "settings": {
            "preprocess_concurrency": 4,
            "max_batch_cost": 8,
            "graph_buckets": list(range(1, 9)),
            "graph_image_sizes": ["500x500"],
            "preprocess_cache_size": 0,
            "queue_wait_ms": 1,
            "max_num_seqs": 64,
            "max_model_len": 2048,
            "vllm_gpu_memory_utilization": 0.4,
        },
        "aiperf_extra_args": AIPERF_EXTRA_ARGS,
        "dynamo_commit": os.environ.get("DYNAMO_BENCHMARK_COMMIT"),
        "dynamo_branch": os.environ.get("DYNAMO_BENCHMARK_BRANCH"),
        "container_image": os.environ.get("DYNAMO_BENCHMARK_IMAGE"),
        "vllm_version": _package_version("vllm"),
        "aiperf_version": _package_version("aiperf")
        or _command_output(["aiperf", "--version"]),
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "python_version": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "workload_manifest_sha256": _sha256(manifest_path),
        "input_sha256": _sha256(workload_dir / INPUT_NAME),
        "gpu": _gpu_metadata(),
        "host": platform.node(),
        "triton_baseline": {
            str(value): {"request_throughput": metrics[0], "p95_ms": metrics[1]}
            for value, metrics in TRITON_BASELINE.items()
        },
    }


def build_config(
    input_file: Path,
    concurrencies: tuple[int, ...],
    output_dir: Path,
    smoke: bool,
) -> SweepConfig:
    workflow = REPO_ROOT / (
        "examples/custom_encoder/launch/agg_qwen2_5_vl_benchmark.sh"
    )
    return SweepConfig(
        model=DECODER_MODEL,
        concurrencies=[1] if smoke else list(concurrencies),
        osl=1 if smoke else TARGET_OSL,
        conversation_num=1 if smoke else REQUESTS,
        warmup_count=1 if smoke else WARMUP_REQUESTS,
        port=8000,
        timeout=1800,
        input_files=[str(input_file.resolve())],
        configs=[BenchmarkConfig(label=RUNTIME, workflow=str(workflow))],
        output_dir=str(output_dir),
        skip_plots=True,
        restart_server_every_benchmark=True,
        env={
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DYN_MAX_MODEL_LEN": "2048",
            "DYN_MAX_NUM_SEQS": "64",
            "DYN_VLLM_GPU_MEMORY_UTILIZATION": "0.4",
            "DYN_QWEN2_VL_ENCODER_MODEL": ENCODER_MODEL,
            "DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE": "1536",
            "DYN_QWEN2_VL_PREPROCESS_CONCURRENCY": "4",
            "DYN_QWEN2_VL_MAX_BATCH_COST": "8",
            "DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS": "1,2,3,4,5,6,7,8",
            "DYN_QWEN2_VL_GRAPH_IMAGE_SIZES": "500x500",
            "DYN_QWEN2_VL_PREPROCESS_CACHE_SIZE": "0",
            "DYN_CUSTOM_ENCODER_QUEUE_WAIT_MS": "1",
            "DYN_CUSTOM_ENCODER_TIMING": "1",
        },
        aiperf_extra_args=AIPERF_EXTRA_ARGS,
    )


def run_matrix(
    workload_dir: Path,
    output_dir: Path,
    concurrencies: tuple[int, ...],
    smoke: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(concurrencies, smoke, workload_dir)
    for required in ("dynamo_commit", "dynamo_branch", "container_image"):
        if not metadata[required]:
            raise RuntimeError(f"missing required benchmark provenance: {required}")
    metadata_path = output_dir / "benchmark_metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError("refusing to resume with different benchmark provenance")
    else:
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    config = build_config(workload_dir / INPUT_NAME, concurrencies, output_dir, smoke)
    config.validate(repo_root=REPO_ROOT)
    run_sweep(config, repo_root=REPO_ROOT)


def _metric(data: dict[str, Any], name: str, statistic: str = "avg") -> float | None:
    value = data.get(name)
    if not isinstance(value, dict) or statistic not in value:
        return None
    return float(value[statistic])


def validate_result(
    path: Path,
    expected_concurrencies: Collection[int] = CONCURRENCIES,
) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    runtime = path.parents[1].name
    loadgen = data.get("input_config", {}).get("loadgen", {})
    concurrency = int(loadgen.get("concurrency", -1))
    command = str(data.get("input_config", {}).get("cli_command", ""))
    if runtime != RUNTIME:
        failures.append("runtime")
    if concurrency not in expected_concurrencies:
        failures.append("concurrency")
    if _metric(data, "request_count") != float(REQUESTS):
        failures.append("request_count")
    if data.get("error_summary"):
        failures.append("errors")
    if data.get("was_cancelled"):
        failures.append("cancelled")
    if not data.get("input_config", {}).get("endpoint", {}).get("streaming", False):
        failures.append("streaming")
    if "--random-seed 42" not in command:
        failures.append("random_seed")
    if f"--concurrency {concurrency}" not in command:
        failures.append("concurrency_cli")
    if "--request-rate" in command:
        failures.append("request_rate_cli")
    for metric_name, expected in (
        ("input_sequence_length", TARGET_ISL),
        ("output_sequence_length", TARGET_OSL),
    ):
        for statistic in ("min", "avg", "max"):
            if _metric(data, metric_name, statistic) != float(expected):
                failures.append(f"{metric_name}_{statistic}")
    if _metric(data, "request_latency", "p95") is None:
        failures.append("request_latency_p95")
    if _metric(data, "request_throughput") is None:
        failures.append("request_throughput")
    command_path = path.parent / "command.txt"
    if not command_path.is_file():
        failures.append("command_artifact")
    return {
        "path": str(path),
        "runtime": runtime,
        "concurrency": concurrency,
        "accepted": not failures,
        "failures": failures,
        "request_throughput": _metric(data, "request_throughput"),
        "p95_ms": _metric(data, "request_latency", "p95"),
        "command": str(command_path),
    }


def validate_matrix(root: Path) -> list[dict[str, Any]]:
    results = [
        validate_result(path)
        for path in sorted(root.rglob("profile_export_aiperf.json"))
        if path.parents[1].name == RUNTIME
    ]
    expected = set(CONCURRENCIES)
    observed = {int(result["concurrency"]) for result in results}
    if observed != expected or len(results) != len(expected):
        raise AssertionError(
            f"expected {len(expected)} unique cells; found {len(results)}, "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    rejected = [result for result in results if not result["accepted"]]
    if rejected:
        details = "; ".join(
            f"concurrency{result['concurrency']}=" f"{','.join(result['failures'])}"
            for result in rejected
        )
        raise AssertionError(f"rejected benchmark artifacts: {details}")
    results.sort(key=lambda result: int(result["concurrency"]))
    validation_path = root / "validation.json"
    validation_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"BENCHMARK_AUDIT=PASS cells={len(results)} validation={validation_path}")
    return results


def _dispatch_counts(root: Path) -> dict[int, Counter[tuple[int, int]]]:
    log_path = root / "sweep.log"
    if not log_path.exists():
        return {}
    active: int | None = None
    counts: dict[int, Counter[tuple[int, int]]] = defaultdict(Counter)
    cell_pattern = re.compile(r"Config: .*\bconcurrency=(\d+)\b")
    dispatch_pattern = re.compile(
        r"custom_encoder_timing stage=vit_forward .*?"
        r"batch_size=(\d+) bucket=(\d+) cost=(\d+)"
    )
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        cell = cell_pattern.search(line)
        if cell:
            active = int(cell.group(1))
        dispatch = dispatch_pattern.search(line)
        if active is not None and dispatch:
            batch_size, bucket, _cost = (int(value) for value in dispatch.groups())
            counts[active][(batch_size, bucket)] += 1
    return dict(counts)


def _comparison_rows(root: Path) -> list[dict[str, Any]]:
    validated = validate_matrix(root)
    rows: list[dict[str, Any]] = []
    for result in validated:
        concurrency = int(result["concurrency"])
        triton_req_s, triton_p95 = TRITON_BASELINE[concurrency]
        dynamo_req_s = float(result["request_throughput"])
        dynamo_p95 = float(result["p95_ms"])
        path = Path(str(result["path"]))
        rows.append(
            {
                "concurrency": concurrency,
                "triton_req_s": triton_req_s,
                "triton_p95_ms": triton_p95,
                "dynamo_req_s": dynamo_req_s,
                "dynamo_p95_ms": dynamo_p95,
                "delta_req_s_pct": (dynamo_req_s / triton_req_s - 1.0) * 100.0,
                "delta_p95_pct": (dynamo_p95 / triton_p95 - 1.0) * 100.0,
                "artifact": str(path.relative_to(root)),
                "command": str((path.parent / "command.txt").relative_to(root)),
            }
        )
    return rows


def summarize(root: Path, markdown_path: Path, csv_path: Path) -> None:
    rows = _comparison_rows(root)
    metadata = json.loads(
        (root / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    settings = metadata["settings"]
    lines = [
        "# Qwen custom-encoder proxy versus supplied Triton baseline",
        "",
        "> **Performance proxy only:** Dynamo uses the Qwen2.5-VL-3B vision "
        "tower with its 2048-wide output truncated to 1536 columns for the "
        "Qwen2.5-1.5B decoder. The supplied Triton baseline used different "
        "encoder weights, so this table is not a same-model or quality comparison.",
        "",
        "Each Dynamo cell uses the same 100-request JSONL, nine reused 500×500 "
        "JPEGs, exact ISL 644, exact OSL 7, and 20 excluded warmups.",
        "",
        "## Runtime",
        "",
        f"- Dynamo commit: `{metadata['dynamo_commit']}`",
        f"- Container image: `{metadata['container_image']}`",
        f"- GPU: `{metadata['gpu']}`",
        f"- Decoder: `{metadata['decoder_model']}`",
        f"- Vision encoder: `{metadata['encoder_model']}`",
        f"- vLLM: `{metadata['vllm_version']}`; Transformers: "
        f"`{metadata['transformers_version']}`; PyTorch: "
        f"`{metadata['torch_version']}`; AIPerf: `{metadata['aiperf_version']}`",
        f"- Preprocess concurrency: {settings['preprocess_concurrency']}; "
        f"maximum batch cost: {settings['max_batch_cost']}; queue wait: "
        f"{settings['queue_wait_ms']} ms",
        f"- CUDA graph buckets: `{settings['graph_buckets']}`; image shape: "
        f"`{settings['graph_image_sizes']}`; preprocessing cache: disabled",
        "",
        "## Results",
        "",
        "| Concurrency | Triton req/s | Triton p95 (ms) | Dynamo req/s | "
        "Dynamo p95 (ms) | Δ req/s | Δ p95 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['concurrency']} | {row['triton_req_s']:.2f} | "
            f"{row['triton_p95_ms']:.1f} | {row['dynamo_req_s']:.2f} | "
            f"{row['dynamo_p95_ms']:.1f} | {row['delta_req_s_pct']:+.1f}% | "
            f"{row['delta_p95_pct']:+.1f}% |"
        )

    dispatch = _dispatch_counts(root)
    lines.extend(
        [
            "",
            "## Observed CUDA graph dispatch",
            "",
            "| Concurrency | Maximum batch | Selected buckets | Dispatches |",
            "| ---: | ---: | --- | --- |",
        ]
    )
    for concurrency in CONCURRENCIES:
        counter = dispatch.get(concurrency, Counter())
        maximum = max((batch for batch, _bucket in counter), default=0)
        buckets = sorted({bucket for _batch, bucket in counter})
        details = ", ".join(
            f"{batch}→{bucket}: {calls}"
            for (batch, bucket), calls in sorted(counter.items())
        )
        lines.append(
            f"| {concurrency} | {maximum} | `{buckets}` | {details or 'none'} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| Concurrency | AIPerf JSON | Exact command |",
            "| ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['concurrency']} | [artifact]({row['artifact']}) | "
            f"[command]({row['command']}) |"
        )
    lines.extend(
        [
            "",
            "- [Validation](validation.json)",
            "- [Workload manifest](../workload/workload_manifest.json)",
            "- [CUDA graph verification](graph_verification.log)",
            "- [Full sweep log](sweep.log)",
            "",
        ]
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"benchmark={markdown_path}")
    print(f"csv={csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--workload-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--concurrencies", type=int, nargs="+", default=list(CONCURRENCIES)
    )
    run.add_argument("--smoke", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("root", type=Path)
    report = subparsers.add_parser("summarize")
    report.add_argument("root", type=Path)
    report.add_argument("--markdown", type=Path, required=True)
    report.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_matrix(
            args.workload_dir.resolve(),
            args.output_dir.resolve(),
            tuple(args.concurrencies),
            smoke=args.smoke,
        )
    elif args.command == "validate":
        validate_matrix(args.root.resolve())
    else:
        summarize(args.root.resolve(), args.markdown.resolve(), args.csv.resolve())


if __name__ == "__main__":
    main()
