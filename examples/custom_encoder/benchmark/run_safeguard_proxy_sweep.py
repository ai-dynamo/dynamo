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
    DEFAULT_IMAGE_SIZE,
    ENCODER_MODEL,
    INPUT_NAME,
    REQUESTS,
    TARGET_ISL,
    TARGET_OSL,
)

RUNTIME_DELAYS_US = {
    "dynamo-custom-encoder-wait-0us": 0,
    "dynamo-custom-encoder-wait-1000us": 1000,
}
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


def _workload_image_size(workload_dir: Path) -> tuple[int, int]:
    manifest = json.loads(
        (workload_dir / "workload_manifest.json").read_text(encoding="utf-8")
    )
    decoded_image = manifest["decoded_image"]
    width = int(decoded_image["width"])
    height = int(decoded_image["height"])
    if width < 1 or height < 1:
        raise ValueError("workload image dimensions must be positive")
    return width, height


def _workload_unique_images(workload_dir: Path) -> int:
    manifest = json.loads(
        (workload_dir / "workload_manifest.json").read_text(encoding="utf-8")
    )
    unique_images = int(manifest["unique_images"])
    if unique_images < 1:
        raise ValueError("workload unique image count must be positive")
    return unique_images


def _metadata(
    concurrencies: tuple[int, ...],
    smoke: bool,
    workload_dir: Path,
) -> dict[str, Any]:
    manifest_path = workload_dir / "workload_manifest.json"
    width, height = _workload_image_size(workload_dir)
    unique_images = _workload_unique_images(workload_dir)
    return {
        "axis": "concurrency",
        "concurrencies": list(concurrencies),
        "runtimes": RUNTIME_DELAYS_US,
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
            "graph_image_sizes": [f"{width}x{height}"],
            "unique_images": unique_images,
            "preprocess_cache_size": 0,
            "batching_policy": (
                "drain immediately available work, then hold up to the configured "
                "deadline unless max_batch_cost is reached"
            ),
            "queue_delays_us": list(RUNTIME_DELAYS_US.values()),
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
    image_size: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
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
        configs=[
            BenchmarkConfig(
                label=label,
                workflow=str(workflow),
                extra_args=[
                    "--custom-encoder-max-queue-delay-us",
                    str(delay_us),
                ],
            )
            for label, delay_us in RUNTIME_DELAYS_US.items()
        ],
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
            "DYN_QWEN2_VL_GRAPH_IMAGE_SIZES": (f"{image_size[0]}x{image_size[1]}"),
            "DYN_QWEN2_VL_PREPROCESS_CACHE_SIZE": "0",
            "DYN_CUSTOM_ENCODER_DISPATCH_LOG": "1",
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
    image_size = _workload_image_size(workload_dir)
    config = build_config(
        workload_dir / INPUT_NAME,
        concurrencies,
        output_dir,
        smoke,
        image_size=image_size,
    )
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
    if runtime not in RUNTIME_DELAYS_US:
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
    if _metric(data, "time_to_first_token", "p95") is None:
        failures.append("time_to_first_token_p95")
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
        "ttft_p95_ms": _metric(data, "time_to_first_token", "p95"),
        "e2e_p95_ms": _metric(data, "request_latency", "p95"),
        "command": str(command_path),
    }


def validate_matrix(root: Path) -> list[dict[str, Any]]:
    results = [
        validate_result(path)
        for path in sorted(root.rglob("profile_export_aiperf.json"))
        if path.parents[1].name in RUNTIME_DELAYS_US
    ]
    expected = {
        (runtime, concurrency)
        for runtime in RUNTIME_DELAYS_US
        for concurrency in CONCURRENCIES
    }
    observed = {
        (str(result["runtime"]), int(result["concurrency"])) for result in results
    }
    if observed != expected or len(results) != len(expected):
        raise AssertionError(
            f"expected {len(expected)} unique cells; found {len(results)}, "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    rejected = [result for result in results if not result["accepted"]]
    if rejected:
        details = "; ".join(
            f"{result['runtime']}/concurrency{result['concurrency']}="
            f"{','.join(result['failures'])}"
            for result in rejected
        )
        raise AssertionError(f"rejected benchmark artifacts: {details}")
    results.sort(
        key=lambda result: (
            RUNTIME_DELAYS_US[str(result["runtime"])],
            int(result["concurrency"]),
        )
    )
    validation_path = root / "validation.json"
    validation_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"BENCHMARK_AUDIT=PASS cells={len(results)} validation={validation_path}")
    return results


DispatchKey = tuple[str, int, int | None]
CellKey = tuple[str, int]


def _dispatch_counts(root: Path) -> dict[CellKey, Counter[DispatchKey]]:
    log_path = root / "sweep.log"
    if not log_path.exists():
        return {}
    active: CellKey | None = None
    counts: dict[CellKey, Counter[DispatchKey]] = defaultdict(Counter)
    cell_pattern = re.compile(
        r"Config: (dynamo-custom-encoder-wait-(?:0|1000)us)\s+" r"concurrency=(\d+)\b"
    )
    dispatch_pattern = re.compile(
        r"custom_encoder_dispatch mode=(graph|eager) "
        r"batch_size=(\d+) bucket=(\d+|None)"
    )
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        cell = cell_pattern.search(line)
        if cell:
            active = (cell.group(1), int(cell.group(2)))
        dispatch = dispatch_pattern.search(line)
        if active is not None and dispatch:
            mode, batch_size_raw, bucket_raw = dispatch.groups()
            bucket = None if bucket_raw == "None" else int(bucket_raw)
            calls_match = re.search(r"\bcalls=(\d+)", line)
            counts[active][(mode, int(batch_size_raw), bucket)] += int(
                calls_match.group(1) if calls_match else 1
            )
    return dict(counts)


def _dispatch_rows(root: Path) -> list[dict[str, Any]]:
    counts = _dispatch_counts(root)
    expected_items = REQUESTS + WARMUP_REQUESTS
    rows: list[dict[str, Any]] = []
    for runtime, delay_us in RUNTIME_DELAYS_US.items():
        for concurrency in CONCURRENCIES:
            counter = counts.get((runtime, concurrency), Counter())
            total_calls = sum(counter.values())
            total_items = sum(
                batch_size * calls
                for (_mode, batch_size, _bucket), calls in counter.items()
            )
            if total_items != expected_items:
                raise AssertionError(
                    f"dispatch log for {runtime}/concurrency{concurrency} accounts "
                    f"for {total_items} items; expected {expected_items}"
                )
            eager_calls = sum(
                calls
                for (mode, _batch_size, _bucket), calls in counter.items()
                if mode == "eager"
            )
            if eager_calls:
                raise AssertionError(
                    f"dispatch log for {runtime}/concurrency{concurrency} contains "
                    f"{eager_calls} eager calls"
                )
            padded_slots = sum(
                (bucket or batch_size) * calls
                for (_mode, batch_size, bucket), calls in counter.items()
            )
            distribution = {
                f"{batch_size}->{bucket}": calls
                for (_mode, batch_size, bucket), calls in sorted(counter.items())
            }
            rows.append(
                {
                    "runtime": runtime,
                    "queue_delay_us": delay_us,
                    "concurrency": concurrency,
                    "forward_calls": total_calls,
                    "items": total_items,
                    "average_batch_size": total_items / total_calls,
                    "graph_utilization_pct": total_items / padded_slots * 100.0,
                    "distribution": distribution,
                }
            )
    (root / "dispatch_distribution.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def _comparison_rows(root: Path) -> list[dict[str, Any]]:
    validated = validate_matrix(root)
    by_cell = {
        (str(result["runtime"]), int(result["concurrency"])): result
        for result in validated
    }
    runtimes = list(RUNTIME_DELAYS_US)
    rows: list[dict[str, Any]] = []
    for concurrency in CONCURRENCIES:
        no_wait = by_cell[(runtimes[0], concurrency)]
        wait = by_cell[(runtimes[1], concurrency)]
        triton_req_s, triton_p95 = TRITON_BASELINE[concurrency]
        no_wait_req_s = float(no_wait["request_throughput"])
        wait_req_s = float(wait["request_throughput"])
        no_wait_ttft = float(no_wait["ttft_p95_ms"])
        wait_ttft = float(wait["ttft_p95_ms"])
        no_wait_e2e = float(no_wait["e2e_p95_ms"])
        wait_e2e = float(wait["e2e_p95_ms"])
        rows.append(
            {
                "concurrency": concurrency,
                "triton_req_s": triton_req_s,
                "triton_p95_ms": triton_p95,
                "no_wait_req_s": no_wait_req_s,
                "wait_1000us_req_s": wait_req_s,
                "delta_req_s_pct": (wait_req_s / no_wait_req_s - 1.0) * 100.0,
                "no_wait_ttft_p95_ms": no_wait_ttft,
                "wait_1000us_ttft_p95_ms": wait_ttft,
                "delta_ttft_p95_pct": (wait_ttft / no_wait_ttft - 1.0) * 100.0,
                "no_wait_e2e_p95_ms": no_wait_e2e,
                "wait_1000us_e2e_p95_ms": wait_e2e,
                "delta_e2e_p95_pct": (wait_e2e / no_wait_e2e - 1.0) * 100.0,
                "no_wait_artifact": str(Path(str(no_wait["path"])).relative_to(root)),
                "wait_1000us_artifact": str(Path(str(wait["path"])).relative_to(root)),
                "no_wait_command": str(
                    (Path(str(no_wait["path"])).parent / "command.txt").relative_to(
                        root
                    )
                ),
                "wait_1000us_command": str(
                    (Path(str(wait["path"])).parent / "command.txt").relative_to(root)
                ),
            }
        )
    return rows


def summarize(root: Path, markdown_path: Path, csv_path: Path) -> None:
    rows = _comparison_rows(root)
    dispatch_rows = _dispatch_rows(root)
    metadata = json.loads(
        (root / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    settings = metadata["settings"]
    image_shape = settings["graph_image_sizes"][0]
    unique_images = int(settings["unique_images"])
    workload_description = (
        f"all requests share one {image_shape.replace('x', '×')} JPEG"
        if unique_images == 1
        else f"requests reuse {unique_images} unique "
        f"{image_shape.replace('x', '×')} JPEGs"
    )
    baseline_workload_note = (
        " The supplied Triton baseline used nine unique images, unlike this "
        "single-image reuse run."
        if unique_images == 1
        else ""
    )
    lines = [
        "# Qwen custom-encoder queue-delay comparison",
        "",
        "> **Performance proxy only:** Dynamo uses the Qwen2.5-VL-3B vision "
        "tower with its 2048-wide output truncated to 1536 columns for the "
        "Qwen2.5-1.5B decoder. The supplied Triton baseline used different "
        "encoder weights, so this table is not a same-model or quality comparison."
        f"{baseline_workload_note}",
        "",
        f"Each Dynamo cell uses the same {REQUESTS:,}-request JSONL; "
        f"{workload_description}; ISL is exactly 644, OSL is exactly 7, and "
        f"{WARMUP_REQUESTS} warmups are excluded from performance metrics. "
        f"Dispatch distributions include all {REQUESTS + WARMUP_REQUESTS:,} "
        "forwarded items.",
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
        f"maximum batch cost: {settings['max_batch_cost']}; queue delays: "
        f"`{settings['queue_delays_us']}` microseconds",
        f"- Batching: {settings['batching_policy']}",
        f"- CUDA graph buckets: `{settings['graph_buckets']}`; image shape: "
        f"`{settings['graph_image_sizes']}`; preprocessing cache: disabled",
        "",
        "## Queue-delay results",
        "",
        "| Concurrency | 0 us req/s | 1000 us req/s | Δ req/s | "
        "0 us TTFT p95 | 1000 us TTFT p95 | Δ TTFT | 0 us E2E p95 | "
        "1000 us E2E p95 | Δ E2E |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['concurrency']} | {row['no_wait_req_s']:.2f} | "
            f"{row['wait_1000us_req_s']:.2f} | {row['delta_req_s_pct']:+.1f}% | "
            f"{row['no_wait_ttft_p95_ms']:.1f} ms | "
            f"{row['wait_1000us_ttft_p95_ms']:.1f} ms | "
            f"{row['delta_ttft_p95_pct']:+.1f}% | "
            f"{row['no_wait_e2e_p95_ms']:.1f} ms | "
            f"{row['wait_1000us_e2e_p95_ms']:.1f} ms | "
            f"{row['delta_e2e_p95_pct']:+.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Observed CUDA graph dispatch",
            "",
            "Distribution percentages are shares of forward calls and forwarded "
            "items, respectively.",
            "",
            "| Queue delay | Concurrency | Forward calls | Average batch | "
            "Graph utilization | Batch→bucket distribution |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for dispatch_row in dispatch_rows:
        total_calls = int(dispatch_row["forward_calls"])
        total_items = int(dispatch_row["items"])
        details = ", ".join(
            f"{key}: {calls} ({calls / total_calls * 100:.1f}% calls, "
            f"{int(key.split('->', 1)[0]) * calls / total_items * 100:.1f}% items)"
            for key, calls in dispatch_row["distribution"].items()
        )
        lines.append(
            f"| {dispatch_row['queue_delay_us']} us | "
            f"{dispatch_row['concurrency']} | {total_calls} | "
            f"{dispatch_row['average_batch_size']:.2f} | "
            f"{dispatch_row['graph_utilization_pct']:.1f}% | {details} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| Concurrency | 0 us artifacts | 1000 us artifacts |",
            "| ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['concurrency']} | "
            f"[AIPerf]({row['no_wait_artifact']}) / "
            f"[command]({row['no_wait_command']}) | "
            f"[AIPerf]({row['wait_1000us_artifact']}) / "
            f"[command]({row['wait_1000us_command']}) |"
        )
    lines.extend(
        [
            "",
            "- [Validation](validation.json)",
            "- [Dispatch distribution](dispatch_distribution.json)",
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
