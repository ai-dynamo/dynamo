# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit and summarize the remote Qwen workflow open-loop benchmark."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import re
import statistics
import urllib.request
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MEASURED_REQUESTS = 1000
WARMUP_REQUESTS = 20
REQUEST_RATE = 50
OUTPUT_TOKENS = 7
TEXT_TOKENS = 644
EXPECTED_AVERAGE_ISL = 874.5
EXPECTED_DECODER_ISLS = {"300x300": 773, "500x500": 976}
EXPECTED_MEASURED_SIZES = {"300x300": 500, "500x500": 500}
EXPECTED_WARMUP_SIZES = {"300x300": 10, "500x500": 10}
EXPECTED_MEASURED_SHA256 = (
    "743e859f895ee0e22df2476f74e5d3fa4d48db059273f5fe517634f31d9ef7cc"
)
EXPECTED_PATCH_COST = 907_800
EXPECTED_GRIDS = {"1x22x22", "1x36x36"}
EXPECTED_GRAPH_CAPTURES = 14
REPETITIONS = 1
TOPOLOGY = "remote"
MIN_ACHIEVED_TO_OFFERED_RATIO = 0.95
MAX_LAST_TO_FIRST_QUARTER_MEDIAN_RATIO = 1.20
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_DISPATCH_RE = re.compile(
    r"custom_encoder_dispatch mode=(?P<mode>\w+).*?patch_cost=(?P<patches>\d+)"
)
_GRID_RE = re.compile(r"\bgrid=(?P<grid>\d+x\d+x\d+)\b")
_CAPTURE_RE = re.compile(r"captured CUDA graph: grid=")
_CAPTURE_COMPLETE_RE = re.compile(r"CUDA graph capture complete: .*?graphs=(\d+)")
_PERF_RE = re.compile(r"\bworkflow_perf (?P<payload>\{.*\})$")


class BenchmarkAuditError(RuntimeError):
    """Raised when benchmark provenance or results drift from the contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkAuditError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise BenchmarkAuditError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _image_size(path: str) -> str:
    for size in EXPECTED_MEASURED_SIZES:
        if f"_{size}_" in Path(path).name:
            return size
    raise BenchmarkAuditError(f"cannot audit image size from {path}")


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
    expected_sizes: Mapping[str, int],
) -> dict[str, Any]:
    if len(rows) != expected_rows:
        raise BenchmarkAuditError(f"expected {expected_rows} rows, found {len(rows)}")
    if any(set(row) != {"session_id", "image", "text"} for row in rows):
        raise BenchmarkAuditError(
            "benchmark rows must contain session_id, image, and text"
        )

    prompts = {str(row["text"]) for row in rows}
    image_paths = [Path(str(row["image"])) for row in rows]
    if len(prompts) != 1:
        raise BenchmarkAuditError("every benchmark row must use one identical prompt")
    if len(set(image_paths)) != expected_rows:
        raise BenchmarkAuditError("every benchmark row must use a unique image")
    missing = next((path for path in image_paths if not path.is_file()), None)
    if missing is not None:
        raise BenchmarkAuditError(f"benchmark image is missing: {missing}")

    sizes = Counter(_image_size(str(path)) for path in image_paths)
    if sizes != Counter(expected_sizes):
        raise BenchmarkAuditError(
            f"image-size mix changed: {dict(sizes)}; expected {dict(expected_sizes)}"
        )
    return {
        "rows": len(rows),
        "unique_images": len(set(image_paths)),
        "image_size_counts": dict(sorted(sizes.items())),
        "prompt_sha256": hashlib.sha256(next(iter(prompts)).encode()).hexdigest(),
    }


def validate_workload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    measured = root / "measured" / "image_custom_1000_textisl644.jsonl"
    warmup = root / "warmup" / "image_custom_20_textisl644.jsonl"
    manifest_path = root / "workload_manifest.json"
    for path in (measured, warmup, manifest_path):
        if not path.is_file():
            raise BenchmarkAuditError(f"required workload artifact is missing: {path}")

    measured_sha256 = _sha256(measured)
    if measured_sha256 != EXPECTED_MEASURED_SHA256:
        raise BenchmarkAuditError(
            f"measured workload SHA-256 changed: {measured_sha256}; "
            f"expected {EXPECTED_MEASURED_SHA256}"
        )
    manifest = _read_json(manifest_path)
    expected_manifest_values = {
        "text_tokens": TEXT_TOKENS,
        "target_osl": OUTPUT_TOKENS,
        "decoder_isls_by_image_size": EXPECTED_DECODER_ISLS,
    }
    for key, expected in expected_manifest_values.items():
        if manifest.get(key) != expected:
            raise BenchmarkAuditError(
                f"workload manifest {key} changed: {manifest.get(key)!r}; "
                f"expected {expected!r}"
            )
    manifest_measured = manifest.get("measured")
    if not isinstance(manifest_measured, Mapping):
        raise BenchmarkAuditError("workload manifest measured entry is missing")
    if manifest_measured.get("sha256") != measured_sha256:
        raise BenchmarkAuditError("workload manifest measured SHA-256 is stale")

    measured_audit = _validate_rows(
        _read_jsonl(measured),
        expected_rows=MEASURED_REQUESTS,
        expected_sizes=EXPECTED_MEASURED_SIZES,
    )
    warmup_audit = _validate_rows(
        _read_jsonl(warmup),
        expected_rows=WARMUP_REQUESTS,
        expected_sizes=EXPECTED_WARMUP_SIZES,
    )
    if measured_audit["prompt_sha256"] != warmup_audit["prompt_sha256"]:
        raise BenchmarkAuditError("measured and warmup prompts differ")

    return {
        "root": str(root),
        "measured_path": str(measured),
        "warmup_path": str(warmup),
        "measured_sha256": measured_sha256,
        "text_token_definition": manifest.get("text_token_definition"),
        "text_tokens": TEXT_TOKENS,
        "target_osl": OUTPUT_TOKENS,
        "decoder_isls_by_image_size": EXPECTED_DECODER_ISLS,
        "measured": measured_audit,
        "warmup": warmup_audit,
        "prompt_sha256": measured_audit["prompt_sha256"],
    }


def audit_encoder_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    mode_counts: Counter[str] = Counter()
    grids: set[str] = set()
    patch_cost = 0
    for line in text.splitlines():
        dispatch = _DISPATCH_RE.search(line)
        if dispatch is None:
            continue
        mode_counts[dispatch.group("mode")] += 1
        patch_cost += int(dispatch.group("patches"))
        grid = _GRID_RE.search(line)
        if grid is not None:
            grids.add(grid.group("grid"))

    captured_graphs = len(_CAPTURE_RE.findall(text))
    capture_totals = [int(value) for value in _CAPTURE_COMPLETE_RE.findall(text)]
    if mode_counts.get("eager", 0):
        raise BenchmarkAuditError(
            f"encoder unexpectedly used eager dispatch {mode_counts['eager']} times"
        )
    if patch_cost != EXPECTED_PATCH_COST:
        raise BenchmarkAuditError(
            f"encoder processed {patch_cost} patches; expected {EXPECTED_PATCH_COST}"
        )
    if grids != EXPECTED_GRIDS:
        raise BenchmarkAuditError(
            f"encoder grids changed: {sorted(grids)}; expected {sorted(EXPECTED_GRIDS)}"
        )
    if captured_graphs != EXPECTED_GRAPH_CAPTURES or capture_totals != [
        EXPECTED_GRAPH_CAPTURES
    ]:
        raise BenchmarkAuditError(
            "encoder CUDA graph audit changed: "
            f"captured_lines={captured_graphs}, completion_totals={capture_totals}"
        )
    return {
        "dispatch_calls": sum(mode_counts.values()),
        "dispatch_modes": dict(sorted(mode_counts.items())),
        "patch_cost": patch_cost,
        "grids": sorted(grids),
        "captured_graphs": captured_graphs,
    }


def _metric_average(profile: Mapping[str, Any], key: str) -> float:
    metric = profile.get(key)
    if not isinstance(metric, Mapping) or "avg" not in metric:
        raise BenchmarkAuditError(f"AIPerf result is missing {key}.avg")
    return float(metric["avg"])


def validate_profile(path: Path, *, expected_requests: int) -> dict[str, Any]:
    profile = _read_json(path)
    errors = profile.get("error_summary")
    if errors not in ({}, []):
        raise BenchmarkAuditError(f"AIPerf reported errors: {errors!r}")
    if profile.get("was_cancelled") is not False:
        raise BenchmarkAuditError(
            "AIPerf run was cancelled or lacks cancellation state"
        )
    if _metric_average(profile, "request_count") != expected_requests:
        raise BenchmarkAuditError(
            f"AIPerf did not complete exactly {expected_requests} requests"
        )
    input_isl = _metric_average(profile, "input_sequence_length")
    output_isl = _metric_average(profile, "output_sequence_length")
    if not math.isclose(input_isl, EXPECTED_AVERAGE_ISL):
        raise BenchmarkAuditError(
            f"average decoder ISL is {input_isl}; expected {EXPECTED_AVERAGE_ISL}"
        )
    if output_isl != OUTPUT_TOKENS:
        raise BenchmarkAuditError(
            f"average output length is {output_isl}; expected {OUTPUT_TOKENS}"
        )
    latency = profile.get("request_latency")
    if not isinstance(latency, Mapping):
        raise BenchmarkAuditError("AIPerf result is missing request latency")
    latency_values = {
        key: float(latency[key]) for key in ("avg", "p50", "p95", "p99", "max")
    }
    return {
        "request_count": int(_metric_average(profile, "request_count")),
        "request_window_throughput_req_s": _metric_average(
            profile, "request_throughput"
        ),
        "output_token_throughput_tok_s": _metric_average(
            profile, "output_token_throughput"
        ),
        "request_latency_ms": latency_values,
        "input_sequence_length": input_isl,
        "output_sequence_length": output_isl,
        "errors": errors,
        "was_cancelled": False,
    }


def _parse_gpu_telemetry(path: Path) -> dict[str, Any]:
    utilization: list[float] = []
    memory: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 3:
            continue
        try:
            utilization.append(float(values[1]))
            memory.append(float(values[2]))
        except ValueError:
            continue
    if not utilization:
        raise BenchmarkAuditError(f"GPU telemetry has no samples: {path}")
    return {
        "samples": len(utilization),
        "utilization_percent_mean": statistics.mean(utilization),
        "utilization_percent_max": max(utilization),
        "memory_used_mib_mean": statistics.mean(memory),
        "memory_used_mib_max": max(memory),
    }


def audit_latency_stability(path: Path) -> dict[str, Any]:
    """Require the final request quarter not to accumulate an open-loop queue."""

    records = _read_jsonl(path)
    if len(records) != MEASURED_REQUESTS:
        raise BenchmarkAuditError(
            f"expected {MEASURED_REQUESTS} request records, found {len(records)}"
        )
    ordered = sorted(
        records,
        key=lambda record: int(record.get("metadata", {}).get("request_start_ns", 0)),
    )
    quarter = MEASURED_REQUESTS // 4

    def latency(record: Mapping[str, Any]) -> float:
        metrics = record.get("metrics")
        metric = (
            metrics.get("request_latency") if isinstance(metrics, Mapping) else None
        )
        if not isinstance(metric, Mapping) or "value" not in metric:
            raise BenchmarkAuditError("request record is missing request_latency")
        return float(metric["value"])

    first_median = statistics.median(latency(record) for record in ordered[:quarter])
    last_median = statistics.median(latency(record) for record in ordered[-quarter:])
    ratio = last_median / first_median if first_median else math.inf
    if ratio > MAX_LAST_TO_FIRST_QUARTER_MEDIAN_RATIO:
        raise BenchmarkAuditError(
            "request latency accumulated across the measured window: "
            f"first-quarter median={first_median:.3f} ms, "
            f"last-quarter median={last_median:.3f} ms, ratio={ratio:.3f}"
        )
    return {
        "first_quarter_median_ms": first_median,
        "last_quarter_median_ms": last_median,
        "last_to_first_ratio": ratio,
        "maximum_ratio": MAX_LAST_TO_FIRST_QUARTER_MEDIAN_RATIO,
        "passed": True,
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_perf_log(path: Path) -> dict[str, Any]:
    """Aggregate sampled workflow timing records from one measured window."""

    records: list[dict[str, Any]] = []
    malformed = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _PERF_RE.search(line)
        if match is None:
            continue
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("event"), str):
            malformed += 1
            continue
        records.append(payload)

    events: dict[str, dict[str, Any]] = {}
    for event in sorted({str(record["event"]) for record in records}):
        event_records = [record for record in records if record["event"] == event]
        numeric_fields: dict[str, dict[str, float | int]] = {}
        numeric_names = sorted(
            {
                key
                for record in event_records
                for key, value in record.items()
                if key not in {"event", "trace_id"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
        )
        for field_name in numeric_names:
            values = [
                float(record[field_name])
                for record in event_records
                if isinstance(record.get(field_name), (int, float))
                and not isinstance(record[field_name], bool)
            ]
            numeric_fields[field_name] = {
                "count": len(values),
                "mean": statistics.mean(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "p99": _percentile(values, 0.99),
                "max": max(values),
            }
        events[event] = {
            "records": len(event_records),
            "unique_trace_ids": len(
                {str(record.get("trace_id")) for record in event_records}
            ),
            "numeric_fields": numeric_fields,
        }
    return {
        "records": len(records),
        "malformed_records": malformed,
        "events": events,
    }


def validate_cell(
    profile_path: Path,
    records_path: Path,
    wall_path: Path,
    server_log: Path,
    perf_log: Path,
    gpu_telemetry: Path,
) -> dict[str, Any]:
    try:
        wall_seconds = float(wall_path.read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise BenchmarkAuditError(f"invalid client wall time in {wall_path}") from error
    if wall_seconds <= 0:
        raise BenchmarkAuditError("client wall time must be positive")
    server_text = server_log.read_text(encoding="utf-8", errors="replace")
    if "(use_kv_events=True)" not in server_text:
        raise BenchmarkAuditError("vLLM KV-event publication is not enabled")
    if "Worker reading KV events for dp_rank=" not in server_text:
        raise BenchmarkAuditError("Dynamo KV-event reader did not start")
    return {
        "full_client_process_wall_s": wall_seconds,
        "full_client_process_throughput_req_s": MEASURED_REQUESTS / wall_seconds,
        "aiperf": validate_profile(profile_path, expected_requests=MEASURED_REQUESTS),
        "latency_stability": audit_latency_stability(records_path),
        "encoder": audit_encoder_log(server_log),
        "kv_cache": {
            "prefix_caching": True,
            "event_publishing": True,
            "event_reader_started": True,
        },
        "gpu": _parse_gpu_telemetry(gpu_telemetry),
        "perf_trace": summarize_perf_log(perf_log),
    }


def _parse_gpu_info(gpu_info: str, torch_gpu_count: int) -> dict[str, Any]:
    values = [value.strip() for value in gpu_info.split(",")]
    if len(values) != 4:
        raise BenchmarkAuditError(f"unexpected nvidia-smi output: {gpu_info!r}")
    name = values[0]
    power_watts = float(values[1])
    max_sm_clock_mhz = int(values[2])
    memory_total_mib = int(values[3])
    if (
        "NVIDIA H100 80GB HBM3" not in name
        or power_watts < 650
        or max_sm_clock_mhz < 1900
    ):
        raise BenchmarkAuditError(f"unsupported benchmark GPU: {gpu_info}")
    if torch_gpu_count != 1:
        raise BenchmarkAuditError(
            f"expected one visible Torch GPU, found {torch_gpu_count}"
        )
    return {
        "name": name,
        "power_limit_watts": power_watts,
        "max_sm_clock_mhz": max_sm_clock_mhz,
        "memory_total_mib": memory_total_mib,
        "torch_device_count": torch_gpu_count,
    }


def capture_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if args.request_rate != REQUEST_RATE:
        raise BenchmarkAuditError(
            f"request rate is {args.request_rate}; expected {REQUEST_RATE}"
        )
    if not args.prefix_caching or not args.kv_event_publishing:
        raise BenchmarkAuditError(
            "qualification requires prefix caching and KV-event publication"
        )
    if args.embedding_transfer_mode != "nixl-write":
        raise BenchmarkAuditError("qualification requires nixl-write transfer mode")
    packages: dict[str, str] = {}
    for package in ("ai-dynamo", "aiperf", "torch", "transformers", "vllm"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    packages["aiperf"] = args.aiperf_version
    return {
        "dynamo_commit": args.source_commit,
        "dynamo_branch": args.source_branch,
        "working_diff_sha256": args.working_diff_sha256,
        "container_image": args.container_image,
        "cuda_visible_devices": args.cuda_visible_devices,
        "gpu": _parse_gpu_info(args.gpu_info, args.torch_gpu_count),
        "versions": packages,
        "topology_order_by_repetition": [
            [TOPOLOGY],
        ],
        "benchmark": {
            "load_mode": "constant",
            "repetitions": REPETITIONS,
            "warmup_requests": WARMUP_REQUESTS,
            "measured_requests": MEASURED_REQUESTS,
            "request_rate_mode": "constant",
            "request_rates": [args.request_rate],
            "concurrency": None,
            "topologies": [TOPOLOGY],
            "response_placement": "inline",
            "streaming": False,
            "max_tokens": OUTPUT_TOKENS,
            "min_tokens": OUTPUT_TOKENS,
            "ignore_eos": True,
            "max_num_seqs": 64,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.4,
            "prefix_caching": bool(args.prefix_caching),
            "kv_event_publishing": bool(args.kv_event_publishing),
            "kv_event_port": args.kv_event_port,
            "max_batch_patches": 41_472,
            "max_batch_items": 64,
            "batch_queue_wait_ms": args.batch_queue_wait_ms,
            "batch_queue_max_wait_ms": args.batch_queue_max_wait_ms,
            "embedding_transfer_mode": args.embedding_transfer_mode,
            "nixl_receive_storage": "pre-registered receiver ring buffer",
            "workflow_provider": args.workflow_provider,
            "perf_trace": bool(args.perf_trace),
            "perf_sample_every": args.perf_sample_every,
            "preprocess_concurrency": 64,
            "preprocess_cache_size": 0,
            "graph_batch_buckets": [1, 2, 4, 8, 16, 32, 64],
            "graph_image_sizes": ["300x300", "500x500"],
        },
    }


def smoke_joined_response(
    input_path: Path,
    *,
    endpoint: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    row = _read_jsonl(input_path)[0]
    image = Path(str(row["image"])).read_bytes()
    image_uri = "data:image/jpeg;base64," + base64.b64encode(image).decode()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": row["text"]},
                ],
            }
        ],
        "max_tokens": OUTPUT_TOKENS,
        "min_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "temperature": 0,
        "stream": False,
        "nvext": {"extra_fields": ["engine_data"]},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise BenchmarkAuditError("joined-response smoke returned a non-object")
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise BenchmarkAuditError("joined-response smoke returned no completion")
    usage = result.get("usage")
    if (
        not isinstance(usage, Mapping)
        or usage.get("completion_tokens") != OUTPUT_TOKENS
    ):
        raise BenchmarkAuditError("joined-response smoke returned the wrong OSL")
    return {
        "completion_tokens": usage["completion_tokens"],
        "finish_reason": choices[0].get("finish_reason"),
    }


def _sample_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _summarize_topology(root: Path, topology: str) -> dict[str, Any]:
    cells = [
        _read_json(root / f"rep-{repetition}" / topology / "cell_audit.json")
        for repetition in range(1, REPETITIONS + 1)
    ]
    walls = [float(cell["full_client_process_wall_s"]) for cell in cells]
    full_rates = [float(cell["full_client_process_throughput_req_s"]) for cell in cells]
    window_rates = [
        float(cell["aiperf"]["request_window_throughput_req_s"]) for cell in cells
    ]
    return {
        "runs": cells,
        "full_client_process_wall_s": {
            "runs": walls,
            "mean": statistics.mean(walls),
            "sample_stdev": _sample_stdev(walls),
        },
        "full_client_process_throughput_req_s": {
            "runs": full_rates,
            "from_mean_wall": MEASURED_REQUESTS / statistics.mean(walls),
            "mean_of_run_rates": statistics.mean(full_rates),
            "sample_stdev": _sample_stdev(full_rates),
        },
        "request_window_throughput_req_s": {
            "runs": window_rates,
            "mean": statistics.mean(window_rates),
            "sample_stdev": _sample_stdev(window_rates),
        },
        "latency_ms": {
            key: statistics.mean(
                float(cell["aiperf"]["request_latency_ms"][key]) for cell in cells
            )
            for key in ("avg", "p50", "p95", "p99", "max")
        },
        "gpu": {
            "utilization_percent_mean": statistics.mean(
                float(cell["gpu"]["utilization_percent_mean"]) for cell in cells
            ),
            "memory_used_mib_max": max(
                float(cell["gpu"]["memory_used_mib_max"]) for cell in cells
            ),
        },
    }


def summarize(root: Path) -> dict[str, Any]:
    metadata = _read_json(root / "benchmark_metadata.json")
    workload = _read_json(root / "workload_audit.json")
    benchmark = metadata.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise BenchmarkAuditError("benchmark metadata is missing benchmark settings")
    request_rates = benchmark.get("request_rates")
    if request_rates != [REQUEST_RATE] or benchmark.get("concurrency") is not None:
        raise BenchmarkAuditError(
            "summary requires 50 req/s constant load without a concurrency limit"
        )
    topology = _summarize_topology(root, TOPOLOGY)
    joined_smokes = [
        _read_json(root / f"rep-{repetition}" / "remote" / "joined_smoke.json")
        for repetition in range(1, REPETITIONS + 1)
    ]
    window = topology["request_window_throughput_req_s"]
    minimum_rate = REQUEST_RATE * MIN_ACHIEVED_TO_OFFERED_RATIO
    run_rates = [float(value) for value in window["runs"]]
    comparison = {
        "topology": TOPOLOGY,
        "offered_request_rate_req_s": REQUEST_RATE,
        "achieved_request_window_req_s": float(window["mean"]),
        "achieved_to_offered_ratio": float(window["mean"]) / REQUEST_RATE,
        "minimum_ratio": MIN_ACHIEVED_TO_OFFERED_RATIO,
        "minimum_rate_req_s": minimum_rate,
        "passed": all(value >= minimum_rate for value in run_rates),
    }
    summary = {
        "metadata": metadata,
        "workload": workload,
        "topologies": {TOPOLOGY: topology},
        "comparison": comparison,
        "gate": {
            "minimum_achieved_to_offered_ratio": MIN_ACHIEVED_TO_OFFERED_RATIO,
            "passed": comparison["passed"],
        },
        "joined_response_smokes": joined_smokes,
    }
    _write_json(root / "summary.json", summary)
    _write_report(root / "report.md", summary)
    return summary


def _format_runs(values: list[Any]) -> str:
    return ", ".join(f"{float(value):.3f}" for value in values)


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    metadata = summary["metadata"]
    workload = summary["workload"]
    topology = summary["topologies"][TOPOLOGY]
    comparison = summary["comparison"]
    lines = [
        "# Remote Qwen workflow open-loop qualification",
        "",
        "## Audited configuration",
        "",
        f"- Commit: `{metadata['dynamo_commit']}`",
        f"- Container: `{metadata['container_image']}`",
        f"- GPU: {metadata['gpu']}",
        f"- Workload SHA-256: `{workload['measured_sha256']}`",
        "- Raw text: 644 tokens plus one image; decoder ISL 773/976",
        "- OSL: 7; rate: 50 req/s; concurrency: unlimited; "
        "1,000 measured requests; 20 warmups",
        "- Response placement: inline",
        "- Non-streaming; TTFT and ITL are intentionally not compared",
        "- Tensor transport: "
        f"{metadata['benchmark']['embedding_transfer_mode']}; "
        f"{metadata['benchmark']['nixl_receive_storage']}",
        "- Encoder queue waits: "
        f"{metadata['benchmark']['batch_queue_wait_ms']} ms quiet / "
        f"{metadata['benchmark']['batch_queue_max_wait_ms']} ms maximum",
        "",
        "## Throughput",
        "",
        "| Topology | Full-process runs (req/s) | From mean wall | "
        "Request-window runs (req/s) | Window mean |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    full = topology["full_client_process_throughput_req_s"]
    window = topology["request_window_throughput_req_s"]
    lines.append(
        f"| {TOPOLOGY} | {_format_runs(full['runs'])} | "
        f"{full['from_mean_wall']:.3f} | {_format_runs(window['runs'])} | "
        f"{window['mean']:.3f} |"
    )
    lines.extend(
        [
            "",
            "Achieved/offered request-window ratio: "
            f"**{comparison['achieved_to_offered_ratio']:.3f}**; "
            f"every run >= {comparison['minimum_rate_req_s']:.3f} req/s: "
            f"**{comparison['passed']}**.",
            "",
            "Overall >=95% achieved/offered in every run gate: "
            f"**{summary['gate']['passed']}**.",
            "",
            "## Correctness",
            "",
            "- Every measured cell completed 1,000 requests with zero errors.",
            "- Every cell produced average decoder ISL 874.5 and exact OSL 7.",
            "- Every cell processed 907,800 patches across both audited grids.",
            "- Every remote repetition returned the exact seven-token completion.",
            "- Final-quarter median latency remained within 1.2× of the first quarter.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    workload = subparsers.add_parser("validate-workload")
    workload.add_argument("root", type=Path)
    workload.add_argument("--output", type=Path, required=True)

    metadata = subparsers.add_parser("capture-metadata")
    metadata.add_argument("--output", type=Path, required=True)
    metadata.add_argument("--source-commit", required=True)
    metadata.add_argument("--source-branch", required=True)
    metadata.add_argument("--working-diff-sha256", required=True)
    metadata.add_argument("--container-image", required=True)
    metadata.add_argument("--cuda-visible-devices", required=True)
    metadata.add_argument("--gpu-info", required=True)
    metadata.add_argument("--torch-gpu-count", type=int, required=True)
    metadata.add_argument("--request-rate", type=int, required=True)
    metadata.add_argument("--aiperf-version", required=True)
    metadata.add_argument("--prefix-caching", type=int, choices=(0, 1), required=True)
    metadata.add_argument(
        "--kv-event-publishing", type=int, choices=(0, 1), required=True
    )
    metadata.add_argument("--kv-event-port", type=int, required=True)
    metadata.add_argument("--batch-queue-wait-ms", type=float, required=True)
    metadata.add_argument("--batch-queue-max-wait-ms", type=float, required=True)
    metadata.add_argument("--embedding-transfer-mode", required=True)
    metadata.add_argument("--workflow-provider", required=True)
    metadata.add_argument("--perf-trace", type=int, choices=(0, 1), required=True)
    metadata.add_argument("--perf-sample-every", type=int, required=True)

    cell = subparsers.add_parser("validate-cell")
    cell.add_argument("--profile", type=Path, required=True)
    cell.add_argument("--records", type=Path, required=True)
    cell.add_argument("--wall-seconds", type=Path, required=True)
    cell.add_argument("--server-log", type=Path, required=True)
    cell.add_argument("--perf-log", type=Path, required=True)
    cell.add_argument("--gpu-telemetry", type=Path, required=True)
    cell.add_argument("--output", type=Path, required=True)

    perf = subparsers.add_parser("summarize-perf")
    perf.add_argument("--log", type=Path, required=True)
    perf.add_argument("--output", type=Path, required=True)

    profile = subparsers.add_parser("validate-profile")
    profile.add_argument("--profile", type=Path, required=True)
    profile.add_argument("--expected-requests", type=int, required=True)
    profile.add_argument("--output", type=Path, required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--input", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument(
        "--endpoint", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    smoke.add_argument("--model", default=MODEL)
    smoke.add_argument("--timeout", type=float, default=300)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("root", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "validate-workload":
        result = validate_workload(args.root)
        _write_json(args.output, result)
    elif args.command == "capture-metadata":
        if args.cuda_visible_devices != "0":
            raise BenchmarkAuditError(
                "benchmark requires CUDA_VISIBLE_DEVICES=0, got "
                f"{args.cuda_visible_devices!r}"
            )
        _write_json(args.output, capture_metadata(args))
    elif args.command == "validate-cell":
        _write_json(
            args.output,
            validate_cell(
                args.profile,
                args.records,
                args.wall_seconds,
                args.server_log,
                args.perf_log,
                args.gpu_telemetry,
            ),
        )
    elif args.command == "summarize-perf":
        _write_json(args.output, summarize_perf_log(args.log))
    elif args.command == "validate-profile":
        _write_json(
            args.output,
            validate_profile(
                args.profile,
                expected_requests=args.expected_requests,
            ),
        )
    elif args.command == "smoke":
        _write_json(
            args.output,
            smoke_joined_response(
                args.input,
                endpoint=args.endpoint,
                model=args.model,
                timeout=args.timeout,
            ),
        )
    elif args.command == "summarize":
        print(json.dumps(summarize(args.root), indent=2))
    else:  # pragma: no cover - argparse enforces the command choices.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
