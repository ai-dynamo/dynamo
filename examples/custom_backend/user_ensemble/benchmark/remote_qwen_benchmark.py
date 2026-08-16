# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit the direct-versus-workflow integrated-encoder benchmark."""

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
SUPPORTED_CONCURRENCY_CAPS = frozenset({64, 512})
SUPPORTED_LOAD_MODES = frozenset({"closed_loop", "constant"})
REQUEST_RATES = (40, 50)
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
REPETITIONS = 3
TOPOLOGIES = ("direct", "workflow")
MIN_WORKFLOW_TO_DIRECT_RATIO = 0.9
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_DISPATCH_RE = re.compile(
    r"custom_encoder_dispatch mode=(?P<mode>\w+).*?patch_cost=(?P<patches>\d+)"
)
_GRID_RE = re.compile(r"\bgrid=(?P<grid>\d+x\d+x\d+)\b")
_CAPTURE_RE = re.compile(r"captured CUDA graph: grid=")
_CAPTURE_COMPLETE_RE = re.compile(r"CUDA graph capture complete: .*?graphs=(\d+)")


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


def validate_cell(
    profile_path: Path,
    wall_path: Path,
    server_log: Path,
    gpu_telemetry: Path,
) -> dict[str, Any]:
    try:
        wall_seconds = float(wall_path.read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise BenchmarkAuditError(f"invalid client wall time in {wall_path}") from error
    if wall_seconds <= 0:
        raise BenchmarkAuditError("client wall time must be positive")
    return {
        "full_client_process_wall_s": wall_seconds,
        "full_client_process_throughput_req_s": MEASURED_REQUESTS / wall_seconds,
        "aiperf": validate_profile(profile_path, expected_requests=MEASURED_REQUESTS),
        "encoder": audit_encoder_log(server_log),
        "gpu": _parse_gpu_telemetry(gpu_telemetry),
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
    if args.load_mode not in SUPPORTED_LOAD_MODES:
        raise BenchmarkAuditError(
            f"unsupported load mode {args.load_mode!r}; expected one of "
            f"{sorted(SUPPORTED_LOAD_MODES)}"
        )
    if args.load_mode == "closed_loop" and args.concurrency != 64:
        raise BenchmarkAuditError(
            f"closed-loop concurrency is {args.concurrency}; expected 64"
        )
    if (
        args.load_mode == "constant"
        and args.concurrency not in SUPPORTED_CONCURRENCY_CAPS
    ):
        raise BenchmarkAuditError(
            f"unsupported concurrency cap {args.concurrency}; expected one of "
            f"{sorted(SUPPORTED_CONCURRENCY_CAPS)}"
        )
    packages: dict[str, str] = {}
    for package in ("ai-dynamo", "aiperf", "torch", "transformers", "vllm"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    packages["aiperf"] = args.aiperf_version
    benchmark: dict[str, Any] = {
        "load_mode": args.load_mode,
        "repetitions": REPETITIONS,
        "warmup_requests": WARMUP_REQUESTS,
        "measured_requests": MEASURED_REQUESTS,
        "request_rate_mode": args.load_mode,
        "concurrency": args.concurrency,
        "streaming": False,
        "max_tokens": OUTPUT_TOKENS,
        "min_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "max_num_seqs": 64,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.4,
        "max_batch_patches": 41_472,
        "max_batch_items": 64,
        "batching_policy": "block for first item, then eager-drain queued work",
        "preprocess_concurrency": 64,
        "preprocess_cache_size": 0,
        "graph_batch_buckets": [1, 2, 4, 8, 16, 32, 64],
        "graph_image_sizes": ["300x300", "500x500"],
    }
    if args.load_mode == "constant":
        benchmark["request_rates"] = list(REQUEST_RATES)
        benchmark["concurrency_cap"] = args.concurrency

    return {
        "dynamo_commit": args.source_commit,
        "dynamo_branch": args.source_branch,
        "working_diff_sha256": args.working_diff_sha256,
        "container_image": args.container_image,
        "cuda_visible_devices": args.cuda_visible_devices,
        "gpu": _parse_gpu_info(args.gpu_info, args.torch_gpu_count),
        "versions": packages,
        "topology_order_by_repetition": [
            ["direct", "workflow"],
            ["workflow", "direct"],
            ["direct", "workflow"],
        ],
        "benchmark": benchmark,
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
    nvext = result.get("nvext")
    engine_data = nvext.get("engine_data") if isinstance(nvext, Mapping) else None
    ensemble = engine_data.get("ensemble") if isinstance(engine_data, Mapping) else None
    scores = (
        ensemble.get("classifier_scores") if isinstance(ensemble, Mapping) else None
    )
    if not isinstance(scores, Mapping) or not scores:
        raise BenchmarkAuditError("joined response is missing classifier scores")
    score_values = [float(value) for value in scores.values()]
    if any(not math.isfinite(value) for value in score_values) or not math.isclose(
        sum(score_values), 1.0
    ):
        raise BenchmarkAuditError(f"invalid classifier scores: {dict(scores)}")
    return {
        "completion_tokens": usage["completion_tokens"],
        "classifier_scores": dict(scores),
        "classifier_score_sum": sum(score_values),
        "finish_reason": choices[0].get("finish_reason"),
    }


def _sample_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _summarize_topology(
    root: Path,
    topology: str,
    *,
    request_rate: int | None = None,
) -> dict[str, Any]:
    cell_root = (
        root / "closed-loop" if request_rate is None else root / f"rate-{request_rate}"
    )
    cells = [
        _read_json(cell_root / f"rep-{repetition}" / topology / "cell_audit.json")
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


def _compare_topologies(topologies: Mapping[str, Any]) -> dict[str, Any]:
    direct_window = topologies["direct"]["request_window_throughput_req_s"]["mean"]
    workflow_window = topologies["workflow"]["request_window_throughput_req_s"]["mean"]
    direct_full = topologies["direct"]["full_client_process_throughput_req_s"][
        "from_mean_wall"
    ]
    workflow_full = topologies["workflow"]["full_client_process_throughput_req_s"][
        "from_mean_wall"
    ]
    window_ratio = workflow_window / direct_window
    full_ratio = workflow_full / direct_full
    return {
        "workflow_to_direct_request_window_ratio": window_ratio,
        "workflow_request_window_delta_percent": (window_ratio - 1.0) * 100,
        "workflow_to_direct_full_process_ratio": full_ratio,
        "workflow_full_process_delta_percent": (full_ratio - 1.0) * 100,
        "minimum_ratio": MIN_WORKFLOW_TO_DIRECT_RATIO,
        "passed": window_ratio >= MIN_WORKFLOW_TO_DIRECT_RATIO,
    }


def summarize(root: Path) -> dict[str, Any]:
    metadata = _read_json(root / "benchmark_metadata.json")
    workload = _read_json(root / "workload_audit.json")
    benchmark = metadata.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise BenchmarkAuditError("benchmark metadata is missing benchmark settings")
    load_mode = str(benchmark.get("load_mode", "constant"))
    if load_mode == "closed_loop":
        topologies = {
            topology: _summarize_topology(root, topology) for topology in TOPOLOGIES
        }
        comparison = _compare_topologies(topologies)
        summary = {
            "metadata": metadata,
            "workload": workload,
            "topologies": topologies,
            "comparison": comparison,
            "gate": {
                "minimum_workflow_to_direct_ratio": MIN_WORKFLOW_TO_DIRECT_RATIO,
                "passed": comparison["passed"],
            },
            "joined_response_smokes": [
                _read_json(
                    root
                    / "closed-loop"
                    / f"rep-{repetition}"
                    / "workflow"
                    / "joined_smoke.json"
                )
                for repetition in range(1, REPETITIONS + 1)
            ],
        }
        _write_json(root / "summary.json", summary)
        _write_report(root / "report.md", summary)
        return summary
    if load_mode != "constant":
        raise BenchmarkAuditError(f"unsupported summary load mode: {load_mode!r}")

    rates: dict[str, Any] = {}
    joined_smokes: list[dict[str, Any]] = []
    gate_results = []
    for request_rate in REQUEST_RATES:
        topologies = {
            topology: _summarize_topology(
                root,
                topology,
                request_rate=request_rate,
            )
            for topology in TOPOLOGIES
        }
        comparison = _compare_topologies(topologies)
        gate_results.append(comparison["passed"])
        rates[str(request_rate)] = {
            "topologies": topologies,
            "comparison": comparison,
        }
        joined_smokes.extend(
            _read_json(
                root
                / f"rate-{request_rate}"
                / f"rep-{repetition}"
                / "workflow"
                / "joined_smoke.json"
            )
            for repetition in range(1, REPETITIONS + 1)
        )
    summary = {
        "metadata": metadata,
        "workload": workload,
        "rates": rates,
        "gate": {
            "minimum_workflow_to_direct_ratio": MIN_WORKFLOW_TO_DIRECT_RATIO,
            "passed": all(gate_results),
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
    benchmark = metadata["benchmark"]
    load_mode = benchmark.get("load_mode", "constant")
    lines = [
        "# Direct versus integrated-encoder workflow",
        "",
        "## Audited configuration",
        "",
        f"- Commit: `{metadata['dynamo_commit']}`",
        f"- Container: `{metadata['container_image']}`",
        f"- GPU: {metadata['gpu']}",
        f"- Workload SHA-256: `{workload['measured_sha256']}`",
        "- Raw text: 644 tokens plus one image; decoder ISL 773/976",
    ]
    if load_mode == "closed_loop":
        lines.append(
            f"- OSL: 7; load: closed-loop; concurrency: {benchmark['concurrency']}"
        )
    else:
        lines.append(
            "- OSL: 7; rates: 40/50 req/s; concurrency cap: "
            f"{benchmark['concurrency_cap']}"
        )
    lines.extend(
        [
            "- 1,000 measured requests; 20 warmups per cell",
            "- Non-streaming; TTFT and ITL are intentionally not compared",
            "",
            "## Throughput",
            "",
        ]
    )
    if load_mode == "closed_loop":
        lines.extend(
            [
                "| Topology | Full-process runs (req/s) | From mean wall | "
                "Request-window runs (req/s) | Window mean |",
                "| --- | --- | ---: | --- | ---: |",
            ]
        )
        for topology in TOPOLOGIES:
            topology_summary = summary["topologies"][topology]
            full = topology_summary["full_client_process_throughput_req_s"]
            window = topology_summary["request_window_throughput_req_s"]
            lines.append(
                f"| {topology} | {_format_runs(full['runs'])} | "
                f"{full['from_mean_wall']:.3f} | {_format_runs(window['runs'])} | "
                f"{window['mean']:.3f} |"
            )
        comparison = summary["comparison"]
        lines.extend(
            [
                "",
                "Closed-loop workflow/direct request-window ratio: "
                f"**{comparison['workflow_to_direct_request_window_ratio']:.3f}** "
                f"({comparison['workflow_request_window_delta_percent']:+.2f}%).",
                "Closed-loop workflow/direct full-process ratio: "
                f"**{comparison['workflow_to_direct_full_process_ratio']:.3f}** "
                f"({comparison['workflow_full_process_delta_percent']:+.2f}%).",
            ]
        )
    else:
        rates = summary["rates"]
        lines.extend(
            [
                "| Rate | Topology | Full-process runs (req/s) | From mean wall | "
                "Request-window runs (req/s) | Window mean |",
                "| ---: | --- | --- | ---: | --- | ---: |",
            ]
        )
        for request_rate in REQUEST_RATES:
            rate = rates[str(request_rate)]
            for topology in TOPOLOGIES:
                topology_summary = rate["topologies"][topology]
                full = topology_summary["full_client_process_throughput_req_s"]
                window = topology_summary["request_window_throughput_req_s"]
                lines.append(
                    f"| {request_rate} | {topology} | {_format_runs(full['runs'])} | "
                    f"{full['from_mean_wall']:.3f} | "
                    f"{_format_runs(window['runs'])} | {window['mean']:.3f} |"
                )
            comparison = rate["comparison"]
            lines.extend(
                [
                    "",
                    f"Rate {request_rate} workflow/direct request-window ratio: "
                    f"**{comparison['workflow_to_direct_request_window_ratio']:.3f}** "
                    f"({comparison['workflow_request_window_delta_percent']:+.2f}%).",
                ]
            )
    lines.extend(
        [
            "",
            f"Overall >=90% gate: **{summary['gate']['passed']}**.",
            "",
            "## Correctness",
            "",
            "- Every measured cell completed 1,000 requests with zero errors.",
            "- Every cell produced average decoder ISL 874.5 and exact OSL 7.",
            "- Every cell processed 907,800 patches across both audited grids.",
            "- Every workflow repetition returned normalized classifier scores.",
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
    metadata.add_argument(
        "--load-mode", choices=sorted(SUPPORTED_LOAD_MODES), required=True
    )
    metadata.add_argument("--concurrency", type=int, required=True)
    metadata.add_argument("--aiperf-version", required=True)

    cell = subparsers.add_parser("validate-cell")
    cell.add_argument("--profile", type=Path, required=True)
    cell.add_argument("--wall-seconds", type=Path, required=True)
    cell.add_argument("--server-log", type=Path, required=True)
    cell.add_argument("--gpu-telemetry", type=Path, required=True)
    cell.add_argument("--output", type=Path, required=True)

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
                args.wall_seconds,
                args.server_log,
                args.gpu_telemetry,
            ),
        )
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
