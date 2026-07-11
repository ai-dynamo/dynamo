# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf command construction, validation, and compact artifact capture."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, TypedDict

from .artifacts import file_digest, write_json_atomic
from .cluster import (
    Cluster,
    registered_mocker_stats,
    verify_active_images,
    wait_for_ha,
)
from .ha import ha_snapshot, valkey_counter_deltas, valkey_telemetry
from .model import ISL, MODEL, OSL, MatrixPoint, request_count, warmup_count


class RuntimeTopology(TypedDict):
    runtime_namespace: str
    attempt_generation: str
    registered_index_stats: list[int]
    frontend_pods: list[str]
    frontend_urls: list[str]
    router_reset: dict[str, list[str]]
    tokenizer_reset: dict[str, list[str]]
    pre_runtime_snapshot: dict[str, dict[str, Any]]
    pre_ha_snapshot: dict[str, Any]
    load_generator_capacity: dict[str, int | str]


def benchmark_contract(point: MatrixPoint, urls: list[str]) -> dict[str, Any]:
    requests = request_count(point.concurrency)
    return {
        "warmup_concurrency": point.concurrency,
        "warmup_requests": warmup_count(point.concurrency),
        "profiling_concurrency": point.concurrency,
        "profiling_requests": requests,
        "dataset_entries": requests,
        "frontend_urls": urls,
        "model": MODEL,
        "tokenizer": MODEL,
        "isl": float(ISL),
        "osl": float(OSL),
        "dataset_name": "main",
        "dataset_type": "synthetic",
        "random_seed": 100,
        "isl_stddev": 0.0,
        "osl_stddev": 0.0,
        "warmup_type": "concurrency",
        "warmup_excluded": True,
        "profiling_type": "concurrency",
        "endpoint_type": "chat",
        "streaming": True,
        "timeout": 300.0,
        "url_strategy": "round_robin",
        "extra": {
            "ignore_eos": True,
            "max_tokens": OSL,
            "min_tokens": OSL,
            "repetition_penalty": 1.0,
            "temperature": 0.0,
        },
        "raw_artifacts": False,
        "artifact_records": ["jsonl"],
        "gpu_telemetry": False,
        "server_metrics": False,
    }


def aiperf_command(point: MatrixPoint, urls: list[str], remote_dir: str) -> list[str]:
    contract = benchmark_contract(point, urls)
    command = [
        "/opt/aiperf-venv/bin/aiperf",
        "profile",
        "--artifact-dir",
        remote_dir,
        "--model",
        str(contract["model"]),
        "--tokenizer",
        str(contract["tokenizer"]),
        "--endpoint-type",
        str(contract["endpoint_type"]),
        "--streaming",
        "--url-strategy",
        str(contract["url_strategy"]),
    ]
    for url in urls:
        command.extend(("--url", url))
    command.extend(
        (
            "--synthetic-input-tokens-mean",
            str(int(contract["isl"])),
            "--synthetic-input-tokens-stddev",
            "0",
            "--output-tokens-mean",
            str(int(contract["osl"])),
            "--output-tokens-stddev",
            "0",
            "--extra-inputs",
            f"max_tokens:{contract['extra']['max_tokens']}",
            "--extra-inputs",
            f"min_tokens:{contract['extra']['min_tokens']}",
            "--extra-inputs",
            "ignore_eos:true",
            "--extra-inputs",
            "repetition_penalty:1.0",
            "--extra-inputs",
            "temperature:0.0",
            "--concurrency",
            str(contract["profiling_concurrency"]),
            "--request-count",
            str(contract["profiling_requests"]),
            "--num-dataset-entries",
            str(contract["dataset_entries"]),
            "--warmup-request-count",
            str(contract["warmup_requests"]),
            "--request-timeout-seconds",
            str(int(contract["timeout"])),
            "--workers-max",
            "64",
            "--record-processors",
            "32",
            "--random-seed",
            str(contract["random_seed"]),
            "--export-level",
            "records",
            "--no-gpu-telemetry",
            "--no-server-metrics",
            "--ui",
            "simple",
        )
    )
    return command


def metric(metrics: dict[str, Any], name: str, statistic: str = "avg") -> float | None:
    value = metrics.get(name)
    if not isinstance(value, dict):
        return None
    number = value.get(statistic)
    return float(number) if isinstance(number, (int, float)) else None


def error_count(metrics: dict[str, Any]) -> float | None:
    explicit = metric(metrics, "error_request_count")
    if explicit is not None:
        return explicit
    summary = metrics.get("error_summary")
    if isinstance(summary, list):
        return float(len(summary))
    return None


def validate_input_config(
    metrics: dict[str, Any], point: MatrixPoint, frontend_urls: list[str] | None = None
) -> None:
    try:
        config = metrics["input_config"]
        phases = {phase["name"]: phase for phase in config["phases"]}
        warmup = phases["warmup"]
        profiling = phases["profiling"]
        dataset = config["datasets"][0]
        endpoint = config["endpoint"]
        model = config["models"]["items"][0]["name"]
        tokenizer = config["tokenizer"]["name"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            f"AIPerf summary has malformed input_config: {error}"
        ) from error
    expected_requests = request_count(point.concurrency)
    expected = benchmark_contract(point, frontend_urls or endpoint.get("urls", []))
    observed = {
        "warmup_concurrency": warmup.get("concurrency"),
        "warmup_requests": warmup.get("requests"),
        "profiling_concurrency": profiling.get("concurrency"),
        "profiling_requests": profiling.get("requests"),
        "dataset_entries": dataset.get("entries"),
        "frontend_urls": endpoint.get("urls", []),
        "model": model,
        "tokenizer": tokenizer,
        "isl": dataset.get("prompts", {}).get("isl", {}).get("mean"),
        "osl": dataset.get("prompts", {}).get("osl", {}).get("mean"),
        "dataset_name": dataset.get("name"),
        "dataset_type": dataset.get("type"),
        "random_seed": dataset.get("random_seed"),
        "isl_stddev": dataset.get("prompts", {}).get("isl", {}).get("stddev"),
        "osl_stddev": dataset.get("prompts", {}).get("osl", {}).get("stddev"),
        "warmup_type": warmup.get("type"),
        "warmup_excluded": warmup.get("exclude_from_results"),
        "profiling_type": profiling.get("type"),
        "endpoint_type": endpoint.get("type"),
        "streaming": endpoint.get("streaming"),
        "timeout": endpoint.get("timeout"),
        "url_strategy": endpoint.get("url_strategy"),
        "extra": endpoint.get("extra"),
        "raw_artifacts": config.get("artifacts", {}).get("raw"),
        "artifact_records": config.get("artifacts", {}).get("records"),
        "gpu_telemetry": config.get("gpu_telemetry", {}).get("enabled"),
        "server_metrics": config.get("server_metrics", {}).get("enabled"),
    }
    if (
        metrics.get("aiperf_version") != "0.10.0"
        or metrics.get("schema_version") != "1.3"
        or observed != expected
    ):
        raise RuntimeError(
            f"AIPerf summary input_config differs from requested point: "
            f"expected={expected}, observed={observed}, "
            f"version={metrics.get('aiperf_version')!r}"
        )
    distributions = (
        "request_latency",
        "time_to_first_token",
        "inter_token_latency",
        "input_sequence_length",
        "output_sequence_length",
        "osl_mismatch_diff_pct",
    )
    counts = {name: metrics.get(name, {}).get("count") for name in distributions}
    if any(count != expected_requests for count in counts.values()):
        raise RuntimeError(
            f"AIPerf distribution counts differ from measured requests: "
            f"expected={expected_requests}, observed={counts}"
        )


def validate_metrics(
    metrics: dict[str, Any], point: MatrixPoint, frontend_urls: list[str] | None = None
) -> dict[str, float]:
    validate_input_config(metrics, point, frontend_urls)
    expected_requests = request_count(point.concurrency)
    required = {
        "completed_requests": metric(metrics, "request_count"),
        "error_requests": error_count(metrics),
        "request_throughput_rps": metric(metrics, "request_throughput"),
        "request_latency_p50_ms": metric(metrics, "request_latency", "p50"),
        "request_latency_p99_ms": metric(metrics, "request_latency", "p99"),
        "ttft_p50_ms": metric(metrics, "time_to_first_token", "p50"),
        "ttft_p99_ms": metric(metrics, "time_to_first_token", "p99"),
        "itl_p50_ms": metric(metrics, "inter_token_latency", "p50"),
        "itl_p99_ms": metric(metrics, "inter_token_latency", "p99"),
        "actual_isl_avg": metric(metrics, "input_sequence_length"),
        "actual_osl_avg": metric(metrics, "output_sequence_length"),
        "osl_mismatch_diff_pct_avg": metric(metrics, "osl_mismatch_diff_pct"),
        "output_token_throughput": metric(metrics, "output_token_throughput"),
    }
    missing_or_nonfinite = {
        name: value
        for name, value in required.items()
        if value is None or not math.isfinite(value)
    }
    if missing_or_nonfinite:
        raise RuntimeError(
            f"AIPerf summary has invalid required metrics: {missing_or_nonfinite}"
        )
    values = {
        name: float(value) for name, value in required.items() if value is not None
    }
    if (
        values["completed_requests"] != expected_requests
        or values["error_requests"] != 0
    ):
        raise RuntimeError(
            f"invalid AIPerf completion counts: completed={values['completed_requests']}, "
            f"expected={expected_requests}, errors={values['error_requests']}"
        )
    positive = (
        "request_throughput_rps",
        "request_latency_p50_ms",
        "request_latency_p99_ms",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "itl_p50_ms",
        "itl_p99_ms",
        "output_token_throughput",
    )
    invalid_positive = {name: values[name] for name in positive if values[name] <= 0}
    if invalid_positive:
        raise RuntimeError(f"AIPerf metrics must be positive: {invalid_positive}")
    isl_tolerance = max(8.0, ISL * 0.02)
    if abs(values["actual_isl_avg"] - ISL) > isl_tolerance:
        raise RuntimeError(
            f"actual_isl_avg={values['actual_isl_avg']} differs from configured "
            f"{ISL} by more than {isl_tolerance} tokens"
        )
    osl_lower = OSL - max(8.0, OSL * 0.02)
    osl_upper = OSL + max(32.0, OSL * 0.10)
    if not osl_lower <= values["actual_osl_avg"] <= osl_upper:
        raise RuntimeError(
            f"actual_osl_avg={values['actual_osl_avg']} is outside the calibrated "
            f"AIPerf chat range [{osl_lower}, {osl_upper}] for configured OSL={OSL}"
        )
    if values["osl_mismatch_diff_pct_avg"] > 10.0:
        raise RuntimeError(
            "AIPerf average OSL mismatch exceeds 10%: "
            f"{values['osl_mismatch_diff_pct_avg']}"
        )
    if metrics.get("was_cancelled") is not False:
        raise RuntimeError("AIPerf summary reports a cancelled or unknown run")
    return values


def capture_aiperf_summary(
    cluster: Cluster, remote_dir: str, local_dir: Path, *, required: bool
) -> dict[str, Any] | None:
    """Copy the compact aggregate summary before deleting remote record artifacts."""
    summary_name = "profile_export_aiperf.json"
    summary = cluster.client_exec(
        ("cat", f"{remote_dir}/{summary_name}"), timeout=60, check=False
    )
    if summary.returncode != 0 or not summary.stdout.strip():
        if required:
            raise RuntimeError("AIPerf did not produce an aggregate summary")
        return None
    summary_path = local_dir / summary_name
    summary_path.write_text(summary.stdout, encoding="utf-8")
    try:
        return json.loads(summary.stdout)
    except json.JSONDecodeError as error:
        if required:
            raise RuntimeError("AIPerf aggregate summary is not JSON") from error
        return None


def run_point(
    cluster: Cluster,
    campaign: str,
    manifest_digest: str,
    point: MatrixPoint,
    topology: RuntimeTopology,
    local_dir: Path,
    binding: dict[str, Any],
) -> dict[str, Any]:
    remote_dir = (
        f"/data/valkey-router-sweep/{campaign}/{point.slug}/"
        f"attempt-{topology['attempt_generation']}"
    )
    command = aiperf_command(point, topology["frontend_urls"], remote_dir)
    pre_valkey = valkey_telemetry(cluster, topology["pre_ha_snapshot"])
    started = time.time()
    failure: BaseException | None = None
    try:
        execution = cluster.client_exec(command, timeout=7200, check=False)
        measured_seconds = time.time() - started
        (local_dir / "aiperf.stdout.log").write_text(
            execution.stdout + execution.stderr, encoding="utf-8"
        )
        post_valkey = valkey_telemetry(cluster, topology["pre_ha_snapshot"])
        if execution.returncode != 0:
            capture_aiperf_summary(cluster, remote_dir, local_dir, required=False)
            raise RuntimeError(
                f"aiperf failed for {point.slug} with exit code {execution.returncode}"
            )
        wait_for_ha(cluster)
        post_runtime_snapshot = verify_active_images(cluster, binding, point)
        post_ha_snapshot = ha_snapshot(cluster)
        post_registered_index_stats = registered_mocker_stats(
            cluster, topology["runtime_namespace"]
        )
        if post_runtime_snapshot != topology["pre_runtime_snapshot"]:
            raise RuntimeError(
                f"pod topology or restart counts changed during {point.slug}: "
                f"before={topology['pre_runtime_snapshot']}, after={post_runtime_snapshot}"
            )
        if post_ha_snapshot != topology["pre_ha_snapshot"]:
            raise RuntimeError(
                f"HA topology changed during {point.slug}: "
                f"before={topology['pre_ha_snapshot']}, after={post_ha_snapshot}"
            )
        if (
            len(post_registered_index_stats) != 3
            or post_registered_index_stats[1] != point.mockers
        ):
            raise RuntimeError(
                f"mocker registration changed during {point.slug}: "
                f"before={topology['registered_index_stats']}, "
                f"after={post_registered_index_stats}"
            )

        metrics = capture_aiperf_summary(cluster, remote_dir, local_dir, required=True)
        assert metrics is not None
        summary_digest = file_digest(local_dir / "profile_export_aiperf.json")
        validated = validate_metrics(metrics, point, topology["frontend_urls"])
        result = {
            "status": "ok",
            "campaign": campaign,
            "manifest_digest": manifest_digest,
            "point": point._asdict(),
            "provenance": binding,
            "slug": point.slug,
            "offered_request_rate": "inf",
            "request_count": request_count(point.concurrency),
            "warmup_request_count": warmup_count(point.concurrency),
            "model": MODEL,
            "configured_isl": ISL,
            "configured_osl": OSL,
            "runtime_namespace": topology["runtime_namespace"],
            "attempt_generation": topology["attempt_generation"],
            "remote_artifact_dir": remote_dir,
            "remote_artifacts_retained": False,
            "aiperf_summary_digest": summary_digest,
            "registered_index_stats": topology["registered_index_stats"],
            "post_registered_index_stats": post_registered_index_stats,
            "frontend_pods": topology["frontend_pods"],
            "frontend_urls": topology["frontend_urls"],
            "pre_runtime_snapshot": topology["pre_runtime_snapshot"],
            "post_runtime_snapshot": post_runtime_snapshot,
            "pre_ha_snapshot": topology["pre_ha_snapshot"],
            "post_ha_snapshot": post_ha_snapshot,
            "load_generator_capacity": topology["load_generator_capacity"],
            "router_reset": topology["router_reset"],
            "tokenizer_reset": topology["tokenizer_reset"],
            "elapsed_seconds": time.time() - started,
            "measurement_seconds": measured_seconds,
            "metrics": validated,
            "valkey": {
                "pre": pre_valkey,
                "post": post_valkey,
                "counter_deltas": valkey_counter_deltas(pre_valkey, post_valkey),
            },
        }
        result["valkey"]["post"]["dynamo-tokenizer"]["dbsize"] = cluster.valkey(
            post_ha_snapshot["dynamo-tokenizer"]["sentinel_master"][0], "DBSIZE"
        )
        write_json_atomic(local_dir / "result.json", result)
        return result
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            cluster.client_exec(("rm", "-rf", "--", remote_dir), timeout=300)
        except Exception as cleanup_error:
            if failure is None:
                raise
            failure.add_note(f"failed to clean AIPerf artifacts: {cleanup_error!r}")
