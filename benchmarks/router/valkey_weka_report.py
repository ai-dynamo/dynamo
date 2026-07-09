"""Summarize and render a Weka-derived in-process versus Valkey A/B."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any


SUMMARY_METRICS = {
    "request_throughput_rps": ("request_throughput_rps",),
    "request_latency_p50_ms": ("request_latency_ms", "p50"),
    "request_latency_p95_ms": ("request_latency_ms", "p95"),
    "request_latency_p99_ms": ("request_latency_ms", "p99"),
    "ttft_p50_ms": ("ttft_ms", "p50"),
    "ttft_p95_ms": ("ttft_ms", "p95"),
    "ttft_p99_ms": ("ttft_ms", "p99"),
    "itl_p50_ms": ("itl_ms", "p50"),
    "itl_p95_ms": ("itl_ms", "p95"),
    "itl_p99_ms": ("itl_ms", "p99"),
    "isl_avg_tokens": ("isl_tokens", "avg"),
    "osl_avg_tokens": ("osl_tokens", "avg"),
}


def metric(result: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = result["aiperf_metrics"]["records"]
    for field in path:
        value = value[field]
    return float(value)


def summarize(
    results: list[dict[str, Any]], schedule: list[dict[str, Any]]
) -> dict[str, Any]:
    passed = [result for result in results if result["status"] == "passed"]
    arms: dict[str, Any] = {}
    for arm in ("inprocess", "valkey_ha"):
        samples = [result for result in passed if result["arm"] == arm]
        arms[arm] = {
            "successful_runs": len(samples),
            "runs": [
                {
                    "sample": sample["sample"],
                    **{
                        name: metric(sample, path)
                        for name, path in SUMMARY_METRICS.items()
                    },
                }
                for sample in samples
            ],
            "median": {
                name: statistics.median(metric(sample, path) for sample in samples)
                for name, path in SUMMARY_METRICS.items()
            }
            if samples
            else {},
        }
    baseline = arms["inprocess"]["median"]
    candidate = arms["valkey_ha"]["median"]
    deltas = {
        name: (candidate[name] / baseline[name] - 1.0) * 100.0
        for name in SUMMARY_METRICS
        if baseline.get(name) not in (None, 0) and candidate.get(name) is not None
    }
    paired_throughput = _paired_throughput(passed)
    return {
        "valid": len(passed) == len(schedule),
        "planned_samples": len(schedule),
        "passed_samples": len(passed),
        "schedule": schedule,
        "arms": arms,
        "valkey_delta_percent": deltas,
        "paired_throughput": paired_throughput,
        "paired_throughput_median_delta_percent": (
            statistics.median(
                pair["valkey_delta_percent"] for pair in paired_throughput
            )
            if paired_throughput
            else None
        ),
        "cache": _cache_summary(passed),
        "validation": _validation_summary(passed),
    }


def _paired_throughput(passed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for repetition in sorted({result["repetition"] for result in passed}):
        controls = [
            result
            for result in passed
            if result["repetition"] == repetition and result["arm"] == "inprocess"
        ]
        candidates = [
            result
            for result in passed
            if result["repetition"] == repetition and result["arm"] == "valkey_ha"
        ]
        if len(controls) != 1 or len(candidates) != 1:
            continue
        control_rps = metric(controls[0], SUMMARY_METRICS["request_throughput_rps"])
        candidate_rps = metric(candidates[0], SUMMARY_METRICS["request_throughput_rps"])
        pairs.append(
            {
                "repetition": repetition,
                "inprocess_rps": control_rps,
                "valkey_ha_rps": candidate_rps,
                "valkey_delta_percent": (candidate_rps / control_rps - 1.0) * 100.0,
            }
        )
    return pairs


def _cache_summary(passed: list[dict[str, Any]]) -> dict[str, Any]:
    def cache_median(arm: str, name: str) -> float:
        values = [
            result["tokenizer_cache_metric_delta"][name]
            for result in passed
            if result["arm"] == arm
        ]
        return statistics.median(values) if values else 0.0

    return {
        "inprocess": {
            "l1_hits": cache_median(
                "inprocess", "dynamo_frontend_tokenizer_cache_hits_total"
            ),
            "l1_misses": cache_median(
                "inprocess", "dynamo_frontend_tokenizer_cache_misses_total"
            ),
        },
        "valkey_ha": {
            "l1_hits": cache_median(
                "valkey_ha", "dynamo_frontend_tokenizer_cache_hits_total"
            ),
            "l1_misses": cache_median(
                "valkey_ha", "dynamo_frontend_tokenizer_cache_misses_total"
            ),
            "l2_hits": cache_median(
                "valkey_ha", "dynamo_frontend_tokenizer_cache_l2_hits_total"
            ),
            "l2_misses": cache_median(
                "valkey_ha", "dynamo_frontend_tokenizer_cache_l2_misses_total"
            ),
            "l2_errors_total": _cache_total(
                passed, "dynamo_frontend_tokenizer_cache_l2_errors_total"
            ),
            "l2_write_errors_total": _cache_total(
                passed, "dynamo_frontend_tokenizer_cache_l2_write_errors_total"
            ),
        },
    }


def _cache_total(passed: list[dict[str, Any]], name: str) -> float:
    return sum(
        result["tokenizer_cache_metric_delta"][name]
        for result in passed
        if result["arm"] == "valkey_ha"
    )


def _validation_summary(passed: list[dict[str, Any]]) -> dict[str, int]:
    records = [result["aiperf_metrics"]["records"] for result in passed]
    return {
        "completed_records": sum(
            record["completed_profiling_records"] for record in records
        ),
        "request_errors": sum(
            record["errored_profiling_records"] for record in records
        ),
        "cancelled_requests": sum(
            record["cancelled_profiling_records"] for record in records
        ),
        "integrity_failure_markers": sum(
            result["log_validation"]["integrity_failure_markers"] for result in passed
        ),
    }


def write_report(path: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    control = summary["arms"]["inprocess"]["median"]
    valkey = summary["arms"]["valkey_ha"]["median"]
    delta = summary["valkey_delta_percent"]
    if not control or not valkey:
        path.write_text(
            "# Weka-Derived In-Process versus HA Valkey A/B\n\n"
            "Status: **invalid**\n\n"
            "The campaign did not produce at least one successful sample for each arm.\n"
        )
        return
    table = _metric_table(control, valkey, delta)
    pairs = _paired_table(summary["paired_throughput"])
    cache = summary["cache"]
    validation = summary["validation"]
    report = f"""# Weka-Derived In-Process versus HA Valkey A/B

Status: **{"passed" if summary["valid"] else "invalid"}**

The campaign alternated complete, fresh topologies over {args.runs} repetitions
per arm. Both arms used three frontends, four logical mock workers, TCP request
transport, synchronized frontend-local admission, the same release binary,
the same dataset, {args.concurrency} closed-loop concurrency, unlimited request
rate, {args.warmup_requests:,} warmup requests, and {args.requests:,} measured
requests per sample.

The control used an in-process router and a per-frontend tokenizer L1. The
candidate used a replicated router Valkey pair plus a separate replicated
tokenizer-cache pair, with both groups discovered through three shared
Sentinels.

## Median Results

{chr(10).join(table)}

Positive throughput delta is better; positive latency delta is worse.

## Per-Repetition Throughput

{chr(10).join(pairs)}

The median paired throughput delta was
{summary["paired_throughput_median_delta_percent"]:+.2f}%. The small absolute
difference should be treated as parity rather than proof of a material speedup.

## Cache and Integrity Validation

- In-process median tokenizer L1 hits/misses:
  {cache["inprocess"]["l1_hits"]:.0f} / {cache["inprocess"]["l1_misses"]:.0f}.
- HA Valkey median tokenizer L1 hits/misses and L2 hits/misses:
  {cache["valkey_ha"]["l1_hits"]:.0f} / {cache["valkey_ha"]["l1_misses"]:.0f}
  and {cache["valkey_ha"]["l2_hits"]:.0f} / {cache["valkey_ha"]["l2_misses"]:.0f}.
- Tokenizer L2 lookup/write errors: {cache["valkey_ha"]["l2_errors_total"]:.0f}
  / {cache["valkey_ha"]["l2_write_errors_total"]:.0f}.
- Completed records across all samples: {validation["completed_records"]:,};
  request errors/cancellations: {validation["request_errors"]} /
  {validation["cancelled_requests"]}.
- Direct-Valkey integrity failure markers: {validation["integrity_failure_markers"]}.

## Caveats

- CPU mock workers used a 100000 speedup ratio; this isolates frontend/router
  overhead and does not model GPU inference capacity.
- The Weka-derived workload caps ISL at 8,192 and OSL at 16 tokens and groups
  source 64-token hashes into stable 512-token hashes for AIPerf 0.10.
- This campaign did not kill a primary during measured traffic.
- Sub-millisecond ITL percentiles are quantized in this fast mocker setup; do
  not interpret their relative percentage changes as production token latency.
"""
    path.write_text(report)


def _metric_table(
    control: dict[str, float], valkey: dict[str, float], delta: dict[str, float]
) -> list[str]:
    rows = (
        ("Request throughput", "request_throughput_rps", "requests/s"),
        ("Request latency p50", "request_latency_p50_ms", "ms"),
        ("Request latency p95", "request_latency_p95_ms", "ms"),
        ("TTFT p50", "ttft_p50_ms", "ms"),
        ("TTFT p95", "ttft_p95_ms", "ms"),
        ("ITL p99", "itl_p99_ms", "ms"),
        ("ISL average", "isl_avg_tokens", "tokens"),
        ("OSL average", "osl_avg_tokens", "tokens"),
    )
    table = [
        "| Metric | In-process median | HA Valkey median | Valkey delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key, unit in rows:
        table.append(
            f"| {label} | {control[key]:.3f} {unit} | {valkey[key]:.3f} {unit} "
            f"| {delta.get(key, 0.0):+.2f}% |"
        )
    return table


def _paired_table(paired_throughput: list[dict[str, Any]]) -> list[str]:
    table = [
        "| Repetition | In-process RPS | HA Valkey RPS | Valkey delta |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for pair in paired_throughput:
        table.append(
            f"| {pair['repetition']} | {pair['inprocess_rps']:.3f} | "
            f"{pair['valkey_ha_rps']:.3f} | {pair['valkey_delta_percent']:+.2f}% |"
        )
    return table
