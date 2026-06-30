#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    rank = (len(values) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    return (
        values[lo] if lo == hi else values[lo] * (hi - rank) + values[hi] * (rank - lo)
    )


def wait_ready(base_url: str, model: str, timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=3) as response:
                ids = {item.get("id") for item in json.load(response).get("data", [])}
                if response.status == 200 and model in ids:
                    return
        except Exception as exc:
            last = exc
        time.sleep(1)
    raise RuntimeError(f"frontend not ready: {last}")


def nvext(args, trigger_sequence_length: int) -> dict:
    if args.mode == "baseline":
        return {
            "routing_constraints": {"required_taints": [args.baseline_taint]},
            "extra_fields": ["completion_token_ids"],
        }
    return {
        "decode_migration": {
            "source": {"required_taints": [args.source_taint]},
            "destination": {"required_taints": [args.destination_taint]},
            "trigger": {"type": "sequence_length", "tokens": trigger_sequence_length},
        },
        "extra_fields": ["completion_token_ids"],
    }


def request_shape(args, shape_index: int) -> tuple[int, int]:
    if args.osl_stddev == 0:
        output_tokens = args.max_tokens
    else:
        seed = (args.shape_seed << 32) ^ (shape_index & 0xFFFFFFFF)
        output_tokens = round(
            random.Random(seed).gauss(args.max_tokens, args.osl_stddev)
        )
    if args.osl_min is not None:
        output_tokens = max(args.osl_min, output_tokens)
    if args.osl_max is not None:
        output_tokens = min(args.osl_max, output_tokens)
    if output_tokens < 2:
        raise ValueError("Each request needs at least two output tokens")
    source_tokens = round(output_tokens * args.source_fraction)
    source_tokens = min(max(1, source_tokens), output_tokens - 1)
    return output_tokens, source_tokens


def stream_one(
    args,
    request_id: int,
    shape_index: int,
    scheduled_at: float,
    benchmark_started: float,
) -> dict:
    delay = scheduled_at - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    started = time.monotonic()
    host, port = args.base_url.removeprefix("http://").rsplit(":", 1)
    conn = http.client.HTTPConnection(host, int(port), timeout=900)
    output_tokens, source_tokens = request_shape(args, shape_index)
    first_visible_index = source_tokens + 1
    body = json.dumps(
        {
            "model": args.model,
            "prompt": [args.prompt_token_id],
            "temperature": 0,
            "seed": args.seed + request_id,
            "n": 1,
            "best_of": 1,
            "max_tokens": output_tokens,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "nvext": nvext(args, 1 + source_tokens),
        }
    )
    conn.request(
        "POST",
        "/v1/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    if response.status != 200:
        message = response.read().decode(errors="replace")
        conn.close()
        raise RuntimeError(f"HTTP {response.status}: {message}")

    token_count = 0
    source_boundary_arrival = None
    boundary_arrival = None
    first_token_arrival = None
    finish_reason = None
    usage = None
    token_id_chunks = 0
    fallback_chunks = 0
    completion_token_ids: list[int] = []
    while True:
        line = response.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            break
        chunk = json.loads(payload)
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        ids = (chunk.get("nvext") or {}).get("completion_token_ids") or []
        if ids:
            increment = len(ids)
            completion_token_ids.extend(ids)
            token_id_chunks += 1
        elif choice.get("finish_reason") is None:
            increment = 1
            fallback_chunks += 1
        else:
            increment = 0
        if increment:
            token_count += increment
            arrival = time.monotonic() - started
            if first_token_arrival is None:
                first_token_arrival = arrival
            if source_boundary_arrival is None and token_count >= source_tokens:
                source_boundary_arrival = arrival
            if boundary_arrival is None and token_count >= first_visible_index:
                boundary_arrival = arrival

    completed = time.monotonic()
    conn.close()
    completion_tokens = (usage or {}).get("completion_tokens", token_count)
    visible_tokens = output_tokens - source_tokens
    visible_duration = (
        (completed - started - boundary_arrival)
        if boundary_arrival is not None
        else None
    )
    visible_rate = (
        (visible_tokens - 1) / visible_duration
        if visible_duration is not None and visible_duration > 0 and visible_tokens > 1
        else 0.0
    )
    return {
        "request_id": request_id,
        "shape_index": shape_index,
        "output_tokens": output_tokens,
        "source_tokens": source_tokens,
        "destination_tokens": visible_tokens,
        "ok": completion_tokens == output_tokens and boundary_arrival is not None,
        "dispatch_offset_s": started - benchmark_started,
        "completion_offset_s": completed - benchmark_started,
        "ttft_s": first_token_arrival,
        "source_boundary_s": source_boundary_arrival,
        "ttfnt_s": boundary_arrival,
        "handoff_gap_s": (
            boundary_arrival - source_boundary_arrival
            if boundary_arrival is not None and source_boundary_arrival is not None
            else None
        ),
        "latency_s": completed - started,
        "visible_rate_tps": visible_rate,
        "completion_tokens": completion_tokens,
        "stream_counted_tokens": token_count,
        "token_id_chunks": token_id_chunks,
        "fallback_chunks": fallback_chunks,
        "finish_reason": finish_reason,
        "completion_token_ids": completion_token_ids,
        "token_fingerprint": (
            hashlib.sha256(
                b"".join(
                    int(token).to_bytes(4, "little") for token in completion_token_ids
                )
            ).hexdigest()
            if completion_token_ids
            else None
        ),
    }


def run(args) -> dict:
    interval = 1.0 / args.arrival_rate
    warmup_requests = math.ceil(args.arrival_rate * args.warmup_seconds)
    cooldown_requests = math.ceil(args.arrival_rate * args.cooldown_seconds)
    total_requests = warmup_requests + args.requests + cooldown_requests
    benchmark_started_wall = time.time()
    benchmark_started = time.monotonic()
    measurement_started = benchmark_started + warmup_requests * interval
    measurement_started_wall = benchmark_started_wall + warmup_requests * interval
    results = []
    max_workers = args.max_concurrency or total_requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                stream_one,
                args,
                args.request_offset + i,
                i - warmup_requests,
                benchmark_started + i * interval,
                measurement_started,
            ): args.request_offset
            + i
            for i in range(total_requests)
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "request_id": futures[future],
                        "ok": False,
                        "error": repr(exc),
                    }
                )

    first_measured_id = args.request_offset + warmup_requests
    first_cooldown_id = first_measured_id + args.requests
    warmup = [
        r for r in results if r.get("request_id", first_measured_id) < first_measured_id
    ]
    measured = [
        r
        for r in results
        if first_measured_id <= r.get("request_id", -1) < first_cooldown_id
    ]
    cooldown = [r for r in results if r.get("request_id", -1) >= first_cooldown_id]
    good = [r for r in measured if r.get("ok")]
    compliant = [r for r in good if r["visible_rate_tps"] >= args.min_visible_rate]
    ordered = sorted(compliant, key=lambda r: r["dispatch_offset_s"])
    half = len(ordered) // 2
    early = [r["ttfnt_s"] for r in ordered[:half]]
    late = [r["ttfnt_s"] for r in ordered[half:]]
    ttfnt = [r["ttfnt_s"] for r in compliant]
    source_boundaries = [r["source_boundary_s"] for r in compliant]
    handoff_gaps = [r["handoff_gap_s"] for r in compliant]
    visible_rates = [r["visible_rate_tps"] for r in compliant]
    wall = max((r.get("completion_offset_s", 0) for r in measured), default=0)
    last_dispatch = max((r.get("dispatch_offset_s", 0) for r in measured), default=0)
    output_lengths = [r["output_tokens"] for r in good]
    source_lengths = [r["source_tokens"] for r in good]
    destination_lengths = [r["destination_tokens"] for r in good]
    summary = {
        "run_label": args.run_label,
        "mode": args.mode,
        "benchmark_started_unix_s": benchmark_started_wall,
        "measurement_started_unix_s": measurement_started_wall,
        "measurement_ended_unix_s": measurement_started_wall + wall,
        "warmup_seconds": args.warmup_seconds,
        "warmup_requests": warmup_requests,
        "warmup_completed": sum(bool(r.get("ok")) for r in warmup),
        "cooldown_seconds": args.cooldown_seconds,
        "cooldown_requests": cooldown_requests,
        "cooldown_completed": sum(bool(r.get("ok")) for r in cooldown),
        "requests": args.requests,
        "completed": len(good),
        "slo_compliant": len(compliant),
        "arrival_rate_rps": args.arrival_rate,
        "gpu_count": args.gpu_count,
        "isl": 1,
        "osl_distribution": {
            "family": "fixed" if args.osl_stddev == 0 else "truncated_normal",
            "mean_parameter": args.max_tokens,
            "stddev_parameter": args.osl_stddev,
            "minimum": args.osl_min,
            "maximum": args.osl_max,
            "shape_seed": args.shape_seed,
            "sampled_mean": statistics.mean(output_lengths) if output_lengths else 0,
            "sampled_p05": percentile(output_lengths, 0.05),
            "sampled_p50": percentile(output_lengths, 0.5),
            "sampled_p95": percentile(output_lengths, 0.95),
        },
        "source_tokens_mean": (
            statistics.mean(source_lengths) if source_lengths else 0
        ),
        "destination_tokens_mean": (
            statistics.mean(destination_lengths) if destination_lengths else 0
        ),
        "source_fraction": args.source_fraction,
        "ttfnt_definition": (
            "arrival of output token round(source_fraction * request_osl) + 1"
        ),
        "measurement_wall_time_s": wall,
        "last_dispatch_offset_s": last_dispatch,
        "drain_time_s": max(0, wall - last_dispatch),
        "offered_goodput_rps": args.arrival_rate * len(compliant) / args.requests,
        "offered_goodput_per_gpu": args.arrival_rate
        * len(compliant)
        / args.requests
        / args.gpu_count,
        "achieved_throughput_rps": len(compliant) / wall,
        "achieved_throughput_per_gpu": len(compliant) / wall / args.gpu_count,
        "p50_source_boundary_s": percentile(source_boundaries, 0.5),
        "p95_source_boundary_s": percentile(source_boundaries, 0.95),
        "p50_handoff_gap_s": percentile(handoff_gaps, 0.5),
        "p95_handoff_gap_s": percentile(handoff_gaps, 0.95),
        "p50_ttfnt_s": percentile(ttfnt, 0.5),
        "p95_ttfnt_s": percentile(ttfnt, 0.95),
        "early_p50_ttfnt_s": percentile(early, 0.5),
        "late_p50_ttfnt_s": percentile(late, 0.5),
        "p50_ttfnt_drift_s": percentile(late, 0.5) - percentile(early, 0.5),
        "p50_visible_rate_tps": (
            statistics.median(visible_rates) if visible_rates else 0
        ),
        "min_visible_rate_tps": min(visible_rates) if visible_rates else 0,
        "min_visible_rate_gate_tps": args.min_visible_rate,
    }
    return {
        "summary": summary,
        "results": sorted(measured, key=lambda r: r.get("request_id", -1)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:18000")
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=("baseline", "migration"), required=True)
    p.add_argument("--run-label", default="")
    p.add_argument("--requests", type=int, default=64)
    p.add_argument(
        "--max-concurrency",
        type=int,
        help="Maximum number of in-flight requests (default: all requests).",
    )
    p.add_argument("--arrival-rate", type=float, required=True)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Fixed OSL, or the mean OSL when --osl-stddev is nonzero.",
    )
    p.add_argument("--osl-stddev", type=float, default=0)
    p.add_argument("--osl-min", type=int)
    p.add_argument("--osl-max", type=int)
    p.add_argument("--shape-seed", type=int, default=20260613)
    p.add_argument("--source-fraction", type=float, default=0.6)
    p.add_argument("--prompt-token-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260617)
    p.add_argument("--request-offset", type=int, default=0)
    p.add_argument("--warmup-seconds", type=float, default=0)
    p.add_argument("--cooldown-seconds", type=float, default=0)
    p.add_argument("--gpu-count", type=int, required=True)
    p.add_argument("--baseline-taint", default="decode/fast")
    p.add_argument("--source-taint", default="decode/fast")
    p.add_argument("--destination-taint", default="decode/slow")
    p.add_argument("--min-visible-rate", type=float, default=20)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 0 < args.source_fraction < 1:
        raise ValueError("source fraction must be between zero and one")
    if args.osl_stddev < 0:
        raise ValueError("osl stddev must be non-negative")
    if args.osl_min is not None and args.osl_max is not None:
        if args.osl_min > args.osl_max:
            raise ValueError("osl minimum cannot exceed osl maximum")
    wait_ready(args.base_url, args.model)
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
