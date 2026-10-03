#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare one live AIPerf run of a cell with offline replay of the same cell and replicate.

Inputs: the generator output of the cell (``--inputs``), the AIPerf artifact directory
(``--live-run``), the cached replay record of the same (policy, cell, k) (``--replay-record``),
the idle calibration (``--idle-inputs`` and ``--idle-run``) and the payload's metric scrapes
(``--marks`` and ``--series``). Output: one JSON document with

- ``latency``: TTFT, per-request mean ITL and E2E quantiles, live and replay, for all completed
  requests and for the arrival window, plus per-request live/replay ratios;
- ``prefix_cache``: replay's reused-token fraction, the live per-request ``cached_tokens``
  fraction, and the vLLM ``prefix_cache_hits / prefix_cache_queries`` delta over the run;
- ``load_balance``: per-worker request, input-token and uncached-prefill shares (sorted, since
  live instance IDs and replay worker indices are different labels), and engine queue series;
- ``goodput``: the A2 windowed goodput of both sides under the AIS E0 and the live-fitted E0;
- ``schedule``: credit-issue lateness and HTTP-start lag against the generated schedule, with
  their drift over the run;
- ``idle_e0``: the live idle fit and its ratio to the AIS E0 at the calibration points.

It reads files only; it never contacts a server.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import score_live
from learned_routing import goodput

QUANTILES = (5, 10, 25, 50, 75, 90, 95, 99)
COUNTERS = (
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
)
GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
)


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dist(values: Sequence[float]) -> dict | None:
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return None
    out = {"n": len(values), "mean": statistics.fmean(values)}
    for q in QUANTILES:
        out[f"p{q}"] = goodput.percentile(values, q)
    out["max"] = max(values)
    return out


def latency_block(rows: Sequence[dict], window: tuple[float, float] | None) -> dict:
    done = [
        r
        for r in rows
        if goodput.completed(r)
        and (window is None or window[0] <= r["arrival_time_ms"] <= window[1])
    ]
    return {
        "completed": len(done),
        "ttft_ms": dist([r["ttft_ms"] for r in done]),
        "itl_mean_ms": dist([goodput.mean_itl_ms(r) for r in done]),
        "e2e_ms": dist([r["e2e_latency_ms"] for r in done]),
    }


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float | None:
    a, b = sorted(a), sorted(b)
    if not a or not b:
        return None
    i = j = 0
    best = 0.0
    while i < len(a) and j < len(b):
        x = min(a[i], b[j])
        while i < len(a) and a[i] <= x:
            i += 1
        while j < len(b) and b[j] <= x:
            j += 1
        best = max(best, abs(i / len(a) - j / len(b)))
    return best


def shares(counts: dict) -> list[float]:
    total = sum(counts.values())
    return sorted((v / total for v in counts.values()), reverse=True) if total else []


def balance(
    rows: Sequence[dict], worker_of: Callable[[dict], object], num_workers: int
) -> dict:
    requests: Counter = Counter()
    inputs: Counter = Counter()
    prefill: Counter = Counter()
    for row in rows:
        worker = worker_of(row)
        if worker is None:
            continue
        requests[worker] += 1
        inputs[worker] += row["input_length"]
        reused = row.get("reused_input_tokens")
        if reused is not None:
            prefill[worker] += max(row["input_length"] - reused, 0)
    out = {
        "workers_used": len(requests),
        "rows_with_worker": sum(requests.values()),
        "request_share_sorted": shares(requests),
        "input_token_share_sorted": shares(inputs),
        "uncached_prefill_share_sorted": shares(prefill) if prefill else None,
    }
    for name, counts in (("requests", requests), ("input_tokens", inputs)):
        values = [counts.get(w, 0) for w in counts] + [0] * (num_workers - len(counts))
        mean = statistics.fmean(values) if values else 0.0
        out[f"{name}_max_over_mean"] = max(values) / mean if mean else None
        out[f"{name}_cv"] = statistics.pstdev(values) / mean if mean else None
    return out


def best_relabel_agreement(live: Sequence, replay: Sequence) -> dict:
    """Fraction of requests on the same worker under the best live-to-replay relabeling."""
    pairs = [(a, b) for a, b in zip(live, replay) if a is not None and b is not None]
    labels = sorted({a for a, _ in pairs})
    targets = sorted({b for _, b in pairs})
    if not pairs or len(labels) > 8:
        return {"pairs": len(pairs), "agreement": None}
    best, mapping = -1, None
    for perm in itertools.permutations(targets, len(labels)):
        relabel = dict(zip(labels, perm))
        hits = sum(relabel[a] == b for a, b in pairs)
        if hits > best:
            best, mapping = hits, relabel
    first_diff = next(
        (i for i, (a, b) in enumerate(pairs) if mapping[a] != b), len(pairs)
    )
    return {
        "pairs": len(pairs),
        "agreement": best / len(pairs),
        "first_disagreement_index": first_diff,
        "mapping": {str(k): v for k, v in mapping.items()},
    }


def counter_deltas(marks: Sequence[dict], start_tag: str, end_tag: str) -> dict:
    by = {(m["tag"], m["target"]): m for m in marks if m.get("ok")}
    out: dict = {}
    for (tag, target), start in by.items():
        if tag != start_tag or (end_tag, target) not in by:
            continue
        end = by[(end_tag, target)]

        def total(samples: dict, name: str) -> float | None:
            hits = [v for k, v in samples.items() if k.split("{", 1)[0] == name]
            return sum(hits) if hits else None

        deltas = {}
        for name in COUNTERS:
            a, b = total(start["samples"], name), total(end["samples"], name)
            if a is not None and b is not None:
                deltas[name] = b - a
        out[target] = {
            "seconds": (end["t_unix_ns"] - start["t_unix_ns"]) / 1e9,
            **deltas,
        }
    return out


def gauge_series(series: Sequence[dict], t0_ns: int, t1_ns: int) -> dict:
    out: dict = defaultdict(lambda: defaultdict(list))
    for scrape in series:
        if not scrape.get("ok") or not t0_ns <= scrape["t_unix_ns"] <= t1_ns:
            continue
        for key, value in scrape["samples"].items():
            name = key.split("{", 1)[0]
            if name in GAUGES:
                out[scrape["target"]][name].append(value)
    return {
        target: {
            name: {
                "n": len(v),
                "mean": statistics.fmean(v),
                "p90": goodput.percentile(v, 90),
                "max": max(v),
            }
            for name, v in gauges.items()
        }
        for target, gauges in out.items()
    }


def schedule_drift(manifest: dict, requests: Sequence[dict], records: Sequence[dict]):
    """Credit-issue lateness by schedule decile (open loop)."""
    if not manifest["open_loop"]:
        return None
    expected = {(r["conversation_id"], r["turn_index"]): r for r in requests}
    pairs = []
    for record in records:
        key = score_live.record_key(record)
        request = expected.get(key)
        if request is None or request["arrival_ms"] is None:
            continue
        pairs.append((request["arrival_ms"], score_live._issue_ns(record["metadata"])))
    if not pairs:
        return None
    origin = min(issue - round(arrival * 1e6) for arrival, issue in pairs)
    lateness = sorted((a, (i - origin) / 1e6 - a) for a, i in pairs)
    n = len(lateness)
    deciles = []
    for d in range(10):
        chunk = [x for _, x in lateness[d * n // 10 : (d + 1) * n // 10]]
        if chunk:
            deciles.append(
                {
                    "from_ms": lateness[d * n // 10][0],
                    "p50": goodput.percentile(chunk, 50),
                    "p99": goodput.percentile(chunk, 99),
                    "max": max(chunk),
                }
            )
    values = [x for _, x in lateness]
    return {
        "deciles": deciles,
        "frac_over_10ms": sum(v > 10 for v in values) / n,
        "frac_over_100ms": sum(v > 100 for v in values) / n,
        "frac_over_1s": sum(v > 1000 for v in values) / n,
    }


def goodput_summary(metrics: dict) -> dict:
    keys = (
        "goodput_rps_window",
        "good_frac_window",
        "window_good",
        "window_requests",
        "goodput_rps",
        "good_frac",
        "completed_rows",
        "incomplete",
        "makespan_ms",
        "slowdown_p50",
        "slowdown_p90",
        "slowdown_p95",
        "ttft_p50",
        "ttft_p90",
        "itl_p50",
        "itl_p90",
    )
    out = {k: metrics.get(k) for k in keys}
    out["rescore_goodput_rps_window"] = {
        scale: v["goodput_rps_window"] for scale, v in metrics["rescore"].items()
    }
    out["guards"] = metrics.get("guards")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--live-run", type=Path, required=True)
    p.add_argument("--replay-record", type=Path, required=True)
    p.add_argument("--idle-inputs", type=Path, default=None)
    p.add_argument("--idle-run", type=Path, default=None)
    p.add_argument("--marks", type=Path, default=None)
    p.add_argument("--series", type=Path, default=None)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows-out", type=Path, default=None)
    p.add_argument("--e0-fit-out", type=Path, default=None)
    args = p.parse_args(argv)

    manifest, requests = score_live.load_inputs(args.inputs)
    replay_record = json.loads(args.replay_record.read_text())
    if replay_record.get("cell_id") != manifest["cell_id"] or int(
        replay_record["repeat"]
    ) != int(manifest["k"]):
        raise SystemExit("replay record does not belong to this cell and replicate")
    if replay_record.get("trace_sha256") != manifest["replicate"]["sha256"]:
        raise SystemExit("replay record was run on a different replicate trace")
    replay_rows = read_jsonl(Path(replay_record["per_request_path"]))

    # Live rows and the AIS-E0 score (the scorer also checks ISL/OSL/prompt/session fidelity).
    live_ais, live_rows = score_live.score_run(
        args.inputs, args.live_run, e0_mode="ais", root=args.root
    )
    raw = score_live.parse_raw(args.live_run / score_live.RAW_FILE)
    for row in live_rows:
        usage = (raw.get(tuple(row["_live_key"])) or {}).get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        row["reused_input_tokens"] = None if cached is None else int(cached)

    ais = score_live.ais_e0(requests, args.root)
    out: dict = {
        "schema": "learned-routing.live-smoke-compare.v1",
        "cell_id": manifest["cell_id"],
        "k": manifest["k"],
        "manifest_id": manifest["manifest_id"],
        "replay_record": {
            "path": str(args.replay_record),
            "cache_key": replay_record["cache_key"],
            "policy_name": replay_record.get("policy_name"),
            "policy_seed": replay_record.get("policy_seed"),
            "build_id": replay_record.get("build_id"),
        },
        "live_fidelity": {
            "fidelity_ok": live_ais["fidelity_ok"],
            **{
                k: live_ais["live"][k]
                for k in (
                    "records",
                    "matched",
                    "missing",
                    "statuses",
                    "isl_mismatch",
                    "osl_mismatch",
                    "prompt_checked",
                    "prompt_mismatch",
                    "usage_isl_mismatch",
                    "session_header_mismatch",
                    "workers_seen",
                )
            },
        },
    }

    # Replay rescored through the same scorer: must reproduce the cached record.
    replay_ais = score_live.score_rows(
        manifest, replay_rows, ais, warmup=score_live.warmup_ids(manifest)
    )
    out["replay_rescore_check"] = {
        "goodput_rps_window": replay_ais["goodput_rps_window"],
        "cached_goodput_rps_window": replay_record.get("goodput_rps_window"),
        "exact": replay_ais["goodput_rps_window"]
        == replay_record.get("goodput_rps_window"),
    }

    window = (
        live_ais["window_start_ms"],
        live_ais["window_end_ms"],
    )
    out["latency"] = {
        "window_ms": list(window),
        "live_all": latency_block(live_rows, None),
        "replay_all": latency_block(replay_rows, None),
        "live_window": latency_block(live_rows, window),
        "replay_window": latency_block(replay_rows, window),
        "ks_all": {
            name: ks_statistic(
                [f(r) for r in live_rows if goodput.completed(r)],
                [f(r) for r in replay_rows if goodput.completed(r)],
            )
            for name, f in (
                ("ttft_ms", lambda r: r["ttft_ms"]),
                ("e2e_ms", lambda r: r["e2e_latency_ms"]),
                ("itl_mean_ms", lambda r: goodput.mean_itl_ms(r) or 0.0),
            )
        },
        "per_request_live_over_replay": score_live.compare_with_replay(
            manifest, live_rows, replay_rows
        ),
    }

    # Prefix cache.
    def reuse(rows):
        known = [r for r in rows if r.get("reused_input_tokens") is not None]
        total = sum(r["input_length"] for r in known)
        return {
            "rows": len(known),
            "reused_tokens": sum(r["reused_input_tokens"] for r in known),
            "input_tokens": total,
            "fraction": sum(r["reused_input_tokens"] for r in known) / total
            if total
            else None,
        }

    prefix = {
        "replay_rows": reuse(replay_rows),
        "replay_record_prefix_reuse": replay_record.get("prefix_reuse"),
        "live_usage_cached_tokens": reuse(live_rows),
    }
    marks = read_jsonl(args.marks) if args.marks and args.marks.exists() else []
    deltas = counter_deltas(marks, "cell-start", "cell-end")
    if deltas:
        hits = sum(
            d.get("vllm:prefix_cache_hits_total", 0.0)
            for t, d in deltas.items()
            if t != "frontend"
        )
        queries = sum(
            d.get("vllm:prefix_cache_queries_total", 0.0)
            for t, d in deltas.items()
            if t != "frontend"
        )
        prefix["live_vllm_counters"] = {
            "hits": hits,
            "queries": queries,
            "hit_rate": hits / queries if queries else None,
            "note": "vLLM counts queried and hit prompt tokens per scheduling, including "
            "recomputation after preemption; replay counts reuse once per request",
        }
    out["prefix_cache"] = prefix
    out["engine_counters"] = deltas

    # Load balance.
    num_workers = int(manifest["num_workers"])
    out["load_balance"] = {
        "live": balance(live_rows, lambda r: r.get("_live_worker_id"), num_workers),
        "replay": balance(replay_rows, goodput.worker_of, num_workers),
        "routing_agreement_best_relabel": best_relabel_agreement(
            [r.get("_live_worker_id") for r in live_rows],
            [
                goodput.worker_of(r)
                for r in replay_rows_in_live_order(manifest, live_rows, replay_rows)
            ],
        ),
    }
    if args.series and args.series.exists() and marks:
        starts = [m["t_unix_ns"] for m in marks if m.get("tag") == "cell-start"]
        ends = [m["t_unix_ns"] for m in marks if m.get("tag") == "cell-end"]
        if starts and ends:
            out["load_balance"]["live_engine_gauges"] = gauge_series(
                read_jsonl(args.series), min(starts), max(ends)
            )

    # Goodput under both E0 modes.
    gp = {
        "ais_e0": {
            "live": goodput_summary(live_ais),
            "replay": goodput_summary(replay_ais),
        }
    }
    if args.idle_inputs and args.idle_run:
        idle_manifest, idle_requests = score_live.load_inputs(args.idle_inputs)
        fit = score_live.fit_live_e0(
            idle_manifest,
            idle_requests,
            read_jsonl(args.idle_run / score_live.RECORDS_FILE),
        )
        if args.e0_fit_out:
            args.e0_fit_out.write_text(
                json.dumps(fit, indent=1, default=score_live._json_default) + "\n"
            )
        live_e0 = score_live.LiveE0(fit)
        live_lr = score_live.score_rows(
            manifest, live_rows, live_e0, warmup=score_live.warmup_ids(manifest)
        )
        replay_lr = score_live.score_rows(
            manifest, replay_rows, live_e0, warmup=score_live.warmup_ids(manifest)
        )
        gp["live_e0"] = {
            "live": goodput_summary(live_lr),
            "replay": goodput_summary(replay_lr),
            "note": "both sides scored against the live idle fit; the headline pairing is "
            "live@live_e0 against replay@ais_e0 (each against its own engine's idle speed)",
        }
        idle_raw = args.idle_run / score_live.RAW_FILE
        idle_rows, _ = score_live.build_rows(
            idle_manifest,
            idle_requests,
            read_jsonl(args.idle_run / score_live.RECORDS_FILE),
            score_live.parse_raw(idle_raw) if idle_raw.exists() else None,
        )
        points = []
        for row in idle_rows:
            if not goodput.completed(row):
                continue
            isl, osl = row["input_length"], row["output_length"]
            points.append(
                {
                    "isl": isl,
                    "osl": osl,
                    "ttft_ms": row["ttft_ms"],
                    "itl_mean_ms": goodput.mean_itl_ms(row),
                    "e2e_ms": row["e2e_latency_ms"],
                    "ais_e0_ms": ais(isl, osl),
                    "live_over_ais_e0": row["e2e_latency_ms"] / ais(isl, osl),
                    "worker": row.get("_live_worker_id"),
                }
            )
        out["idle_e0"] = {
            "fit": {k: fit[k] for k in ("method", "prefill_points", "decode", "check")},
            "points": points,
        }
    for label, block in gp.items():
        live_w = block["live"]["goodput_rps_window"]
        replay_w = block["replay"]["goodput_rps_window"]
        block["live_over_replay_goodput_window"] = (
            live_w / replay_w if live_w is not None and replay_w else None
        )
    if "live_e0" in gp:
        live_w = gp["live_e0"]["live"]["goodput_rps_window"]
        replay_w = gp["ais_e0"]["replay"]["goodput_rps_window"]
        gp["own_reference_live_over_replay"] = (
            live_w / replay_w if live_w is not None and replay_w else None
        )
    out["goodput"] = gp

    out["schedule"] = {
        "schedule_lateness_ms": live_ais["live"]["schedule_lateness_ms"],
        "start_lag_ms": live_ais["live"]["start_lag_ms"],
        "drift": schedule_drift(
            manifest, requests, read_jsonl(args.live_run / score_live.RECORDS_FILE)
        ),
        "live_makespan_ms": live_ais["makespan_ms"],
        "replay_makespan_ms": replay_record.get("makespan_ms"),
    }
    if args.rows_out:
        score_live._write_rows(args.rows_out, live_rows)
    args.out.write_text(
        json.dumps(out, indent=1, sort_keys=True, default=score_live._json_default)
        + "\n"
    )
    print(
        json.dumps(
            {
                "fidelity_ok": out["live_fidelity"]["fidelity_ok"],
                "goodput_window": {
                    k: (
                        v["live"]["goodput_rps_window"],
                        v["replay"]["goodput_rps_window"],
                    )
                    for k, v in gp.items()
                    if isinstance(v, dict)
                },
            }
        )
    )
    return 0 if out["live_fidelity"]["fidelity_ok"] else 2


def replay_rows_in_live_order(
    manifest: dict, live_rows: Sequence[dict], replay_rows: Sequence[dict]
) -> list[dict]:
    """Replay rows paired to live rows by request identity (the ``compare`` join)."""

    def key_of(row: dict) -> tuple:
        if manifest["session_header"]:
            return (row["session_id"], row["turn_index"])
        return (
            row["arrival_time_ms"],
            row["input_length"],
            row["requested_output_length"],
        )

    pools: dict[tuple, list[dict]] = defaultdict(list)
    for row in replay_rows:
        pools[key_of(row)].append(row)
    paired = []
    for row in live_rows:
        pool = pools.get(key_of(row))
        paired.append(pool.pop(0) if pool else {"decode_worker_idx": None})
    return paired


if __name__ == "__main__":
    sys.exit(main())
