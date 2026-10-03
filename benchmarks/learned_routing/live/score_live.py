#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score an AIPerf run of a live-lane input with the harness's A2 scorer (CONTRACT A2, A13).

The adapter rebuilds replay-shaped per-request rows (:data:`learned_routing.goodput.COMPACT_FIELDS`)
from AIPerf's per-request export (``profile_export.jsonl``). It then calls the harness's own
:func:`learned_routing.goodput.compute_metrics`, so the definitions of "good", the windows, the
warm-up, the occupancy rule and the SLO rescoring are the harness's code, not a copy.

**Identity.** Every AIPerf ``conversation_id`` is replay's session key (``gen_aiperf_inputs``), and
``turn_index`` is the turn. Each record therefore maps to exactly one generated request.
Unmatched, duplicate and missing records are counted; a missing request counts as an in-window
miss, as in the harness.

**Time base.** The live origin is the instant that corresponds to replay time 0.

- Open loop: the origin is the earliest ``credit_issued_ns - timestamp`` over first turns. AIPerf's
  scheduler can only be late, so this is the tightest estimate. A first turn's
  ``arrival_time_ms`` is its generated timestamp, which is bit-identical to replay's, so window
  membership by arrival is decided by identity.
- Closed loop: the origin is the first credit issue.
- Every other arrival (later turns, closed loop) is the credit issue time relative to the origin:
  the instant AIPerf decided the request was ready, the analogue of replay's ready time.

``ttft_ms`` and ``e2e_latency_ms`` are AIPerf's ``time_to_first_token`` and ``request_latency``,
measured from the HTTP request start. ``terminal_time_ms`` is the wall-clock ``request_end_ns``,
on the same clock as the credit issues, so a closed-loop slot handoff never shows more than ``C``
occupied slots. ``first_admit_ms`` is set to the arrival, because live has no admission event;
the scorer only tests whether it is present.

**Windows, warm-up, CRN.** Everything comes from the cell, copied into the generator's manifest:

- ``measure`` (arrival basis with ``warmup_ms``/``window_ms`` in replay time, or completion basis
  with ``warmup_trace_ms`` identity warm-up and the ``full_occupancy`` end);
- the occupancy cap (``load.value``);
- replicate ``k`` and its policy seed ``k + 1``.

The warm-up session IDs come from the same replicate trace, through
:func:`learned_routing.goodput.warmup_ids_from_trace`.

**E0 (the E2E slowdown reference).**

- ``--e0 ais`` (default): the harness's AIS table (:mod:`learned_routing.e0`,
  ``ais-chunked-estimator-v2``), the reference the simulator uses. The generator precomputes it
  per request. An (ISL, OSL) pair the live run produced but the generator did not, for example
  after a truncated output, falls back to the AIS table itself.
- ``--e0 live``: a live-fitted idle E0 from ``fit-e0`` over an idle calibration run
  (``gen_aiperf_inputs.py idle``). TTFT(ISL) is piecewise-linear through the per-ISL median idle
  TTFTs. The batch-1 decode step is linear in KV context, ``d0 + d1 * ctx``, least squares over
  per-request mean ITLs at mean context ``ISL + OSL/2 + 1``. E0 follows replay's convention,
  ``TTFT(ISL) + sum_{j=0}^{OSL-2} step(ISL + j + 2)``. Live latencies include the frontend and
  transport, which the AIS table does not model; the live E0 does include them.

Subcommands: ``score`` (one run), ``fit-e0`` (idle calibration), and ``compare``, which joins live
rows with a replay ``per_request`` file of the same cell and replicate.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from learned_routing import goodput

SCHEMA = "learned-routing.live-score.v1"
SCORER_VERSION = "lr-live-score-v1"
E0_LIVE_METHOD = "live-idle-v1"
RECORDS_FILE = "profile_export.jsonl"
RAW_FILE = "profile_export_raw.jsonl"
UNIT_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3, "sec": 1e3}


class ScoreError(ValueError):
    pass


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_inputs(inputs_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((inputs_dir / "manifest.json").read_text())
    requests = read_jsonl(inputs_dir / "requests.jsonl")
    if len(requests) != manifest["num_requests"]:
        raise ScoreError(
            f"requests.jsonl has {len(requests)} rows, manifest says {manifest['num_requests']}"
        )
    return manifest, requests


def metric_value(record: dict, tag: str, unit: str = "ms") -> float | None:
    """A metric of an AIPerf record in ``unit`` (``ms``) or as a count (``unit=None``)."""
    entry = (record.get("metrics") or {}).get(tag)
    if entry is None:
        return None
    if not isinstance(entry, dict):
        return float(entry)
    value = entry.get("value")
    if value is None:
        return None
    if unit is None:
        return float(value)
    scale = UNIT_TO_MS.get(str(entry.get("unit", "ms")).lower())
    if scale is None:
        raise ScoreError(f"metric {tag}: unknown unit {entry.get('unit')!r}")
    return float(value) * scale


def record_key(record: dict) -> tuple[str, int]:
    meta = record.get("metadata") or {}
    conversation = meta.get("conversation_id")
    turn = meta.get("turn_index")
    if conversation is None or turn is None:
        raise ScoreError(f"record without conversation_id/turn_index: {meta}")
    return str(conversation), int(turn)


def _token_digest(tokens: Sequence[int]) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(tokens, dtype="<u4").tobytes()).hexdigest()


def _walk(value, found: dict) -> None:
    if isinstance(value, dict):
        worker = value.get("worker_id")
        if isinstance(worker, dict):
            for key in ("decode_worker_id", "prefill_worker_id"):
                if worker.get(key) is not None:
                    found.setdefault(key, worker[key])
        usage = value.get("usage")
        if isinstance(usage, dict):
            found["usage"] = usage
        for item in value.values():
            _walk(item, found)
    elif isinstance(value, list):
        for item in value:
            _walk(item, found)


def parse_raw(path: Path) -> dict[tuple[str, int], dict]:
    """Per request from ``profile_export_raw.jsonl``: prompt digest, usage, worker IDs, status."""
    out: dict[tuple[str, int], dict] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record_key(record)
            info: dict = {"status": record.get("status")}
            payload = record.get("payload") or {}
            prompt = payload.get("prompt")
            if isinstance(prompt, list):
                info["prompt_sha256"] = _token_digest(prompt)
                info["prompt_tokens_sent"] = len(prompt)
            info["max_tokens"] = payload.get("max_tokens")
            info["min_tokens"] = payload.get("min_tokens")
            info["ignore_eos"] = payload.get("ignore_eos")
            info["session_header"] = next(
                (
                    value
                    for name, value in (record.get("request_headers") or {}).items()
                    if name.lower() == "x-dynamo-session-id"
                ),
                None,
            )
            found: dict = {}
            for response in record.get("responses") or []:
                for packet in response.get("packets") or []:
                    if packet.get("name") != "data":
                        continue
                    text = packet.get("value") or ""
                    if not text.startswith("{"):
                        continue
                    try:
                        _walk(json.loads(text), found)
                    except json.JSONDecodeError:
                        continue
                if "text" in response:
                    try:
                        _walk(json.loads(response["text"]), found)
                    except (json.JSONDecodeError, TypeError):
                        pass
            info.update(found)
            if key in out:
                raise ScoreError(f"duplicate raw record {key}")
            out[key] = info
    return out


# --------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------


def _quantiles(values: Sequence[float]) -> dict | None:
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "p50": goodput.percentile(values, 50),
        "p99": goodput.percentile(values, 99),
        "max": max(values),
    }


def _issue_ns(meta: dict) -> int:
    issued = meta.get("credit_issued_ns")
    return int(issued if issued is not None else meta["request_start_ns"])


def build_rows(
    manifest: dict,
    requests: Sequence[dict],
    records: Sequence[dict],
    raw: dict[tuple[str, int], dict] | None = None,
) -> tuple[list[dict], dict]:
    """Replay-shaped rows (module docstring) and the live-fidelity diagnostics."""
    by_key: dict[tuple[str, int], dict] = {}
    duplicates = unmatched = warmup_records = 0
    expected = {(r["conversation_id"], r["turn_index"]): r for r in requests}
    for record in records:
        meta = record.get("metadata") or {}
        if str(meta.get("benchmark_phase", "profiling")).lower() != "profiling":
            warmup_records += 1
            continue
        key = record_key(record)
        if key not in expected:
            unmatched += 1
            continue
        if key in by_key:
            duplicates += 1
            continue
        by_key[key] = record
    if duplicates or unmatched:
        raise ScoreError(
            f"{duplicates} duplicate and {unmatched} unmatched AIPerf records: the export does not "
            "belong to this input"
        )
    if warmup_records:
        raise ScoreError(
            f"{warmup_records} non-profiling records: run without an AIPerf warm-up"
        )

    open_loop = bool(manifest["open_loop"])
    if not by_key:
        raise ScoreError("no AIPerf records matched the input")
    if open_loop:
        offsets = [
            _issue_ns(by_key[key]["metadata"]) - round(request["arrival_ms"] * 1e6)
            for key, request in expected.items()
            if key in by_key and request["arrival_ms"] is not None
        ]
        if not offsets:
            raise ScoreError("open loop: no first-turn record to anchor the origin")
        origin = min(offsets)
    else:
        origin = min(_issue_ns(record["metadata"]) for record in by_key.values())

    emit_session = bool(manifest["session_header"])
    rows: list[dict] = []
    lateness, start_lag, think_residual = [], [], []
    isl_mismatch = (
        osl_mismatch
    ) = prompt_mismatch = prompt_checked = usage_isl_mismatch = 0
    header_mismatch = 0
    statuses: Counter = Counter()
    workers: dict[tuple[str, int], int | None] = {}
    previous_end: dict[tuple[str, int], float] = {}
    for request in requests:
        key = (request["conversation_id"], request["turn_index"])
        record = by_key.get(key)
        if record is None:
            continue
        meta = record["metadata"]
        issue_ms = (_issue_ns(meta) - origin) / 1e6
        start_ms = (int(meta["request_start_ns"]) - origin) / 1e6
        end_ms = (int(meta["request_end_ns"]) - origin) / 1e6
        if request["arrival_ms"] is not None:
            arrival = request["arrival_ms"]
            lateness.append(issue_ms - arrival)
            start_lag.append(start_ms - arrival)
        else:
            arrival = issue_ms
        previous = previous_end.get(
            (request["conversation_id"], request["turn_index"] - 1)
        )
        if previous is not None:
            think_residual.append(issue_ms - previous - (request["delay_ms"] or 0.0))
        previous_end[key] = end_ms

        ttft = metric_value(record, "time_to_first_token")
        e2e = metric_value(record, "request_latency")
        osl = metric_value(record, "output_sequence_length", unit=None)
        isl = metric_value(record, "input_sequence_length", unit=None)
        error = record.get("error")
        if meta.get("was_cancelled"):
            status = "cancelled"
        elif error is not None:
            status = "error"
        elif ttft is None or e2e is None or osl is None:
            status = "incomplete"
        else:
            status = "completed"
        statuses[status] += 1
        if isl is not None and int(isl) != request["input_length"]:
            isl_mismatch += 1
        if status == "completed" and int(osl) != request["osl_sent"]:
            osl_mismatch += 1
        info = (raw or {}).get(key)
        worker = None
        if info is not None:
            if "prompt_sha256" in info and request.get("prompt_sha256"):
                prompt_checked += 1
                prompt_mismatch += info["prompt_sha256"] != request["prompt_sha256"]
            usage = info.get("usage") or {}
            if usage.get("prompt_tokens") is not None and int(
                usage["prompt_tokens"]
            ) != (request["input_length"]):
                usage_isl_mismatch += 1
            if (info.get("session_header") is not None) != emit_session:
                header_mismatch += 1
            worker = info.get("decode_worker_id", info.get("prefill_worker_id"))
        workers[key] = worker
        completed = status == "completed"
        rows.append(
            {
                "session_id": request["conversation_id"] if emit_session else None,
                "turn_index": request["turn_index"] if emit_session else None,
                "arrival_time_ms": arrival,
                "first_admit_ms": arrival if completed else None,
                "first_token_ms": start_ms + ttft if completed else None,
                # Wall-clock end, on the same clock as every credit issue: a closed-loop slot is
                # reissued strictly after it, so measured occupancy never exceeds the cap.
                "last_token_ms": end_ms if completed else None,
                "terminal_time_ms": end_ms,
                "ttft_ms": ttft if completed else None,
                "e2e_latency_ms": e2e if completed else None,
                "input_length": request["input_length"],
                "requested_output_length": request["output_length"],
                "output_length": int(osl) if completed else 0,
                "reused_input_tokens": None,
                "prefill_worker_idx": None,
                "decode_worker_idx": None,
                "terminal_status": status,
                "play_id": None,
                "lane_id": None,
                "routing_workers": [],
                "queue_wait_ms": 0.0,
                "_live_key": list(key),
                "_live_start_ms": start_ms,
                "_live_worker_id": worker,
            }
        )
    # Logical worker indices: live instance IDs in ascending order.
    instance_ids = sorted({w for w in workers.values() if w is not None})
    index_of = {w: i for i, w in enumerate(instance_ids)}
    for row in rows:
        worker = row["_live_worker_id"]
        if worker is not None:
            row["decode_worker_idx"] = index_of[worker]
            row["routing_workers"] = [index_of[worker]]
    diagnostics = {
        "records": len(records),
        "matched": len(by_key),
        "missing": len(requests) - len(by_key),
        "statuses": dict(statuses),
        "origin_ns": origin,
        "isl_mismatch": isl_mismatch,
        "osl_mismatch": osl_mismatch,
        "raw_checked": raw is not None,
        "prompt_checked": prompt_checked,
        "prompt_mismatch": prompt_mismatch,
        "usage_isl_mismatch": usage_isl_mismatch,
        "session_header_mismatch": header_mismatch if raw is not None else None,
        "workers_seen": len(instance_ids),
        "worker_instance_ids": instance_ids,
        "rows_with_worker": sum(row["decode_worker_idx"] is not None for row in rows),
        "schedule_lateness_ms": _quantiles(lateness),
        "start_lag_ms": _quantiles(start_lag),
        "think_residual_ms": _quantiles(think_residual),
    }
    if not open_loop:
        diagnostics["concurrency_cap"] = manifest["concurrency"]
    return rows, diagnostics


# --------------------------------------------------------------------------------------------
# E0
# --------------------------------------------------------------------------------------------


def ais_e0(requests: Sequence[dict], root: Path | None) -> Callable[[int, int], float]:
    table = {
        (r["input_length"], r["osl_sent"]): r["e0_ms"]
        for r in requests
        if r.get("e0_ms") is not None
    }
    fallback = None

    def e0(isl: int, osl: int) -> float:
        nonlocal fallback
        value = table.get((int(isl), int(osl)))
        if value is not None:
            return value
        if fallback is None:
            from learned_routing.e0 import E0Table
            from learned_routing.paths import Layout

            layout = Layout.resolve(root)
            fallback = E0Table(
                json.loads(layout.engine_json.read_text()), layout.e0_dir
            )
        return fallback(isl, osl)

    return e0


class LiveE0:
    """Live-fitted idle E0 (module docstring)."""

    def __init__(self, fit: dict):
        if fit.get("method") != E0_LIVE_METHOD:
            raise ScoreError(f"unsupported live E0 method {fit.get('method')!r}")
        points = sorted((float(i), float(t)) for i, t in fit["prefill_points"])
        if len(points) < 2:
            raise ScoreError("live E0 needs at least two prefill points")
        self.isl = [p[0] for p in points]
        self.ttft = [p[1] for p in points]
        self.d0 = float(fit["decode"]["d0_ms"])
        self.d1 = float(fit["decode"]["d1_ms_per_token"])

    def prefill_ms(self, isl: float) -> float:
        xs, ys = self.isl, self.ttft
        j = min(max(bisect.bisect_left(xs, isl), 1), len(xs) - 1)
        x0, x1, y0, y1 = xs[j - 1], xs[j], ys[j - 1], ys[j]
        return max(y0 + (y1 - y0) * (isl - x0) / (x1 - x0), 0.0)

    def __call__(self, isl: int, osl: int) -> float:
        n = max(int(osl) - 1, 0)
        contexts = n * int(isl) + n * (n - 1) / 2 + 2 * n  # sum_{j<n} (isl + j + 2)
        return self.prefill_ms(isl) + n * self.d0 + self.d1 * contexts


def fit_live_e0(
    manifest: dict, requests: Sequence[dict], records: Sequence[dict]
) -> dict:
    rows, diagnostics = build_rows(manifest, requests, records)
    done = [r for r in rows if r["terminal_status"] == "completed"]
    if len(done) < 4:
        raise ScoreError(f"only {len(done)} completed calibration requests")
    by_isl: dict[int, list[float]] = defaultdict(list)
    xs, ys = [], []
    for row in done:
        by_isl[row["input_length"]].append(row["ttft_ms"])
        itl = goodput.mean_itl_ms(row)
        if itl is not None:
            xs.append(row["input_length"] + row["output_length"] / 2.0 + 1.0)
            ys.append(itl)
    points = [[isl, statistics.median(v)] for isl, v in sorted(by_isl.items())]
    if len(xs) < 2 or len(set(xs)) < 2:
        raise ScoreError("decode fit needs ITL samples at two or more contexts")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    d1 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    d0 = my - d1 * mx
    ss_res = sum((y - d0 - d1 * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    fit = {
        "method": E0_LIVE_METHOD,
        "prefill_points": points,
        "prefill_spread_ms": {
            str(isl): [min(v), max(v)] for isl, v in sorted(by_isl.items())
        },
        "decode": {
            "d0_ms": d0,
            "d1_ms_per_token": d1,
            "n": len(xs),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        },
        "calibration_manifest_id": manifest.get("manifest_id"),
        "diagnostics": diagnostics,
    }
    model = LiveE0(fit)
    fit["check"] = {
        "e2e_over_e0": _quantiles(
            [
                r["e2e_latency_ms"] / model(r["input_length"], r["output_length"])
                for r in done
            ]
        )
    }
    return fit


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------


def warmup_ids(manifest: dict) -> frozenset | None:
    warmup_trace_ms = (manifest.get("measure") or {}).get("warmup_trace_ms")
    if warmup_trace_ms is None:
        return None
    path = Path(manifest["replicate"]["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["replicate"]["sha256"]:
        raise ScoreError(f"{path} does not match the manifest's replicate SHA-256")
    return goodput.warmup_ids_from_trace(
        data.decode().splitlines(), float(warmup_trace_ms)
    )


def score_rows(
    manifest: dict,
    rows: Sequence[dict],
    e0: Callable[[int, int], float],
    *,
    warmup: frozenset | None,
) -> dict:
    scored = [
        {k: v for k, v in row.items() if not k.startswith("_live_")} for row in rows
    ]
    completed = [r for r in scored if goodput.completed(r)]
    makespan = max((r["terminal_time_ms"] or 0.0) for r in scored) if scored else 0.0
    summary = {
        "num_requests": manifest["num_requests"],
        "duration_ms": makespan,
        "completed_requests": len(completed),
    }
    metrics = goodput.compute_metrics(
        scored,
        summary,
        sla=manifest["sla"],
        measure=manifest["measure"],
        open_loop=bool(manifest["open_loop"]),
        num_workers=int(manifest["num_workers"]),
        e0=e0,
        warmup_ids=warmup,
        occupancy_cap=None if manifest["open_loop"] else int(manifest["concurrency"]),
    )
    itls = [v for r in completed for v in [goodput.mean_itl_ms(r)] if v is not None]
    slowdowns = [
        r["e2e_latency_ms"] / e0(r["input_length"], r["output_length"])
        for r in completed
    ]
    metrics["itl_mean_ms"] = statistics.fmean(itls) if itls else None
    metrics["slowdown_mean"] = statistics.fmean(slowdowns) if slowdowns else None
    return metrics


def score_run(
    inputs_dir: Path,
    run_dir: Path,
    *,
    e0_mode: str = "ais",
    e0_live_path: Path | None = None,
    root: Path | None = None,
    policy: str | None = None,
    use_raw: bool = True,
    measure_override: dict | None = None,
) -> tuple[dict, list[dict]]:
    manifest, requests = load_inputs(inputs_dir)
    if measure_override is not None:
        # Smoke subsets only: the cell's window can lie beyond a subset's arrivals.
        manifest = {**manifest, "measure": measure_override}
    records = read_jsonl(run_dir / RECORDS_FILE)
    raw_path = run_dir / RAW_FILE
    raw = parse_raw(raw_path) if use_raw and raw_path.exists() else None
    rows, diagnostics = build_rows(manifest, requests, records, raw)
    if e0_mode == "ais":
        e0 = ais_e0(requests, root)
        e0_info = {"mode": "ais", "method": (manifest.get("e0") or {}).get("method")}
    elif e0_mode == "live":
        if e0_live_path is None:
            raise ScoreError("--e0 live needs --e0-live")
        fit = json.loads(e0_live_path.read_text())
        e0 = LiveE0(fit)
        e0_info = {
            "mode": "live",
            "method": E0_LIVE_METHOD,
            "fit": str(e0_live_path),
            "fit_sha256": hashlib.sha256(e0_live_path.read_bytes()).hexdigest(),
        }
    else:
        raise ScoreError(f"unknown E0 mode {e0_mode!r}")
    metrics = score_rows(manifest, rows, e0, warmup=warmup_ids(manifest))
    record = {
        "schema": SCHEMA,
        "scorer": SCORER_VERSION,
        "cell_id": manifest["cell_id"],
        "repeat": manifest["k"],
        "policy": policy,
        "policy_seed": manifest["replicate"]["policy_seed"],
        "replicate_protocol": manifest["replicate"]["protocol"],
        "manifest_id": manifest["manifest_id"],
        "family": manifest["family"],
        "num_workers": manifest["num_workers"],
        "load": manifest["load"],
        "sla": manifest["sla"],
        "measure": manifest["measure"],
        "subset": manifest.get("subset"),
        "measure_override": measure_override is not None,
        "e0": e0_info,
        "run_dir": str(run_dir),
        "records_sha256": hashlib.sha256(
            (run_dir / RECORDS_FILE).read_bytes()
        ).hexdigest(),
        "live": diagnostics,
        **metrics,
    }
    record["fidelity_ok"] = (
        diagnostics["missing"] == 0
        and diagnostics["isl_mismatch"] == 0
        and diagnostics["osl_mismatch"] == 0
        and diagnostics["prompt_mismatch"] == 0
        and diagnostics["usage_isl_mismatch"] == 0
        and not diagnostics["session_header_mismatch"]
    )
    return record, rows


# --------------------------------------------------------------------------------------------
# Live vs replay join
# --------------------------------------------------------------------------------------------


def compare_with_replay(
    manifest: dict, live_rows: Sequence[dict], replay_rows: Sequence[dict]
) -> dict:
    """Per-request live/replay latency ratios, joined by request identity.

    With session metadata the key is ``(session_id, turn_index)``. Without it (open-loop
    single-turn) the key is replay's ``(arrival_time_ms, input_length, requested_output_length)``,
    which the generator reproduces bit for bit; equal keys pair in order.
    """

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
    ttft_ratio, e2e_ratio, itl_ratio = [], [], []
    unmatched = 0
    for row in live_rows:
        pool = pools.get(key_of(row))
        if not pool:
            unmatched += 1
            continue
        other = pool.pop(0)
        if not (goodput.completed(row) and goodput.completed(other)):
            continue
        if other["ttft_ms"] > 0:
            ttft_ratio.append(row["ttft_ms"] / other["ttft_ms"])
        if other["e2e_latency_ms"] > 0:
            e2e_ratio.append(row["e2e_latency_ms"] / other["e2e_latency_ms"])
        a, b = goodput.mean_itl_ms(row), goodput.mean_itl_ms(other)
        if a is not None and b:
            itl_ratio.append(a / b)
    return {
        "unmatched_live": unmatched,
        "unmatched_replay": sum(len(p) for p in pools.values()),
        "ttft_live_over_replay": _quantiles(ttft_ratio),
        "e2e_live_over_replay": _quantiles(e2e_ratio),
        "itl_live_over_replay": _quantiles(itl_ratio),
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _write_rows(path: Path, rows: Sequence[dict]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(gzip.compress(text.encode(), mtime=0))
    os.replace(tmp, path)


def _json_default(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="score one AIPerf run")
    p_score.add_argument(
        "--inputs", type=Path, required=True, help="gen_aiperf_inputs output"
    )
    p_score.add_argument(
        "--run-dir", type=Path, required=True, help="AIPerf artifact directory"
    )
    p_score.add_argument("--out", type=Path, required=True, help="score record (JSON)")
    p_score.add_argument("--rows-out", type=Path, default=None, help="rows (.jsonl.gz)")
    p_score.add_argument("--e0", choices=("ais", "live"), default="ais")
    p_score.add_argument("--e0-live", type=Path, default=None)
    p_score.add_argument("--policy", default=None, help="policy label for the record")
    p_score.add_argument("--root", type=Path, default=None, help="campaign root (CR)")
    p_score.add_argument(
        "--no-raw", action="store_true", help="ignore profile_export_raw.jsonl"
    )
    p_score.add_argument(
        "--measure-json",
        default=None,
        help="override the cell's measurement rule (smoke subsets only; recorded in the output)",
    )

    p_fit = sub.add_parser("fit-e0", help="fit the live idle E0 from a calibration run")
    p_fit.add_argument("--inputs", type=Path, required=True)
    p_fit.add_argument("--run-dir", type=Path, required=True)
    p_fit.add_argument("--out", type=Path, required=True)

    p_cmp = sub.add_parser(
        "compare", help="join live rows with replay per_request rows"
    )
    p_cmp.add_argument("--inputs", type=Path, required=True)
    p_cmp.add_argument(
        "--live-rows", type=Path, required=True, help="score --rows-out file"
    )
    p_cmp.add_argument("--replay-per-request", type=Path, required=True)
    p_cmp.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "score":
        record, rows = score_run(
            args.inputs,
            args.run_dir,
            e0_mode=args.e0,
            e0_live_path=args.e0_live,
            root=args.root,
            policy=args.policy,
            use_raw=not args.no_raw,
            measure_override=None
            if args.measure_json is None
            else json.loads(args.measure_json),
        )
        if args.rows_out:
            _write_rows(args.rows_out, rows)
            record["rows_path"] = str(args.rows_out)
        args.out.write_text(
            json.dumps(record, indent=1, sort_keys=True, default=_json_default)
        )
        keys = (
            "cell_id",
            "repeat",
            "policy",
            "good_frac_window",
            "goodput_rps_window",
            "fidelity_ok",
        )
        print(json.dumps({k: record.get(k) for k in keys}, sort_keys=True))
        return 0 if record["fidelity_ok"] else 2
    if args.command == "fit-e0":
        manifest, requests = load_inputs(args.inputs)
        fit = fit_live_e0(manifest, requests, read_jsonl(args.run_dir / RECORDS_FILE))
        fit["run_dir"] = str(args.run_dir)
        args.out.write_text(json.dumps(fit, indent=1, sort_keys=True))
        print(
            json.dumps({"decode": fit["decode"], "points": len(fit["prefill_points"])})
        )
        return 0
    manifest, _ = load_inputs(args.inputs)
    report = compare_with_replay(
        manifest, read_jsonl(args.live_rows), read_jsonl(args.replay_per_request)
    )
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
