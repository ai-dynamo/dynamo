# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live scoring adapter: AIPerf records -> replay-shaped rows -> the harness's A2 scorer."""

from __future__ import annotations

import gzip
import heapq
import json
import random
from dataclasses import asdict
from pathlib import Path

import gen_aiperf_inputs as gen
import numpy as np
import pytest
import score_live as score
from learned_routing import goodput
from learned_routing.paths import Layout

ORIGIN_NS = 1_760_000_000_000_000_000
SLA = {"itl_ms": 30.0, "e2e_slowdown": 2.0, "ttft_ms": None}


def e0_const(isl: int, osl: int) -> float:
    return 100.0 + 0.01 * isl + 20.0 * max(osl - 1, 0)


def manifest_for(
    requests, *, open_loop, session_header, concurrency=None, measure=None
):
    return {
        "cell_id": "unit",
        "k": 0,
        "open_loop": open_loop,
        "session_header": session_header,
        "concurrency": concurrency,
        "num_requests": len(requests),
        "num_workers": 2,
        "sla": SLA,
        "measure": measure or {"basis": "arrival"},
    }


def replay_row(request, arrival, ttft, e2e, worker=0, session=True):
    return {
        "session_id": request["conversation_id"] if session else None,
        "turn_index": request["turn_index"] if session else None,
        "arrival_time_ms": arrival,
        "first_admit_ms": arrival,
        "first_token_ms": arrival + ttft,
        "last_token_ms": arrival + e2e,
        "terminal_time_ms": arrival + e2e,
        "ttft_ms": ttft,
        "e2e_latency_ms": e2e,
        "input_length": request["input_length"],
        "requested_output_length": request["output_length"],
        "output_length": request["osl_sent"],
        "reused_input_tokens": 0,
        "prefill_worker_idx": None,
        "decode_worker_idx": worker,
        "terminal_status": "completed",
        "play_id": None,
        "lane_id": None,
        "routing_workers": [worker],
        "queue_wait_ms": 0.0,
    }


def record_from(request, row, *, lateness_ms=0.0, start_extra_ms=0.0):
    """An AIPerf profile_export.jsonl record whose timing equals a replay row's."""
    issue = ORIGIN_NS + round((row["arrival_time_ms"] + lateness_ms) * 1e6)
    start = issue + round(start_extra_ms * 1e6)
    end = ORIGIN_NS + round(row["terminal_time_ms"] * 1e6)
    return {
        "metadata": {
            "conversation_id": request["conversation_id"],
            "turn_index": request["turn_index"],
            "credit_issued_ns": issue,
            "request_start_ns": start,
            "request_end_ns": end,
            "benchmark_phase": "profiling",
        },
        "metrics": {
            "time_to_first_token": {"value": row["ttft_ms"], "unit": "ms"},
            "request_latency": {"value": row["e2e_latency_ms"], "unit": "ms"},
            "output_sequence_length": {"value": row["output_length"], "unit": "tokens"},
            "input_sequence_length": {"value": row["input_length"], "unit": "tokens"},
        },
    }


def open_requests(n=200, seed=0, speedup=1.7):
    rng = random.Random(seed)
    rows = [
        {
            "timestamp": rng.randint(0, 60) * 1000,
            "input_length": rng.randint(1, 4000),
            "output_length": rng.randint(1, 60),
            "hash_ids": list(range(i * 100, i * 100 + 70)),
        }
        for i in range(n)
    ]
    sessions = gen.parse_mooncake([json.dumps(r) for r in rows], 64)
    plan = gen.plan_requests(
        sessions, open_loop=True, speedup=speedup, max_model_len=None
    )
    return [asdict(r) for r in plan]


def test_open_loop_scoring_equals_harness_scoring_of_the_same_timings():
    requests = open_requests()
    rng = random.Random(1)
    rows = []
    for r in requests:
        ttft = rng.uniform(50, 3000)
        rows.append(
            replay_row(
                r,
                r["arrival_ms"],
                ttft,
                ttft + rng.uniform(0, 40) * max(r["osl_sent"] - 1, 0),
                session=False,
            )
        )
    measure = {"basis": "arrival", "warmup_ms": 5000.0, "window_ms": 20000.0}
    manifest = manifest_for(
        requests, open_loop=True, session_header=False, measure=measure
    )
    # the scheduler is late by up to 3 ms and HTTP starts 1 ms after the credit; neither moves
    # window membership, which is decided by the generated (replay-identical) timestamps
    records = [
        record_from(
            r,
            row,
            lateness_ms=(0.0 if i == 7 else rng.uniform(0, 3)),
            start_extra_ms=1.0,
        )
        for i, (r, row) in enumerate(zip(requests, rows))
    ]
    random.Random(2).shuffle(records)
    live_rows, diag = score.build_rows(manifest, requests, records)
    assert diag["matched"] == len(requests) and diag["missing"] == 0
    assert diag["origin_ns"] == ORIGIN_NS
    assert (
        0.0 <= diag["schedule_lateness_ms"]["min"]
        and diag["schedule_lateness_ms"]["max"] < 3.0
    )
    live = score.score_rows(manifest, live_rows, e0_const, warmup=None)
    summary = {
        "num_requests": len(rows),
        "duration_ms": max(r["terminal_time_ms"] for r in rows),
    }
    harness = goodput.compute_metrics(
        rows,
        summary,
        sla=SLA,
        measure=measure,
        open_loop=True,
        num_workers=2,
        e0=e0_const,
    )
    for key in (
        "good_frac_window",
        "window_requests",
        "window_good",
        "window_start_ms",
        "window_end_ms",
        "goodput_rps_window",
        "slowdown_p50",
        "itl_p90",
    ):
        assert live[key] == pytest.approx(harness[key], rel=1e-9, abs=1e-9), key
    for scale, values in harness["rescore"].items():
        assert live["rescore"][scale]["good_frac_window"] == values["good_frac_window"]


def simulate_closed(requests, concurrency, seed=0):
    """Replay-shaped rows of a C-slot closed loop: sessions start in order as slots free up."""
    rng = random.Random(seed)
    sessions = {}
    for r in requests:
        sessions.setdefault(r["conversation_id"], []).append(r)
    free = [0.0] * concurrency
    heapq.heapify(free)
    rows = []
    for turns in sessions.values():
        t = heapq.heappop(free)
        for r in turns:
            if r["turn_index"] > 0:
                t += r["delay_ms"]
            ttft = rng.uniform(20, 500)
            e2e = ttft + rng.uniform(5, 45) * max(r["osl_sent"] - 1, 0)
            rows.append(replay_row(r, t, ttft, e2e))
            t += e2e
        heapq.heappush(free, t)
    return rows


def test_closed_loop_scoring_with_identity_warmup_equals_harness():
    rng = random.Random(3)
    trace = []
    for s in range(120):
        for t in range(rng.randint(1, 4)):
            row = {
                "session_id": f"s{s:03d}",
                "input_length": rng.randint(10, 900),
                "output_length": rng.randint(1, 30),
                "hash_ids": [s * 10 + j for j in range(15)],
            }
            if t == 0:
                row["timestamp"] = s * 1000
            else:
                row["delay"] = rng.uniform(0, 2000)
            trace.append(row)
    lines = [json.dumps(r) for r in trace]
    sessions = gen.parse_mooncake(lines, 64)
    requests = [
        asdict(r)
        for r in gen.plan_requests(
            sessions, open_loop=False, speedup=None, max_model_len=None
        )
    ]
    rows = simulate_closed(requests, concurrency=8)
    measure = {
        "basis": "completion",
        "end": "full_occupancy",
        "warmup_trace_ms": 20000.0,
    }
    warmup = goodput.warmup_ids_from_trace(lines, 20000.0)
    manifest = manifest_for(
        requests, open_loop=False, session_header=True, concurrency=8, measure=measure
    )
    records = [record_from(r, row) for r, row in zip(requests, rows)]
    live_rows, diag = score.build_rows(manifest, requests, records)
    assert diag["think_residual_ms"]["max"] == pytest.approx(0.0, abs=1e-5)
    live = score.score_rows(manifest, live_rows, e0_const, warmup=warmup)
    summary = {
        "num_requests": len(rows),
        "duration_ms": max(r["terminal_time_ms"] for r in rows),
    }
    harness = goodput.compute_metrics(
        rows,
        summary,
        sla=SLA,
        measure=measure,
        open_loop=False,
        num_workers=2,
        e0=e0_const,
        warmup_ids=warmup,
        occupancy_cap=8,
    )
    assert harness["warmup_excluded_rows"] > 0 and harness["occupancy_peak"] == 8
    for key in (
        "good_frac_window",
        "window_requests",
        "window_good",
        "warmup_excluded_rows",
        "occupancy_peak",
        "full_occupancy_end_ms",
        "goodput_rps_window",
    ):
        assert live[key] == pytest.approx(harness[key], rel=1e-9, abs=1e-6), key


def test_errors_mismatches_and_missing_records_are_counted():
    requests = open_requests(n=6)
    rows = [
        replay_row(
            r, r["arrival_ms"], 100.0, 100.0 + 10.0 * (r["osl_sent"] - 1), session=False
        )
        for r in requests
    ]
    records = [record_from(r, row) for r, row in zip(requests, rows)]
    records[1]["error"] = {"code": 500, "message": "boom"}
    records[2]["metadata"]["was_cancelled"] = True
    records[3]["metrics"]["output_sequence_length"]["value"] = (
        requests[3]["osl_sent"] + 1
    )
    del records[4]
    manifest = manifest_for(requests, open_loop=True, session_header=False)
    live_rows, diag = score.build_rows(manifest, requests, records)
    assert diag["statuses"] == {"completed": 3, "error": 1, "cancelled": 1}
    assert diag["missing"] == 1 and diag["osl_mismatch"] == 1
    metrics = score.score_rows(manifest, live_rows, e0_const, warmup=None)
    assert metrics["missing_rows"] == 1 and metrics["good"] <= 3
    with pytest.raises(score.ScoreError, match="duplicate"):
        score.build_rows(manifest, requests, records + [records[0]])
    foreign = json.loads(json.dumps(records[0]))
    foreign["metadata"]["conversation_id"] = "elsewhere"
    with pytest.raises(score.ScoreError, match="unmatched"):
        score.build_rows(manifest, requests, records + [foreign])


def test_raw_export_checks_prompts_headers_and_maps_workers(tmp_path):
    requests = open_requests(n=3)
    prompts = [list(range(r["input_length"])) for r in requests]
    for r, p in zip(requests, prompts):
        r["prompt_sha256"] = gen.token_digest(np.asarray(p))
    rows = [replay_row(r, r["arrival_ms"], 10.0, 20.0, session=False) for r in requests]
    records = [record_from(r, row) for r, row in zip(requests, rows)]
    raw_path = tmp_path / score.RAW_FILE
    with raw_path.open("w") as handle:
        for r, rec, p, worker in zip(requests, records, prompts, (77, 12, 77)):
            usage = {
                "prompt_tokens": r["input_length"],
                "completion_tokens": r["osl_sent"],
            }
            done = {
                "choices": [],
                "usage": usage,
                "nvext": {"worker_id": {"decode_worker_id": worker}},
            }
            raw = {
                "metadata": rec["metadata"],
                "payload": {"prompt": p if worker != 12 else p[:-1] + [p[-1] + 1]},
                "request_headers": {"X-Request-ID": "x"},
                "responses": [
                    {
                        "perf_ns": 1,
                        "packets": [
                            {
                                "name": "data",
                                "value": json.dumps({"choices": [{"text": "a"}]}),
                            }
                        ],
                    },
                    {
                        "perf_ns": 2,
                        "packets": [{"name": "data", "value": json.dumps(done)}],
                    },
                    {"perf_ns": 3, "packets": [{"name": "data", "value": "[DONE]"}]},
                ],
            }
            handle.write(json.dumps(raw) + "\n")
    raw = score.parse_raw(raw_path)
    manifest = manifest_for(requests, open_loop=True, session_header=False)
    live_rows, diag = score.build_rows(manifest, requests, records, raw)
    assert diag["prompt_checked"] == 3 and diag["prompt_mismatch"] == 1
    assert diag["session_header_mismatch"] == 0 and diag["usage_isl_mismatch"] == 0
    assert diag["worker_instance_ids"] == [12, 77]
    assert [row["decode_worker_idx"] for row in live_rows] == [1, 0, 1]


def test_live_e0_fit_recovers_an_affine_idle_model():
    isls = (256, 1024, 8192, 32768)
    requests, records = [], []
    t = 0.0
    for rep in range(3):
        for isl in isls:
            r = {
                "conversation_id": f"idle-r{rep}-isl{isl}",
                "turn_index": 0,
                "input_length": isl,
                "output_length": 64,
                "osl_sent": 64,
                "arrival_ms": None,
                "delay_ms": None,
            }
            ttft = 15.0 + 0.08 * isl + rep * 0.01
            decode = sum(22.0 + 2e-4 * (isl + j + 2) for j in range(63))
            row = replay_row(r, t, ttft, ttft + decode)
            requests.append(r)
            records.append(record_from(r, row))
            t += ttft + decode + 5.0
    manifest = manifest_for(
        requests, open_loop=False, session_header=False, concurrency=1
    )
    fit = score.fit_live_e0(manifest, requests, records)
    assert fit["decode"]["d0_ms"] == pytest.approx(22.0, rel=1e-6)
    assert fit["decode"]["d1_ms_per_token"] == pytest.approx(2e-4, rel=1e-6)
    model = score.LiveE0(fit)
    for isl in isls:
        expected = (
            15.0
            + 0.08 * isl
            + 0.01
            + sum(22.0 + 2e-4 * (isl + j + 2) for j in range(99))
        )
        assert model(isl, 100) == pytest.approx(expected, rel=1e-9)
    assert model.prefill_ms(2048) == pytest.approx(15.0 + 0.08 * 2048 + 0.01, rel=1e-9)
    assert fit["check"]["e2e_over_e0"]["max"] == pytest.approx(1.0, rel=1e-4)


def test_compare_joins_live_and_replay_by_identity():
    requests = open_requests(n=20)
    rows = [
        replay_row(r, r["arrival_ms"], 100.0, 400.0, session=False) for r in requests
    ]
    records = [record_from(r, row) for r, row in zip(requests, rows)]
    manifest = manifest_for(requests, open_loop=True, session_header=False)
    live_rows, _ = score.build_rows(manifest, requests, records)
    report = score.compare_with_replay(manifest, live_rows, rows)
    assert report["unmatched_live"] == report["unmatched_replay"] == 0
    assert report["e2e_live_over_replay"]["p50"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------
# Real campaign data: the scorer reproduces lr-eval's records from replay-identical timings.
# --------------------------------------------------------------------------------------------

CR = Layout.resolve().root
CACHED = [
    (
        "train",
        "mooncake-w0-base-n4-open-L1",
        "c1/c12f7a1e3e21e3feb27806537932182d2db5a146e95a87a8dc327c5be0f8bcc6",
    ),
    (
        "train",
        "mooncake-w0-base-n8-closed-L2",
        "fd/fd9dc8dd9ebb67c65e1fdb623c9f82dd76853ba66e4c1a22060dc464627204f1",
    ),
    (
        "train",
        "sessions-s0-base-n4-open-L2",
        "24/24d33200a9f597dbf894f9710d1dc60cd9756f90aa9881f6703906d36922ccee",
    ),
    (
        "train",
        "sessions-s0-base-n8-closed-L2",
        "39/392ed310677d81349f9b8d0cf8f9aae61f8ed7a425e681b0729f65f3d0315711",
    ),
]


@pytest.mark.parametrize("split, cell_id, key", CACHED, ids=[c[1] for c in CACHED])
def test_scorer_reproduces_cached_lr_eval_records(split, cell_id, key):
    base = CR / "runs" / "cache" / "results" / key
    if not (
        base.with_suffix(".json").exists()
        and Path(f"{base}.per_request.jsonl.gz").exists()
    ):
        pytest.skip("campaign cache not available")
    from learned_routing.cells import load_cells
    from learned_routing.e0 import E0Table
    from learned_routing.paths import Layout

    layout = Layout.resolve(CR)
    cell = next(
        c
        for c in load_cells(CR / "cells" / f"{split}.jsonl", layout)
        if c.cell_id == cell_id
    )
    cached = json.loads(base.with_suffix(".json").read_text())
    with gzip.open(f"{base}.per_request.jsonl.gz", "rt") as handle:
        replay = [json.loads(line) for line in handle]
    plan = gen.plan_cell(cell, cached["repeat"])
    assert plan.rep.trace_sha256 == cached["trace_sha256"]
    requests = [asdict(r) for r in plan.requests]
    manifest = {
        "cell_id": cell_id,
        "k": plan.k,
        "open_loop": plan.open_loop,
        "session_header": plan.session_header,
        "concurrency": plan.concurrency,
        "num_requests": len(requests),
        "num_workers": cell.num_workers,
        "sla": cell.sla,
        "measure": cell.measure,
        "replicate": {
            "path": str(plan.rep.trace_path),
            "sha256": plan.rep.trace_sha256,
        },
    }
    if plan.session_header:
        index = {(r["conversation_id"], r["turn_index"]): r for r in requests}
        pairs = [(index[(row["session_id"], row["turn_index"])], row) for row in replay]
    else:
        pool = {}
        for r in requests:
            pool.setdefault(
                (r["arrival_ms"], r["input_length"], r["output_length"]), []
            ).append(r)
        pairs = [
            (
                pool[
                    (
                        row["arrival_time_ms"],
                        row["input_length"],
                        row["requested_output_length"],
                    )
                ].pop(),
                row,
            )
            for row in replay
        ]
    records = [record_from(r, row) for r, row in pairs]
    live_rows, diag = score.build_rows(manifest, requests, records)
    assert diag["missing"] == 0 and diag["osl_mismatch"] == 0
    e0 = E0Table(json.loads(layout.engine_json.read_text()), layout.e0_dir)
    live = score.score_rows(manifest, live_rows, e0, warmup=score.warmup_ids(manifest))
    for field in (
        "good_frac_window",
        "goodput_rps_window",
        "window_requests",
        "window_good",
        "warmup_excluded_rows",
        "slowdown_p50",
        "itl_p50",
    ):
        assert live[field] == pytest.approx(cached[field], rel=1e-9, abs=1e-9), field
