# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: real replays through the campaign slot pool, into a throwaway cache."""

import gzip
import json
import os
from pathlib import Path

import pytest
from learned_routing import goodput
from learned_routing.cells import Cell
from learned_routing.e0 import E0Table
from learned_routing.evaluate import Evaluator, Task
from learned_routing.paths import Layout
from learned_routing.policy import BUILTIN_SPECS, spec_from_dict

pytestmark = pytest.mark.replay

# A Mooncake trace of at least 120 rows (the campaign used the first 1,000 rows of the public
# Mooncake trace); set LR_TEST_MOONCAKE_TRACE to run this test.
SOURCE = Path(
    os.environ.get("LR_TEST_MOONCAKE_TRACE") or "/nonexistent/mooncake_trace.jsonl"
)
CAMPAIGN = Layout.resolve()


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    pytest.importorskip("dynamo.replay")
    if not SOURCE.exists() or not CAMPAIGN.engine_json.exists():
        pytest.skip("campaign trace or engine.json missing")
    root = tmp_path_factory.mktemp("root")
    trace = root / "trace.jsonl"
    trace.write_text("".join(SOURCE.read_text().splitlines(keepends=True)[:120]))
    layout = Layout(root)
    cell = Cell(
        raw={
            "cell_id": "e2e",
            "trace_files": [str(trace)],
            "trace_format": "mooncake",
            "trace_block_size": 512,
            "load": {"mode": "open_speedup", "value": 1.0},
            "num_workers": 4,
            "sla": {"itl_ms": 30.0, "e2e_slowdown": 8.0, "ttft_ms": None},
            "engine_ref": str(CAMPAIGN.engine_json),
        },
        layout=layout,
    )
    specs = [
        spec_from_dict(BUILTIN_SPECS["round_robin"]),
        spec_from_dict(BUILTIN_SPECS["default"]),
    ]
    return layout, cell, specs


def run(layout, tasks, **kwargs):
    with Evaluator(
        layout, concurrency=4, slots_dir=CAMPAIGN.slots_dir, verbose=False, **kwargs
    ) as ev:
        return ev.evaluate(tasks)


def test_replays_score_cache_and_reproduce(setup):
    layout, cell, specs = setup
    tasks = [Task(spec, cell, k) for spec in specs for k in (0, 1)]
    first = run(layout, tasks)
    assert all(r is not None and r["error"] is None for r in first), [
        r and r["error"] for r in first
    ]
    engine = json.loads(CAMPAIGN.engine_json.read_text())
    e0 = E0Table(engine, layout.e0_dir)
    for record in first:
        assert record["goodput_check_rel_err"] <= goodput.NATIVE_REL_TOL
        # independent recomputation from the stored per-request rows
        rows = [
            json.loads(line)
            for line in gzip.decompress(
                Path(record["per_request_path"]).read_bytes()
            ).splitlines()
        ]
        assert len(rows) == record["num_requests"] == 120
        token_good = sum(goodput.token_form_good(r, None, 30.0) for r in rows)
        assert token_good / (record["duration_ms"] / 1000.0) == pytest.approx(
            record["goodput_rps_report"], rel=1e-12
        )
        a2 = [goodput.a2_good(r, record["sla"], e0) for r in rows]
        assert sum(a2) == record["good"]
    assert (
        first[2]["policy_seed"] == 1
        and first[3]["policy_seed"] == 2
        and first[0]["policy_seed"] is None
    )
    # CRN: both policies on replicate k replay the same workload
    assert (
        first[0]["trace_sha256"] == first[2]["trace_sha256"] != first[1]["trace_sha256"]
    )

    cached = run(layout, tasks)
    assert all(r["cached"] for r in cached)
    renamed = run(layout, [Task(specs[1].with_values(name="default-alias"), cell, 0)])
    assert renamed[0]["cached"] and renamed[0]["policy_name"] == "default-alias"
    again = run(layout, tasks, refresh=True)
    for a, b in zip(first, again):
        assert not b["cached"]
        assert a["per_request_canonical_sha256"] == b["per_request_canonical_sha256"]
        assert a["goodput_rps_window"] == b["goodput_rps_window"]


def test_lr_eval_cli_caches_refreshes_and_reports_exit_codes(setup, tmp_path):
    from learned_routing import eval_cli

    layout, cell, _ = setup
    cells = tmp_path / "cells.jsonl"
    cells.write_text(json.dumps(cell.raw) + "\n")
    out = tmp_path / "results.jsonl"
    base = [
        "--root", str(layout.root), "--slots-dir", str(CAMPAIGN.slots_dir),
        "--policy-spec", "round_robin", "--cells", str(cells), "--out", str(out),
        "--repeats", "1", "--repeat-offset", "7", "--slots", "2", "--quiet",
    ]  # fmt: skip
    assert eval_cli.main(base) == 0
    assert eval_cli.main(base) == 0
    assert eval_cli.main([*base, "--refresh"]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["cached"] for r in rows] == [False, True, False]
    assert len({r["per_request_canonical_sha256"] for r in rows}) == 1
    # a deadline that has already passed still runs the first wave (progress guarantee)
    assert eval_cli.main([*base, "--refresh", "--max-wall-seconds", "0"]) == 0


# B2's AgentX A2 lanes trace: 11 plays, 486 requests (build audit r1 F1's cell).
AGENTX_A2 = (
    CAMPAIGN.root
    / "traces/derived/59ee3018f1a327095cd39cdf6d837c268820fd2e0ec3c783945b224043dfa4ef.jsonl"
)
FULL_OCCUPANCY = {"basis": "completion", "end": "full_occupancy"}


def per_request(record):
    return [
        json.loads(line)
        for line in gzip.decompress(Path(record["per_request_path"]).read_bytes())
        .decode()
        .splitlines()
    ]


def test_full_occupancy_closed_loop_ends_after_the_last_admission(setup):
    layout, cell, specs = setup
    raw = {
        **cell.raw,
        "cell_id": "e2e-closed",
        "load": {"mode": "closed_concurrency", "value": 4},
        "measure": FULL_OCCUPANCY,
    }
    closed = Cell(raw=raw, layout=layout)
    no_end = Cell(
        raw={**raw, "cell_id": "e2e-closed-no-end", "measure": {"basis": "completion"}},
        layout=layout,
    )
    record, rejected = run(
        layout, [Task(specs[1], closed, 0), Task(specs[1], no_end, 0)]
    )
    assert record["error"] is None, record["error"]
    # a completion-basis cell without an end rule fails at plan time, before any replay
    assert rejected["error"].startswith(
        "plan_error: ValueError: basis completion needs"
    )
    rows = per_request(record)
    last_admission = max(r["arrival_time_ms"] for r in rows)
    first_departure_after = min(
        r["terminal_time_ms"] for r in rows if r["terminal_time_ms"] > last_admission
    )
    assert record["occupancy_peak"] == record["occupancy_cap"] == 4
    assert record["window_end_ms"] == record["full_occupancy_end_ms"]
    assert record["window_end_ms"] == first_departure_after
    assert record["window_below_cap_ms"] == 0.0
    in_window = [
        r
        for r in rows
        if goodput.completed(r)
        and record["window_start_ms"] <= r["terminal_time_ms"] < record["window_end_ms"]
    ]
    assert record["window_requests"] == len(in_window) < len(rows)


def test_full_occupancy_lanes_end_at_the_first_lane_exhaustion(setup):
    layout, cell, specs = setup
    if not AGENTX_A2.exists():
        pytest.skip("campaign AgentX trace missing")
    lanes = Cell(
        raw={
            **cell.raw,
            "cell_id": "e2e-lanes",
            "trace_files": [str(AGENTX_A2)],
            "trace_format": "weka",
            "trace_block_size": 64,
            "load": {"mode": "agentic_lanes", "value": 4},
            "measure": FULL_OCCUPANCY,
        },
        layout=layout,
    )
    (record,) = run(layout, [Task(specs[1], lanes, 0)])
    assert record["error"] is None, record["error"]
    rows = per_request(record)
    # The driver's own lane identity: each lane is busy until its last play releases it.
    lane_end = {}
    for r in rows:
        lane_end[r["lane_id"]] = max(
            lane_end.get(r["lane_id"], 0.0), r["terminal_time_ms"]
        )
    assert len(lane_end) == 4 and len({r["play_id"] for r in rows}) == 11
    assert (
        record["window_end_ms"]
        == record["full_occupancy_end_ms"]
        == min(lane_end.values())
    )
    assert record["occupancy_peak"] == 4 and record["window_below_cap_ms"] == 0.0
    # the former end, the last dispatch, lies well inside the drain
    assert max(r["arrival_time_ms"] for r in rows) > record["window_end_ms"]


# (ISL, OSL): prefill only, one decode step, chunked prefill (chunk 8192), unaligned blocks.
IDLE_REQUESTS = ((1, 1), (100, 2), (1000, 10), (8192, 64), (8193, 300), (20000, 129))


def test_idle_requests_are_good_at_unit_slowdown(setup):
    """Build audit goodput r2 F1: a request alone on an idle worker with no prefix reuse has
    e2e == E0, so it is good at S * scale = 1 under every policy, including round-robin.
    """
    layout, cell, specs = setup
    trace = layout.root / "idle.jsonl"
    lines, next_hash = [], 0
    for i, (isl, osl) in enumerate(IDLE_REQUESTS):
        blocks = -(-isl // 512)
        hash_ids = list(range(next_hash, next_hash + blocks))  # fresh: no reuse
        next_hash += blocks
        lines.append(
            json.dumps(
                {
                    "timestamp": i * 600_000,  # far longer than any e2e: idle workers
                    "input_length": isl,
                    "output_length": osl,
                    "hash_ids": hash_ids,
                }
            )
        )
    trace.write_text("\n".join(lines) + "\n")
    idle = Cell(
        raw={
            **cell.raw,
            "cell_id": "e2e-idle",
            "trace_files": [str(trace)],
            "num_workers": 2,
            "sla": {"itl_ms": None, "e2e_slowdown": 1.0, "ttft_ms": None},
        },
        layout=layout,
    )
    records = run(layout, [Task(spec, idle, 0) for spec in specs])
    engine = json.loads(CAMPAIGN.engine_json.read_text())
    e0 = E0Table(engine, layout.e0_dir)
    n = len(IDLE_REQUESTS)
    for record in records:
        assert record["error"] is None, record["error"]
        rows = per_request(record)
        assert len(rows) == n and all(goodput.completed(r) for r in rows)
        assert all(not r["reused_input_tokens"] for r in rows)
        for r in rows:
            reference = e0(r["input_length"], r["output_length"])
            assert r["e2e_latency_ms"] == pytest.approx(reference, rel=1e-7), r
        assert record["good"] == record["window_good"] == n
        assert record["good_frac_window"] == 1.0
        assert record["slowdown_atom_frac"] == 1.0
        # the tolerance is float-noise sized: a tighter scale fails every request
        assert record["rescore"]["0.75"]["good_frac_window"] == 0.0


def test_timing_perturbed_idle_requests_match_the_consistent_e0(setup):
    """Phase2-mid sim-exploitation F2: on a perturbed engine (speedup 0.8, decode x1.25) an idle,
    no-reuse request runs exactly E0' = prefill / s + decode / (s d). lr-eval's nominal E0 makes
    every one of them miss a unit slowdown; the consistent rescoring makes every one good.
    """
    from learned_routing.rescore_timing import ConsistentE0, rescore_record

    layout, cell, specs = setup
    trace = layout.root / "idle_pert.jsonl"
    lines, next_hash = [], 0
    for i, (isl, osl) in enumerate(IDLE_REQUESTS):
        blocks = -(-isl // 512)
        hash_ids = list(range(next_hash, next_hash + blocks))  # fresh: no reuse
        next_hash += blocks
        lines.append(
            json.dumps(
                {
                    "timestamp": i * 600_000,
                    "input_length": isl,
                    "output_length": osl,
                    "hash_ids": hash_ids,
                }
            )
        )
    trace.write_text("\n".join(lines) + "\n")
    s, d = 0.8, 1.25
    pert = Cell(
        raw={
            **cell.raw,
            "cell_id": "e2e-idle-pert",
            "trace_files": [str(trace)],
            "num_workers": 2,
            "sla": {"itl_ms": None, "e2e_slowdown": 1.0, "ttft_ms": None},
            "engine_overrides": {"speedup_ratio": s, "decode_speedup_ratio": d},
        },
        layout=layout,
    )
    records = run(layout, [Task(specs[1], pert, 0)])
    engine = json.loads(CAMPAIGN.engine_json.read_text())
    table = E0Table(engine, layout.e0_dir)
    consistent = ConsistentE0(table, s, d)
    n = len(IDLE_REQUESTS)
    (record,) = records
    assert record["error"] is None, record["error"]
    rows = per_request(record)
    assert len(rows) == n and all(not r["reused_input_tokens"] for r in rows)
    for r in rows:
        isl, osl = r["input_length"], r["output_length"]
        assert r["e2e_latency_ms"] == pytest.approx(consistent(isl, osl), rel=1e-7), r
        # prefill runs at 1 / s = 1.25x, decode at 1 / (s d) = 1x: slower than the nominal E0
        assert r["e2e_latency_ms"] > table(isl, osl) * (1 + 1e-4)
    # lr-eval scores against the nominal E0: the slower engine also tightened the SLA
    assert record["good"] == 0
    both = rescore_record(record, pert, table)
    assert both["nominal_e0"]["goodput_rps_window"] == record["goodput_rps_window"]
    assert both["nominal_e0"]["good_frac_window"] == 0.0
    assert both["consistent_e0"]["good_frac_window"] == 1.0
    assert both["consistent_e0"]["slowdown_atom_frac"] == 1.0
    assert (both["speedup_ratio"], both["decode_speedup_ratio"]) == (s, d)
