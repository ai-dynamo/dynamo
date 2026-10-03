# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invariants of the synthetic-session generator and the split plan."""

from __future__ import annotations

import json
import math
from collections import defaultdict

import pytest
from learned_routing import goodput
from learned_routing.replicates import permute_mooncake_ties
from learned_routing.workloads import cells
from learned_routing.workloads.synthetic import SessionSpec, generate
from learned_routing.workloads.transform import TransformSpec, transform_mooncake

SMALL = {"duration_s": 120.0, "session_rate": 2.0, "num_prefix_groups": 4}


def lcp(a: list, b: list) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def by_session(rows: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for row in rows:
        out[row["session_id"]].append(row)
    return out


def test_turns_extend_the_previous_context():
    spec = SessionSpec.from_dict({**SMALL, "seed": 1})
    sessions = by_session(generate(spec))
    multi = [turns for turns in sessions.values() if len(turns) > 1]
    assert multi, "the spec should produce multi-turn sessions"
    for turns in sessions.values():
        for prev, cur in zip(turns, turns[1:]):
            full = prev["input_length"] // spec.block_size
            assert cur["hash_ids"][:full] == prev["hash_ids"][:full]
            assert cur["input_length"] > prev["input_length"] + prev["output_length"]
            if prev["input_length"] % spec.block_size:
                assert (
                    prev["hash_ids"][full] not in cur["hash_ids"]
                )  # a filled block is new content


def test_rows_are_replayable_mooncake_sessions():
    spec = SessionSpec.from_dict({**SMALL, "seed": 2})
    rows = generate(spec)
    for turns in by_session(rows).values():
        assert "timestamp" in turns[0] and "delay" not in turns[0]
        assert turns[0]["timestamp"] % spec.arrival_grid_ms == 0
        assert all("delay" in t and "timestamp" not in t for t in turns[1:])
    for row in rows:
        assert len(row["hash_ids"]) == math.ceil(row["input_length"] / spec.block_size)
        assert row["input_length"] + row["output_length"] <= spec.context_cap
    # Sessions are contiguous, so the replicate protocol and the window transform see whole sessions.
    runs = [
        s
        for i, s in enumerate(r["session_id"] for r in rows)
        if i == 0 or s != rows[i - 1]["session_id"]
    ]
    assert len(runs) == len(set(runs))


def test_system_prompts_are_shared_within_a_group_only():
    spec = SessionSpec.from_dict({**SMALL, "seed": 3})
    firsts = [t[0] for t in by_session(generate(spec)).values()]
    roots = {f["hash_ids"][0] for f in firsts}
    assert 1 < len(roots) <= spec.num_prefix_groups


def test_generator_is_seeded():
    a = generate(SessionSpec.from_dict({**SMALL, "seed": 4}))
    b = generate(SessionSpec.from_dict({**SMALL, "seed": 4}))
    c = generate(SessionSpec.from_dict({**SMALL, "seed": 5}))
    assert a == b and a != c


def test_arrival_grid_gives_the_replicate_protocol_ties():
    lines = [
        json.dumps(r) for r in generate(SessionSpec.from_dict({**SMALL, "seed": 6}))
    ]
    _, stats = permute_mooncake_ties(lines, seed=11)
    assert not stats.degenerate and stats.units_in_ties > 0


def test_think_and_isl_transforms_keep_sessions_replayable():
    spec = SessionSpec.from_dict({**SMALL, "seed": 7})
    rows = generate(spec)
    out, _ = transform_mooncake(
        rows, TransformSpec(isl_unique_mult=2.0, think_mult=0.5), spec.block_size
    )
    src, dst = by_session(rows), by_session(out)
    for session, turns in dst.items():
        for (p0, c0), (p1, c1) in zip(
            zip(src[session], src[session][1:]), zip(turns, turns[1:])
        ):
            assert lcp(p1["hash_ids"], c1["hash_ids"]) == lcp(
                p0["hash_ids"], c0["hash_ids"]
            )
        for row in turns:
            assert len(row["hash_ids"]) == math.ceil(
                row["input_length"] / spec.block_size
            )
    delays_in = [r["delay"] for r in rows if "delay" in r]
    delays_out = [r["delay"] for r in out if "delay" in r]
    assert delays_out == pytest.approx([d * 0.5 for d in delays_in])


# --- split plan ---------------------------------------------------------------------------------


def fake_cells() -> dict[str, list[dict]]:
    """Cells from the plan tables without materializing traces (validate() needs no IO)."""
    plays = cells.allocate_plays({f"{i:04d}-p.json": i % 4 == 0 for i in range(82)})
    out = {}
    for split, plans in cells.PLANS.items():
        members = []
        for plan in plans:
            if plan.segment.startswith("agentx:"):
                family, segment = "agentx", plan.segment
            else:
                definition = cells.segments()[plan.segment]
                family, segment = definition["family"], definition.get(
                    "segment", plan.segment
                )
            spec = TransformSpec.from_dict(cells.TRANSFORMS[plan.tag]).to_dict()
            members.append(
                {
                    "cell_id": f"{plan.segment}-{plan.tag}-n{plan.n}-{plan.mode}-{plan.level}",
                    "family": family,
                    "segment": segment,
                    "num_workers": plan.n,
                    "holdout_axis": plan.axis,
                    "transform": spec,
                }
            )
        out[split] = members
    return out, plays


def test_plan_satisfies_the_split_invariants():
    built, plays = fake_cells()
    summary = cells.validate(built, plays)
    assert {"mooncake:w3", "mooncake:w5"} <= set(summary["segments_by_split"]["test"])
    assert (
        len(built["train"]) <= 40
        and len(built["val"]) <= 15
        and len(built["test"]) <= 60
    )


def test_windowed_segments_share_no_trace_rows():
    """build audit S1: a window's warm-up must not be another window's rows, in any split."""
    assert cells.overlapping_slices() == []
    defs = cells.segments()
    for name, d in defs.items():
        if d["family"] in cells.WINDOWED_FAMILIES:
            lo, hi = d["slice"]
            start, end = d["measure"]
            assert lo < start < end == hi, name
    original = cells.mooncake_window
    try:
        cells.mooncake_window = lambda k: {
            "slice": (8 * k * cells.MINUTE_MS, (12 + 8 * k) * cells.MINUTE_MS),
            "measure": ((4 + 8 * k) * cells.MINUTE_MS, (12 + 8 * k) * cells.MINUTE_MS),
        }
        assert ("mooncake:w0", "mooncake:w1") in cells.overlapping_slices()
        built, plays = fake_cells()
        with pytest.raises(ValueError, match="share trace rows"):
            cells.validate(built, plays)
    finally:
        cells.mooncake_window = original


def test_validation_has_three_segments_per_load_mode():
    seg_by_mode = defaultdict(set)
    for plan in cells.VAL:
        definition = cells.segments().get(plan.segment, {})
        seg_by_mode[plan.mode].add(definition.get("segment", plan.segment))
    assert all(len(v) >= 3 for v in seg_by_mode.values()), dict(seg_by_mode)


def test_validate_rejects_a_segment_leak_and_a_bad_worker_count():
    built, plays = fake_cells()
    leaked = {k: [dict(c) for c in v] for k, v in built.items()}
    leaked["train"][0]["segment"] = "mooncake:w3"
    with pytest.raises(ValueError, match="several splits"):
        cells.validate(leaked, plays)
    bad = {k: [dict(c) for c in v] for k, v in built.items()}
    bad["train"][0]["num_workers"] = 16
    with pytest.raises(ValueError, match="not allowed"):
        cells.validate(bad, plays)


def test_play_allocation_is_disjoint_complete_and_stratified():
    corpus = {f"{i:04d}-p.json": i % 4 == 0 for i in range(82)}
    plays = cells.allocate_plays(corpus)
    members = [p for group in plays.values() for p in group]
    assert sorted(members) == sorted(corpus)
    sizes = {name: size for name, _, size in cells.AGENTX_SEGMENTS}
    assert {k: len(v) for k, v in plays.items()} == sizes
    with_sub = [sum(corpus[p] for p in group) for group in plays.values()]
    assert max(with_sub) - min(with_sub) <= 1
    assert plays == cells.allocate_plays(corpus)


def test_every_planned_measure_rule_is_accepted_and_ends_at_full_occupancy():
    """build audit r1 F1: no completion-basis cell may end its window at the last dispatch."""
    for plans in cells.PLANS.values():
        for plan in plans:
            rule = cells.measure_rule(plan.mode)
            basis = goodput.validate_measure(rule, open_loop=plan.mode == "open")
            assert basis == "arrival" or rule["end"] == "full_occupancy", plan
