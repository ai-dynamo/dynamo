# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invariants of the recycled AgentX lowering (learned_routing.workloads.agentx_lowered)."""

from __future__ import annotations

import json
import math
from collections import defaultdict

import pytest
from learned_routing.paths import Layout
from learned_routing.workloads import agentx_lowered as ax

CR = Layout.resolve().root
REAL_BASES = (CR / "traces/agentx_lowered/base_cap300/MANIFEST.json").exists() and (
    CR / "cells/SPLIT_MANIFEST.json"
).exists()


def _row(rid, session, nb, hashes, deps=(), out=10):
    return {
        "request_id": f"weka:{rid}",
        "play_id": "weka:ns:play:p",
        "source_play_ordinal": 0,
        "session_id": f"weka:{session}",
        "model": "m",
        "input_length": 64 * len(hashes),
        "output_length": out,
        "hash_ids": list(hashes),
        "not_before_ms": nb,
        "recorded_api_time_ms": 5.0,
        **({"dependencies": [dict(d) for d in deps]} if deps else {}),
    }


def _dep(target, relation="sequence", trigger="completion", delay=7.5):
    return {
        "request_id": f"weka:{target}",
        "trigger": trigger,
        "delay_ms": delay,
        "relation": relation,
    }


def play_a() -> ax.BasePlay:
    """Root, a sequenced follow-up sharing its prefix, a dispatch-spawned subagent and a join."""
    rows = [
        _row("ns:request:outer:0", "ns:session:root", 0.0, [11, 12, 13]),
        _row(
            "ns:request:outer:1",
            "ns:session:root",
            900.0,
            [11, 12, 13, 14],
            [_dep("ns:request:outer:0")],
        ),
        _row(
            "ns:request:outer:1:inner:0",
            "ns:session:subagent:1:a",
            400.0,
            [21, 22],
            [_dep("ns:request:outer:0", "spawn", "dispatch", 400.0)],
            out=0,
        ),
        _row(
            "ns:request:outer:2",
            "ns:session:root",
            2000.0,
            [11, 12, 13, 14, 15],
            [
                _dep("ns:request:outer:1"),
                _dep("ns:request:outer:1:inner:0", "join", "completion", 3.0),
            ],
        ),
    ]
    rows.sort(key=lambda r: r["request_id"])
    return ax.BasePlay(
        name="a.json",
        header={"block_size": 64, "hash_id_scope": "local"},
        rows=tuple(rows),
    )


def play_b() -> ax.BasePlay:
    rows = [
        _row("nb:request:outer:0", "nb:session:root", 0.0, [11, 99]),
        _row(
            "nb:request:outer:1",
            "nb:session:root",
            50.0,
            [11, 99, 100],
            [_dep("nb:request:outer:0")],
        ),
    ]
    return ax.BasePlay(
        name="b.json",
        header={"block_size": 64, "hash_id_scope": "local"},
        rows=tuple(rows),
    )


def by_copy(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[":".join(row["request_id"].split(":")[:2])].append(row)
    return groups


def strip(text: str) -> str:
    return text.split(":", 2)[2]


def equality_pattern(rows):
    flat = [h for row in rows for h in row["hash_ids"]]
    return [[flat.index(h) for h in row["hash_ids"]] for row in rows]


@pytest.mark.parametrize("starts", [[None] * 6, [0.0, 10.0, 10.0, 25.5, 40.0, 41.0]])
def test_copy_isolation_no_hash_collisions_across_copies(starts):
    bases = [play_a(), play_b(), play_a(), play_a(), play_b(), play_a()]
    rows, copies = ax.assemble(bases, starts_ms=starts)
    groups = by_copy(rows)
    seen = {}
    for copy in copies:
        hashes = {h for row in groups[copy["label"]] for h in row["hash_ids"]}
        lo, hi = copy["hash_range"]
        assert hashes == set(range(lo, hi))
        for h in hashes:
            assert seen.setdefault(h, copy["label"]) == copy["label"]
    # consecutive, disjoint ranges
    for prev, nxt in zip(copies, copies[1:]):
        assert prev["hash_range"][1] == nxt["hash_range"][0]
    # equal source hashes stay equal within a copy (same equality pattern as the base)
    for base, copy in zip(bases, copies):
        assert equality_pattern(groups[copy["label"]]) == equality_pattern(base.rows)


def test_ids_unique_and_session_structure_kept():
    bases = [play_a(), play_a(), play_b()]
    rows, copies = ax.assemble(bases, starts_ms=[None] * 3)
    ids = [row["request_id"] for row in rows]
    assert len(ids) == len(set(ids))
    groups = by_copy(rows)
    sessions_seen, plays_seen = set(), set()
    for base, copy in zip(bases, copies):
        mine = groups[copy["label"]]
        sessions = {row["session_id"] for row in mine}
        assert len(sessions) == len({row["session_id"] for row in base.rows})
        assert {strip(s) for s in sessions} == {row["session_id"] for row in base.rows}
        assert not sessions & sessions_seen
        sessions_seen |= sessions
        (play_id,) = {row["play_id"] for row in mine}
        assert play_id not in plays_seen
        plays_seen.add(play_id)
        local = {row["request_id"] for row in mine}
        for row in mine:
            for dep in row.get("dependencies", []):
                assert dep["request_id"] in local
    assert rows == sorted(rows, key=lambda r: r["request_id"])
    assert [c["ordinal"] for c in copies] == [0, 1, 2]
    for copy in copies:
        assert {r["source_play_ordinal"] for r in groups[copy["label"]]} == {
            copy["ordinal"]
        }


@pytest.mark.parametrize("mode", ["closed", "open"])
def test_dependency_edges_and_relative_timing_unchanged(mode):
    bases = [play_a(), play_b(), play_a()]
    starts = [None] * 3 if mode == "closed" else [0.0, 123.456, 999.0]
    rows, copies = ax.assemble(bases, starts_ms=starts)
    groups = by_copy(rows)
    for base, copy, start in zip(bases, copies, starts):
        mine = {strip(row["request_id"]): row for row in groups[copy["label"]]}
        root_nb = ax.root_not_before_ms(base.rows)
        for original in base.rows:
            new = mine[original["request_id"]]
            assert [
                {**d, "request_id": strip(d["request_id"])}
                for d in new.get("dependencies", [])
            ] == original.get("dependencies", [])
            for key in (
                "model",
                "input_length",
                "output_length",
                "recorded_api_time_ms",
            ):
                assert new[key] == original[key]
            if start is None:
                assert new["not_before_ms"] == original["not_before_ms"]
            else:
                assert new["not_before_ms"] == start + max(
                    original["not_before_ms"] - root_nb, 0.0
                )


def test_label_ranks_change_only_labels():
    bases = [play_a(), play_b(), play_a()]
    rows, copies = ax.assemble(bases, starts_ms=[None] * 3, label_ranks=[2, 0, 1])
    assert [c["label"] for c in copies] == [
        ax.copy_label(2),
        ax.copy_label(0),
        ax.copy_label(1),
    ]
    assert [c["ordinal"] for c in copies] == [0, 1, 2]
    with pytest.raises(ValueError):
        ax.assemble(bases, starts_ms=[None] * 3, label_ranks=[0, 0, 1])


def test_poisson_seeding_determinism():
    a = ax.poisson_arrivals_ms(5000, 7, 0.25)
    assert a == ax.poisson_arrivals_ms(5000, 7, 0.25)
    assert a != ax.poisson_arrivals_ms(5000, 8, 0.25)
    assert a[0] == 0.0 and all(x <= y for x, y in zip(a, a[1:]))
    assert all(round(x * 1e6) == pytest.approx(x * 1e6, abs=1e-3) for x in a[:200])
    mean_gap_s = a[-1] / 1000.0 / (len(a) - 1)
    assert mean_gap_s == pytest.approx(4.0, rel=0.05)
    # the normalized arrival sequence is shared across rates (CRN across worker counts)
    b = ax.poisson_arrivals_ms(5000, 7, 0.5)
    assert all(abs(2 * y - x) <= 2e-6 for x, y in zip(a, b))
    with pytest.raises(ValueError):
        ax.poisson_arrivals_ms(3, 0, 0.0)


def test_draws_are_seeded_and_stay_in_the_pool():
    pool = [f"p{i}.json" for i in range(7)]
    draws = ax.draw_plays(pool, "val", 400, 3)
    assert draws == ax.draw_plays(pool, "val", 400, 3)
    assert draws != ax.draw_plays(pool, "val", 400, 4)
    assert set(draws) == set(pool)
    assert draws[:50] == ax.draw_plays(pool, "val", 50, 3)


def test_split_pool_purity(tmp_path):
    manifest = {
        "agentx_play_subsets": {
            "A1": {"split": "train", "plays": ["b.json", "a.json"]},
            "A2": {"split": "train", "plays": ["c.json"]},
            "V1": {"split": "val", "plays": ["d.json"]},
            "T1": {"split": "test", "plays": ["e.json", "f.json"]},
        }
    }
    path = tmp_path / "SPLIT_MANIFEST.json"
    path.write_text(json.dumps(manifest))
    pools = ax.split_play_pools(path)
    assert pools == {
        "train": ["a.json", "b.json", "c.json"],
        "val": ["d.json"],
        "test": ["e.json", "f.json"],
    }
    manifest["agentx_play_subsets"]["V1"]["plays"].append("a.json")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ax.split_play_pools(path)


@pytest.mark.skipif(not REAL_BASES, reason="campaign base lowering not built")
def test_real_pools_and_generation_are_pure_and_deterministic():
    pools = ax.split_play_pools(CR / "cells/SPLIT_MANIFEST.json")
    assert {k: len(v) for k, v in pools.items()} == {"train": 33, "val": 21, "test": 28}
    assert not (
        set(pools["train"]) & set(pools["val"])
        or set(pools["train"]) & set(pools["test"])
    )
    assert not set(pools["val"]) & set(pools["test"])
    for split in ax.SPLITS:
        spec = ax.GenSpec(split=split, mode="closed", num_copies=60, seed=1)
        header, rows, meta = ax.generate(spec, CR)
        assert {c["play"] for c in meta["copies"]} <= set(pools[split])
        assert header["block_size"] == 64 and header["hash_id_scope"] == "local"
        again = ax.generate(spec, CR)
        assert ax.agentic_bytes(header, rows) == ax.agentic_bytes(again[0], again[1])
    spec = ax.GenSpec(
        split="val", mode="open", num_copies=30, seed=2, play_rate_per_s=0.01
    )
    header, rows, meta = ax.generate(spec, CR)
    groups = by_copy(rows)
    for copy in meta["copies"]:
        roots = [r for r in groups[copy["label"]] if not r.get("dependencies")]
        assert [r["not_before_ms"] for r in roots] == [copy["start_ms"]]
    with pytest.raises(ValueError):
        ax.GenSpec(split="val", mode="open", num_copies=3, seed=0).validate()


def test_occupancy_and_compare_helpers():
    occ = ax._occupancy([(0.0, 150.0), (50.0, 100.0), (250.0, 260.0)], 100.0, 3)
    assert occ == pytest.approx([1.5, 0.5, 0.1])
    assert ax._merge([(0, 2), (1, 3), (5, 6)]) == [(0, 3), (5, 6)]
    ref = [
        {
            "request_id": "weka:x:request:0",
            "ttft_ms": 1.0,
            "agentic": {"root_id": "weka:x:request:0"},
        }
    ]
    cand = [
        {
            "request_id": "lrx:000003:weka:x:request:0",
            "ttft_ms": 1.0,
            "agentic": {"root_id": "lrx:000003:weka:x:request:0"},
        }
    ]
    assert ax.compare_per_request(ref, cand, ax.prefix_renamer({"lrx:000003:": ""}))[
        "identical"
    ]
    cand[0]["ttft_ms"] = 2.0
    result = ax.compare_per_request(ref, cand, ax.prefix_renamer({"lrx:000003:": ""}))
    assert not result["identical"] and result["differing_fields"] == {"ttft_ms": 1}
    assert not math.isnan(occ[0])
