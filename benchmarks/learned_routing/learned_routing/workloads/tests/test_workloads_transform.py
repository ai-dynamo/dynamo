# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invariants of the derived-trace transforms (learned_routing.workloads.transform)."""

from __future__ import annotations

import itertools
import json
import random

import pytest
from learned_routing.workloads.transform import (
    TransformSpec,
    busy_periods,
    materialize,
    transform_mooncake,
    transform_weka,
)

BS = 16


def chained_trace(
    seed: int = 0, roots: int = 3, families: int = 12, rows: int = 160
) -> list[dict]:
    """Prefix-chained single-turn rows: root segment, family prefix, then a unique suffix."""
    rng = random.Random(seed)
    next_id = itertools.count()
    root_blocks = [
        [next(next_id) for _ in range(rng.randint(1, 4))] for _ in range(roots)
    ]
    family_blocks = []
    for f in range(families):
        root = root_blocks[f % roots]
        family_blocks.append(root + [next(next_id) for _ in range(rng.randint(1, 5))])
    out = []
    for i in range(rows):
        fam = family_blocks[rng.randrange(families)]
        depth = rng.randint(1, len(fam))
        ids = fam[:depth] + [next(next_id) for _ in range(rng.randint(0, 6))]
        tail = rng.randint(1, BS)
        out.append(
            {
                "timestamp": 1000 * (i // 4),
                "input_length": BS * (len(ids) - 1) + tail,
                "output_length": rng.randint(1, 50),
                "hash_ids": ids,
            }
        )
    return out


def lcp(a: list, b: list) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def shared_blocks(rows: list[dict]) -> set:
    counts: dict = {}
    for row in rows:
        for h in set(row["hash_ids"]):
            counts[h] = counts.get(h, 0) + 1
    return {h for h, c in counts.items() if c >= 2}


def test_identity_transform_is_a_hash_bijection():
    rows = chained_trace()
    out, stats = transform_mooncake(rows, TransformSpec(), BS)
    assert stats.rows_out == len(rows)
    forward: dict = {}
    for a, b in zip(rows, out):
        assert {k: v for k, v in a.items() if k != "hash_ids"} == {
            k: v for k, v in b.items() if k != "hash_ids"
        }
        assert len(a["hash_ids"]) == len(b["hash_ids"])
        for x, y in zip(a["hash_ids"], b["hash_ids"]):
            assert forward.setdefault(x, y) == y
    assert len(set(forward.values())) == len(forward)
    assert sorted(forward.values()) == list(range(len(forward)))  # dense u32-safe ids


def test_materialize_is_deterministic_and_content_addressed(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in chained_trace()))
    spec = TransformSpec.from_dict(
        {
            "isl_prefix_mult": 1.5,
            "isl_unique_mult": 0.7,
            "prefix_root_mult": 2,
            "seed": 3,
        }
    )
    a = materialize(src, "mooncake", spec, BS, tmp_path / "d1")
    b = materialize(src, "mooncake", spec, BS, tmp_path / "d2")
    assert a.sha256 == b.sha256 and a.path.read_bytes() == b.path.read_bytes()
    assert a.path.name == f"{a.sha256}.jsonl"
    assert a.meta["source"]["sha256"] == b.meta["source"]["sha256"]
    other = materialize(
        src,
        "mooncake",
        TransformSpec.from_dict({**spec.to_dict(), "seed": 4}),
        BS,
        tmp_path / "d3",
    )
    assert other.spec_key != a.spec_key


def test_window_keeps_whole_sessions_and_rebases_to_first_arrival():
    rows = [
        {
            "session_id": "a",
            "timestamp": 500,
            "input_length": 16,
            "output_length": 1,
            "hash_ids": [0],
        },
        {
            "session_id": "a",
            "delay": 10.0,
            "input_length": 32,
            "output_length": 1,
            "hash_ids": [0, 1],
        },
        {
            "session_id": "b",
            "timestamp": 2500,
            "input_length": 16,
            "output_length": 1,
            "hash_ids": [2],
        },
        {
            "session_id": "b",
            "delay": 99999.0,
            "input_length": 32,
            "output_length": 1,
            "hash_ids": [2, 3],
        },
        {
            "session_id": "c",
            "timestamp": 4000,
            "input_length": 16,
            "output_length": 1,
            "hash_ids": [4],
        },
    ]
    out, stats = transform_mooncake(rows, TransformSpec(window=(1000.0, 4000.0)), BS)
    assert [r["session_id"] for r in out] == ["b", "b"]
    assert (
        out[0]["timestamp"] == 0
        and "timestamp" not in out[1]
        and out[1]["delay"] == 99999.0
    )
    assert stats.extra["window_rebase_ms"] == 2500.0


def test_integer_prefix_mult_scales_every_shared_prefix_exactly():
    rows = chained_trace(seed=1)
    shared = shared_blocks(rows)
    out, _ = transform_mooncake(rows, TransformSpec(isl_prefix_mult=2.0), BS)
    for (i, a), (j, b) in itertools.combinations(enumerate(rows), 2):
        assert lcp(out[i]["hash_ids"], out[j]["hash_ids"]) == 2 * lcp(
            a["hash_ids"], b["hash_ids"]
        )
    for a, b in zip(rows, out):
        n_shared = sum(h in shared for h in a["hash_ids"])
        assert len(b["hash_ids"]) == 2 * n_shared + (len(a["hash_ids"]) - n_shared)
        tail = a["input_length"] - BS * (len(a["hash_ids"]) - 1)
        assert b["input_length"] == BS * (len(b["hash_ids"]) - 1) + tail


def test_fractional_prefix_mult_keeps_hash_identity_consistent():
    rows = chained_trace(seed=2, rows=300)
    out, _ = transform_mooncake(rows, TransformSpec(isl_prefix_mult=1.5, seed=7), BS)
    # Consistency: two pairs that share the same source prefix share the same output prefix.
    by_prefix: dict = {}
    for (i, a), (j, b) in itertools.combinations(enumerate(rows), 2):
        k = lcp(a["hash_ids"], b["hash_ids"])
        if k:
            by_prefix.setdefault(tuple(a["hash_ids"][:k]), set()).add(
                lcp(out[i]["hash_ids"], out[j]["hash_ids"])
            )
        else:
            assert lcp(out[i]["hash_ids"], out[j]["hash_ids"]) == 0  # no false sharing
    assert all(len(lengths) == 1 for lengths in by_prefix.values())
    shared = shared_blocks(rows)
    src_shared = sum(1 for h in shared)
    out_shared = len(shared_blocks(out))
    assert 1.25 < out_shared / src_shared < 1.75


def test_unique_mult_scales_only_the_unique_suffix():
    rows = chained_trace(seed=3)
    shared = shared_blocks(rows)
    out, _ = transform_mooncake(rows, TransformSpec(isl_unique_mult=3.0), BS)
    for a, b in zip(rows, out):
        n_shared = sum(h in shared for h in a["hash_ids"])
        n_unique = len(a["hash_ids"]) - n_shared
        assert len(b["hash_ids"]) == max(1, n_shared + 3 * n_unique)
    for (i, a), (j, b) in itertools.combinations(enumerate(rows), 2):
        assert lcp(out[i]["hash_ids"], out[j]["hash_ids"]) == lcp(
            a["hash_ids"], b["hash_ids"]
        )


def test_prefix_root_mult_splits_roots_and_keeps_families():
    rows = chained_trace(seed=4, roots=2, families=20, rows=200)
    out, stats = transform_mooncake(rows, TransformSpec(prefix_root_mult=3, seed=5), BS)
    assert stats.prefix_root_roots == 2
    first_src = {r["hash_ids"][0] for r in rows}
    first_out = {r["hash_ids"][0] for r in out}
    assert len(first_src) < len(first_out) <= 3 * len(first_src)
    common: dict = {}
    for r in rows:
        root = r["hash_ids"][0]
        common[root] = (
            r["hash_ids"]
            if root not in common
            else common[root][: lcp(common[root], r["hash_ids"])]
        )
    for (i, a), (j, b) in itertools.combinations(enumerate(rows), 2):
        src = lcp(a["hash_ids"], b["hash_ids"])
        got = lcp(out[i]["hash_ids"], out[j]["hash_ids"])
        segment = len(common[a["hash_ids"][0]])
        assert got in (
            0,
            src,
        )  # a pair either lands in one copy (sharing kept) or in two (none)
        if src > segment:
            assert (
                got == src
            )  # sharing beyond the root segment means the same family, same copy


def test_prefix_root_mult_one_is_identity_up_to_renumbering():
    rows = chained_trace(seed=5)
    a, _ = transform_mooncake(rows, TransformSpec(), BS)
    b, _ = transform_mooncake(rows, TransformSpec(prefix_root_mult=1, seed=9), BS)
    assert a == b


def test_osl_mult_and_context_cap():
    rows = [
        {
            "timestamp": 0,
            "input_length": 100,
            "output_length": 10,
            "hash_ids": list(range(7)),
        },
        {
            "timestamp": 0,
            "input_length": 200,
            "output_length": 3,
            "hash_ids": list(range(10, 23)),
        },
        {
            "timestamp": 0,
            "input_length": 300,
            "output_length": 1,
            "hash_ids": list(range(30, 49)),
        },
    ]
    out, stats = transform_mooncake(
        rows, TransformSpec(osl_mult=0.25, max_model_len=250), BS
    )
    assert [r["input_length"] for r in out] == [100, 200]
    assert stats.dropped_isl_at_cap == 1
    assert out[0]["output_length"] == 2  # round(2.5) with banker's rounding
    assert out[1]["output_length"] == 1  # floored at one token
    out, stats = transform_mooncake(
        rows[:1], TransformSpec(osl_mult=40.0, max_model_len=250), BS
    )
    assert out[0]["output_length"] == 150 and stats.clamped_osl_at_cap == 1


def test_think_mult_scales_session_delays_and_rejects_flat_traces():
    rows = [
        {
            "session_id": "a",
            "timestamp": 0,
            "input_length": 16,
            "output_length": 1,
            "hash_ids": [0],
        },
        {
            "session_id": "a",
            "delay": 400.0,
            "input_length": 32,
            "output_length": 1,
            "hash_ids": [0, 1],
        },
    ]
    out, _ = transform_mooncake(rows, TransformSpec(think_mult=2.5), BS)
    assert out[1]["delay"] == 1000.0 and out[0]["timestamp"] == 0
    out, stats = transform_mooncake(rows, TransformSpec(think_cap_s=0.1), BS)
    assert out[1]["delay"] == 100.0 and stats.extra["capped_delays"] == 1
    with pytest.raises(ValueError, match="sessions"):
        transform_mooncake(chained_trace(), TransformSpec(think_mult=2.0), BS)


def weka_play() -> dict:
    """Root stream with a long idle gap, plus a blocking subagent with absolute timestamps."""

    def req(t, api, inp=128, out=10, hashes=2):
        return {
            "type": "n",
            "t": t,
            "api_time": api,
            "model": "m",
            "in": inp,
            "out": out,
            "hash_ids": list(range(hashes)),
        }

    return {
        "id": "p1",
        "block_size": 64,
        "hash_id_scope": "local",
        "models": ["m"],
        "requests": [
            req(100.0, 5.0),
            req(110.0, 4.0, inp=192, hashes=3),
            {
                "type": "subagent",
                "t": 115.0,
                "agent_id": "x",
                "subagent_type": "t",
                "status": "completed",
                "duration_ms": 20000,
                "requests": [req(116.0, 3.0), req(125.0, 6.0, inp=192, hashes=3)],
            },
            req(5000.0, 2.0, inp=256, hashes=4),
            req(5010.0, 2.0, inp=320, out=0, hashes=5),
        ],
    }


def all_times(play: dict) -> list[float]:
    out = []
    for e in play["requests"]:
        out.append(e["t"])
        for r in e.get("requests", []):
            out.append(r["t"])
    return out


def test_weka_idle_cap_compresses_only_long_gaps_and_keeps_busy_periods():
    play = weka_play()
    (out,), stats = transform_weka([("p1.json", play)], TransformSpec(think_cap_s=60.0))
    before, after = busy_periods(play), busy_periods(out)
    assert [round(e - s, 9) for s, e in before] == [round(e - s, 9) for s, e in after]
    gaps_before = [b[0] - a[1] for a, b in zip(before, before[1:])]
    gaps_after = [b[0] - a[1] for a, b in zip(after, after[1:])]
    for g0, g1 in zip(gaps_before, gaps_after):
        assert g1 == pytest.approx(min(g0, 60.0))
    times = all_times(out)
    assert all_times(play) == sorted(all_times(play)) and times == sorted(times)
    assert stats.extra["capped_idle_gaps"] == 1
    # The subagent's recorded end still falls after its last request and before the join target.
    marker = out["requests"][2]
    end = marker["t"] + marker["duration_ms"] / 1000.0
    last_inner = marker["requests"][-1]
    assert (
        last_inner["t"] + last_inner["api_time"] <= end + 1e-3 < out["requests"][3]["t"]
    )


def test_weka_think_mult_dilates_every_delay():
    play = weka_play()
    (out,), _ = transform_weka([("p1.json", play)], TransformSpec(think_mult=2.0))
    t0, t1 = all_times(play), all_times(out)
    for (a0, a1), (b0, b1) in itertools.combinations(list(zip(t0, t1)), 2):
        assert b1 - a1 == pytest.approx(2.0 * (b0 - a0))
    assert out["requests"][0]["api_time"] == 10.0
    assert out["requests"][2]["duration_ms"] == 40000


def test_weka_osl_keeps_zero_outputs_and_clamps_at_cap():
    play = weka_play()
    (out,), stats = transform_weka(
        [("p1.json", play)], TransformSpec(osl_mult=100.0, max_model_len=1200)
    )
    outs = [
        r["out"]
        for e in out["requests"]
        for r in ([e] if e["type"] == "n" else e["requests"])
    ]
    ins = [
        r["in"]
        for e in out["requests"]
        for r in ([e] if e["type"] == "n" else e["requests"])
    ]
    assert 0 in outs
    assert all(i + max(o, 1) <= 1200 for i, o in zip(ins, outs))
    assert stats.clamped_osl_at_cap > 0


def test_weka_rejects_mooncake_only_transforms():
    for spec in ({"isl_unique_mult": 2.0}, {"prefix_root_mult": 2}, {"window": [0, 1]}):
        with pytest.raises(ValueError):
            transform_weka([("p1.json", weka_play())], TransformSpec.from_dict(spec))


def test_spec_rejects_unknown_keys_and_bad_values():
    with pytest.raises(ValueError, match="unknown"):
        TransformSpec.from_dict({"isl_mult": 2.0})
    with pytest.raises(ValueError):
        TransformSpec.from_dict({"osl_mult": 0})
    with pytest.raises(ValueError):
        TransformSpec.from_dict({"prefix_root_mult": 1.5})
