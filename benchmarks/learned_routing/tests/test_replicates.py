# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter

import pytest
from learned_routing import noise, replicates


def mooncake_line(timestamp, input_length, **extra):
    row = {"timestamp": timestamp, "input_length": input_length, "output_length": 8}
    row.update(extra)
    row["hash_ids"] = [input_length]
    return json.dumps(row)


def burst_trace(bursts=5, size=12):
    return [
        mooncake_line(3000 * burst, 1000 * burst + index)
        for burst in range(bursts)
        for index in range(size)
    ]


def test_mooncake_permutation_only_reorders_within_equal_arrivals():
    lines = burst_trace()
    permuted, stats = replicates.permute_mooncake_ties(lines, seed=7)

    assert Counter(permuted) == Counter(lines)
    timestamps = [json.loads(line)["timestamp"] for line in permuted]
    assert timestamps == sorted(timestamps)
    assert stats.tie_groups == 5 and stats.units_in_ties == 60
    assert stats.moved > 0 and permuted != lines


def test_replicates_differ_by_k_and_repeat_exactly():
    lines = burst_trace()
    seeds = [replicates.replicate_seed("trace-sha", k) for k in range(4)]
    orders = [replicates.permute_mooncake_ties(lines, s)[0] for s in seeds]

    assert len({tuple(order) for order in orders}) == 4
    assert replicates.permute_mooncake_ties(lines, seeds[2])[0] == orders[2]
    assert all(0 <= seed < 2**63 for seed in seeds)


def test_sessions_keep_turn_order_and_move_as_a_unit():
    lines = [
        json.dumps(
            {
                "session_id": "a",
                "timestamp": 0,
                "input_length": 1,
                "output_length": 1,
                "hash_ids": [1],
            }
        ),
        json.dumps(
            {
                "session_id": "b",
                "timestamp": 0,
                "input_length": 2,
                "output_length": 1,
                "hash_ids": [2],
            }
        ),
        json.dumps(
            {
                "session_id": "a",
                "delay": 5.0,
                "input_length": 3,
                "output_length": 1,
                "hash_ids": [1, 3],
            }
        ),
        json.dumps(
            {
                "session_id": "c",
                "timestamp": 0,
                "input_length": 4,
                "output_length": 1,
                "hash_ids": [4],
            }
        ),
        json.dumps(
            {
                "session_id": "b",
                "delay": 7.0,
                "input_length": 5,
                "output_length": 1,
                "hash_ids": [2, 5],
            }
        ),
    ]
    seen_first = set()
    for k in range(16):
        permuted, stats = replicates.permute_mooncake_ties(
            lines, replicates.replicate_seed("sessions", k)
        )
        rows = [json.loads(line) for line in permuted]
        for session in "abc":
            lengths = [r["input_length"] for r in rows if r["session_id"] == session]
            assert lengths == sorted(lengths)
        seen_first.add(rows[0]["session_id"])
        assert stats.units == 3 and stats.units_in_ties == 3
    assert seen_first == {"a", "b", "c"}


def test_trace_without_ties_is_degenerate_and_unchanged():
    lines = [mooncake_line(t, t + 1) for t in range(10)]
    permuted, stats = replicates.permute_mooncake_ties(lines, seed=3)

    assert permuted == lines
    assert stats.degenerate


def test_dependency_rows_are_rejected():
    with pytest.raises(ValueError, match="wait_for"):
        replicates.permute_mooncake_ties([mooncake_line(0, 1, wait_for=["x"])], seed=1)


def test_weka_directory_loads_in_loader_order_and_permutes_plays(tmp_path):
    out_dir = tmp_path.parent / f"{tmp_path.name}-out"
    plays = {
        "b/2.json": {"id": "p2", "requests": []},
        "a.json": {"id": "p0", "requests": []},
        "b/1.jsonl": None,
    }
    (tmp_path / "b").mkdir()
    for name, play in plays.items():
        if play is not None:
            (tmp_path / name).write_text(json.dumps(play, indent=2))
    (tmp_path / "b/1.jsonl").write_text(
        json.dumps({"id": "p1a", "requests": []})
        + "\n"
        + json.dumps({"id": "p1b", "requests": []})
        + "\n"
    )

    loaded = [json.loads(p)["id"] for p in replicates.load_weka_plays(tmp_path)]
    assert loaded == ["p0", "p1a", "p1b", "p2"]

    with pytest.raises(ValueError, match="inside the source"):
        replicates.materialize_replicate(tmp_path, "weka", 0, tmp_path / "out")
    first = replicates.materialize_replicate(tmp_path, "weka", 0, out_dir)
    again = replicates.materialize_replicate(tmp_path, "weka", 0, out_dir)
    assert first.sha256 == again.sha256 and first.path == again.path
    ids = [json.loads(line)["id"] for line in first.path.read_text().splitlines()]
    assert sorted(ids) == sorted(loaded)
    assert first.policy_seed == 1 and not first.stats.degenerate


def test_materialized_mooncake_replicates_share_source_key(tmp_path):
    source = tmp_path / "trace.jsonl"
    source.write_text("\n".join(burst_trace()) + "\n")

    reps = [
        replicates.materialize_replicate(source, "mooncake", k, tmp_path / "out")
        for k in range(3)
    ]
    assert len({r.source_sha256 for r in reps}) == 1
    assert len({r.sha256 for r in reps}) == 3
    assert [r.policy_seed for r in reps] == [1, 2, 3]
    for r in reps:
        assert Counter(r.path.read_text().splitlines()) == Counter(burst_trace())


def test_identical_reruns_cannot_certify_a_difference():
    # A deterministic policy re-run on one workload has zero spread: the false floor of audit F1.
    ratios = {
        "c1": [1.05, 1.05, 1.05],
        "c2": [1.04, 1.04, 1.04],
        "c3": [1.06, 1.06, 1.06],
    }
    verdict = noise.differs_beyond_noise(ratios, {c: c for c in ratios})

    assert verdict.degenerate and not verdict.differs


def test_order_noise_hides_a_small_gain_but_not_a_large_one():
    segments = {f"c{i}": f"s{i}" for i in range(4)}
    noisy_small = {
        "c0": [1.03, 0.99, 1.06, 1.00],
        "c1": [0.98, 1.04, 1.01, 1.05],
        "c2": [1.02, 1.07, 0.98, 1.00],
        "c3": [1.05, 0.99, 1.03, 1.01],
    }
    small = noise.differs_beyond_noise(noisy_small, segments)
    assert not small.cell_clause and not small.differs

    large = {cell: [r + 0.12 for r in ratios] for cell, ratios in noisy_small.items()}
    verdict = noise.differs_beyond_noise(large, segments)
    assert verdict.cell_clause and verdict.segment_clause and verdict.differs


def test_cells_sharing_a_segment_count_once():
    ratios = {"a4": [1.2, 1.1], "a8": [1.2, 1.1], "b4": [1.2, 1.1]}
    verdict = noise.differs_beyond_noise(ratios, {"a4": "a", "a8": "a", "b4": "b"})

    assert verdict.segments == 2 and not verdict.segment_clause


def test_paired_ratios_require_common_replicates():
    policy = {("c", 0): 2.0, ("c", 1): 2.2}
    with pytest.raises(ValueError, match="differ"):
        noise.paired_ratios(policy, {("c", 0): 2.0, ("c", 2): 2.0})
    assert noise.paired_ratios(policy, {("c", 0): 2.0, ("c", 1): 2.0}) == {
        "c": [1.0, 1.1]
    }
