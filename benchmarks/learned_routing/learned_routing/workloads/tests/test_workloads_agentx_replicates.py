# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRN replicates of recycled AgentX traces and per-segment play subsets (calibration stage)."""

from __future__ import annotations

import json

import pytest
from learned_routing import replicates
from learned_routing.cells import Cell
from learned_routing.paths import Layout
from learned_routing.workloads import agentx_lowered as ax
from test_workloads_agentx_lowered import REAL_BASES, play_a, play_b

CR = Layout.resolve().root


def _file_bytes(bases, header) -> bytes:
    rows, _ = ax.assemble(bases, starts_ms=[None] * len(bases))
    return ax.agentic_bytes(header, rows)


HEADER = {
    "schema": ax.AGENTIC_SCHEMA,
    "version": ax.AGENTIC_VERSION,
    "block_size": 64,
    "hash_id_scope": "local",
    "source": {"format": "weka", "digest": "test"},
}


@pytest.mark.parametrize("seed", [0, 1, 7, 12345])
def test_permutation_equals_generating_the_permuted_draws(seed):
    bases = [play_a(), play_b(), play_a(), play_b(), play_b(), play_a(), play_a()]
    source = _file_bytes(bases, HEADER).decode().splitlines()
    lines, stats = replicates.permute_agentic_copies(source, seed)
    order = replicates._shuffle(list(range(len(bases))), seed, "copies")
    expected = _file_bytes([bases[i] for i in order], HEADER)
    assert ("\n".join(lines) + "\n").encode() == expected
    assert stats.units == len(bases) and not stats.degenerate
    assert stats.moved == sum(i != j for i, j in enumerate(order))


def test_permutation_is_a_bijection_on_copies():
    bases = [play_a(), play_b(), play_b(), play_a()]
    source = _file_bytes(bases, HEADER).decode().splitlines()
    lines, _ = replicates.permute_agentic_copies(source, 3)
    rows = [json.loads(line) for line in lines[1:]]
    ordinals = sorted({row["source_play_ordinal"] for row in rows})
    assert ordinals == [0, 1, 2, 3]
    # every row's ids carry its own ordinal's label, and dependencies stay inside the copy
    for row in rows:
        label = f"lrx:{row['source_play_ordinal']:06d}:"
        assert row["request_id"].startswith(label)
        assert row["play_id"].startswith(label)
        assert row["session_id"].startswith(label)
        for dep in row.get("dependencies", []):
            assert dep["request_id"].startswith(label)
    # hash ranges stay contiguous and disjoint in the new copy order
    ranges = {}
    for row in rows:
        lo, hi = ranges.get(row["source_play_ordinal"], (10**18, -1))
        ranges[row["source_play_ordinal"]] = (
            min(lo, *row["hash_ids"]),
            max(hi, *row["hash_ids"]),
        )
    expected_low = 1
    for ordinal in ordinals:
        lo, hi = ranges[ordinal]
        assert lo == expected_low
        expected_low = hi + 1


def test_permutation_rejects_unlabeled_rows():
    header = json.dumps(HEADER)
    row = json.dumps({"request_id": "weka:x", "hash_ids": [1]})
    with pytest.raises(ValueError, match="lrx"):
        replicates.permute_agentic_copies([header, row], 0)


def test_materialize_agentic_replicate(tmp_path):
    bases = [play_a(), play_b(), play_a()]
    source = tmp_path / "src.jsonl"
    source.write_bytes(_file_bytes(bases, HEADER))
    rep0 = replicates.materialize_replicate(
        source, "agentic_mooncake", 0, tmp_path / "r"
    )
    rep0b = replicates.materialize_replicate(
        source, "agentic_mooncake", 0, tmp_path / "r"
    )
    rep1 = replicates.materialize_replicate(
        source, "agentic_mooncake", 1, tmp_path / "r"
    )
    assert rep0.sha256 == rep0b.sha256
    assert rep0.policy_seed == 1 and rep1.policy_seed == 2
    assert rep0.path.read_bytes().startswith(source.read_bytes().split(b"\n")[0])


def _lanes_cell(tmp_path, trace_format):
    raw = {
        "cell_id": "c",
        "num_workers": 2,
        "trace_format": trace_format,
        "trace_files": [str(tmp_path / "t.jsonl")],
        "load": {"mode": "agentic_lanes", "value": 6, "level": "L2"},
    }
    return Cell(raw=raw, layout=Layout(tmp_path))


def test_lanes_accept_agentic_mooncake(tmp_path):
    assert _lanes_cell(tmp_path, "agentic_mooncake").load_kwargs(None) == {
        "agentic_lanes": 6
    }
    assert _lanes_cell(tmp_path, "weka").load_kwargs(None) == {"agentic_lanes": 6}
    with pytest.raises(ValueError, match="agentic_lanes"):
        _lanes_cell(tmp_path, "mooncake").load_kwargs(None)


def test_genspec_plays_key_and_validation():
    base = ax.GenSpec(split="train", mode="closed", num_copies=4, seed=0)
    assert "plays" not in base.key_dict()
    sub = ax.GenSpec(
        split="train", mode="closed", num_copies=4, seed=0, plays=("a.json", "b.json")
    )
    assert sub.key_dict()["plays"] == ["a.json", "b.json"]
    with pytest.raises(ValueError, match="sorted"):
        ax.GenSpec(
            split="train", mode="closed", num_copies=4, seed=0, plays=("b", "a")
        ).validate()


@pytest.mark.skipif(not REAL_BASES, reason="campaign traces not present")
def test_real_subset_draws_stay_in_the_subset():
    manifest = CR / "cells" / "SPLIT_MANIFEST.json"
    plays = ax.subset_plays(manifest, "train", ["A1"])
    assert len(plays) == 11
    spec = ax.GenSpec(split="train", mode="closed", num_copies=40, seed=3, plays=plays)
    _, rows, meta = ax.generate(spec, CR)
    assert set(meta["draw_counts"]) <= set(plays)
    assert meta["pool"] == list(plays)
    with pytest.raises(ValueError, match="belongs to split"):
        ax.subset_plays(manifest, "train", ["T1"])
    outside = ax.subset_plays(manifest, "test", ["T1"])
    with pytest.raises(ValueError, match="split purity"):
        ax.generate(
            ax.GenSpec(
                split="train", mode="closed", num_copies=4, seed=0, plays=outside
            ),
            CR,
        )
