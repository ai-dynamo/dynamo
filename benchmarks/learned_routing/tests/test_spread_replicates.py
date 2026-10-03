# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arrival-spread CRN replicates (protocol crn-spread-v1) for slot-quantized Mooncake traces."""

from __future__ import annotations

import json

import pytest
from learned_routing import replicates
from learned_routing.cells import Cell, resolve_replicate
from learned_routing.paths import Layout


def _trace() -> list[str]:
    rows = []
    for slot in range(4):
        for j in range(5):
            rows.append(
                {
                    "timestamp": slot * 3000,
                    "input_length": 10 + j,
                    "output_length": 2,
                    "hash_ids": [slot, j],
                }
            )
    # one two-turn session in slot 1
    rows.append(
        {
            "session_id": "s",
            "timestamp": 3000,
            "input_length": 7,
            "output_length": 3,
            "hash_ids": [9],
        }
    )
    rows.append(
        {
            "session_id": "s",
            "delay": 50.0,
            "input_length": 12,
            "output_length": 3,
            "hash_ids": [9, 10],
        }
    )
    return [json.dumps(r) for r in rows]


def test_spread_moves_first_arrivals_inside_their_slot_only():
    lines = _trace()
    out, stats = replicates.spread_mooncake_arrivals(lines, seed=11, spread_ms=3000.0)
    rows = [json.loads(x) for x in out]
    assert len(rows) == len(lines) and not stats.degenerate
    original = [json.loads(x) for x in lines]
    by_content = {(r["input_length"], tuple(r["hash_ids"])): r for r in original}
    stamps = []
    for row in rows:
        src = by_content[(row["input_length"], tuple(row["hash_ids"]))]
        if "timestamp" in src:
            assert src["timestamp"] <= row["timestamp"] < src["timestamp"] + 3000
            stamps.append(row["timestamp"])
        else:
            assert row["delay"] == src["delay"] and "timestamp" not in row
    assert stamps == sorted(stamps)
    assert len(set(stamps)) > 15  # the bursts are broken up
    # later turns stay right after their session's first turn
    first = next(i for i, r in enumerate(rows) if r.get("session_id") == "s")
    assert rows[first + 1]["session_id"] == "s" and "delay" in rows[first + 1]


def test_spread_is_seeded():
    lines = _trace()
    a, _ = replicates.spread_mooncake_arrivals(lines, seed=1, spread_ms=3000.0)
    b, _ = replicates.spread_mooncake_arrivals(lines, seed=1, spread_ms=3000.0)
    c, _ = replicates.spread_mooncake_arrivals(lines, seed=2, spread_ms=3000.0)
    assert a == b and a != c


def test_spread_cells_resolve_to_their_own_protocol(tmp_path):
    trace = tmp_path / "t.jsonl"
    trace.write_text("\n".join(_trace()) + "\n")
    base = {
        "cell_id": "c",
        "num_workers": 2,
        "trace_format": "mooncake",
        "trace_files": [str(trace)],
        "load": {"mode": "open_speedup", "value": 1.0},
    }
    layout = Layout(tmp_path)
    plain = resolve_replicate(Cell(raw=dict(base), layout=layout), 0)
    spread = resolve_replicate(
        Cell(raw={**base, "arrival_spread_ms": 3000.0}, layout=layout), 0
    )
    assert plain.protocol == replicates.PROTOCOL
    assert spread.protocol == replicates.SPREAD_PROTOCOL
    assert plain.trace_path != spread.trace_path
    with pytest.raises(ValueError, match="mooncake"):
        Cell(
            raw={**base, "trace_format": "weka", "arrival_spread_ms": 1.0},
            layout=layout,
        ).arrival_spread_ms


def test_materialized_replicates_are_reused_and_checked(tmp_path, monkeypatch):
    trace = tmp_path / "t.jsonl"
    trace.write_text("\n".join(_trace()) + "\n")
    first = replicates.materialize_replicate(
        trace, "mooncake", 2, tmp_path / "r", spread_ms=3000.0
    )
    calls = []
    real = replicates.spread_mooncake_arrivals
    monkeypatch.setattr(
        replicates,
        "spread_mooncake_arrivals",
        lambda *a, **k: calls.append(1) or real(*a, **k),
    )
    again = replicates.materialize_replicate(
        trace, "mooncake", 2, tmp_path / "r", spread_ms=3000.0
    )
    assert again == first and not calls
    # a replicate file that no longer matches its record is regenerated
    first.path.write_text("truncated\n")
    fixed = replicates.materialize_replicate(
        trace, "mooncake", 2, tmp_path / "r", spread_ms=3000.0
    )
    assert calls and fixed.sha256 == first.sha256
    assert replicates._sha256_file(fixed.path) == first.sha256
