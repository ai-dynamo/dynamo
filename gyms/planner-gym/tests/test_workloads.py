# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the workload generator.

Pure-Python — no Dynamo runtime needed. Verifies: registry shape, record schema,
determinism, and that each axis (arrival / shape / prefix) actually does what it
claims.
"""

from __future__ import annotations

import json
import random
import statistics

import pytest
from autoscaling_arena.workloads import (
    WORKLOADS,
    Workload,
    get_workload,
    list_workloads,
    registry,
    validate_mooncake_trace,
)

# The public Mooncake anchor is optional at runtime and lives in a Dynamo checkout.
REAL = {"mooncake"}
EXPECTED = {
    "flat",
    "staircase",
    "square_wave",
    "flash_crowd",
    "diurnal",
    "decode_heavy",
    "shared_prefix",
} | REAL
SYNTHETIC = EXPECTED - REAL


def _records(name, seed=0):
    return get_workload(name).generate_records(seed)


def test_registry_has_all_workloads():
    assert set(list_workloads()) == EXPECTED
    assert len(WORKLOADS) == 8


def test_mooncake_is_real_anchor():
    wl = get_workload("mooncake")
    assert not wl.is_synthetic
    assert wl.static_trace.name == "mooncake_trace_1000.jsonl"


@pytest.mark.parametrize("name", sorted(SYNTHETIC))
def test_record_schema_and_ordering(name):
    recs = _records(name)
    assert 500 < len(recs) < 1600, f"{name}: unexpected request count {len(recs)}"
    last = -1
    dur_ms = get_workload(name).duration_s * 1000
    for r in recs:
        assert isinstance(r["timestamp"], int) and 0 <= r["timestamp"] <= dur_ms
        assert isinstance(r["input_length"], int) and r["input_length"] > 0
        assert isinstance(r["output_length"], int) and r["output_length"] > 0
        assert isinstance(r["hash_ids"], list) and r["hash_ids"]
        assert all(isinstance(h, int) for h in r["hash_ids"])
        assert r["timestamp"] >= last  # sorted
        last = r["timestamp"]


@pytest.mark.parametrize("name", sorted(SYNTHETIC))
def test_determinism(name):
    assert _records(name, seed=0) == _records(name, seed=0)
    assert _records(name, seed=0) != _records(name, seed=1)


# --- arrival axis ---------------------------------------------------------


def test_staircase_rate_increases():
    recs = _records("staircase")
    first_third = sum(1 for r in recs if r["timestamp"] < 60_000)
    last_third = sum(1 for r in recs if r["timestamp"] >= 120_000)
    assert last_third > 2 * first_third  # rate climbs over time


def test_flat_is_roughly_uniform():
    recs = _records("flat")
    first_half = sum(1 for r in recs if r["timestamp"] < 90_000)
    second_half = len(recs) - first_half
    assert 0.7 < first_half / second_half < 1.4


def test_flash_crowd_has_a_spike():
    recs = _records("flash_crowd")
    in_spike = [r for r in recs if 80_000 <= r["timestamp"] < 95_000]
    out_spike = [r for r in recs if not (80_000 <= r["timestamp"] < 95_000)]
    rate_in = len(in_spike) / 15.0
    rate_out = len(out_spike) / (180.0 - 15.0)
    assert rate_in > 3 * rate_out  # the spike is real


# --- shape axis -----------------------------------------------------------


def test_prefill_heavy_vs_decode_heavy():
    pf = _records("flat")  # prefill-heavy shape
    dh = _records("decode_heavy")
    pf_isl = statistics.median(r["input_length"] for r in pf)
    pf_osl = statistics.median(r["output_length"] for r in pf)
    dh_isl = statistics.median(r["input_length"] for r in dh)
    dh_osl = statistics.median(r["output_length"] for r in dh)
    assert pf_isl > 10 * pf_osl  # prefill-heavy: long in, short out
    assert dh_osl > dh_isl  # decode-heavy: long out
    assert pf_isl > dh_isl  # and a genuinely different shape


# --- prefix axis ----------------------------------------------------------


def test_no_sharing_has_unique_blocks():
    recs = _records("flat")  # NoSharing
    all_ids = [h for r in recs for h in r["hash_ids"]]
    assert len(all_ids) == len(set(all_ids))  # every block hash unique


def test_shared_prefix_reuses_leading_blocks():
    recs = _records("shared_prefix")  # SharedPrefix(8)
    with_block0 = sum(1 for r in recs if 0 in r["hash_ids"])
    assert with_block0 > 0.9 * len(recs)  # the shared prefix is in ~every request
    all_ids = [h for r in recs for h in r["hash_ids"]]
    assert len(all_ids) > len(set(all_ids))  # there IS reuse


# --- materialization ------------------------------------------------------


def test_materialize_synthetic_roundtrips(tmp_path):
    path = get_workload("flat").materialize(tmp_path, seed=0)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == _records("flat", seed=0)


def test_materialize_static_returns_anchor(tmp_path):
    workload = _static_workload(tmp_path)
    assert workload.materialize(tmp_path / "out") == workload.static_trace


def test_materialize_unsorted_static_trace_creates_sorted_copy(tmp_path):
    trace = tmp_path / "unsorted.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "input_length": 8,
                    "output_length": 2,
                    "hash_ids": [timestamp],
                }
            )
            for timestamp in (20, 10)
        )
        + "\n"
    )
    workload = Workload(
        name="recorded-trace",
        description="test trace",
        static_trace=trace,
        presorted=False,
    )

    materialized = workload.materialize(tmp_path / "out")

    assert materialized != trace
    assert [
        json.loads(line)["timestamp"] for line in materialized.read_text().splitlines()
    ] == [10, 20]


def test_force_copy_static_stages_presorted_trace(tmp_path):
    workload = _static_workload(tmp_path)

    materialized = workload.materialize(tmp_path / "staged", force_copy_static=True)

    assert materialized != workload.static_trace
    assert materialized.read_bytes() == workload.static_trace.read_bytes()


def test_validate_recorded_trace_contract(tmp_path):
    trace = tmp_path / "recorded.jsonl"
    trace.write_text(
        '{"timestamp": 0, "input_length": 65, "output_length": 2, '
        '"hash_ids": [1, 2]}\n'
    )

    assert validate_mooncake_trace(trace, block_size=64, presorted=True) == 1


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"timestamp": 0}, "input_length"),
        (
            {
                "timestamp": 0,
                "input_length": 65,
                "output_length": 2,
                "hash_ids": [1],
            },
            "requires 2 hashes",
        ),
    ],
)
def test_validate_recorded_trace_rejects_malformed_rows(tmp_path, record, message):
    trace = tmp_path / "invalid.jsonl"
    trace.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match=message):
        validate_mooncake_trace(trace, block_size=64)


def test_validate_recorded_trace_honors_presorted_declaration(tmp_path):
    trace = tmp_path / "unsorted.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "input_length": 8,
                    "output_length": 2,
                    "hash_ids": [timestamp],
                }
            )
            for timestamp in (10, 5)
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="timestamps are not sorted"):
        validate_mooncake_trace(trace, block_size=512, presorted=True)


# --- max_requests cap + arrival_speedup ----------------------------------


def test_max_requests_caps_synthetic_count():
    full = _records("flat")
    capped = get_workload("flat").generate_records(0, max_requests=20)
    assert len(capped) == 20
    assert capped == full[:20]  # keeps the earliest-arriving N


def test_max_requests_field_on_workload():
    wl = WORKLOADS["flat"]
    import dataclasses

    capped = dataclasses.replace(wl, max_requests=15)
    assert len(capped.generate_records(0)) == 15


def test_materialize_max_requests(tmp_path):
    path = get_workload("flat").materialize(tmp_path, seed=0, max_requests=12)
    assert len(path.read_text().splitlines()) == 12


def test_materialize_static_truncates_to_copy(tmp_path):
    workload = _static_workload(tmp_path)
    path = workload.materialize(tmp_path / "out", max_requests=2)
    assert path != workload.static_trace
    assert len(path.read_text().splitlines()) == 2


def test_arrival_speedup_compresses_timestamps(tmp_path):
    import json

    base = [
        json.loads(line)
        for line in get_workload("staircase")
        .materialize(tmp_path / "a", seed=0)
        .read_text()
        .splitlines()
    ]
    fast = [
        json.loads(line)
        for line in get_workload("staircase")
        .materialize(tmp_path / "b", seed=0, arrival_speedup=2.0)
        .read_text()
        .splitlines()
    ]
    assert len(base) == len(fast)  # same requests, just compressed in time
    assert (
        max(r["timestamp"] for r in fast) <= max(r["timestamp"] for r in base) / 2 + 1
    )


def _static_workload(tmp_path):
    trace = tmp_path / "input.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "input_length": 8,
                    "output_length": 2,
                    "hash_ids": [timestamp],
                }
            )
            for timestamp in (0, 100, 200)
        )
        + "\n"
    )
    return Workload(
        name="recorded",
        description="test trace",
        static_trace=trace,
        presorted=True,
    )


def test_mooncake_discovery_uses_ancestor_repository_regardless_of_name(
    tmp_path, monkeypatch
):
    root = tmp_path / "arbitrary-checkout-name"
    (root / "lib/bench/testdata").mkdir(parents=True)
    source = root / "gyms/planner-gym/src/autoscaling_arena/workloads/registry.py"
    monkeypatch.delenv("DYNAMO_DIR", raising=False)
    monkeypatch.setattr(registry, "__file__", str(source))
    assert registry._dynamo_dir() == root


def test_mooncake_discovery_respects_explicit_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMO_DIR", str(tmp_path))
    assert registry._dynamo_dir() == tmp_path


@pytest.mark.parametrize("block_size", [16, 512])
def test_shared_prefix_partial_terminal_blocks_have_unique_hashes(block_size):
    from autoscaling_arena.workloads.axes import SharedPrefix

    policy = SharedPrefix(8)
    rng = random.Random(0)
    first = policy.assign(rng, block_size - 4, block_size)
    second = policy.assign(rng, block_size + 7, block_size)
    full = policy.assign(rng, block_size * 2, block_size)
    another_partial = policy.assign(rng, block_size + 7, block_size)
    assert second[0] == full[0] == another_partial[0] == 0
    assert full == [0, 1]
    assert len({first[-1], second[-1], another_partial[-1]}) == 3
    assert not ({first[-1], second[-1], another_partial[-1]} & set(range(8)))


def test_shared_prefix_hashes_keep_one_token_length_across_generated_trace():
    workload = get_workload("shared_prefix")
    lengths = {}
    for record in workload.generate_records(seed=0):
        for index, hash_id in enumerate(record["hash_ids"]):
            size = min(
                workload.block_size,
                record["input_length"] - index * workload.block_size,
            )
            assert size > 0
            assert lengths.setdefault(hash_id, size) == size


@pytest.mark.parametrize(
    "missing", ["arrival", "shape", "prefix_factory", "duration_s"]
)
def test_synthetic_workload_missing_axis_names_the_configuration_error(missing):
    from dataclasses import replace

    workload = replace(get_workload("flat"), **{missing: None})
    with pytest.raises(ValueError, match=f"missing required axes: {missing}"):
        workload.generate_records(seed=0)


def test_synthetic_workload_lists_all_missing_axes():
    with pytest.raises(ValueError, match="arrival, shape, prefix_factory, duration_s"):
        Workload(name="incomplete", description="test").generate_records(seed=0)
