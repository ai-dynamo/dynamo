# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construction of unevenly split prefill batches.

The properties worth pinning here are the ones a plausible-looking refactor
would break silently: the batches only mean something as a difference against
the uniform batch of the same cell, and the difference is only the spread if
both token totals are conserved exactly.
"""

import pytest

from dynamo.vllm.benchmark_imbalance import (
    build_manifest,
    classify,
    mla_work,
    plan_cell,
    segments_for,
    work_columns,
)
from dynamo.vllm.benchmark_points import BenchmarkPoints

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

TOPK = 2048


def _uniform(point: dict) -> bool:
    return len({tuple(row) for row in point["rows"]}) == 1


def _cell_of(point: dict) -> tuple[int, int, int]:
    b = point["batch_size"]
    return b, point["total_prefill_tokens"] // b, point["total_kv_read_tokens"] // b


# --------------------------------------------------------------- work model


def test_attention_work_is_capped_at_the_index_budget():
    """Past topk a query reads topk keys, not all of them, so the work stops
    growing with the prefix. Everything downstream assumes this cap."""
    assert mla_work(100, TOPK, TOPK) == 100 * TOPK
    assert mla_work(100, 4 * TOPK, TOPK) == 100 * TOPK


def test_a_row_at_the_bound_is_dense():
    """The classification turns on whether the top-k selection had anything to
    discard, so a request whose whole length equals the budget did not."""
    assert classify([(TOPK, 0)], TOPK) == "unsat"
    assert classify([(TOPK + 1, 0)], TOPK) == "sat"
    assert classify([(TOPK + 1, 0), (8, 0)], TOPK) == "mixed"


def test_available_segments_follow_from_the_conserved_total():
    """Which regimes a cell can express is arithmetic, not search: the total
    length is conserved however the tokens are redistributed."""
    assert segments_for(4, 64, 0, TOPK) == ("unsat",)
    assert segments_for(4, 1024, 0, TOPK) == ("unsat", "mixed")
    assert segments_for(4, 4096, 0, TOPK) == ("mixed", "sat")


def test_uniform_batch_has_no_deviation():
    rows = [(1024, 512)] * 4
    assert work_columns(rows, 1024, 512, TOPK) == (0.0, 0.0, 0.0)


def test_columns_credit_the_subtrahend_to_the_average_point_s_own_kernels():
    """A dense average point subtracts through the dense column, so a batch
    that moves rows onto the sparse path shows a NEGATIVE dense deviation --
    which is the shape of the label, not a bug in the sign."""
    rows = [(4096, 0), (128, 0), (128, 0), (128, 0)]
    x_idx, x_sparse, x_dense = work_columns(rows, 1120, 0, TOPK)
    assert x_idx > 0 and x_sparse > 0
    assert x_dense < 0


# --------------------------------------------------------------- planning


def test_plan_conserves_both_totals_exactly():
    """A batch whose totals drift is not a measurement of the spread: the
    label would carry the cost of the extra tokens as well."""
    plan = plan_cell(8, 4096, 1024, TOPK, kv_block=64)
    assert plan is not None
    for batch in plan.batches:
        new_total, kv_total = batch.totals
        assert new_total == 8 * 4096
        assert kv_total == 8 * 1024


def test_planned_batches_stay_inside_the_regime_they_claim():
    plan = plan_cell(8, 4096, 1024, TOPK, kv_block=64)
    assert plan is not None
    for batch in plan.batches:
        assert classify(batch.rows, TOPK) == batch.regime


def test_kv_lengths_are_whole_cache_blocks():
    """A prefix cache is looked up per block, so a ragged KV length is served
    a different length than asked for and its label measures another batch."""
    plan = plan_cell(8, 4096, 1024, TOPK, kv_block=64)
    assert plan is not None
    for batch in plan.batches:
        assert all(kv % 64 == 0 for _, kv in batch.rows)


def test_every_row_carries_at_least_one_new_token():
    plan = plan_cell(8, 4096, 1024, TOPK, kv_block=64)
    assert plan is not None
    for batch in plan.batches:
        assert all(new >= 1 for new, _ in batch.rows)


def test_a_cell_with_no_room_to_redistribute_plans_nothing():
    assert plan_cell(1, 4096, 0, TOPK, kv_block=64) is None


# --------------------------------------------------------------- manifest


def test_manifest_emits_the_uniform_batch_for_every_cell():
    """This manifest REPLACES the generated grid rather than adding to it, so
    a cell dropped here is a coordinate the switch-off run would have measured
    and this one silently would not. It holds even for cells around which no
    spread can be built."""
    cells = [(1, 4096, 1024), (2, 128, 64), (8, 32768, 8192)]
    manifest, _ = build_manifest(cells, TOPK, repeats=1, kv_block=64)
    emitted = {_cell_of(p) for p in manifest["prefill"] if _uniform(p)}
    for b, total_new, total_kv in cells:
        assert (b, total_new // b, total_kv // b) in emitted


def test_manifest_is_valid_against_the_schema():
    manifest, _ = build_manifest([(8, 32768, 8192)], TOPK, repeats=2, kv_block=64)
    parsed = BenchmarkPoints.model_validate(manifest)
    assert parsed.schema_version == 3
    assert all(point.rows is not None for point in parsed.prefill)


def test_repeats_multiply_shapes_not_coordinates():
    one, _ = build_manifest([(8, 32768, 8192)], TOPK, repeats=1, kv_block=64)
    five, _ = build_manifest([(8, 32768, 8192)], TOPK, repeats=5, kv_block=64)
    assert len(five["prefill"]) == 5 * len(one["prefill"])


def test_identical_shapes_are_not_emitted_twice():
    manifest, _ = build_manifest(
        [(8, 32768, 8192), (8, 32768, 8192)], TOPK, repeats=1, kv_block=64
    )
    shapes = [tuple(map(tuple, p["rows"])) for p in manifest["prefill"]]
    assert len(shapes) == len(set(shapes))


def test_unplannable_cells_are_reported_rather_than_dropped_silently():
    """A cell with no constructible spread pins no coefficient, and that is
    invisible in the results file -- which only ever shows what did run."""
    _, notes = build_manifest([(2, 256, 128)], TOPK, repeats=1, kv_block=64)
    assert notes


def test_a_single_request_gets_its_uniform_point_and_no_spread():
    manifest, _ = build_manifest([(1, 4096, 1024)], TOPK, repeats=1, kv_block=64)
    assert len(manifest["prefill"]) == 1
    assert _uniform(manifest["prefill"][0])
