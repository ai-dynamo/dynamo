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
    idx_work,
    mla_work,
    plan_cell,
    reference_rows,
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
    assert work_columns(rows, rows, TOPK) == (0.0, 0.0)


def test_columns_credit_the_subtrahend_per_row():
    """The subtrahend is the reference batch's own rows, credited to the same
    columns by the same rule. Here the reference sits below the bound, so it
    contributes nothing to x_idx and the whole scan of the row that crossed
    survives into that column, while x_mla carries only what the batch's
    attention gained over the reference's."""
    rows = [(4096, 0), (128, 0), (128, 0), (128, 0)]
    reference = [(1120, 0)] * 4
    x_idx, x_mla = work_columns(rows, reference, TOPK)

    assert x_idx == pytest.approx(idx_work(4096, 0, TOPK))
    assert x_mla == pytest.approx(
        sum(mla_work(s, p, TOPK) for s, p in rows)
        - sum(mla_work(s, p, TOPK) for s, p in reference)
    )
    assert x_idx > 0 and x_mla > 0


def test_indexer_column_leaves_out_rows_below_the_budget():
    """Short rows do run the indexer, and their scan is still kept out of
    x_idx. Below the bound a row's scan and its attention are the same
    trapezoid up to a half-token that the conserved total cancels, so counting
    both would make the two columns exactly collinear and a pure unsaturated
    segment would identify nothing. Left out, that cost lands in c_mla."""
    s_bar, p_bar, b = 512, 0, 4
    rows = [(1024, 0), (256, 0), (256, 0), (512, 0)]
    assert all(s + p <= TOPK for s, p in rows)
    assert sum(s for s, _ in rows) == b * s_bar

    x_idx, x_mla = work_columns(rows, [(s_bar, p_bar)] * b, TOPK)
    assert x_idx == 0.0
    assert x_mla > 0.0

    # What that exclusion buys: counting the short rows would have made the two
    # columns identical, leaving only their sum determined.
    naive = sum(idx_work(s, p, TOPK) for s, p in rows) - b * idx_work(
        s_bar, p_bar, TOPK
    )
    assert naive == pytest.approx(x_mla)


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


def test_every_shape_conserves_the_cell_s_exact_totals():
    """The label is a difference against the reference batch, so a shape that
    conserved a floored total instead would carry a token-count change into it.
    At the C+1 coordinates the sweep samples on purpose that difference is not
    one token of work: 4097 tokens pad up to the next capture size while 4096
    does not, so the label would price a whole CUDA-graph jump as a spread."""
    for b, total_new, total_kv in [(4, 4097, 0), (8, 32769, 8192), (4, 4095, 256)]:
        manifest, _ = build_manifest(
            [(b, total_new, total_kv)], TOPK, repeats=1, kv_block=64
        )
        assert manifest["prefill"], (b, total_new, total_kv)
        for point in manifest["prefill"]:
            assert point["total_prefill_tokens"] == total_new
            assert point["total_kv_read_tokens"] == total_kv
            assert sum(row[0] for row in point["rows"]) == total_new
            assert sum(row[1] for row in point["rows"]) == total_kv


def test_cells_sample_variance_when_only_the_average_prefix_is_ragged():
    """The block deal is over the cell's exact KV total, so a cell is runnable
    whenever that total is whole blocks -- the floored per-request average need
    not be. Gating on the average rejected cells whose exact total distributes
    perfectly well, and those cells then contributed a reference point and no
    variance at all."""
    b, total_new, total_kv, kv_block = 3, 3072, 256, 64
    assert (total_kv // b) % kv_block, "not the case under test"
    assert total_kv % kv_block == 0

    manifest, _ = build_manifest(
        [(b, total_new, total_kv)], TOPK, repeats=1, kv_block=kv_block
    )
    points = manifest["prefill"]
    reference = reference_rows(b, total_new, total_kv, kv_block)
    assert [list(map(list, reference))] == [points[0]["rows"]]
    assert len(points) > 1, "cell emitted its reference and no spread"
    for point in points:
        assert point["total_prefill_tokens"] == total_new
        assert point["total_kv_read_tokens"] == total_kv
        assert all(row[1] % kv_block == 0 for row in point["rows"])


def test_manifest_emits_the_reference_batch_for_every_cell():
    """This manifest REPLACES the generated grid rather than adding to it, so
    a cell dropped here is a coordinate the switch-off run would have measured
    and this one silently would not. It holds even for cells around which no
    spread can be built.

    The coordinate is the pair of totals, not the pair of averages: a cell whose
    totals do not divide evenly is still measured, by a batch that carries the
    remainder rather than one that truncates it away."""
    cells = [(1, 4096, 1024), (2, 128, 64), (8, 32768, 8192)]
    manifest, _ = build_manifest(cells, TOPK, repeats=1, kv_block=64)
    emitted = {
        (p["batch_size"], p["total_prefill_tokens"], p["total_kv_read_tokens"])
        for p in manifest["prefill"]
    }
    for cell in cells:
        assert cell in emitted


def test_reference_batch_conserves_both_totals_in_whole_blocks():
    """The floor-divided batch was wrong twice over: it measured a coordinate up
    to ``b - 1`` tokens off the one the sweep chose -- enough to fall off the
    CUDA-graph capture size the point was picked for -- and it left every row's
    prefix at ``total_kv // b``, which the loader rejects when that is not a
    whole block, aborting the run instead of measuring the cell."""
    kv_block = 64
    for b, total_new, total_kv in [(2, 128, 64), (3, 3072, 256), (8, 32769, 8192)]:
        rows = reference_rows(b, total_new, total_kv, kv_block)
        assert rows is not None
        assert len(rows) == b
        assert sum(s for s, _ in rows) == total_new
        assert sum(p for _, p in rows) == total_kv
        assert all(p % kv_block == 0 for _, p in rows)
        # Still the equal batch to within the remainder it has to carry.
        assert max(s for s, _ in rows) - min(s for s, _ in rows) <= 1
        assert max(p for _, p in rows) - min(p for _, p in rows) <= kv_block


def test_reference_batch_reports_a_cell_it_cannot_express():
    """A KV total that is not a whole number of blocks cannot be split into rows
    that are. Emitting it anyway is what aborts the run, so the cell is reported
    and skipped instead."""
    assert reference_rows(2, 128, 100, 64) is None
    manifest, notes = build_manifest([(2, 128, 100)], TOPK, repeats=1, kv_block=64)
    assert manifest["prefill"] == []
    assert any("not expressible" in note for note in notes)


def test_repeats_are_laid_out_as_passes_not_back_to_back():
    """Every shape is sampled alike, the reference included: the label is a
    difference between two measured points, so sampling one harder than the
    other buys nothing the difference can use.

    The extra passes go over the whole sweep rather than back to back per shape,
    so identical shapes do not sit inside one thermal excursion and average a
    correlated error."""
    manifest, _ = build_manifest([(8, 32768, 8192)], TOPK, repeats=3, kv_block=64)
    shapes = [tuple(map(tuple, p["rows"])) for p in manifest["prefill"]]
    per_pass = len(set(shapes))

    assert len(shapes) == 3 * per_pass
    reference = tuple(map(tuple, reference_rows(8, 32768, 8192, 64)))
    assert shapes.count(reference) == 3
    for start in range(0, len(shapes), per_pass):
        assert len(set(shapes[start : start + per_pass])) == per_pass
    assert shapes[:per_pass] == shapes[per_pass : 2 * per_pass]


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
