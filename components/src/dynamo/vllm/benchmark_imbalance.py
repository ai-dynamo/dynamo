# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration-batch planning for the intra-batch prefill work delta.

A forward-pass estimate prices a prefill batch from its per-request means, so
two batches with the same totals and different spreads get the same number.
They do not cost the same: attention work is quadratic in the new-token count,
and a sparse indexer's cost depends on which requests sit above its top-k.
Measuring that difference needs batches built on purpose, because natural
traffic offers only one imbalance ratio per shape and the resulting equations
are parallel.

Around one cell ``(b, s_bar, p_bar)`` the batches hold both conserved sums
exactly, ``N = b * s_bar`` and ``P = b * p_bar``, so the label

    y = T_batch - T_uniform

cancels every term that depends on the totals alone and leaves the cost of the
spread.

Three prices, not two. A request whose full length stays at or below ``topk``
short-circuits the indexer and runs the dense kernel; one above it runs the
indexer and the capped-attention kernel. Those are different kernels with
different prices, so a batch that straddles the bound cannot be charged with a
single attention coefficient:

    y = c_idx * x_idx  +  c_mla_sparse * x_mla_sparse  +  c_mla_mha * x_mla_dense

The three columns are what :func:`work_columns` returns. Splitting them is not
a refinement -- in the measured grid the dense rows carry between 1% and 88% of
a mixed batch's attention work, so lumping them prices most of that work at the
wrong rate.

Which column a segment can move follows from the regime, and that is what lets
one segment pin one coefficient without a joint fit:

    pure saturated    every row is above topk, the indexer is pinned there for
                      all of them, so the attention term is linear in s and its
                      deviation cancels exactly -- only x_idx survives.
    pure unsaturated  every row short-circuits the indexer, so x_idx is
                      identically zero -- only x_mla_dense survives.
    mixed             all three are live; the segment is solved against
                      whichever coefficients the cell's other segment and the
                      global dense fit already fixed.

``c_mla_mha`` is a property of the dense kernel rather than of a cell, and at a
cell whose average request is long it cannot be measured at all: a dense row is
capped at ``topk`` tokens, so the dense column tops out at ``(b - 1) * topk^2 /
2`` while the cell's own work grows with ``s_bar^2``. It is therefore fitted
once from the unsaturated cells, where it carries 8-196% of the work, and
treated as known everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Regime",
    "CalibrationBatch",
    "CellPlan",
    "plan_cell",
    "segments_for",
    "idx_work",
    "mla_work",
    "work_columns",
    "key_column",
    "short_row_new_tokens",
]

UNSAT = "unsat"
MIXED = "mixed"
SAT = "sat"
Regime = str


# --------------------------------------------------------------- work terms


def idx_work(s: float, p: float, topk: int) -> float:
    """Indexer scoring work for one request: every query against every key.

    NOT gated on ``topk``. The gate was an assumption and it is wrong: vLLM's
    ``sparse_attn_indexer`` has no short-circuit for sequences at or below the
    index budget. It scores the whole sequence with ``fp8_fp4_mqa_logits`` and
    then runs the top-k selection regardless, padding the index buffer with -1
    when there are fewer candidates than ``topk``. The only skip in that path is
    for an empty sequence.

    Charging zero here made every request below the bound look free, which is
    exactly the requests a mixed batch is full of.
    """
    return s * (2.0 * p + s) / 2.0


def mla_work(s: float, p: float, topk: int) -> float:
    """Attention pairs actually read for one request, in three segments.

    Each of the ``s`` new queries reads ``min(p + i + 1, topk)`` keys. Which
    segment applies depends on where the request sits relative to ``topk``:
    entirely below it, entirely capped by it, or straddling.
    """
    if s + p <= topk:
        return s * p + s * (s + 1) / 2.0
    if p >= topk:
        return s * topk
    free = topk - p
    return free * p + free * (free + 1) / 2.0 + (s - free) * topk


def runs_sparse(s: float, p: float, topk: int) -> bool:
    """Whether this request's attention is actually truncated by the budget.

    Every request runs the indexer, so this no longer selects a kernel -- it
    only says whether the top-k selection had anything to discard. Requests at
    or below the bound read all of their keys; longer ones are capped.
    """
    return s + p > topk


def work_columns(
    rows: list[tuple[int, int]], s_bar: float, p_bar: float, topk: int
) -> tuple[float, float, float]:
    """``(x_idx, x_mla_sparse, x_mla_dense)`` against the cell's uniform batch.

    Each row's attention work is credited to the column for the kernel that row
    actually runs on. The subtrahend goes to whichever pair of columns the
    UNIFORM batch runs on -- that is decided by the average point's own length,
    not by the calibration batch's regime -- so an unsaturated average point
    contributes nothing to the indexer column and its attention work is
    subtracted from the dense one.
    """
    b = len(rows)
    x_idx = sum(idx_work(s, p, topk) for s, p in rows if runs_sparse(s, p, topk))
    x_sp = sum(mla_work(s, p, topk) for s, p in rows if runs_sparse(s, p, topk))
    x_dn = sum(mla_work(s, p, topk) for s, p in rows if not runs_sparse(s, p, topk))
    if runs_sparse(s_bar, p_bar, topk):
        x_idx -= b * idx_work(s_bar, p_bar, topk)
        x_sp -= b * mla_work(s_bar, p_bar, topk)
    else:
        x_dn -= b * mla_work(s_bar, p_bar, topk)
    return x_idx, x_sp, x_dn


def key_column(
    columns: tuple[float, float, float], regime: Regime, avg_is_sat: bool
) -> float:
    """Magnitude of the column this segment has to move to be worth measuring.

    A segment that leaves its own column near zero produces a label the fit
    cannot attribute, so this is what the relative-deviation floor is applied
    to -- not the total work change, which a mixed batch can inflate by an order
    of magnitude through a column whose coefficient is already known.
    """
    x_idx, x_sp, x_dn = columns
    if regime == SAT:
        return abs(x_idx)
    if regime == UNSAT:
        return abs(x_dn)
    # A mixed batch at a saturated average point is solved for the sparse
    # attention price alone: the gated price came from the cell's own saturated
    # segment and the dense price from the global fit. At an unsaturated average
    # point only the dense price is known, so two columns have to carry signal.
    return abs(x_sp) if avg_is_sat else min(abs(x_idx), abs(x_sp))


# ------------------------------------------------------------------ regimes


def classify(rows: list[tuple[int, int]], topk: int) -> Regime:
    """Which side of the indexer each request falls on, from its length ``s + p``."""
    lengths = [s + p for s, p in rows]
    if min(lengths) > topk:
        return SAT
    if max(lengths) <= topk:
        return UNSAT
    return MIXED


def admits(rows: list[tuple[int, int]], want: Regime, topk: int) -> bool:
    """Whether a constructed batch satisfies the segment it was built for."""
    lengths = [s + p for s, p in rows]
    if want == UNSAT:
        return max(lengths) <= topk
    if want == SAT:
        return min(lengths) > topk
    return min(lengths) <= topk < max(lengths)


def segments_for(b: int, s_bar: int, p_bar: int, topk: int) -> tuple[Regime, ...]:
    """Regimes a cell can express, fixed by its conserved token total.

    Every regime is a statement about each request's full length ``s + p``, and
    ``T = sum(s + p) = b * (s_bar + p_bar)`` is conserved however the tokens are
    redistributed, so availability follows from arithmetic alone. Note the band
    boundary coincides with the average point's own regime: ``T > b * topk`` is
    the same statement as ``s_bar + p_bar > topk``.
    """
    total = b * (s_bar + p_bar)
    if total < topk + b:
        return (UNSAT,)
    if total <= b * topk:
        return (UNSAT, MIXED)
    return (MIXED, SAT)


# ---------------------------------------------------------------- the floor

# A request shorter than this stops being a scaled-down version of the workload
# and becomes a different one: below roughly two query tiles the prefill kernel
# is launch-bound rather than work-bound, so its latency stops tracking the work
# term and the row contributes a constant that the fit charges to a coefficient.
MIN_NEW_TOKENS = 64

# Above this an absolute floor is more restrictive than it needs to be, so the
# floor tracks the cell until it reaches here and then stops. Note both bounds
# are on the NEW-TOKEN count and not on the row length: a pure regime holds
# ``p = p_bar`` on every row, so a floor written on the length would demand
# ``s < 0`` at any cell whose ``p_bar`` exceeds it.
SHORT_ROW_CAP_TOKENS = 256
SHORT_ROW_FRACTION = 8


def short_row_new_tokens(s_bar: int) -> int:
    """Fewest new tokens a short request may carry at this cell."""
    return max(MIN_NEW_TOKENS, min(s_bar // SHORT_ROW_FRACTION, SHORT_ROW_CAP_TOKENS))


# Deviations smaller than this fraction of the cell's own work are not worth
# measuring: at a fixed work the engine produces a spread of latencies, so a
# perturbation below that spread yields a difference whose sign is not even
# reliable.
MIN_RELATIVE_DELTA = 0.05


# ---------------------------------------------------------------- templates


def uniform_rows(b: int, s_bar: int, p_bar: int) -> list[tuple[int, int]]:
    """The equal-length batch of the cell: the subtrahend every label uses."""
    return [(s_bar, p_bar)] * b


def _settle(
    s: list[int], p: list[int], new_tokens: int, kv_tokens: int, sink: int
) -> list[tuple[int, int]] | None:
    """Push every integer-division remainder into one row so both totals are exact.

    A batch whose totals drift is not a measurement of the spread -- the label
    would carry the cost of the extra tokens as well. Which row absorbs the
    remainder is not free: it lengthens that row, so it has to be a row the
    regime leaves headroom on. An unsaturated batch pins its LONG rows at the
    bound, so the remainder goes to a short one; the other two regimes pin their
    SHORT rows, so it goes to a long one.
    """
    s[sink] += new_tokens - sum(s)
    p[0] += kv_tokens - sum(p)
    if min(s) < 1 or min(p) < 0:
        return None
    return list(zip(s, p))


def pure_rows(
    b: int, s_bar: int, p_bar: int, m: int, want: Regime, topk: int, reach: float = 1.0
) -> list[tuple[int, int]] | None:
    """``m`` long and ``b - m`` short requests, all holding ``p = p_bar``.

    A pure regime constrains EVERY row, so the prefix cannot move: shifting it
    would push some row across ``topk`` and out of the segment. Only the new
    tokens split, and the regime pins whichever group is its binding one --
    saturated pins the shortest row one token above the bound, unsaturated pins
    the longest row at it -- with conservation fixing the other.
    """
    if not 1 <= m < b:
        return None
    new_tokens, kv_tokens = b * s_bar, b * p_bar
    total = new_tokens + kv_tokens
    short_count = b - m

    mean_len = total / b
    if want == SAT:
        # Two lower bounds meet on the short row and the binding one wins: the
        # regime needs it past topk, and the kernel needs it to carry real new
        # tokens. At a cell whose p_bar already reaches topk the regime bound
        # alone would leave the row a single new token.
        pin = max(topk + 1, p_bar + MIN_NEW_TOKENS)
        # ``reach`` walks the pinned side from the regime's own bound back
        # toward the mean. The bound is the most lopsided the regime allows;
        # everything between it and the mean is still inside the regime, so
        # this is a magnitude ladder that costs no validity. It is what carries
        # the ladder when ``m`` cannot: at b = 2 the only split is 1 v 1.
        short_len = int(mean_len - (mean_len - pin) * reach)
        long_len = (total - short_count * short_len) // m
        if long_len <= short_len:
            return None
    else:
        floor_len = short_row_new_tokens(s_bar) + p_bar
        pin = min(topk, (total - short_count * floor_len) // m)
        long_len = int(mean_len + (pin - mean_len) * reach)
        short_len = (total - m * long_len) // short_count
        if short_len >= long_len:
            return None

    s = [long_len - p_bar] * m + [short_len - p_bar] * short_count
    if min(s) < 1:
        return None
    # Saturated pins the short rows, so the long rows have headroom for the
    # remainder; unsaturated pins the long rows, so it has to go the other way.
    rows = _settle(s, [p_bar] * b, new_tokens, kv_tokens, 0 if want == SAT else b - 1)
    if rows is None or min(a for a, _ in rows) < MIN_NEW_TOKENS:
        return None
    return rows if classify(rows, topk) == want else None


def mixed_rows(
    b: int, s_bar: int, p_bar: int, m: int, short_len: int, topk: int, kv_block: int = 1
) -> list[tuple[int, int]] | None:
    """``m`` long requests carrying ALL the prefix, ``b - m`` short ones with none.

    Concentrating the prefix is what gives the attention columns room to move:
    spread evenly their deviation stays within a few percent however skewed the
    new-token counts are. The short rows are placed and the long ones take the
    remainder, which is what guarantees they clear ``topk``.

    The prefix is dealt out in whole ``kv_block``s rather than by dividing the
    token count. A prefix cache is looked up per block, so a request asking for
    a ragged number of KV tokens is served a different length than the plan
    specified and its label measures some other batch. Splitting ``P // m``
    evenly produces exactly that whenever ``m`` does not divide the block count.
    """
    if not 1 <= m < b:
        return None
    new_tokens, kv_tokens = b * s_bar, b * p_bar
    total = new_tokens + kv_tokens
    short_count = b - m
    long_len = (total - short_count * short_len) // m
    if long_len <= topk:
        return None
    if kv_tokens % kv_block:
        return None

    blocks, base = kv_tokens // kv_block, (kv_tokens // kv_block) // m
    spare = blocks - base * m
    p = [(base + (1 if i < spare else 0)) * kv_block for i in range(m)] + [
        0
    ] * short_count
    # Each long row is held at the same total length, so the new-token count
    # absorbs whatever the block deal gave that row.
    s = [long_len - q for q in p[:m]] + [short_len] * short_count
    rows = _settle(s, p, new_tokens, kv_tokens, 0)
    if rows is None or min(a for a, _ in rows) < MIN_NEW_TOKENS:
        return None
    return rows if classify(rows, topk) == MIXED else None


# ------------------------------------------------------------------ planning


@dataclass(frozen=True)
class CalibrationBatch:
    """One constructed batch and the three work columns it moves."""

    regime: Regime
    m: int
    short_len: int
    rows: list[tuple[int, int]]
    x_idx: float
    x_mla_sparse: float
    x_mla_dense: float

    @property
    def columns(self) -> tuple[float, float, float]:
        return self.x_idx, self.x_mla_sparse, self.x_mla_dense

    @property
    def totals(self) -> tuple[int, int]:
        return sum(s for s, _ in self.rows), sum(p for _, p in self.rows)

    @property
    def label(self) -> str:
        return f"{self.regime}:{self.m}v{len(self.rows) - self.m}@{self.short_len}"


@dataclass
class CellPlan:
    b: int
    s_bar: int
    p_bar: int
    topk: int
    segments: tuple[Regime, ...]
    batches: list[CalibrationBatch] = field(default_factory=list)
    # Segments the cell could express in principle but cannot measure, with the
    # reason. Reported rather than silently dropped: a missing segment means a
    # coefficient this cell will not pin.
    unusable: list[str] = field(default_factory=list)

    @property
    def avg_is_sat(self) -> bool:
        """A uniform batch has b identical rows, so it is never mixed."""
        return self.s_bar + self.p_bar > self.topk

    @property
    def uniform(self) -> list[tuple[int, int]]:
        return uniform_rows(self.b, self.s_bar, self.p_bar)

    def by_regime(self, regime: Regime) -> list[CalibrationBatch]:
        return [x for x in self.batches if x.regime == regime]


def _knobs(b: int, s_bar: int, want: Regime, topk: int, avg_is_sat: bool):
    """Every ``(m, short_len)`` the segment could be built at.

    Which knob carries the ladder depends on the average point's regime, because
    it decides which columns have to be separated. At a saturated average point
    the two attention columns are the unknowns and their ratio is
    ``-2 * topk / (short_len + 1)`` -- it depends on the short row's length and
    on nothing else, so varying ``m`` would leave the design matrix rank one
    while varying the length conditions it. Everywhere else it is ``m``.

    This is a sweep of candidates, not a search for a valid batch: every one of
    them is solved from the regime and either satisfies it or is arithmetically
    impossible. What the sweep buys is the ladder -- the imbalance is unimodal
    in ``m`` and the peak moves with the cell, so a fixed set of splits lands on
    the flat end at some cells and misses the usable range entirely.
    """
    if want == MIXED and avg_is_sat:
        lengths = {
            short_row_new_tokens(s_bar),
            topk // 8,
            topk // 4,
            topk // 2,
            3 * topk // 4,
            topk - 1,
        }
        return [(1, L, 1.0) for L in sorted(lengths) if L >= MIN_NEW_TOKENS]
    if want == MIXED:
        return [(m, short_row_new_tokens(s_bar), 1.0) for m in range(1, b)]
    # Pure segments sweep both knobs. ``m`` alone is empty at b = 2 and coarse
    # at b = 4, and the pinned side reaches from the regime's bound to the mean
    # without ever leaving the regime.
    return [(m, None, r) for m in range(1, b) for r in (1.0, 0.7, 0.45, 0.25)]


def _pick_rungs(
    rungs: list[tuple[float, "CalibrationBatch"]], want: int
) -> list["CalibrationBatch"]:
    """Spread the chosen rungs across the segment's usable range.

    Taking the ``want`` largest would cluster them at one magnitude, and a
    coefficient fitted at one magnitude says nothing about whether the relation
    is linear. Spanning the range is what makes the residual a test rather than
    a formality.
    """
    ranked = sorted(rungs, key=lambda kv: kv[0])
    if len(ranked) <= want:
        return [c for _, c in ranked]
    picks = {round(i * (len(ranked) - 1) / (want - 1)) for i in range(want)}
    return [ranked[i][1] for i in sorted(picks)]


# A mixed segment at a saturated average point is solved for one unknown, so two
# rungs already leave a degree of freedom; at an unsaturated one it carries two
# unknowns and needs three. A pure segment carries one.
_RUNGS_NEEDED = {SAT: 2, UNSAT: 2, MIXED: 2}


def plan_cell(
    b: int,
    s_bar: int,
    p_bar: int,
    topk: int,
    *,
    max_model_len: int = 131072,
    kv_block: int = 1,
) -> CellPlan | None:
    """Plan the calibration batches around one average point.

    Returns ``None`` when no segment yields a usable batch, which happens for
    cells whose totals leave no room to redistribute.
    """
    plan = CellPlan(b, s_bar, p_bar, topk, segments_for(b, s_bar, p_bar, topk))
    avg_is_sat = plan.avg_is_sat
    reference = b * (idx_work(s_bar, p_bar, topk) + mla_work(s_bar, p_bar, topk))
    if reference <= 0:
        return None
    # A pure segment holds p = p_bar on every row, so the whole cell is
    # unrunnable if the average prefix is not a whole number of cache blocks.
    if p_bar % kv_block:
        plan.unusable.append(
            f"avg_p={p_bar} is not a multiple of the kv block size {kv_block}"
        )
        return None

    for regime in plan.segments:
        rungs, tried = [], 0
        for m, short_len, reach in _knobs(b, s_bar, regime, topk, avg_is_sat):
            tried += 1
            rows = (
                mixed_rows(b, s_bar, p_bar, m, short_len, topk, kv_block)
                if regime == MIXED
                else pure_rows(b, s_bar, p_bar, m, regime, topk, reach)
            )
            if rows is None or not admits(rows, regime, topk):
                continue
            if max(s + p for s, p in rows) > max_model_len:
                continue
            columns = work_columns(rows, s_bar, p_bar, topk)
            key = key_column(columns, regime, avg_is_sat) / reference
            if key < MIN_RELATIVE_DELTA:
                continue
            rungs.append(
                (
                    key,
                    CalibrationBatch(
                        regime, m, min(s + p for s, p in rows), rows, *columns
                    ),
                )
            )

        needed = 3 if (regime == MIXED and not avg_is_sat) else _RUNGS_NEEDED[regime]
        if len(rungs) < needed:
            plan.unusable.append(
                f"{regime}: {len(rungs)} of {tried} shapes clear the "
                f"{MIN_RELATIVE_DELTA:.0%} floor on the key column, {needed} needed"
            )
            continue
        plan.batches.extend(_pick_rungs(rungs, max(needed, 3)))
    return plan if plan.batches else None


# ------------------------------------------------------- manifest assembly


def _point(rows: list[tuple[int, int]]) -> dict:
    return {
        "batch_size": len(rows),
        "total_prefill_tokens": sum(s for s, _ in rows),
        "total_kv_read_tokens": sum(p for _, p in rows),
        "rows": [[int(s), int(p)] for s, p in rows],
    }


def build_manifest(
    cells,
    topk: int,
    *,
    repeats: int = 5,
    max_model_len: int = 131072,
    kv_block: int = 1,
    decode: list | None = None,
) -> tuple[dict, list[str]]:
    """Turn uniform sweep coordinates into a schema-v3 manifest with spreads.

    ``cells`` is an iterable of ``(batch_size, total_prefill_tokens,
    total_kv_read_tokens)`` -- the points the ordinary grid would have measured.
    Each becomes its own uniform batch plus the calibration rungs planned around
    it, so the manifest is a superset of the sweep it was derived from: the
    uniform batch is emitted for every cell, including cells around which no
    spread can be built, because this manifest replaces the generated grid
    rather than adding to it.

    Returns the manifest dict and the list of cells that could not be planned,
    with the reason. Those are reported rather than dropped silently -- a cell
    with no spread is a coefficient nothing will pin.
    """
    prefill: list[dict] = []
    notes: list[str] = []
    seen: set[tuple] = set()

    def emit(rows) -> None:
        key = tuple(map(tuple, rows))
        if key in seen:
            return
        seen.add(key)
        point = _point(rows)
        prefill.extend(dict(point) for _ in range(repeats))

    for batch_size, total_new, total_kv in cells:
        s_bar, p_bar = total_new // batch_size, total_kv // batch_size
        if s_bar < 1:
            continue
        # The sweep's own coordinate goes in first and unconditionally. It is
        # emitted even when no spread can be built around it, because this
        # manifest REPLACES the generated grid: a cell dropped here is a
        # coordinate the switch-off run would have measured and this one
        # silently would not.
        emit(uniform_rows(batch_size, s_bar, p_bar))
        if batch_size < 2:
            continue  # a single request has no spread
        plan = plan_cell(
            batch_size,
            s_bar,
            p_bar,
            topk,
            max_model_len=max_model_len,
            kv_block=kv_block,
        )
        if plan is None:
            notes.append(
                f"b={batch_size} avg_s={s_bar} avg_p={p_bar}: no constructible spread"
            )
            continue
        for batch in plan.batches:
            emit(batch.rows)
        notes.extend(
            f"b={batch_size} avg_s={s_bar} avg_p={p_bar}: {reason}"
            for reason in plan.unusable
        )
    return {"schema_version": 3, "prefill": prefill, "decode": decode or []}, notes
