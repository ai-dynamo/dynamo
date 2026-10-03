# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-request goodput and every derived metric of one replay (CONTRACT Amendments A1, A2).

**Good request (A2).** A request is good iff it completed (``terminal_status == "completed"`` and
admitted, with a first and last token) and

- ITL: its mean ITL ``(e2e - ttft) / (output_length - 1)`` is ``<= I``, skipped when
  ``output_length <= 1`` (the replay report's own definition, aisimulate-core ``report.rs``
  ``SlaThresholds::is_good``); and
- E2E slowdown: ``e2e_latency_ms <= S * E0(ISL, OSL) * (1 + E2E_REL_TOL)``
  (:mod:`learned_routing.e0`); and
- TTFT (legacy, normally unset under A2): ``ttft_ms <= T``.

``E2E_REL_TOL`` (1e-6) absorbs float noise only. A request that ran alone with no prefix reuse
has ``e2e == E0`` up to about 2e-8 relative, and must pass at ``S * scale = 1``; AgentX has an
atom of such requests at ``e2e / E0 = 1`` (build audit goodput r2 F1: 11-12% of default's rows,
39-42% of round-robin's). Any real contention (one more request in a batch, one queued pass)
costs far more than 1e-6. ``slowdown_atom_frac`` reports that atom per record.

Unset thresholds are not checked. Rejected, errored, or missing requests are never good.

**Native cross-check (A2.4).** The replay is called with the token-form thresholds (T, I) only, so
its ``goodput_request_throughput_rps`` counts requests passing T and I over the report's
``duration_ms``. :func:`token_form_good` recomputes exactly that from ``per_request``; a relative
disagreement above ``1e-6`` is an error.

**Measurement window (A2.3, LR-01).** ``cell.measure`` sets the rule:

- ``warmup_ms`` (default 0): the window starts this long after the first arrival;
- ``window_ms``: the window length (arrival basis default: through the last arrival);
- ``basis``: ``arrival`` (open-loop default) or ``completion`` (closed-loop and lanes default).
- ``end`` (completion basis only, required there; build audit r1 F1): ``full_occupancy`` or
  ``fixed``, see below.

- ``warmup_trace_ms`` (completion basis only; build audit S1): exclude the warm-up by identity
  instead of by replay time. Every request of a session whose first arrival in the trace is less
  than ``warmup_trace_ms`` after the trace's first arrival is a warm-up request
  (:func:`warmup_ids_from_trace`); warm-up requests are never scored, and the window starts at
  the first dispatch of a non-warm-up request. Closed-loop replay dispatches by concurrency, not
  by trace time, so a replay-time warm-up cannot say which trace rows only warmed the cache.

**Completion-basis end rule (build audit r1 F1).** A closed loop runs at its load only while every
slot is occupied; afterwards the run drains, and drain completions are easy, under-loaded
requests. Agentic lanes drain for most of the run: replay deals plays to lanes once
(``play_index % lanes``), so lanes run out of plays at very different times.

- ``end: full_occupancy`` ends the window at the last instant at which ``occupancy_cap`` (the
  cell's ``load.value``: concurrency or lanes) slots were occupied (:func:`occupancy_profile`).
  A slot is held by an occupancy unit (:func:`occupancy_unit`): an agentic play (a lane runs one
  play at a time, and replay activates a lane's next play the instant the previous one releases)
  or a closed-loop session (held across its turns and think time). A unit spans its first
  arrival to its last terminal. For a work-conserving closed loop this is the first departure
  after the last admission; for lanes it is the first lane's exhaustion. Occupancy above the cap
  or never reaching it, and a window that would end before it starts, are errors.
- ``end: fixed`` ends the window ``window_ms`` after its start (``window_ms > 0`` required).
- A completion-basis rule without ``end`` is an error: the former default, the last dispatch,
  scored the drain.

Every completion-basis record carries the occupancy diagnostics ``occupancy_cap``,
``occupancy_peak``, ``occupancy_units``, ``full_occupancy_end_ms`` and
``window_below_cap_ms``/``window_below_cap_frac`` (time inside the window with fewer than
``occupancy_cap`` occupied slots; 0 under ``full_occupancy``).

Arrival basis: ``good_frac_window`` is the good fraction among requests that ARRIVE inside the
window (late finishers still count; incomplete ones are not good), and ``goodput_rps_window`` is
good in-window arrivals over the window length. Completion basis: good completions in
``[start, end)`` over the window length, and the good fraction among those completions (the ISL
buckets' ``good_frac_window`` uses the same set). An arrival-basis window of zero length (all
arrivals simultaneous) falls back to ``[start, makespan]`` and sets ``window_fallback``.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

SCALES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
ISL_BUCKETS = ((0, 2048), (2048, 8192), (8192, 32768), (32768, 65536), (65536, None))
NATIVE_REL_TOL = 1e-6
E2E_REL_TOL = 1e-6

COMPACT_FIELDS = (
    "session_id",
    "turn_index",
    "arrival_time_ms",
    "first_admit_ms",
    "first_token_ms",
    "last_token_ms",
    "terminal_time_ms",
    "ttft_ms",
    "e2e_latency_ms",
    "input_length",
    "requested_output_length",
    "output_length",
    "reused_input_tokens",
    "prefill_worker_idx",
    "decode_worker_idx",
    "terminal_status",
    "play_id",
)

E0Fn = Callable[[int, int], float]


def compact_row(record: dict) -> dict:
    """The per-request fields the harness keeps (no UUID: it is random per run)."""
    row = {key: record.get(key) for key in COMPACT_FIELDS}
    status = row["terminal_status"]
    row["terminal_status"] = None if status is None else str(status)
    row["lane_id"] = (record.get("agentic") or {}).get("lane_id")
    history = record.get("routing_history") or []
    row["routing_workers"] = [route.get("logical_worker_id") for route in history]
    row["queue_wait_ms"] = sum(
        float(route.get("queue_wait_ms") or 0.0) for route in history
    )
    return row


def canonical_rows(rows: Sequence[dict]) -> list[dict]:
    """Rows in a run-independent order (native per_request order follows random UUIDs)."""
    import json

    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def worker_of(row: dict) -> int | None:
    worker = row.get("decode_worker_idx")
    return row.get("prefill_worker_idx") if worker is None else worker


def completed(row: dict) -> bool:
    return (
        row.get("terminal_status") == "completed"
        and row.get("first_admit_ms") is not None
        and row.get("ttft_ms") is not None
        and row.get("e2e_latency_ms") is not None
    )


def mean_itl_ms(row: dict) -> float | None:
    osl = row["output_length"]
    if osl <= 1:
        return None
    return (row["e2e_latency_ms"] - row["ttft_ms"]) / (osl - 1.0)


def token_form_good(row: dict, ttft_ms: float | None, itl_ms: float | None) -> bool:
    """The replay report's ``SlaThresholds::is_good`` for a token-form SLA."""
    if not completed(row):
        return False
    if ttft_ms is not None and row["ttft_ms"] > ttft_ms:
        return False
    if itl_ms is not None and row["output_length"] > 1:
        if (row["e2e_latency_ms"] - row["ttft_ms"]) / (
            row["output_length"] - 1.0
        ) > itl_ms:
            return False
    return True


def a2_good(row: dict, sla: dict, e0: E0Fn | None, scale: float = 1.0) -> bool:
    ttft = sla.get("ttft_ms")
    itl = sla.get("itl_ms")
    if not token_form_good(
        row,
        None if ttft is None else scale * ttft,
        None if itl is None else scale * itl,
    ):
        return False
    slowdown = sla.get("e2e_slowdown")
    if slowdown is not None:
        if e0 is None:
            raise ValueError("an e2e_slowdown SLA needs an E0 function")
        bound = scale * slowdown * e0(row["input_length"], row["output_length"])
        if row["e2e_latency_ms"] > bound * (1.0 + E2E_REL_TOL):
            return False
    return True


def at_uncontended_latency(row: dict, e0: E0Fn) -> bool:
    """A completed request whose e2e equals its E0 within ``E2E_REL_TOL`` (the e2e/E0 atom)."""
    reference = e0(row["input_length"], row["output_length"])
    return abs(row["e2e_latency_ms"] - reference) <= E2E_REL_TOL * reference


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy's default method); None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = q / 100.0 * (len(ordered) - 1)
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


MEASURE_KEYS = {"basis", "warmup_ms", "window_ms", "warmup_trace_ms", "end"}
COMPLETION_ENDS = ("full_occupancy", "fixed")
OCCUPANCY_FIELDS = (
    "occupancy_cap",
    "occupancy_peak",
    "occupancy_units",
    "full_occupancy_end_ms",
    "window_below_cap_ms",
    "window_below_cap_frac",
)


def validate_measure(measure: dict, *, open_loop: bool) -> str:
    """Check a measurement rule (module docstring) without rows; returns its basis."""
    unknown = sorted(set(measure) - MEASURE_KEYS)
    if unknown:
        raise ValueError(f"unknown measure keys {unknown}")
    basis = measure.get("basis") or ("arrival" if open_loop else "completion")
    if basis not in ("arrival", "completion"):
        raise ValueError(f"measure.basis must be arrival or completion, got {basis!r}")
    if measure.get("warmup_trace_ms") is not None and (
        basis != "completion" or measure.get("warmup_ms")
    ):
        raise ValueError(
            "measure.warmup_trace_ms needs basis completion and no warmup_ms"
        )
    end = measure.get("end")
    if basis == "arrival":
        if end is not None:
            raise ValueError(
                f"measure.end applies to basis completion only, got {end!r}"
            )
        return basis
    if end not in COMPLETION_ENDS:
        raise ValueError(
            f"basis completion needs measure.end in {list(COMPLETION_ENDS)}, got {end!r} "
            "(the last dispatch scores the drain: build audit r1 F1)"
        )
    window_ms = measure.get("window_ms")
    if end == "fixed" and not (window_ms is not None and float(window_ms) > 0):
        raise ValueError(f"measure.end fixed needs window_ms > 0, got {window_ms!r}")
    if end == "full_occupancy" and window_ms is not None:
        raise ValueError("measure.end full_occupancy takes no window_ms")
    return basis


def occupancy_unit(row: dict) -> str:
    """The closed-loop slot holder of a request: its agentic play, else its session."""
    unit = row.get("play_id")
    if unit is None:
        unit = row.get("session_id")
    if unit is None:
        raise ValueError("occupancy needs a play_id or session_id on every row")
    return str(unit)


def occupancy_profile(rows: Sequence[dict], cap: int) -> dict:
    """Occupied slots over time: each unit holds one from its first arrival to its last terminal.

    ``steps`` lists ``(t, occupied)`` from each change point until the next. ``full_end_ms`` is
    the end of the last step with at least ``cap`` occupied slots (None if there is none).
    """
    spans: dict[str, tuple[float, float]] = {}
    for row in rows:
        start, end = row.get("arrival_time_ms"), row.get("terminal_time_ms")
        if start is None or end is None:
            raise ValueError("occupancy needs arrival_time_ms and terminal_time_ms")
        unit = occupancy_unit(row)
        lo, hi = spans.get(unit, (start, end))
        spans[unit] = (min(lo, start), max(hi, end))
    # Net change per instant: a unit that ends as another starts leaves the count unchanged.
    deltas: dict[float, int] = defaultdict(int)
    for lo, hi in spans.values():
        if hi > lo:
            deltas[lo] += 1
            deltas[hi] -= 1
    times = sorted(deltas)
    steps, occupied, peak, full_end = [], 0, 0, None
    for index, t in enumerate(times):
        occupied += deltas[t]
        peak = max(peak, occupied)
        steps.append((t, occupied))
        if occupied >= cap and index + 1 < len(times):
            full_end = times[index + 1]
    return {"units": len(spans), "peak": peak, "full_end_ms": full_end, "steps": steps}


def time_below_cap(
    steps: Sequence[tuple[float, int]], cap: int, lo: float, hi: float
) -> float:
    """Time inside ``[lo, hi]`` with fewer than ``cap`` occupied slots."""
    below = 0.0
    bounds = [t for t, _ in steps[1:]] + [math.inf]
    if not steps or lo < steps[0][0]:
        below += min(hi, steps[0][0] if steps else hi) - lo
    for (t, occupied), t_next in zip(steps, bounds):
        if occupied < cap:
            below += max(0.0, min(t_next, hi) - max(t, lo))
    return below


def session_key(row: dict, line_number: int) -> str:
    """The session ID replay gives a trace row: its own, else ``request_<line>`` (1-based).

    Mirrors aisimulate-core ``replay/loadgen/trace.rs`` (``MooncakeTraceBuilder::push``), which is
    what closed-loop replay reports as ``per_request.session_id``.
    """
    session = row.get("session_id")
    return f"request_{line_number}" if session is None else str(session)


def warmup_ids_from_trace(lines: Sequence[str], warmup_trace_ms: float) -> frozenset:
    """Session IDs whose first arrival is less than ``warmup_trace_ms`` after the first arrival.

    ``lines`` are the Mooncake-format JSONL lines replay ran (the replicate trace). Later turns of
    a session carry no timestamp; they belong to their session's first arrival.
    """
    rows = [(n, json.loads(line)) for n, line in enumerate(lines, 1) if line.strip()]
    stamps = [float(row["timestamp"]) for _, row in rows if "timestamp" in row]
    if not stamps:
        raise ValueError("warmup_trace_ms needs a trace with timestamps")
    origin = min(stamps)
    return frozenset(
        session_key(row, n)
        for n, row in rows
        if "timestamp" in row and float(row["timestamp"]) - origin < warmup_trace_ms
    )


def warmup_mask(
    rows: Sequence[dict], measure: dict, warmup_ids: frozenset | None
) -> list[bool]:
    """Per row: True if the identity rule excludes it as warm-up (all False without the rule)."""
    if measure.get("warmup_trace_ms") is None:
        return [False] * len(rows)
    if warmup_ids is None:
        raise ValueError(
            "measure.warmup_trace_ms needs the trace's warm-up session IDs"
        )
    missing = sum(row.get("session_id") is None for row in rows)
    if missing:
        raise ValueError(
            f"measure.warmup_trace_ms needs per-request session IDs; {missing} rows have none"
        )
    return [row["session_id"] in warmup_ids for row in rows]


def measurement_window(
    rows: Sequence[dict],
    *,
    open_loop: bool,
    measure: dict,
    makespan_ms: float,
    excluded: Sequence[bool] | None = None,
    occupancy_cap: int | None = None,
) -> dict:
    basis = validate_measure(measure, open_loop=open_loop)
    by_identity = measure.get("warmup_trace_ms") is not None
    arrivals = [
        row["arrival_time_ms"] for row in rows if row.get("arrival_time_ms") is not None
    ]
    first = min(arrivals) if arrivals else 0.0
    last = max(arrivals) if arrivals else 0.0
    warmup = float(measure.get("warmup_ms") or 0.0)
    start = first + warmup
    if by_identity:
        measured = [
            row["arrival_time_ms"]
            for row, skip in zip(rows, excluded or [False] * len(rows))
            if not skip and row.get("arrival_time_ms") is not None
        ]
        start = min(measured) if measured else last
        warmup = start - first
    window = {
        "basis": basis,
        "start_ms": start,
        "warmup_ms": warmup,
        "fallback": False,
        "first_arrival_ms": first,
        "last_arrival_ms": last,
    }
    if basis == "completion":
        return {**window, **_completion_end(rows, measure, start, occupancy_cap)}
    end = start + float(measure["window_ms"]) if measure.get("window_ms") else last
    if end <= start:
        end = max(makespan_ms, start)
        window["fallback"] = True
    return {**window, "end_ms": end}


def _completion_end(
    rows: Sequence[dict], measure: dict, start: float, cap: int | None
) -> dict:
    if cap is None or int(cap) < 1:
        raise ValueError(
            f"basis completion needs occupancy_cap >= 1 (load.value), got {cap!r}"
        )
    cap = int(cap)
    profile = occupancy_profile(rows, cap)
    if profile["peak"] > cap:
        raise ValueError(
            f"occupancy {profile['peak']} exceeds the cap {cap}: the occupancy units do not "
            "match the load driver's slots"
        )
    full_end = profile["full_end_ms"]
    if measure["end"] == "fixed":
        end = start + float(measure["window_ms"])
    elif full_end is None:
        raise ValueError(
            f"measure.end full_occupancy: occupancy never reaches the cap {cap} "
            f"(peak {profile['peak']} over {profile['units']} units)"
        )
    else:
        end = full_end
    if end <= start:
        raise ValueError(
            f"completion window is empty: end {end!r} <= start {start!r} "
            f"(measure.end {measure['end']})"
        )
    below = time_below_cap(profile["steps"], cap, start, end)
    return {
        "end_ms": end,
        "occupancy_cap": cap,
        "occupancy_peak": profile["peak"],
        "occupancy_units": profile["units"],
        "full_occupancy_end_ms": full_end,
        "window_below_cap_ms": below,
        "window_below_cap_frac": below / (end - start),
    }


def _window_counts(rows, good, window, missing: int, excluded) -> tuple[int, int]:
    start, end = window["start_ms"], window["end_ms"]
    if window["basis"] == "arrival":
        inside = [
            index
            for index, row in enumerate(rows)
            if start <= row["arrival_time_ms"] <= end or window["fallback"]
        ]
        # Requests absent from per_request have unknown arrival; count them as in-window misses.
        return sum(good[i] for i in inside), len(inside) + missing
    inside = [
        index
        for index, row in enumerate(rows)
        if not excluded[index] and _completes_inside(row, window)
    ]
    return sum(good[i] for i in inside), len(inside)


def _completes_inside(row: dict, window: dict) -> bool:
    """Completion basis: completed in ``[start, end)``. The full-occupancy end is itself a
    completion (the one that empties a slot for good); counting it would add one completion to
    every run (build fixer r1)."""
    return (
        completed(row)
        and window["start_ms"] <= row["terminal_time_ms"] < window["end_ms"]
    )


def _window_rates(
    good_in: int, n_in: int, window: dict
) -> tuple[float | None, float | None]:
    length_s = (window["end_ms"] - window["start_ms"]) / 1000.0
    frac = good_in / n_in if n_in else None
    rate = good_in / length_s if length_s > 0 else None
    return frac, rate


def compute_metrics(
    rows: Sequence[dict],
    summary: dict,
    *,
    sla: dict,
    measure: dict,
    open_loop: bool,
    num_workers: int,
    e0: E0Fn | None,
    warmup_ids: frozenset | None = None,
    occupancy_cap: int | None = None,
) -> dict:
    """Every per-evaluation metric; pure in its arguments.

    ``occupancy_cap`` (the cell's ``load.value``) is required for completion basis.
    """
    num_requests = int(summary.get("num_requests", len(rows)))
    missing = max(num_requests - len(rows), 0)
    makespan_ms = float(summary.get("duration_ms") or 0.0)
    makespan_s = max(makespan_ms / 1000.0, 1e-9)
    done = [completed(row) for row in rows]
    excluded = warmup_mask(rows, measure, warmup_ids)
    window = measurement_window(
        rows,
        open_loop=open_loop,
        measure=measure,
        makespan_ms=makespan_ms,
        excluded=excluded,
        occupancy_cap=occupancy_cap,
    )
    needs_e0 = sla.get("e2e_slowdown") is not None
    if needs_e0 and e0 is None:
        raise ValueError("an e2e_slowdown SLA needs an E0 function")

    # Native-comparable token-form pass (A2.4).
    token_good = [
        token_form_good(row, sla.get("ttft_ms"), sla.get("itl_ms")) for row in rows
    ]
    report = summary.get("goodput_request_throughput_rps")
    itl_rps = sum(token_good) / makespan_s
    check_err = None
    if report is not None:
        check_err = abs(itl_rps - report) / max(abs(report), 1e-12)

    good = [a2_good(row, sla, e0) for row in rows]
    good_in, n_in = _window_counts(rows, good, window, missing, excluded)
    frac_w, rate_w = _window_rates(good_in, n_in, window)

    rescore = {}
    for scale in SCALES:
        scaled = (
            good if scale == 1.0 else [a2_good(row, sla, e0, scale) for row in rows]
        )
        g_in, n_scaled = _window_counts(rows, scaled, window, missing, excluded)
        f, r = _window_rates(g_in, n_scaled, window)
        rescore[repr(scale)] = {
            "good_frac_window": f,
            "goodput_rps_window": r,
            "good_frac": sum(scaled) / num_requests if num_requests else None,
        }

    ttfts = [row["ttft_ms"] for row, ok in zip(rows, done) if ok]
    itls = [
        v
        for row, ok in zip(rows, done)
        if ok
        for v in [mean_itl_ms(row)]
        if v is not None
    ]
    slowdowns, atom = [], None
    if e0 is not None:
        finished = [row for row, ok in zip(rows, done) if ok]
        slowdowns = [
            row["e2e_latency_ms"] / e0(row["input_length"], row["output_length"])
            for row in finished
        ]
        if finished:
            atom = sum(at_uncontended_latency(row, e0) for row in finished) / len(
                finished
            )

    buckets = {}
    for low, high in ISL_BUCKETS:
        label = f"{low}-{'' if high is None else high}"
        index = [
            i
            for i, row in enumerate(rows)
            if row["input_length"] >= low
            and (high is None or row["input_length"] < high)
        ]
        if not index:
            continue
        if window["basis"] == "completion":
            in_window = [
                i
                for i in index
                if not excluded[i] and _completes_inside(rows[i], window)
            ]
        else:
            in_window = [
                i
                for i in index
                if window["fallback"]
                or window["start_ms"] <= rows[i]["arrival_time_ms"] <= window["end_ms"]
            ]
        bucket_ttft = [rows[i]["ttft_ms"] for i in index if done[i]]
        buckets[label] = {
            "n": len(index),
            "good_frac": sum(good[i] for i in index) / len(index),
            "good_frac_window": (
                sum(good[i] for i in in_window) / len(in_window) if in_window else None
            ),
            "ttft_p50": percentile(bucket_ttft, 50),
            "ttft_p90": percentile(bucket_ttft, 90),
            "worker_share_max": _max_share([worker_of(rows[i]) for i in index]),
        }

    return {
        "num_requests": num_requests,
        "per_request_rows": len(rows),
        "missing_rows": missing,
        "completed": int(summary.get("completed_requests", sum(done))),
        "completed_rows": sum(done),
        "incomplete": num_requests - sum(done),
        "good": sum(good),
        "good_frac": sum(good) / num_requests if num_requests else None,
        "goodput_rps": sum(good) / makespan_s,
        "goodput_rps_itl": itl_rps,
        "goodput_rps_report": report,
        "goodput_check_rel_err": check_err,
        "good_frac_window": frac_w,
        "goodput_rps_window": rate_w,
        "window_good": good_in,
        "window_requests": n_in,
        "window_basis": window["basis"],
        "window_start_ms": window["start_ms"],
        "window_end_ms": window["end_ms"],
        "window_fallback": window["fallback"],
        **{key: window.get(key) for key in OCCUPANCY_FIELDS},
        "warmup_ms": window["warmup_ms"],
        "warmup_excluded_rows": sum(excluded),
        "duration_ms": makespan_ms,
        "makespan_ms": makespan_ms,
        "arrival_span_ms": window["last_arrival_ms"] - window["first_arrival_ms"],
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p90": percentile(ttfts, 90),
        "ttft_p99": percentile(ttfts, 99),
        "itl_p50": percentile(itls, 50),
        "itl_p90": percentile(itls, 90),
        "itl_tok_p99": summary.get("p99_itl_ms"),
        "slowdown_p50": percentile(slowdowns, 50),
        "slowdown_p90": percentile(slowdowns, 90),
        "slowdown_p95": percentile(slowdowns, 95),
        "slowdown_atom_frac": atom,
        "throughput_tok_s": summary.get("output_throughput_tok_s"),
        "total_throughput_tok_s": summary.get("total_throughput_tok_s"),
        "prefix_reuse": summary.get("prefix_cache_reused_ratio"),
        "rescore": rescore,
        "isl_buckets": buckets,
        "guards": guard_metrics(rows, done, sla, e0, num_workers),
    }


def _max_share(workers: Sequence[int | None]) -> float | None:
    counts = Counter(w for w in workers if w is not None)
    total = sum(counts.values())
    return max(counts.values()) / total if total else None


def guard_metrics(rows, done, sla, e0, num_workers: int) -> dict:
    """LR-13 guard metrics: concentration, segregation, stickiness, tail slowdown."""
    requests = Counter()
    prefill_tokens: dict[int, float] = defaultdict(float)
    for row in rows:
        worker = worker_of(row)
        if worker is None:
            continue
        requests[worker] += 1
        prefill_tokens[worker] += max(
            row["input_length"] - (row.get("reused_input_tokens") or 0), 0
        )
    total = sum(requests.values())
    mean_prefill = sum(prefill_tokens.values()) / max(num_workers, 1)
    sessions: dict[str, set] = defaultdict(set)
    session_rows: Counter = Counter()
    for row in rows:
        session = row.get("session_id")
        worker = worker_of(row)
        if session is None or worker is None:
            continue
        sessions[session].add(worker)
        session_rows[session] += 1
    multi = [s for s, n in session_rows.items() if n >= 2]
    slowdown = sla.get("e2e_slowdown")
    ttft_bound = sla.get("ttft_ms")
    finished = [row for row, ok in zip(rows, done) if ok]
    out = {
        "worker_share_max": max(requests.values()) / total if total else None,
        "worker_share_cap": min(2.0 / num_workers, 1.0 / num_workers + 0.25),
        "workers_used": len(requests),
        "worker_prefill_max_over_mean": (
            max(prefill_tokens.values()) / mean_prefill if mean_prefill > 0 else None
        ),
        "sessions_multi_turn": len(multi),
        "session_workers_mean": _mean([len(sessions[s]) for s in multi]),
        "session_split_frac": (
            sum(len(sessions[s]) > 1 for s in multi) / len(multi) if multi else None
        ),
        "frac_ttft_gt_3T": None,
        "frac_slowdown_gt_3S": None,
        "slowdown_clip_mean": None,
        "queue_wait_p99_ms": percentile(
            [row.get("queue_wait_ms") or 0.0 for row in rows], 99
        ),
    }
    if ttft_bound is not None and finished:
        out["frac_ttft_gt_3T"] = sum(
            r["ttft_ms"] > 3 * ttft_bound for r in finished
        ) / len(finished)
    if slowdown is not None and e0 is not None and finished:
        ratios = [
            r["e2e_latency_ms"] / (slowdown * e0(r["input_length"], r["output_length"]))
            for r in finished
        ]
        out["frac_slowdown_gt_3S"] = sum(x > 3.0 for x in ratios) / len(ratios)
        out["slowdown_clip_mean"] = sum(min(x, 10.0) for x in ratios) / len(ratios)
    return out
