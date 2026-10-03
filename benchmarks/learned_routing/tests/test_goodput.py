# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from learned_routing import goodput


def row(
    arrival=0.0,
    ttft=100.0,
    e2e=1000.0,
    osl=10,
    isl=1000,
    status="completed",
    worker=0,
    **extra,
):
    base = {
        "session_id": None,
        "turn_index": None,
        "arrival_time_ms": arrival,
        "first_admit_ms": None if status != "completed" else arrival,
        "first_token_ms": None if ttft is None else arrival + ttft,
        "last_token_ms": None if e2e is None else arrival + e2e,
        "terminal_time_ms": arrival + (e2e or 0.0),
        "ttft_ms": ttft,
        "e2e_latency_ms": e2e,
        "input_length": isl,
        "requested_output_length": osl,
        "output_length": osl,
        "reused_input_tokens": 0,
        "prefill_worker_idx": None,
        "decode_worker_idx": worker,
        "terminal_status": status,
        "routing_workers": [worker],
        "queue_wait_ms": 0.0,
    }
    base.update(extra)
    return base


def test_itl_check_matches_report_definition_and_is_inclusive():
    # mean ITL = (e2e - ttft) / (osl - 1) = (1000 - 100) / 9 = 100.0 exactly
    assert goodput.token_form_good(row(), None, 100.0)
    assert not goodput.token_form_good(row(), None, 99.999)
    # output_length <= 1 skips the ITL check, as the replay report does
    assert goodput.token_form_good(row(osl=1, e2e=10_000.0), None, 1.0)
    # TTFT bound (legacy form) is inclusive too
    assert goodput.token_form_good(row(), 100.0, None)
    assert not goodput.token_form_good(row(), 99.0, None)


def test_incomplete_and_rejected_requests_are_never_good():
    assert not goodput.token_form_good(
        row(status="rejected", ttft=None, e2e=None), None, None
    )
    assert not goodput.a2_good(row(status="rejected", ttft=None, e2e=None), {}, None)
    # completed but tokenless rows are not good under a token-form SLA (report: is_good_without_tokens)
    assert not goodput.token_form_good(row(ttft=None, e2e=None), None, 50.0)


def test_e2e_slowdown_uses_e0_of_isl_osl_and_scales():
    e0 = lambda isl, osl: 250.0  # noqa: E731
    sla = {"itl_ms": None, "e2e_slowdown": 4.0, "ttft_ms": None}
    assert goodput.a2_good(row(e2e=1000.0), sla, e0)  # 1000 <= 4 * 250
    assert not goodput.a2_good(row(e2e=1000.1), sla, e0)
    assert goodput.a2_good(row(e2e=1500.0), sla, e0, scale=1.5)
    assert not goodput.a2_good(row(e2e=1500.0), sla, e0, scale=1.0)
    with pytest.raises(ValueError):
        goodput.a2_good(row(), sla, None)


def test_uncontended_request_is_good_at_unit_slowdown_scale():
    """Build audit goodput r2 F1: e2e == E0 up to float noise must pass at S * scale = 1."""
    e0_ms = 3123.456789
    e0 = lambda isl, osl: e0_ms  # noqa: E731
    sla = {"itl_ms": None, "e2e_slowdown": 1.0, "ttft_ms": None}
    for noise in (-2.4e-9, 0.0, 2.4e-9, 1.6e-8, 9e-7):
        assert goodput.a2_good(row(e2e=e0_ms * (1 + noise)), sla, e0), noise
    # S = 2 at rescore scale 0.5 is the same unit bound
    assert goodput.a2_good(
        row(e2e=e0_ms * (1 + 1.6e-8)), {**sla, "e2e_slowdown": 2.0}, e0, scale=0.5
    )
    # the tolerance is float-noise sized: real slowdowns still fail
    assert not goodput.a2_good(row(e2e=e0_ms * (1 + 2e-6)), sla, e0)
    assert not goodput.a2_good(row(e2e=e0_ms * 1.001), sla, e0)


def test_slowdown_atom_counts_requests_at_their_uncontended_latency():
    e0 = lambda isl, osl: 1000.0  # noqa: E731
    e2es = [1000.0 * (1 + 1e-8), 1000.0, 999.0, 1001.0, 2000.0]  # reuse, contention
    rows = [row(arrival=float(i), e2e=e) for i, e in enumerate(e2es)]
    rows.append(row(arrival=9.0, status="rejected", ttft=None, e2e=None))
    m = goodput.compute_metrics(
        rows,
        summary_for(rows),
        sla={"itl_ms": None, "e2e_slowdown": 1.0, "ttft_ms": None},
        measure={},
        open_loop=True,
        num_workers=2,
        e0=e0,
    )
    assert m["slowdown_atom_frac"] == pytest.approx(2 / 5)
    assert m["good"] == 3  # the atom plus the reuse row; the rejection is a miss
    assert m["slowdown_p95"] == pytest.approx(1.001 + 0.8 * (2.0 - 1.001))


def summary_for(rows, report=None, duration=None):
    duration = (
        duration if duration is not None else max(r["terminal_time_ms"] for r in rows)
    )
    out = {
        "num_requests": len(rows),
        "completed_requests": len(rows),
        "duration_ms": duration,
    }
    if report is not None:
        out["goodput_request_throughput_rps"] = report
    return out


def test_native_cross_check_recomputes_report_goodput():
    rows = [row(arrival=i * 100.0, e2e=1000.0 + 50 * i) for i in range(10)]
    sla = {"itl_ms": 105.0, "e2e_slowdown": None, "ttft_ms": None}
    # ITL_i = (900 + 50 i) / 9 <= 105  <=>  i <= 0.9  -> only i = 0 is good
    duration = max(r["terminal_time_ms"] for r in rows)
    native = 1 / (duration / 1000.0)
    metrics = goodput.compute_metrics(
        rows,
        summary_for(rows, native),
        sla=sla,
        measure={},
        open_loop=True,
        num_workers=2,
        e0=None,
    )
    assert metrics["goodput_rps_itl"] == native
    assert metrics["goodput_check_rel_err"] == 0.0
    wrong = goodput.compute_metrics(
        rows,
        summary_for(rows, native * 1.01),
        sla=sla,
        measure={},
        open_loop=True,
        num_workers=2,
        e0=None,
    )
    assert wrong["goodput_check_rel_err"] > goodput.NATIVE_REL_TOL


def test_arrival_window_excludes_warmup_keeps_late_finishers_and_counts_misses():
    rows = [
        row(arrival=0.0),  # warm-up: excluded
        row(
            arrival=1000.0, e2e=50_000.0
        ),  # in window, finishes long after: counted, but bad ITL
        row(arrival=2000.0),  # in window, good
        row(
            arrival=3000.0, status="rejected", ttft=None, e2e=None
        ),  # in window, not good
        row(arrival=4000.0),  # last arrival = window end (inclusive), good
    ]
    sla = {"itl_ms": 200.0, "e2e_slowdown": None, "ttft_ms": None}
    m = goodput.compute_metrics(
        rows,
        summary_for(rows),
        sla=sla,
        measure={"warmup_ms": 500.0},
        open_loop=True,
        num_workers=2,
        e0=None,
    )
    assert (m["window_start_ms"], m["window_end_ms"]) == (500.0, 4000.0)
    assert m["window_requests"] == 4 and m["window_good"] == 2
    assert m["good_frac_window"] == 0.5
    assert m["goodput_rps_window"] == pytest.approx(2 / 3.5)


def test_requests_missing_from_per_request_count_as_in_window_misses():
    rows = [row(arrival=0.0), row(arrival=1000.0)]
    summary = summary_for(rows)
    summary["num_requests"] = 3
    m = goodput.compute_metrics(
        rows,
        summary,
        sla={"itl_ms": 200.0},
        measure={},
        open_loop=True,
        num_workers=1,
        e0=None,
    )
    assert m["missing_rows"] == 1 and m["window_requests"] == 3
    assert m["good_frac_window"] == pytest.approx(2 / 3)
    assert m["good_frac"] == pytest.approx(2 / 3)


def test_completion_window_counts_good_completions_per_second():
    # closed loop, fixed end: window = [first arrival, first arrival + 2500)
    rows = [
        row(arrival=0.0, e2e=1000.0, session_id="a"),  # completes at 1000: inside
        row(arrival=0.0, e2e=2000.0, session_id="b"),  # completes at 2000: inside
        row(arrival=1000.0, e2e=4000.0, session_id="c"),  # 5000: outside
        row(arrival=2000.0, e2e=1000.0, session_id="d"),  # 3000: outside
    ]
    m = goodput.compute_metrics(
        rows,
        summary_for(rows),
        sla={"itl_ms": 1e9},
        measure={"end": "fixed", "window_ms": 2500.0},
        open_loop=False,
        num_workers=2,
        e0=None,
        occupancy_cap=2,
    )
    assert m["window_basis"] == "completion"
    assert (m["window_start_ms"], m["window_end_ms"]) == (0.0, 2500.0)
    assert m["window_requests"] == 2  # completions at 1000 and 2000
    assert m["goodput_rps_window"] == pytest.approx(2 / 2.5)
    # slots hand over at 1000 (a -> c) and 2000 (b -> d); full occupancy ends with d at 3000
    assert m["window_below_cap_ms"] == 0.0 and m["full_occupancy_end_ms"] == 3000.0


def test_completion_basis_needs_an_explicit_end_rule():
    rows = [row(arrival=0.0, session_id="a"), row(arrival=0.0, session_id="b")]
    base = dict(sla={"itl_ms": 1e9}, num_workers=2, e0=None, occupancy_cap=2)
    for measure, match in (
        ({}, "measure.end"),
        ({"basis": "completion"}, "measure.end"),
        ({"end": "last_dispatch"}, "measure.end"),
        ({"end": "fixed"}, "window_ms > 0"),
        ({"end": "fixed", "window_ms": 0}, "window_ms > 0"),
        ({"end": "full_occupancy", "window_ms": 5.0}, "takes no window_ms"),
    ):
        with pytest.raises(ValueError, match=match):
            goodput.compute_metrics(
                rows, summary_for(rows), measure=measure, open_loop=False, **base
            )
    with pytest.raises(ValueError, match="completion only"):
        goodput.validate_measure({"end": "full_occupancy"}, open_loop=True)
    with pytest.raises(ValueError, match="occupancy_cap"):
        goodput.compute_metrics(
            rows,
            summary_for(rows),
            measure={"end": "full_occupancy"},
            open_loop=False,
            **{**base, "occupancy_cap": None},
        )


def lanes_rows():
    """Two lanes (cap 2): lane 0 runs play p1 then p3; lane 1 runs only p2.

    p1 has a root session and a concurrent subagent session, so counting sessions instead of
    plays would show three occupied slots.
    """
    return [
        row(arrival=0.0, e2e=400.0, session_id="p1:root", play_id="p1"),
        row(arrival=100.0, e2e=500.0, session_id="p1:sub", play_id="p1"),
        row(
            arrival=600.0, e2e=400.0, session_id="p1:root", play_id="p1"
        ),  # p1 ends 1000
        row(
            arrival=0.0, e2e=1500.0, session_id="p2:root", play_id="p2"
        ),  # lane 1 ends 1500
        row(
            arrival=1000.0, e2e=200.0, session_id="p3:root", play_id="p3"
        ),  # 1200: inside
        row(
            arrival=1400.0, e2e=600.0, session_id="p3:root", play_id="p3"
        ),  # 2000: drain
        row(
            arrival=2000.0, e2e=1000.0, session_id="p3:root", play_id="p3"
        ),  # 3000: drain
    ]


def test_full_occupancy_ends_at_the_first_lane_exhaustion():
    rows = lanes_rows()
    m = goodput.compute_metrics(
        rows,
        summary_for(rows),
        sla={"itl_ms": 1e9},
        measure={"basis": "completion", "end": "full_occupancy"},
        open_loop=False,
        num_workers=2,
        e0=None,
        occupancy_cap=2,
    )
    # p1 hands lane 0 to p3 at 1000 without a gap; lane 1 runs out at 1500
    assert (m["window_start_ms"], m["window_end_ms"]) == (0.0, 1500.0)
    assert m["full_occupancy_end_ms"] == 1500.0 and m["window_below_cap_ms"] == 0.0
    assert (m["occupancy_peak"], m["occupancy_units"]) == (2, 3)
    # completions at 400, 600, 1000 and 1200 count; p2's final one at 1500 ends the window
    # (half-open), and the drain at 2000 and 3000 does not count
    assert m["window_requests"] == 4
    assert m["goodput_rps_window"] == pytest.approx(4 / 1.5)
    # The former default (end at the last dispatch, 2000) would also score the drain; a fixed
    # window that long reports its partial-occupancy time.
    fixed = goodput.compute_metrics(
        rows,
        summary_for(rows),
        sla={"itl_ms": 1e9},
        measure={"end": "fixed", "window_ms": 2000.0},
        open_loop=False,
        num_workers=2,
        e0=None,
        occupancy_cap=2,
    )
    assert fixed["window_requests"] == 5
    assert fixed["window_below_cap_ms"] == 500.0
    assert fixed["window_below_cap_frac"] == pytest.approx(0.25)
    # Session keys alone (no play_id) put three units in two slots: an error, not a silent pass.
    sessions_only = [{**r, "play_id": None} for r in rows]
    with pytest.raises(ValueError, match="exceeds the cap 2"):
        goodput.compute_metrics(
            sessions_only,
            summary_for(sessions_only),
            sla={"itl_ms": 1e9},
            measure={"end": "full_occupancy"},
            open_loop=False,
            num_workers=2,
            e0=None,
            occupancy_cap=2,
        )


def test_full_occupancy_closed_loop_ends_at_the_first_departure_after_the_last_admission():
    # C = 2, work-conserving: s3 takes s2's slot at 700, s4 takes s1's at 1000 (last admission).
    # s3 holds its slot through its think time (900-1100). The first departure after the last
    # admission is s4's at 1300.
    rows = [
        row(arrival=0.0, e2e=1000.0, session_id="s1"),
        row(arrival=0.0, e2e=700.0, session_id="s2"),
        row(arrival=700.0, e2e=200.0, session_id="s3"),  # 900
        row(arrival=1100.0, e2e=500.0, session_id="s3"),  # 1600: drain
        row(arrival=1000.0, e2e=300.0, session_id="s4"),  # 1300: last full instant
    ]
    m = goodput.compute_metrics(
        rows,
        summary_for(rows),
        sla={"itl_ms": 1e9},
        measure={"end": "full_occupancy"},
        open_loop=False,
        num_workers=2,
        e0=None,
        occupancy_cap=2,
    )
    assert m["window_end_ms"] == 1300.0 and m["window_below_cap_ms"] == 0.0
    assert (
        m["window_requests"] == 3
    )  # 700, 900 and 1000; s4's departure at 1300 ends it
    # never reaching the cap, or a warm-up past the end of full occupancy, is an error
    with pytest.raises(ValueError, match="never reaches the cap 3"):
        goodput.compute_metrics(
            rows,
            summary_for(rows),
            sla={"itl_ms": 1e9},
            measure={"end": "full_occupancy"},
            open_loop=False,
            num_workers=2,
            e0=None,
            occupancy_cap=3,
        )
    with pytest.raises(ValueError, match="window is empty"):
        goodput.compute_metrics(
            rows,
            summary_for(rows),
            sla={"itl_ms": 1e9},
            measure={"end": "full_occupancy", "warmup_ms": 1300.0},
            open_loop=False,
            num_workers=2,
            e0=None,
            occupancy_cap=2,
        )


def test_zero_length_units_never_occupy_a_slot():
    rows = [
        row(arrival=0.0, e2e=1000.0, session_id="a"),
        row(arrival=500.0, status="rejected", ttft=None, e2e=None, session_id="r"),
    ]
    profile = goodput.occupancy_profile(rows, 1)
    assert profile["peak"] == 1 and profile["full_end_ms"] == 1000.0


def test_warmup_ids_follow_replay_session_keys():
    """Flat rows get replay's request_<line> IDs; later turns belong to their session."""
    import json

    lines = [
        json.dumps({"timestamp": 1000, "input_length": 1, "output_length": 1}),
        json.dumps({"timestamp": 1000, "input_length": 1, "output_length": 1}),
        json.dumps({"timestamp": 4999, "input_length": 1, "output_length": 1}),
        json.dumps({"timestamp": 5000, "input_length": 1, "output_length": 1}),
        json.dumps({"session_id": "s", "timestamp": 2000, "input_length": 1}),
        json.dumps({"session_id": "s", "delay": 10, "input_length": 1}),
        json.dumps({"session_id": "t", "timestamp": 9000, "input_length": 1}),
    ]
    ids = goodput.warmup_ids_from_trace(lines, 4000.0)
    assert ids == {"request_1", "request_2", "request_3", "s"}


def test_closed_loop_identity_warmup_excludes_warmup_rows_and_starts_after_them():
    # C = 2. Warm-up sessions w1, w2 are dispatched first; the window opens at the first
    # measured dispatch (500). m3 takes m2's slot at 1800 (last admission); the first departure
    # after it, m3's at 2100, ends full occupancy.
    rows = [
        row(
            arrival=0.0, e2e=500.0, session_id="w1"
        ),  # warm-up, completes at 500: excluded
        row(
            arrival=0.0, e2e=900.0, session_id="w2"
        ),  # warm-up, completes in window: excluded
        row(arrival=500.0, e2e=1000.0, session_id="m1"),  # completes at 1500: inside
        row(arrival=900.0, e2e=900.0, session_id="m2", osl=2),  # 1800: inside, ITL miss
        row(arrival=1500.0, e2e=800.0, session_id="m4"),  # 2300: drain
        row(arrival=1800.0, e2e=300.0, session_id="m3"),  # 2100: ends full occupancy
    ]
    base = dict(
        sla={"itl_ms": 200.0}, open_loop=False, num_workers=2, e0=None, occupancy_cap=2
    )
    rule = {"basis": "completion", "end": "full_occupancy"}
    m = goodput.compute_metrics(
        rows,
        summary_for(rows),
        measure={**rule, "warmup_trace_ms": 60_000.0},
        warmup_ids=frozenset({"w1", "w2"}),
        **base,
    )
    assert (m["window_start_ms"], m["window_end_ms"]) == (500.0, 2100.0)
    assert m["warmup_excluded_rows"] == 2 and m["warmup_ms"] == 500.0
    assert (m["window_requests"], m["window_good"]) == (2, 1)  # m1, m2
    assert m["goodput_rps_window"] == pytest.approx(1 / 1.6)
    # The same rows without the identity rule score the warm-up completions at 500 and 900 too.
    plain = goodput.compute_metrics(rows, summary_for(rows), measure=rule, **base)
    assert plain["window_requests"] == 4 and plain["warmup_excluded_rows"] == 0
    # The rule needs the IDs, per-request session IDs, the completion basis and no warmup_ms.
    with pytest.raises(ValueError, match="warm-up session IDs"):
        goodput.compute_metrics(
            rows,
            summary_for(rows),
            measure={**rule, "warmup_trace_ms": 1.0},
            **base,
        )
    with pytest.raises(ValueError, match="session IDs; 1 rows"):
        goodput.compute_metrics(
            rows + [row(arrival=1.0)],
            summary_for(rows + [row(arrival=1.0)]),
            measure={**rule, "warmup_trace_ms": 1.0},
            warmup_ids=frozenset(),
            **base,
        )
    for measure in (
        {"basis": "arrival", "warmup_trace_ms": 1.0},
        {**rule, "warmup_trace_ms": 1.0, "warmup_ms": 5.0},
    ):
        with pytest.raises(ValueError, match="basis completion"):
            goodput.compute_metrics(
                rows,
                summary_for(rows),
                measure=measure,
                warmup_ids=frozenset(),
                **base,
            )


def test_simultaneous_arrivals_fall_back_to_makespan():
    rows = [row(arrival=0.0, e2e=1000.0 * (i + 1)) for i in range(4)]
    m = goodput.compute_metrics(
        rows,
        summary_for(rows),
        sla={"itl_ms": 1e9},
        measure={},
        open_loop=True,
        num_workers=2,
        e0=None,
    )
    assert m["window_fallback"] and m["window_end_ms"] == 4000.0
    assert m["window_requests"] == 4
    assert m["goodput_rps_window"] == pytest.approx(4 / 4.0)


def test_rescore_scales_relax_and_tighten_both_thresholds():
    e0 = lambda isl, osl: 100.0  # noqa: E731
    rows = [row(arrival=float(i), e2e=300.0 + 100.0 * i) for i in range(6)]
    sla = {"itl_ms": None, "e2e_slowdown": 4.0, "ttft_ms": None}
    m = goodput.compute_metrics(
        rows,
        summary_for(rows),
        sla=sla,
        measure={},
        open_loop=True,
        num_workers=2,
        e0=e0,
    )
    fracs = [m["rescore"][repr(s)]["good_frac_window"] for s in goodput.SCALES]
    assert fracs == sorted(fracs)  # monotone in the scale
    assert (
        m["rescore"]["1.0"]["good_frac_window"]
        == m["good_frac_window"]
        == pytest.approx(2 / 6)
    )
    assert m["rescore"]["3.0"]["good_frac_window"] == 1.0


def test_guards_flag_concentration_and_session_splits():
    rows = [row(arrival=float(i), worker=0, session_id="s1") for i in range(3)]
    rows += [
        row(arrival=10.0, worker=1, session_id="s1"),
        row(arrival=11.0, worker=0, session_id="s2"),
    ]
    m = goodput.compute_metrics(
        rows,
        summary_for(rows),
        sla={"itl_ms": 1e9},
        measure={},
        open_loop=True,
        num_workers=4,
        e0=None,
    )
    guards = m["guards"]
    assert guards["worker_share_max"] == pytest.approx(4 / 5)
    assert guards["workers_used"] == 2
    assert guards["sessions_multi_turn"] == 1 and guards["session_split_frac"] == 1.0
    assert guards["worker_share_cap"] == pytest.approx(0.5)


def test_compact_row_drops_uuid_and_sums_queue_wait():
    record = {k: None for k in goodput.COMPACT_FIELDS}
    record.update(uuid="abc", terminal_status="completed")
    record["routing_history"] = [
        {"logical_worker_id": 2, "queue_wait_ms": 1.5},
        {"logical_worker_id": 3, "queue_wait_ms": None},
    ]
    record.update(play_id="p", agentic={"play_id": "p", "lane_id": "lane:3"})
    compact = goodput.compact_row(record)
    assert "uuid" not in compact and "agentic" not in compact
    assert compact["routing_workers"] == [2, 3] and compact["queue_wait_ms"] == 1.5
    assert (compact["play_id"], compact["lane_id"]) == ("p", "lane:3")
