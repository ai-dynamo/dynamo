# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from learned_routing.cells import Cell
from learned_routing.paths import Layout
from learned_routing.window_guards import GuardError, cap_for, nmi, record_guards


def row(i, worker, isl, e2e, arrival):
    return {
        "session_id": f"request_{i}",
        "arrival_time_ms": arrival,
        "first_admit_ms": arrival,
        "ttft_ms": 1.0,
        "e2e_latency_ms": e2e,
        "terminal_time_ms": arrival + e2e,
        "terminal_status": "completed",
        "input_length": isl,
        "output_length": 1,
        "decode_worker_idx": worker,
    }


@pytest.fixture
def cell(tmp_path):
    return Cell(
        raw={
            "cell_id": "c",
            "num_workers": 4,
            "load": {"mode": "open_speedup", "value": 1.0},
            "sla": {"itl_ms": None, "e2e_slowdown": 2.0},
            "measure": {"basis": "arrival"},
        },
        layout=Layout(tmp_path),
    )


def test_a_dump_worker_of_long_requests_is_flagged(cell):
    # E0 = 1 ms for every request, S = 2. Worker 3 holds only 70K-token requests, all late (bad);
    # workers 0..2 share the short ones, all good. Worker 0 takes half of all requests.
    rows = []
    for i in range(20):
        rows.append(row(i, 0, 500, 1.0, i))
    for i in range(20, 32):
        rows.append(row(i, 1 + i % 2, 4000, 1.0, i))
    for i in range(32, 40):
        rows.append(row(i, 3, 70_000, 5.0, i))
    record = {
        "cell_id": "c",
        "repeat": 0,
        "window_basis": "arrival",
        "window_start_ms": 0.0,
        "window_end_ms": 100.0,
        "window_good": 32,
    }
    g = record_guards(rows, record, cell, lambda isl, osl: 1.0)
    assert g["n_window"] == 40 and g["share_max_window"] == 0.5
    assert g["cap"] == cap_for(4) == 0.5 and not g["violates_cap"]
    assert g["worker_good_frac_min"] == 0.0 and g["dump_workers"] == 1
    assert g["good_isl_ge_64k"] == 0.0 and g["n_isl_ge_64k"] == 8
    assert g["good_isl_ge_32k"] == 0.0 and g["good_giants"] == 0.0
    assert g["good_top_isl_decile"] == 0.0
    assert g["hot_worker"] == 0 and g["hot_good_frac"] == 1.0
    assert g["others_good_frac"] == pytest.approx(12 / 20)
    assert g["nmi_worker_islq"] > 0.5
    # the guard refuses a window it cannot reproduce
    with pytest.raises(GuardError, match="window_good"):
        record_guards(rows, {**record, "window_good": 31}, cell, lambda isl, osl: 1.0)


def test_nmi_is_one_for_full_segregation_and_zero_for_none():
    assert nmi([(0, "a"), (1, "b")] * 5) == pytest.approx(1.0)
    assert nmi([(w, x) for w in (0, 1) for x in ("a", "b")]) == pytest.approx(0.0)
