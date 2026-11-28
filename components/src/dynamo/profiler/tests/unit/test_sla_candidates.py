# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for collecting and writing SLA-compliant candidate configs."""

import os

import pandas as pd
import pytest

from dynamo.profiler.utils.sla_candidates import (
    SLA_CANDIDATES_FILE_NAME,
    collect_sla_candidates,
    write_sla_candidates,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _agg_row(tp, ttft, tpot, tps_gpu, tps_user, request_latency=0.0):
    return {
        "parallel": f"tp{tp}",
        "tp": tp,
        "ttft": ttft,
        "tpot": tpot,
        "request_latency": request_latency,
        "tokens/s/gpu": tps_gpu,
        "tokens/s/user": tps_user,
        "num_total_gpus": tp,
    }


def _disagg_row(p_tp, d_tp, ttft, tpot, tps_gpu, tps_user):
    return {
        "(p)parallel": f"tp{p_tp}",
        "(d)parallel": f"tp{d_tp}",
        "(p)tp": p_tp,
        "(d)tp": d_tp,
        "ttft": ttft,
        "tpot": tpot,
        "request_latency": 0.0,
        "tokens/s/gpu": tps_gpu,
        "tokens/s/user": tps_user,
        "num_total_gpus": p_tp + d_tp,
    }


def test_keeps_only_rows_within_ttft_and_tpot():
    frontier = pd.DataFrame(
        [
            _agg_row(1, ttft=500.0, tpot=40.0, tps_gpu=900.0, tps_user=25.0),
            _agg_row(2, ttft=500.0, tpot=60.0, tps_gpu=1200.0, tps_user=16.0),
            _agg_row(4, ttft=2500.0, tpot=20.0, tps_gpu=700.0, tps_user=50.0),
        ]
    )

    candidates = collect_sla_candidates(
        {"agg": frontier}, {}, target_ttft=2000.0, target_tpot=50.0
    )

    assert candidates["tp"].tolist() == [1]
    assert candidates["mode"].tolist() == ["agg"]
    assert candidates["selected"].tolist() == [False]


def test_request_latency_target_replaces_tpot():
    frontier = pd.DataFrame(
        [
            _agg_row(1, 300.0, 80.0, 900.0, 12.0, request_latency=9000.0),
            _agg_row(2, 300.0, 10.0, 800.0, 100.0, request_latency=15000.0),
        ]
    )

    candidates = collect_sla_candidates(
        {"agg": frontier},
        {},
        target_ttft=1000.0,
        target_tpot=20.0,
        target_request_latency=10000.0,
    )

    # The TPOT target alone would have kept tp2 and dropped tp1.
    assert candidates["tp"].tolist() == [1]


def test_marks_rank1_pick_and_deduplicates_it_against_frontier():
    picked = _disagg_row(2, 4, ttft=400.0, tpot=30.0, tps_gpu=1500.0, tps_user=33.0)
    runner_up = _disagg_row(1, 4, ttft=600.0, tpot=35.0, tps_gpu=1400.0, tps_user=28.0)
    ranked = pd.DataFrame([picked, runner_up])
    frontier = pd.DataFrame(
        [
            picked,
            _disagg_row(1, 2, ttft=300.0, tpot=15.0, tps_gpu=800.0, tps_user=66.0),
        ]
    )

    candidates = collect_sla_candidates(
        {"disagg": frontier},
        {"disagg": ranked},
        target_ttft=1000.0,
        target_tpot=40.0,
        chosen_exp="disagg",
    )

    assert len(candidates) == 3
    assert candidates["selected"].sum() == 1
    selected = candidates[candidates["selected"]].iloc[0]
    assert (selected["(p)tp"], selected["(d)tp"]) == (2, 4)
    # Sorted by tokens/s/gpu, highest first.
    assert candidates["tokens/s/gpu"].tolist() == [1500.0, 1400.0, 800.0]


def test_merges_modes_and_puts_summary_columns_first():
    agg = pd.DataFrame([_agg_row(2, 500.0, 30.0, 1000.0, 33.0)])
    disagg = pd.DataFrame([_disagg_row(1, 2, 400.0, 25.0, 1300.0, 40.0)])

    candidates = collect_sla_candidates(
        {"agg": agg, "disagg": disagg},
        {"agg": agg, "disagg": disagg},
        target_ttft=1000.0,
        target_tpot=50.0,
        chosen_exp="disagg",
    )

    assert candidates["mode"].tolist() == ["disagg", "agg"]
    assert candidates["selected"].tolist() == [True, False]
    assert list(candidates.columns[:4]) == ["mode", "selected", "ttft", "tpot"]
    # Columns only one mode has are kept and left empty for the other.
    assert pd.isna(candidates.loc[1, "(p)tp"])
    assert pd.isna(candidates.loc[0, "tp"])


def test_keeps_selected_config_even_when_it_misses_the_sla():
    # When nothing meets the SLA, AIC still ranks a best effort config and
    # the profiler deploys it, so the CSV must show it.
    ranked = pd.DataFrame([_agg_row(8, 3000.0, 90.0, 500.0, 11.0)])

    candidates = collect_sla_candidates(
        {"agg": ranked},
        {"agg": ranked},
        target_ttft=1000.0,
        target_tpot=50.0,
        chosen_exp="agg",
    )

    assert candidates["tp"].tolist() == [8]
    assert candidates["selected"].tolist() == [True]


def test_missing_latency_column_fails_closed():
    frontier = pd.DataFrame([{"parallel": "tp1", "ttft": 100.0}])

    candidates = collect_sla_candidates(
        {"agg": frontier}, None, target_ttft=1000.0, target_tpot=50.0
    )

    assert candidates.empty


@pytest.mark.parametrize("fronts", [None, {}, {"agg": None}, {"agg": pd.DataFrame()}])
def test_empty_inputs_give_empty_table(fronts):
    candidates = collect_sla_candidates(fronts, None, 1000.0, 50.0)

    assert candidates.empty
    assert list(candidates.columns) == ["mode", "selected"]


def test_write_creates_csv_even_when_empty(tmp_path):
    empty = collect_sla_candidates(None, None, 1000.0, 50.0)

    name = write_sla_candidates(empty, str(tmp_path))

    assert name == SLA_CANDIDATES_FILE_NAME
    written = pd.read_csv(os.path.join(tmp_path, name))
    assert written.empty
    assert list(written.columns) == ["mode", "selected"]


def test_write_round_trips_rows(tmp_path):
    frontier = pd.DataFrame([_agg_row(1, 500.0, 40.0, 900.0, 25.0)])
    candidates = collect_sla_candidates({"agg": frontier}, None, 1000.0, 50.0)

    write_sla_candidates(candidates, str(tmp_path))

    written = pd.read_csv(os.path.join(tmp_path, SLA_CANDIDATES_FILE_NAME))
    assert written["tp"].tolist() == [1]
    assert written["ttft"].tolist() == [500.0]
