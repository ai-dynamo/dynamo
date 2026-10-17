# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from autoscaling_arena.rank_tables import (
    all_rank_tables,
    average_ranks,
    format_markdown,
    methods_for_metric,
    pairwise_win_rate,
    select_workloads,
)

METHODS = ["planner", "static-4", "cloudai-rl"]
WORKLOADS = ["flat", "w2", "w3", "w4", "w5", "w6", "w7", "w8"]  # last 6 = golden set
SLAS = ["interactive-ttft300ms-itl50ms", "interactive-ttft500ms-itl100ms"]


def _report():
    matrix, results = [], []
    for w_i, w in enumerate(WORKLOADS):
        for sla in SLAS:
            for m_i, m in enumerate(METHODS):
                for rep in range(2):
                    row = {
                        "autoscaler": m,
                        "workload": w,
                        "sla": sla,
                        "repetition": rep,
                    }
                    matrix.append(dict(row))
                    # planner always best on goodput_per_gpu; everyone ties on good_rate
                    # for flat; static-4 wins good_rate elsewhere.
                    gp = 10.0 - m_i + rep * 0.1
                    gr = 0.5 if w == "flat" else (0.9 if m == "static-4" else 0.6)
                    if sla == SLAS[1]:
                        gp += 100.0  # the other SLA must be ignored when not selected
                    results.append(
                        {
                            **row,
                            "status": "ok",
                            "metrics": {"goodput_per_gpu": gp, "good_rate": gr},
                        }
                    )
    return {"matrix": matrix, "results": results}


def test_average_ranks_share_positions_on_ties():
    assert average_ranks({"a": 3.0, "b": 1.0, "c": 3.0, "d": 0.5}) == {
        "a": 1.5,
        "c": 1.5,
        "b": 3.0,
        "d": 4.0,
    }


def test_pairwise_win_rate_counts_ties_as_half():
    table = {"w": {"a": 2.0, "b": 2.0, "c": 1.0}}
    assert pairwise_win_rate(table, ["a", "b", "c"]) == {"a": 0.75, "b": 0.75, "c": 0.0}


def test_methods_for_metric_drops_rl_entries_from_other_metrics():
    assert methods_for_metric(METHODS, "goodput_per_gpu") == [
        "planner",
        "static-4",
        "cloudai-rl",
    ]
    assert methods_for_metric(METHODS, "good_rate") == ["planner", "static-4"]


def test_golden_set_is_the_last_six_workloads_in_matrix_order():
    report = _report()
    assert select_workloads(report, "all") == WORKLOADS
    assert select_workloads(report, "golden") == WORKLOADS[-6:]
    with pytest.raises(ValueError):
        select_workloads(report, "recorded")


def test_all_rank_tables_default_first_sla_and_goodput_only():
    tables = all_rank_tables(_report())
    # goodput_per_gpu is the only evaluation metric; good_rate is never ranked.
    assert [(t["metric"], t["workload_set"]) for t in tables] == [
        ("goodput_per_gpu", "all"),
        ("goodput_per_gpu", "golden"),
    ]
    assert all(t["sla"] == SLAS[0] for t in tables)

    gp_all = tables[0]
    assert gp_all["rl_entry"] == "cloudai-rl"
    assert gp_all["workloads"] == WORKLOADS and gp_all["dropped_workloads"] == []
    assert [r["method"] for r in gp_all["rows"]] == [
        "planner",
        "static-4",
        "cloudai-rl",
    ]
    assert (
        gp_all["rows"][0]["mean_rank"] == 1.0
        and gp_all["rows"][0]["pairwise_win_rate"] == 1.0
    )

    text = format_markdown(tables)
    assert text.count("| Method | Mean rank | Pairwise win rate |") == 2
    assert "good_rate" not in text
    assert "RL entry `cloudai-rl`" in text and "Dropped" not in text


def test_workloads_missing_a_method_are_dropped_and_reported():
    report = _report()
    # Every static-4 row for w3 failed: w3 cannot be ranked fairly and is dropped.
    for row in report["results"]:
        if row["workload"] == "w3" and row["autoscaler"] == "static-4":
            row["status"] = "failed"
            row["metrics"] = None
    tables = all_rank_tables(report)
    gp_all, gp_golden = tables
    assert gp_all["dropped_workloads"] == ["w3"] and "w3" not in gp_all["workloads"]
    assert gp_golden["dropped_workloads"] == ["w3"] and len(gp_golden["workloads"]) == 5
    assert (
        gp_all["rows"][0]["method"] == "planner"
        and gp_all["rows"][0]["mean_rank"] == 1.0
    )
    text = format_markdown(tables)
    assert (
        "all workloads (7)" in text
        and "Dropped (no ok repetition for every method): `w3`" in text
    )


def test_rl_label_is_omitted_when_no_rl_entry_ran():
    report = _report()
    report["matrix"] = [
        r for r in report["matrix"] if not r["autoscaler"].startswith("cloudai-rl")
    ]
    report["results"] = [
        r for r in report["results"] if not r["autoscaler"].startswith("cloudai-rl")
    ]
    tables = all_rank_tables(report)
    assert tables[0]["rl_entry"] is None
    assert "RL entry" not in format_markdown(tables)
