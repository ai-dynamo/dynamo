# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Final evaluation tables: per-method aggregates, per-workload table, RL cross-mode summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from autoscaling_arena.evaluation_table import (
    main,
    method_rows,
    rank_matrix,
    render,
    write_workbook,
)

WORKLOADS = [f"w{i}" for i in range(8)]  # the last six are "golden"
METHODS = {"planner": 0.5, "cloudai-rl": 1.0, "cloudai-rl-v3-lstm": 0.8}


def _report(path: Path, *, scale: float = 1.0) -> Path:
    results, matrix = [], []
    for workload in WORKLOADS:
        for method, value in METHODS.items():
            for repetition in range(2):
                row = {
                    "run_id": f"{method}-{workload}-{repetition}",
                    "backend": "sim",
                    "autoscaler": method,
                    "workload": workload,
                    "sla": "relaxed",
                    "repetition": repetition,
                    "seed": repetition,
                    "status": "ok",
                    "metrics": {
                        "goodput_per_gpu": value * scale + 0.1 * repetition,
                        "good_rate": 0.5,
                        "gpu_hours": 0.25,
                        "mean_ttft_ms": 100.0 + repetition,
                        "mean_itl_ms": 20.0,
                        "scale_events": 3,
                    },
                    "raw_metrics": {"average_gpus": 2.0},
                }
                results.append(row)
                matrix.append(
                    {
                        k: row[k]
                        for k in (
                            "run_id",
                            "backend",
                            "autoscaler",
                            "workload",
                            "sla",
                            "repetition",
                            "seed",
                        )
                    }
                )
    summary = {
        "rank_by": "goodput_per_gpu",
        "metrics": [
            "good_rate",
            "goodput_per_gpu",
            "gpu_hours",
            "mean_ttft_ms",
            "mean_itl_ms",
            "scale_events",
        ],
    }
    path.write_text(
        json.dumps({"summary": summary, "matrix": matrix, "results": results})
    )
    return path


def test_method_rows_average_repetitions_then_workloads_per_set(tmp_path: Path):
    report = json.loads(_report(tmp_path / "r.json").read_text())
    rows = method_rows(report, sla="relaxed")
    assert set(rows) == set(METHODS)
    # repetitions 0 and 1 average to value + 0.05, in good requests per GPU-second.
    rl = rows["cloudai-rl"]
    assert rl["all"]["goodput_per_gpu"] == pytest.approx(1.05)
    assert rl["golden"]["goodput_per_gpu"] == pytest.approx(1.05)
    assert rl["all"]["mean_rank"] == 1.0 and rows["planner"]["all"]["mean_rank"] == 3.0
    assert rl["all"]["win_rate"] == 1.0 and rl["golden"]["win_rate"] == 1.0
    planner = rows["planner"]
    assert "gpu_hours" not in planner["all"]
    assert planner["all"]["avg_gpus"] == pytest.approx(2.0) and planner["golden"][
        "good_rate"
    ] == pytest.approx(0.5)
    assert rl["per_workload"]["w0"] == pytest.approx(1.05)


def test_render_and_cli_report_all_and_golden_in_every_column(tmp_path: Path, capsys):
    agg = _report(tmp_path / "agg.json")
    disagg = _report(tmp_path / "disagg.json", scale=2.0)
    text = render({"agg": agg, "disagg": disagg})
    assert "## agg mode" in text and "## disagg mode" in text
    assert "## CloudAI RL planner by mode" in text
    assert "| Mode | cloudai-rl | cloudai-rl-v3-lstm | Best other method |" in text
    assert (
        "| **cloudai-rl** | 1.050 (1.050) | 1.00 (1.00) | 1.000 (1.000) | 50.0% (50.0%) | 2.00 (2.00) |"
        in text
    )
    assert (
        "## How to read these tables" in text and "GPU-hours |" not in text
    )  # no GPU-hours column
    assert "| agg | 1.050 (1.050) | 0.850 (0.850) | planner: 0.550 (0.550) |" in text
    assert "| disagg | 2.050 (2.050) | 1.650 (1.650) | planner: 1.050 (1.050) |" in text
    assert "| w0 |" not in text  # wide tables live in the workbook, not the markdown
    out = tmp_path / "table.md"
    assert (
        main(
            [
                "--results",
                f"agg={agg}",
                "--results",
                f"disagg={disagg}",
                "--out",
                str(out),
                "--no-xlsx",
            ]
        )
        == 0
    )
    assert out.read_text().strip() == text.strip()
    assert not (tmp_path / "table.xlsx").exists()
    assert "Autoscaler evaluation" in capsys.readouterr().out


def test_rank_matrix_ranks_each_workload_with_shared_ties(tmp_path: Path):
    report = json.loads(_report(tmp_path / "r.json").read_text())
    report["results"][0]["metrics"][
        "goodput_per_gpu"
    ] = 1.1  # planner w0: mean (1.1 + 0.6) / 2 = 0.85 == v3-lstm
    matrix = rank_matrix(report, sla="relaxed")
    assert matrix["cloudai-rl"] == {w: 1.0 for w in WORKLOADS}
    assert matrix["planner"]["w0"] == 2.5 and matrix["cloudai-rl-v3-lstm"]["w0"] == 2.5
    assert matrix["planner"]["w1"] == 3.0 and matrix["cloudai-rl-v3-lstm"]["w1"] == 2.0
    (tmp_path / "r.json").write_text(json.dumps(report))
    text = render({"agg": tmp_path / "r.json"}, workbook=tmp_path / "r.xlsx")
    assert (
        "r.xlsx" in text and "rank matrix" not in text.lower().split("## agg mode")[1]
    )


def test_workbook_holds_summary_goodput_ranks_and_data(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    agg = _report(tmp_path / "agg.json")
    report = json.loads(agg.read_text())
    report["results"][0]["metrics"][
        "goodput_per_gpu"
    ] = 1.1  # planner w0 ties v3-lstm at 0.85
    agg.write_text(json.dumps(report))
    path = write_workbook({"agg": agg}, tmp_path / "eval.xlsx")
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == [
        "README",
        "agg summary",
        "agg goodput_per_gpu",
        "agg ranks",
        "agg data",
    ]
    summary = wb["agg summary"]
    assert [c.value for c in summary[1]][:3] == [
        "Method",
        "Goodput/GPU-s (all)",
        "Goodput/GPU-s (golden)",
    ]
    assert summary["A2"].value == "cloudai-rl" and summary["B2"].value == pytest.approx(
        1.05
    )
    assert (
        summary["D2"].value == 1.0 and summary["F2"].value == 1.0
    )  # mean rank, win rate (all)
    goodput = wb["agg goodput_per_gpu"]
    assert [c.value for c in goodput[1]] == [
        "Method (goodput_per_gpu)",
        *WORKLOADS,
        "Mean (all)",
        "Mean (golden)",
    ]
    assert (
        goodput["B2"].value == pytest.approx(1.05) and goodput["B2"].font.bold
    )  # best per workload in bold
    ranks = wb["agg ranks"]
    assert [c.value for c in ranks[1]][:2] == ["Method (rank by goodput_per_gpu)", "w0"]
    by_method = {
        ranks.cell(row=r, column=1).value: r for r in range(2, ranks.max_row + 1)
    }
    assert ranks.cell(row=by_method["planner"], column=2).value == 2.5
    assert ranks.cell(row=by_method["cloudai-rl-v3-lstm"], column=2).value == 2.5
    assert (
        ranks.cell(row=by_method["cloudai-rl"], column=len(WORKLOADS) + 2).value == 1.0
    )
    assert ranks.cell(
        row=by_method["planner"], column=len(WORKLOADS) + 2
    ).value == pytest.approx((2.5 + 7 * 3) / 8)
    data = wb["agg data"]
    assert data.max_row == 1 + len(WORKLOADS) * len(METHODS)
    assert [c.value for c in data[1]] == [
        "Method",
        "Workload",
        "Set",
        "goodput_per_gpu (good req / GPU-s)",
        "Good rate",
        "Avg GPUs",
        "Mean TTFT (ms)",
        "Mean ITL / TPOT (ms)",
        "Scale events",
        "Repetitions",
    ]
    assert data["C2"].value == "registry" and data["J2"].value == 2
    assert (
        data["G2"].value == pytest.approx(100.5)
        and data["H2"].value == 20.0
        and data["I2"].value == 3
    )
    assert all(
        cell.font.name == "Arial"
        for row in summary.iter_rows()
        for cell in row
        if cell.value is not None
    )
    # CLI: --out implies a sibling workbook.
    out = tmp_path / "report.md"
    assert main(["--results", f"agg={agg}", "--out", str(out)]) == 0
    assert (tmp_path / "report.xlsx").exists() and "report.xlsx" in out.read_text()
