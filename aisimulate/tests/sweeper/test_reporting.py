# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable Sweeper terminal and file result contracts."""

import csv
import json

from aisimulate.sweeper.config import Candidate, SmartSearchConfig
from aisimulate.sweeper.reporting import (
    format_sweep_results,
    serialize_sweep_results,
    write_sweep_results,
)


def _config(*, pareto: bool = False) -> SmartSearchConfig:
    return SmartSearchConfig(
        search_space={"model_name": "model", "hardware_sku": "h200_sxm"},
        workload={
            "isl": 128,
            "osl": 16,
            "concurrency": 2,
            "num_request_ratio": 3,
        },
        goal={"target": "pareto" if pareto else "throughput"},
    )


def _candidate(score: float, *, objectives=None) -> Candidate:
    return Candidate(
        config={
            "backend": "vllm",
            "deployment_mode": "agg",
            "engine": {"max_num_seqs": int(score)},
        },
        used_gpus=4,
        score=score,
        metrics={"output_throughput_tok_s": score},
        objectives=objectives,
    )


def test_scalar_result_envelope_and_top_n_csv_are_complete(tmp_path):
    config = _config()
    candidates = [_candidate(30.0), _candidate(20.0), _candidate(10.0)]

    paths = write_sweep_results(tmp_path, config, candidates, top_n=2)

    assert paths == (
        tmp_path / "sweep_results.json",
        tmp_path / "best_config_topn.csv",
    )
    envelope = json.loads(paths[0].read_text())
    assert envelope["schema_version"] == 1
    assert envelope["result_type"] == "ranked_candidates"
    assert [item["score"] for item in envelope["candidates"]] == [30.0, 20.0, 10.0]
    with paths[1].open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert [row["rank"] for row in rows] == ["1", "2"]
    assert rows[0]["config.engine.max_num_seqs"] == "30"
    assert rows[0]["metrics.output_throughput_tok_s"] == "30.0"


def test_pareto_output_preserves_the_complete_front(tmp_path):
    config = _config(pareto=True)
    candidates = [
        _candidate(
            30.0,
            objectives={"throughput_per_gpu": 30.0, "throughput_per_user": 10.0},
        ),
        _candidate(
            20.0,
            objectives={"throughput_per_gpu": 20.0, "throughput_per_user": 20.0},
        ),
    ]

    paths = write_sweep_results(tmp_path, config, candidates, top_n=1)

    assert paths[1].name == "pareto.csv"
    with paths[1].open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2
    assert rows[1]["objectives.throughput_per_user"] == "20.0"
    assert serialize_sweep_results(config, candidates)["result_type"] == "pareto_front"


def test_terminal_result_includes_configuration_and_metrics():
    output = format_sweep_results(_config(), [_candidate(30.0)], top_n=5)

    assert "top 1 of 1 candidates" in output
    assert 'config={"backend":"vllm"' in output
    assert 'metrics={"output_throughput_tok_s":30.0}' in output
