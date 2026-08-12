# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public predict/recommend CLI contracts."""

from __future__ import annotations

import json

import pytest

import aisimulate.cli as cli
from aisimulate.sweeper.config import Candidate

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]


class _FakeSweeper:
    config = None
    show_progress = None

    def __init__(self, *, runner_factory, show_progress):
        del runner_factory
        type(self).show_progress = show_progress

    def run(self, config):
        type(self).config = config
        return [
            Candidate(
                config={
                    "backend": config.search_space.backend[0],
                    "deployment_mode": config.search_space.deployment_mode[0],
                    "tp": 8,
                    "replicas": 2,
                },
                used_gpus=16,
                score=123.0,
                metrics={
                    "output_throughput_tok_s": 123.0,
                    "mean_ttft_ms": 45.0,
                },
            )
        ]


def test_predict_scalar_flags_lower_to_one_candidate(monkeypatch, capsys):
    monkeypatch.setattr(cli, "Sweeper", _FakeSweeper)
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: object())

    result = cli.main(
        [
            "predict",
            "--stack",
            "engine",
            "--backend",
            "vllm",
            "--model",
            "meta-llama/Meta-Llama-3.1-8B",
            "--system",
            "h200_sxm",
            "--tp-size",
            "8",
            "--replicas",
            "2",
            "--isl",
            "1024",
            "--osl",
            "128",
            "--output",
            "json",
        ]
    )

    assert result == 0
    config = _FakeSweeper.config
    assert config.search_space.parallel_configs == [{"tp": 8, "replicas": 2}]
    assert config.search_space.gpu_budget == 16
    assert config.workload.concurrency == 1
    assert config.workload.num_request_ratio == 10
    assert config.sweep.max_rounds == 1
    assert config.sweep.parallel_evals == 1
    assert _FakeSweeper.show_progress is False
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "predict"
    assert payload["stack"] == "engine"
    assert payload["candidate"]["metrics"]["mean_ttft_ms"] == 45.0


def test_predict_config_must_resolve_exactly_one_candidate(tmp_path, capsys):
    config_path = tmp_path / "predict.yaml"
    config_path.write_text(
        "search_space:\n"
        "  deployment_mode: [agg]\n"
        "  backend: [vllm, sglang]\n"
        "  parallel_configs: [{tp: 8, replicas: 1}]\n"
        "  model_name: example/model\n"
        "  hardware_sku: h200_sxm\n"
        "  gpu_budget: 8\n"
        "  agg_max_num_batched_tokens: [8192]\n"
        "  agg_max_num_seqs: [256]\n"
        "workload:\n"
        "  isl: 1024\n"
        "  osl: 128\n"
        "  concurrency: 1\n"
        "  num_request_ratio: 10\n"
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "predict",
                "--stack",
                "engine",
                "--config",
                str(config_path),
            ]
        )

    assert "exactly one resolved value at search_space.backend" in (
        capsys.readouterr().err
    )


def test_recommend_flag_defaults_select_all_backends_and_dynamo_adapters(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli, "Sweeper", _FakeSweeper)
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: object())

    result = cli.main(
        [
            "recommend",
            "--stack",
            "dynamo",
            "--model",
            "meta-llama/Meta-Llama-3.1-8B",
            "--system",
            "h200_sxm",
            "--total-gpus",
            "16",
            "--isl",
            "1024",
            "--osl",
            "128",
            "--sla-ttft-ms",
            "100",
            "--sla-itl-ms",
            "20",
            "--strict-sla",
            "--output-dir",
            str(tmp_path),
            "--output",
            "json",
        ]
    )

    assert result == 0
    config = _FakeSweeper.config
    assert config.search_space.backend == ["vllm", "sglang", "trtllm"]
    assert config.search_space.deployment_mode == ["agg", "disagg"]
    assert set(config.adapters) == {"dynamo.planner", "dynamo.router"}
    assert config.goal.sla is not None and config.goal.sla.strict is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "recommend"
    assert payload["result_type"] == "ranked_candidates"
    assert payload["schema_version"] == 1
    assert (tmp_path / "sweep_results.json").is_file()
    assert (tmp_path / "best_config_topn.csv").is_file()


def test_help_exposes_only_predict_and_recommend(capsys):
    with pytest.raises(SystemExit, match="0"):
        cli.main(["--help"])

    help_text = capsys.readouterr().out
    assert "{predict,recommend}" in help_text
    assert "Evaluate exactly one" in help_text
    assert "Search and rank" in help_text
