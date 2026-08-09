# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Sweeper's experimental module entry point."""

import subprocess
import sys

import pytest

import aisimulate.sweeper.__main__ as cli
from aisimulate import sweeper


def test_package_is_marked_experimental():
    assert "experimental" in (sweeper.__doc__ or "").lower()


@pytest.mark.timeout(30)
def test_cli_help_is_marked_experimental():
    result = subprocess.run(
        [sys.executable, "-m", "aisimulate.sweeper", "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "[EXPERIMENTAL]" in result.stdout


@pytest.mark.timeout(30)
def test_cli_rejects_malformed_yaml(tmp_path):
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("search_space: [")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisimulate.sweeper",
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert "malformed YAML" in result.stderr


@pytest.mark.timeout(30)
def test_cli_rejects_invalid_config(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("search_space:\n  gpu_budget: 0\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisimulate.sweeper",
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert "invalid config" in result.stderr


def test_cli_requires_an_injected_replay_runtime(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "valid.yaml"
    config_path.write_text(
        "search_space:\n"
        "  model_name: example/model\n"
        "  hardware_sku: example_sku\n"
        "workload:\n"
        "  isl: 128\n"
        "  osl: 16\n"
        "  request_rate: 1\n"
        "  num_request_ratio: 3\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["aisimulate.sweeper", "--config", str(config_path)],
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main()

    error = capsys.readouterr().err
    assert "no default replay runtime" in error
    assert "Sweeper(runner_factory=...).run(config)" in error


def test_runner_wrapper_preserves_no_candidate_exit(capsys):
    config = sweeper.SmartSearchConfig(
        search_space={"model_name": "model", "hardware_sku": "h200_sxm"},
        workload={
            "isl": 128,
            "osl": 16,
            "request_rate": 1,
            "num_request_ratio": 3,
        },
    )

    with pytest.raises(SystemExit, match="1"):
        cli.print_candidates_or_exit(config, [])

    assert "no feasible candidate found" in capsys.readouterr().err


def test_runner_wrapper_preserves_pareto_objectives_and_concurrency(capsys):
    config = sweeper.SmartSearchConfig(
        search_space={"model_name": "model", "hardware_sku": "h200_sxm"},
        workload={
            "isl": 128,
            "osl": 16,
            "kv_load_ratio": [0.0, 1.0],
            "num_request_ratio": 3,
        },
        goal={"target": "pareto"},
    )
    candidate = sweeper.Candidate(
        config={"concurrency": 8},
        used_gpus=4,
        score=12.0,
        metrics={"output_throughput_tok_s": 48.0},
        objectives={"throughput_per_gpu": 12.0, "throughput_per_user": 6.0},
    )

    cli.print_candidates_or_exit(config, [candidate])

    output = capsys.readouterr().out
    assert "pareto front (1 non-dominated)" in output
    assert "throughput_per_gpu=12" in output
    assert "throughput_per_user=6" in output
    assert "concurrency=8" in output
