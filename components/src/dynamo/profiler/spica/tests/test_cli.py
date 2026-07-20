# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Spica's experimental module entry point."""

import subprocess
import sys

import pytest

import dynamo.profiler.spica as spica
import dynamo.profiler.spica.__main__ as cli


def test_package_is_marked_experimental():
    assert "experimental" in (spica.__doc__ or "").lower()


def test_cli_help_is_marked_experimental():
    result = subprocess.run(
        [sys.executable, "-m", "dynamo.profiler.spica", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[EXPERIMENTAL]" in result.stdout


def test_cli_rejects_malformed_yaml(tmp_path):
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("search_space: [")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dynamo.profiler.spica",
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "malformed YAML" in result.stderr


def test_cli_rejects_invalid_config(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("search_space:\n  gpu_budget: 0\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dynamo.profiler.spica",
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "invalid config" in result.stderr


def test_cli_reports_no_feasible_candidates(monkeypatch, tmp_path, capsys):
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
    monkeypatch.setattr(cli, "run_smart_search", lambda config: [])
    monkeypatch.setattr(
        sys,
        "argv",
        ["dynamo.profiler.spica", "--config", str(config_path)],
    )

    with pytest.raises(SystemExit, match="1"):
        cli.main()

    assert "no feasible candidate found" in capsys.readouterr().err
