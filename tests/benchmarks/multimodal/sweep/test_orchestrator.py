# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.multimodal.sweep.config import BenchmarkConfig, SweepConfig
from benchmarks.multimodal.sweep.orchestrator import run_sweep

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _make_config(
    tmp_path: Path,
    num_configs: int = 2,
    num_input_files: int = 2,
    request_rates: List[int] | None = None,
    restart_server_every_benchmark: bool = True,
) -> SweepConfig:
    """Create a SweepConfig with dummy workflow scripts and input files."""
    if request_rates is None:
        request_rates = [4, 8]

    # Create dummy workflow scripts
    configs: List[BenchmarkConfig] = []
    for i in range(num_configs):
        script = tmp_path / f"workflow_{i}.sh"
        script.write_text("#!/bin/bash\necho ok")
        script.chmod(0o755)
        configs.append(
            BenchmarkConfig(
                label=f"cfg-{i}",
                workflow=str(script),
                extra_args=[],
            )
        )

    # Create dummy input files
    input_files: List[str] = []
    for i in range(num_input_files):
        f = tmp_path / f"input_{i}.jsonl"
        f.write_text("{}\n")
        input_files.append(str(f))

    return SweepConfig(
        model="test-model",
        request_rates=request_rates,
        concurrencies=None,
        osl=10,
        conversation_num=1,
        warmup_count=1,
        port=8000,
        timeout=60,
        input_files=input_files,
        configs=configs,
        output_dir=str(tmp_path / "results"),
        skip_plots=True,
        restart_server_every_benchmark=restart_server_every_benchmark,
    )


@patch("benchmarks.multimodal.sweep.orchestrator.run_aiperf_single")
@patch("benchmarks.multimodal.sweep.orchestrator.ServerManager")
def test_loop_order_bench_cfg_outer(
    mock_server_cls: MagicMock,
    mock_aiperf: MagicMock,
    tmp_path: Path,
) -> None:
    """bench_cfg is the outer loop, input_file middle, sweep_values inner."""
    config = _make_config(tmp_path)
    run_sweep(config, repo_root=tmp_path)

    # Extract (label, input_file_stem, sweep_value) from each aiperf call.
    calls: list[tuple[str, str, int]] = []
    for c in mock_aiperf.call_args_list:
        artifact_dir = Path(c.kwargs["artifact_dir"])
        # Structure: results / <file_tag> / <label> / <mode><value>
        label = artifact_dir.parent.name
        value = c.kwargs["sweep_value"]
        input_stem = Path(c.kwargs["input_file"]).stem
        calls.append((label, input_stem, value))

    # All cfg-0 calls come before all cfg-1 calls.
    cfg0_calls = [c for c in calls if c[0] == "cfg-0"]
    cfg1_calls = [c for c in calls if c[0] == "cfg-1"]
    assert len(cfg0_calls) == 4  # 2 files x 2 rates
    assert len(cfg1_calls) == 4
    # cfg-0 block ends before cfg-1 block starts.
    last_cfg0_idx = max(i for i, c in enumerate(calls) if c[0] == "cfg-0")
    first_cfg1_idx = min(i for i, c in enumerate(calls) if c[0] == "cfg-1")
    assert last_cfg0_idx < first_cfg1_idx


@patch("benchmarks.multimodal.sweep.orchestrator.run_aiperf_single")
@patch("benchmarks.multimodal.sweep.orchestrator.ServerManager")
def test_server_lifecycle_restart_every(
    mock_server_cls: MagicMock,
    mock_aiperf: MagicMock,
    tmp_path: Path,
) -> None:
    """restart_server_every_benchmark=True: start/stop per run."""
    config = _make_config(tmp_path, restart_server_every_benchmark=True)
    mock_server = mock_server_cls.return_value
    mock_server.is_running = False

    run_sweep(config, repo_root=tmp_path)

    total_runs = 2 * 2 * 2  # configs x files x rates
    assert mock_server.start.call_count == total_runs
    assert mock_server.stop.call_count == total_runs


@patch("benchmarks.multimodal.sweep.orchestrator.run_aiperf_single")
@patch("benchmarks.multimodal.sweep.orchestrator.ServerManager")
def test_server_lifecycle_restart_per_config(
    mock_server_cls: MagicMock,
    mock_aiperf: MagicMock,
    tmp_path: Path,
) -> None:
    """restart_server_every_benchmark=False: start/stop once per config."""
    config = _make_config(tmp_path, num_configs=3, restart_server_every_benchmark=False)
    mock_server = mock_server_cls.return_value
    mock_server.is_running = False

    run_sweep(config, repo_root=tmp_path)

    assert mock_server.start.call_count == 3
    assert mock_server.stop.call_count == 3


@patch("benchmarks.multimodal.sweep.orchestrator.run_aiperf_single")
@patch("benchmarks.multimodal.sweep.orchestrator.ServerManager")
def test_skip_existing_results(
    mock_server_cls: MagicMock,
    mock_aiperf: MagicMock,
    tmp_path: Path,
) -> None:
    """Runs with existing profile_export_aiperf.json are skipped."""
    config = _make_config(tmp_path, num_configs=1, num_input_files=1)
    mock_server = mock_server_cls.return_value
    mock_server.is_running = False

    # Pre-create result for rate=4
    input_tag = Path(config.input_files[0]).stem
    artifact_dir = tmp_path / "results" / input_tag / "cfg-0" / "request_rate4"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "profile_export_aiperf.json").write_text("{}")

    run_sweep(config, repo_root=tmp_path)

    # Only rate=8 should have run.
    assert mock_aiperf.call_count == 1
    assert mock_aiperf.call_args.kwargs["sweep_value"] == 8


@patch("benchmarks.multimodal.sweep.orchestrator.run_aiperf_single")
@patch("benchmarks.multimodal.sweep.orchestrator.ServerManager")
def test_skip_all_results_no_server_start(
    mock_server_cls: MagicMock,
    mock_aiperf: MagicMock,
    tmp_path: Path,
) -> None:
    """When all runs have results, the server should never start."""
    config = _make_config(
        tmp_path,
        num_configs=1,
        num_input_files=1,
        restart_server_every_benchmark=False,
    )
    mock_server = mock_server_cls.return_value
    mock_server.is_running = False

    # Pre-create results for both rates.
    input_tag = Path(config.input_files[0]).stem
    for rate in [4, 8]:
        artifact_dir = (
            tmp_path / "results" / input_tag / "cfg-0" / f"request_rate{rate}"
        )
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "profile_export_aiperf.json").write_text("{}")

    run_sweep(config, repo_root=tmp_path)

    assert mock_aiperf.call_count == 0
    assert mock_server.start.call_count == 0
