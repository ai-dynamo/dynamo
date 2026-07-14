# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.multimodal.sweep.config import load_config
from benchmarks.multimodal.sweep.runner import _build_aiperf_cmd

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_aiperf_extra_args_are_loaded_and_appended(tmp_path: Path) -> None:
    input_file = tmp_path / "input.jsonl"
    input_file.write_text("{}\n", encoding="utf-8")
    workflow = tmp_path / "workflow.sh"
    workflow.write_text("#!/bin/bash\n", encoding="utf-8")
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        f"""
model: test-model
request_rates: [16]
input_files: [{input_file}]
configs:
  - label: test
    workflow: {workflow}
aiperf_extra_args:
  - --random-seed
  - 42
  - --workers-max
  - 20
""",
        encoding="utf-8",
    )

    config = load_config(str(config_path))
    assert config.aiperf_extra_args == [
        "--random-seed",
        "42",
        "--workers-max",
        "20",
    ]

    command = _build_aiperf_cmd(
        model=config.model,
        port=config.port,
        sweep_mode="request_rate",
        sweep_value=16,
        conversation_num=1000,
        warmup_count=20,
        input_file=str(input_file),
        osl=70,
        artifact_dir=tmp_path / "artifacts",
        extra_args=config.aiperf_extra_args,
    )
    assert command[-4:] == config.aiperf_extra_args
    assert command[command.index("--request-rate") + 1] == "16"
    assert "--streaming" in command
