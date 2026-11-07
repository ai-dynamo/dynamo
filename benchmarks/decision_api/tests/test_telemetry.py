# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
from dynamo_decision_perf.telemetry import record_sample, snapshot

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_cpu_sample_has_cumulative_work_without_command_secrets(tmp_path):
    value = snapshot([os.getpid(), 999999999])
    assert len(value["processes"]) >= 1
    current = next(p for p in value["processes"] if p["pid"] == os.getpid())
    assert current["rss_bytes"] > 0 and current["cpu_user_seconds"] >= 0
    assert "cmdline" not in current and "environ" not in current
    path = tmp_path / "cpu.jsonl"
    record_sample(path, [os.getpid()])
    record_sample(path, [os.getpid()])
    assert len([json.loads(line) for line in path.read_text().splitlines()]) == 2
