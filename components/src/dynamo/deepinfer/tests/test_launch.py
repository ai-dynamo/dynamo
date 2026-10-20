# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import signal
import subprocess
from types import SimpleNamespace

import pytest

from dynamo.deepinfer.launch import launch_ranks, rank_environments

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_rank_gpu_partition_and_ports(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 3,4,5,6,7,8,9")
    monkeypatch.setenv("DYN_SYSTEM_PORT", "0")
    config = SimpleNamespace(
        engine_args={"data_parallel_size": 4, "context_parallel_size": 2}
    )
    envs = rank_environments(config)
    assert [e["CUDA_VISIBLE_DEVICES"] for e in envs] == ["2,3", "4,5", "6,7", "8,9"]
    assert all(e["DYN_SYSTEM_PORT"] == "0" for e in envs)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,4")
    with pytest.raises(ValueError, match="DP \\* CP"):
        rank_environments(config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6, 6")
    with pytest.raises(ValueError, match="repeat"):
        rank_environments(config)


@pytest.mark.timeout(10)
def test_rank_failure_stops_all_process_groups_and_bounds_blocked_shutdown(monkeypatch):
    import dynamo.deepinfer.launch as launch

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("DYN_SYSTEM_PORT", "0")
    processes = []
    signals = []

    class Process:
        def __init__(self, command, *, env, start_new_session):
            self.pid = 100 + len(processes)
            self.returncode = 1 if len(processes) == 2 else None
            self.command = command
            self.env = env
            assert start_new_session
            processes.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if (self.pid, signal.SIGKILL) not in signals:
                raise subprocess.TimeoutExpired(self.command, timeout)
            self.returncode = self.returncode or -signal.SIGKILL
            return self.returncode

    monkeypatch.setattr(launch.subprocess, "Popen", Process)
    monkeypatch.setattr(
        launch.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    config = SimpleNamespace(
        engine_args={"data_parallel_size": 4, "expert_parallel_size": 4}
    )
    assert launch_ranks(config, ["--model", "/model"]) == 1
    assert len(processes) == 4
    for rank, process in enumerate(processes):
        assert process.env["CUDA_VISIBLE_DEVICES"] == str(rank)
        assert json.loads(process.command[-1])["engine_rank"] == rank
        assert (process.pid, signal.SIGTERM) in signals
        assert (process.pid, signal.SIGKILL) in signals
        assert process.returncode is not None
