# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level shutdown contracts using real Rust bindings and CPU engines."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
]


@pytest.mark.parametrize("mode", ["sdk-host", "sdk-wedged", "sdk-failed"])
def test_sdk_shutdown_watchdog_lifetime(mode):
    # Regression: a completed Worker.run killed its embedding host; disarming
    # early instead would leave blocked cleanup without its native watchdog.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("shutdown_probe.py")), mode],
        env={
            **os.environ,
            "DYN_SYSTEM_PORT": "0",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_TCP_RPC_PORT": "0",
            "DYN_EVENT_PLANE": "zmq",
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert "CLEANUP_STARTED" in result.stdout, result.stdout + result.stderr
    expected_code = 70 if mode == "sdk-wedged" else 0
    assert result.returncode == expected_code, result.stdout + result.stderr
    if mode == "sdk-wedged":
        origin = float(
            next(
                line.split("=", 1)[1]
                for line in result.stdout.splitlines()
                if line.startswith("SIGTERM_AT=")
            )
        )
        assert time.monotonic() - origin < 1.5, result.stdout + result.stderr
    if mode == "sdk-host":
        assert "ENGINE_CLEANED" in result.stdout
        assert "WORKER_RETURNED" in result.stdout
        assert "HOST_SURVIVED" in result.stdout
    if mode == "sdk-failed":
        assert "HOST_SURVIVED" in result.stdout
        assert "CLEANUP_FAILED" in result.stdout
        assert "ENGINE_CLEANED" not in result.stdout


@pytest.mark.parametrize(
    "mode",
    [
        "python-pull",
        "python-push",
        "python-escalate",
        "python-wedged",
        "embedding-idle",
        "embedding-slow",
        "gateway-group",
        "gateway-wedged",
    ],
)
def test_python_shutdown_signal_to_exit(mode):
    # Regression: early cleanup or a child-monitor-generated second signal can
    # terminate admitted work instead of draining requests and child processes.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("shutdown_probe.py")), mode],
        env={
            **os.environ,
            "DYN_SYSTEM_PORT": "0",
            "DYN_TCP_RPC_PORT": "0",
            "DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS": "0",
            "DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS": "3",
            "DYN_WORKER_SHUTDOWN_CLEANUP_TIMEOUT_SECS": "1",
        },
        capture_output=True,
        text=True,
        timeout=20,
        start_new_session=True,
    )
    expected_code = 70 if mode in ("python-escalate", "python-wedged") else 0
    assert result.returncode == expected_code, result.stdout + result.stderr
    if mode == "python-wedged":
        origin = float(
            next(
                line.split("=", 1)[1]
                for line in result.stdout.splitlines()
                if line.startswith("SIGTERM_AT=")
            )
        )
        assert time.monotonic() - origin < 4.0, result.stdout + result.stderr
    if expected_code == 0:
        assert "ADMISSION_CLOSED" in result.stdout
        assert "ENGINE_CLEANED" in result.stdout
        assert "RUNTIME_FINISHED" in result.stdout
        if mode.startswith("embedding-") or mode == "gateway-group":
            assert "CHILDREN_DRAINED" in result.stdout
