# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level shutdown contracts using real Rust bindings and CPU engines."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
]


@pytest.mark.parametrize("mode", ["sdk-host", "sdk-wedged"])
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
    if mode == "sdk-host":
        assert "ENGINE_CLEANED" in result.stdout
        assert "WORKER_RETURNED" in result.stdout
        assert "HOST_SURVIVED" in result.stdout
