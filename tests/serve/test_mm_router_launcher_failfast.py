# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM multimodal router launcher must exit when a worker dies at startup.

Without a liveness check, the launcher polls a dead worker's /health port until
its 900 s deadline, so the serve test hits its pytest timeout instead of
failing fast and going to the GPU orchestrator's retry.
"""

import os
import signal
import stat
import subprocess
from pathlib import Path

import pytest

from tests.utils.constants import DynamoPortRange
from tests.utils.port_utils import reserved_ports

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

LAUNCHER = (
    Path(__file__).parents[2] / "examples/backends/vllm/launch/agg_multimodal_router.sh"
)
# Far below the launcher's 900 s readiness deadline, so a hang fails the test.
FAIL_FAST_TIMEOUT_S = 60

# Stands in for a vLLM worker whose engine core crashes during initialisation.
STUB_PYTHON = """#!/bin/bash
if [[ "$1" == "-m" && "$2" == "dynamo.vllm" ]]; then
    echo "stub: Engine core initialization failed" >&2
    exit 1
fi
exec python3 "$@"
"""


def test_launcher_exits_when_worker_dies_during_startup(tmp_path: Path) -> None:
    """Exit non-zero and name the dead backend instead of polling its port."""
    stub = tmp_path / "python"
    stub.write_text(STUB_PYTHON)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)

    env = dict(os.environ)
    env.pop("DYN_MANAGED_PORTS", None)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"
    env["SINGLE_GPU"] = "true"
    env["NUM_WORKERS"] = "2"

    with reserved_ports(2, DynamoPortRange.SERVE.value) as ports:
        env["DYN_SYSTEM_PORT1"] = str(ports[0])
        env["DYN_SYSTEM_PORT2"] = str(ports[1])
        # The launcher's EXIT trap runs `kill 0`; a new session keeps that
        # signal away from pytest.
        proc = subprocess.Popen(
            ["bash", str(LAUNCHER)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            _, stderr = proc.communicate(timeout=FAIL_FAST_TIMEOUT_S)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()

    assert proc.returncode != 0
    assert "exited with status 1 before becoming ready" in stderr
    assert "vLLM backend 1 (pid" in stderr
