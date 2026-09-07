# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only process smokes for sample multimodal worker handoffs."""

from __future__ import annotations

import os
import signal
import subprocess
import uuid
from pathlib import Path

import pytest

from tests.utils.port_utils import allocate_ports, deallocate_ports

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unified,
    pytest.mark.timeout(270),
]

REPO_ROOT = Path(__file__).parents[2]
LAUNCH_DIR = REPO_ROOT / "examples" / "backends" / "sample" / "launch"


@pytest.mark.parametrize(
    "script_name",
    ["multimodal_agg.sh", "multimodal_disagg.sh"],
)
def test_sample_multimodal_smoke(script_name, request, runtime_services_dynamic_ports):
    del runtime_services_dynamic_ports
    ports = allocate_ports(3, 18000)
    request.addfinalizer(lambda: deallocate_ports(ports))

    env = os.environ.copy()
    env.update(
        {
            "DYN_SYSTEM_PORT": str(ports[0]),
            "DYN_SYSTEM_PORT1": str(ports[0]),
            "DYN_SYSTEM_PORT2": str(ports[1]),
            "DYN_SYSTEM_PORT3": str(ports[2]),
            "NAMESPACE": f"sample-mm-{uuid.uuid4().hex}",
        }
    )
    process = subprocess.Popen(
        ["bash", str(LAUNCH_DIR / script_name)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
        pytest.fail(f"{script_name} timed out\n{output}")

    assert process.returncode == 0, f"{script_name} failed\n{output}"
