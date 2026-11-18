# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the NIXL telemetry sink used by co-located TRT-LLM ranks.

An enabled Prometheus exporter can collide on the pod's shared port and prevent
rank startup. The file-sink workaround must initialize every agent and create a
separate buffer per rank in an existing directory, even with that port occupied.
Run against the installed native library, including the TRT-LLM image's preload.
No model, GPU, transport backend, or Dynamo service is needed.
"""

import importlib.util
import json
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.utils.constants import DynamoPortRange
from tests.utils.port_utils import reserved_ports

pytestmark = [
    pytest.mark.trtllm,
    pytest.mark.framework_agnostic,  # Only NIXL is required, not the engine itself.
    pytest.mark.core,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.timeout(90),
    pytest.mark.skipif(
        importlib.util.find_spec("nixl") is None, reason="NIXL is not installed"
    ),
]

# Each fresh interpreter reads its own telemetry environment before constructing
# an agent. Empty backends avoid initializing CUDA or RDMA transports.
_CREATE_AGENT = """
import json
import sys
from nixl._api import nixl_agent, nixl_agent_config

agent = nixl_agent(sys.argv[1], nixl_agent_config(backends=[]))
print(json.dumps({"name": agent.name}))
"""


@pytest.mark.parametrize(
    "enable", ["y", "n", None], ids=["file-sink", "off", "default"]
)
def test_colocated_agents_start_with_occupied_telemetry_port(tmp_path, enable):
    telemetry_dir = tmp_path / "telemetry"
    # BUFFER needs an existing directory; a missing directory can silently lose
    # telemetry on NIXL versions that tolerate exporter construction failure.
    telemetry_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("NIXL_TELEMETRY_")}
    if enable is not None:
        env["NIXL_TELEMETRY_ENABLE"] = enable
    env["NIXL_TELEMETRY_EXPORTER"] = "" if enable == "y" else "prometheus"
    env["NIXL_TELEMETRY_DIR"] = str(telemetry_dir)
    names = [f"rank-{rank}" for rank in range(4 if enable == "y" else 1)]

    with reserved_ports(1, DynamoPortRange.NIXL) as ports, socket.socket() as occupied:
        occupied.bind(("0.0.0.0", ports[0]))
        occupied.listen()
        env["NIXL_TELEMETRY_PROMETHEUS_PORT"] = str(ports[0])
        # Keep the socket occupied through every child's lifetime. This makes
        # accidental Prometheus selection deterministic regardless of scheduling.
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = [
                pool.submit(
                    subprocess.run,
                    [sys.executable, "-c", _CREATE_AGENT, name],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                for name in names
            ]
            for name, future in zip(names, futures):
                result = future.result()
                assert (
                    result.returncode == 0
                ), f"{name} failed to initialize:\n{result.stdout}\n{result.stderr}"
                report = json.loads(result.stdout.splitlines()[-1])
                assert report["name"] == name

    files = list(telemetry_dir.iterdir())
    if enable == "y":
        # Successful startup alone would miss a silently disabled exporter.
        assert {path.name for path in files} == set(names)
        assert all(path.is_file() and path.stat().st_size > 0 for path in files)
    else:
        assert files == [], "Disabled/default telemetry must not create file sinks"
