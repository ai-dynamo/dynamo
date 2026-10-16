# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Python-to-Rust launcher boundary without an inference engine."""

import os
import socket
import sys

import pytest

from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import reserved_ports

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.core,
]


@pytest.fixture(scope="module", autouse=True)
def nats_and_etcd():
    """Override the suite fixture: these subprocesses use in-memory discovery."""


@pytest.fixture
def sidecar_env():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("DYN_", "ETCD_", "NATS_"))
    }
    env.update(
        DYN_DISCOVERY_BACKEND="mem",
        DYN_REQUEST_PLANE="tcp",
        DYN_EVENT_PLANE="zmq",
        DYN_SYSTEM_HOST="127.0.0.1",
        DYN_ENABLE_OTEL="false",
    )
    return env


@pytest.mark.parametrize("engine", ["vllm", "sglang", "trtllm"])
@pytest.mark.timeout(40)
def test_python_sidecar_probes_during_initialization(
    engine, sidecar_env, tmp_path, monkeypatch
):
    # ManagedProcess checks health in this process; bypass ambient proxies.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    # All launchers must serve probes independently of engine initialization.
    with socket.socket() as engine_listener, reserved_ports(1, 10000) as ports:
        engine_listener.bind(("127.0.0.1", 0))
        engine_listener.listen()
        engine_listener.settimeout(15)
        port = ports[0]
        sidecar_env["DYN_SYSTEM_PORT"] = str(port)
        args = [
            sys.executable,
            "-m",
            f"dynamo.{engine}.sidecar",
            "--grpc-endpoint",
            f"http://127.0.0.1:{engine_listener.getsockname()[1]}",
            "--grpc-startup-deadline-secs",
            "60",
        ]
        if engine == "trtllm":
            args.extend(["--model-path", "unused"])
        with ManagedProcess(
            command=args,
            env=sidecar_env,
            log_dir=str(tmp_path),
            display_name=f"{engine}-sidecar",
            terminate_all_matching_process_names=False,
            health_check_urls=[
                f"http://127.0.0.1:{port}/{path}" for path in ("live", "health")
            ],
            timeout=15,
        ) as child:
            # The listener never completes gRPC, including during the helper's
            # health checks. Accept confirms that engine bootstrap has started.
            connection, _ = engine_listener.accept()
            with connection:
                assert child.proc.poll() is None, child.read_logs()
