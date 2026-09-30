# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe and exception contracts of the real Python vLLM launcher, without a GPU."""

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

import pytest
import requests

from tests.utils.constants import DynamoPortRange
from tests.utils.port_utils import reserved_ports

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.timeout(60),
]


@pytest.fixture(scope="module", autouse=True)
def nats_and_etcd():
    # Override the binding suite's shared default-port services. These tests own
    # their dependencies and intentionally exercise unavailable ones.
    yield


def sidecar_env(port):
    return {
        **os.environ,
        "DYN_SYSTEM_HOST": "127.0.0.1",
        "DYN_SYSTEM_PORT": str(port),
        "DYN_SYSTEM_LIVE_PATH": "/live",
        "DYN_SYSTEM_HEALTH_PATH": "/health",
        "DYN_DISCOVERY_BACKEND": "mem",
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_EVENT_PLANE": "zmq",
        "DYN_HEALTH_CHECK_ENABLED": "false",
        "DYN_RUNTIME_NUM_WORKER_THREADS": "2",
        "DYN_COMPUTE_THREADS": "0",
    }


@contextmanager
def process(command, env, log_path):
    with log_path.open("w") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            yield child
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def wait_status(child, url, expected, log_path):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        assert child.poll() is None, log_path.read_text()
        try:
            if requests.get(url, timeout=2).status_code == expected:
                return
        except requests.ConnectionError:
            pass
        time.sleep(0.02)
    pytest.fail(f"{url} never returned {expected}: {log_path.read_text()}")


@pytest.mark.parametrize("dependencies_ready", [False, True])
def test_python_sidecar_probes_before_engine_discovery(tmp_path, dependencies_ready):
    # Regression: waiting for the engine before binding /live deadlocks a native
    # Kubernetes sidecar; SIGTERM must also cancel either startup phase promptly.
    with (
        reserved_ports(1, DynamoPortRange.SERVE.value) as ports,
        socket.socket() as engine,
        socket.socket() as dependency,
    ):
        engine.bind(("127.0.0.1", 0))
        engine.listen()
        dependency.bind(("127.0.0.1", 0))
        dependency.listen()
        env = sidecar_env(ports[0])
        if not dependencies_ready:
            env.update(
                DYN_EVENT_PLANE="nats",
                NATS_SERVER=f"nats://127.0.0.1:{dependency.getsockname()[1]}",
                DYN_SIDECAR_GRPC_ENDPOINT=f"127.0.0.1:{dependency.getsockname()[1]}",
            )
        command = [
            sys.executable,
            "-m",
            "dynamo.vllm.sidecar",
            "--grpc-endpoint",
            f"127.0.0.1:{engine.getsockname()[1]}",
            "--grpc-startup-deadline-secs",
            "60",
        ]
        log_path = tmp_path / "sidecar.log"
        with process(command, env, log_path) as child:
            base = f"http://127.0.0.1:{ports[0]}"
            wait_status(child, base + "/live", 200, log_path)
            wait_status(
                child, base + "/health", 200 if dependencies_ready else 503, log_path
            )
            child.terminate()
            assert child.wait(timeout=5) == 0, log_path.read_text()


@pytest.mark.parametrize(
    "args,exception",
    [
        (["--help"], "SystemExit"),
        (["--dyn-tool-call-parser", "test"], "ValueError"),
        (["--disaggregation-mode", "decode", "--route-to-encoder"], "RuntimeError"),
    ],
)
def test_python_sidecar_argument_errors_precede_dependency_connection(args, exception):
    # Regression: moving construction into an async runner can hide invalid
    # arguments behind dependency waits or change the Python exception category.
    script = """import sys
from dynamo._core import backend
try:
    backend._run_vllm_sidecar(sys.argv[1:])
except (SystemExit, ValueError, RuntimeError) as error:
    print(type(error).__name__)
else:
    raise AssertionError('invalid arguments were accepted')
"""
    env = sidecar_env(-1)
    with socket.socket() as dependency:
        dependency.bind(("127.0.0.1", 0))
        dependency.listen()
        env.update(
            DYN_EVENT_PLANE="nats",
            NATS_SERVER=f"nats://127.0.0.1:{dependency.getsockname()[1]}",
            DYN_SIDECAR_GRPC_ENDPOINT=f"127.0.0.1:{dependency.getsockname()[1]}",
        )
        result = subprocess.run(
            [sys.executable, "-c", script, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == exception


def test_python_sidecar_readiness_tracks_nats_recovery(tmp_path):
    # Regression: using initialization or cached discovery as readiness would
    # leave /health successful during an actual runtime transport outage.
    with (
        reserved_ports(2, DynamoPortRange.SERVE.value) as ports,
        socket.socket() as engine,
    ):
        engine.bind(("127.0.0.1", 0))
        engine.listen()
        env = sidecar_env(ports[0])
        env.update(DYN_EVENT_PLANE="nats", NATS_SERVER=f"nats://127.0.0.1:{ports[1]}")
        nats_command = ["nats-server", "--addr", "127.0.0.1", "--port", str(ports[1])]
        sidecar_command = [
            sys.executable,
            "-m",
            "dynamo.vllm.sidecar",
            "--grpc-endpoint",
            f"127.0.0.1:{engine.getsockname()[1]}",
            "--grpc-startup-deadline-secs",
            "60",
        ]
        log_path = tmp_path / "sidecar.log"
        with process(nats_command, env, tmp_path / "nats.log") as nats:
            deadline = time.monotonic() + 5
            while True:
                assert nats.poll() is None
                try:
                    with socket.create_connection(("127.0.0.1", ports[1]), timeout=1):
                        break
                except ConnectionRefusedError:
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
            with process(sidecar_command, env, log_path) as sidecar:
                base = f"http://127.0.0.1:{ports[0]}"
                wait_status(sidecar, base + "/health", 200, log_path)
                nats.terminate()
                nats.wait(timeout=5)
                wait_status(sidecar, base + "/health", 503, log_path)
                wait_status(sidecar, base + "/live", 200, log_path)
                with process(nats_command, env, tmp_path / "nats-recovered.log"):
                    wait_status(sidecar, base + "/health", 200, log_path)
                sidecar.terminate()
                assert sidecar.wait(timeout=5) == 0, log_path.read_text()
