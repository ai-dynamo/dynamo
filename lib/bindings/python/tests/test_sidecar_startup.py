# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Python-to-Rust launcher boundary without an inference engine."""

import os
import random
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

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
def test_python_sidecar_probes_during_initialization(engine, sidecar_env, tmp_path):
    # Regression: the Python launcher used to discover metadata before binding
    # HTTP. A connected but silent engine must permit independent probes.
    with socket.socket() as engine_listener, socket.socket() as reservation:
        engine_listener.bind(("127.0.0.1", 0))
        engine_listener.listen()
        for _ in range(100):
            port = random.randrange(10000, 32000)  # system port is currently i16
            try:
                reservation.bind(("127.0.0.1", port))
                break
            except OSError:
                continue
        else:
            pytest.fail("no available system port")
        sidecar_env["DYN_SYSTEM_PORT"] = str(port)
        reservation.close()
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
        log_path = tmp_path / "sidecar.log"
        with log_path.open("w") as log:
            child = subprocess.Popen(args, env=sidecar_env, stdout=log, stderr=log)
        connection = None
        try:
            engine_listener.settimeout(15)
            # Observe an actual engine connection, then keep it silent until
            # termination so shutdown is exercised during engine initialization.
            connection, _ = engine_listener.accept()
            http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

            def wait_status(path, expected):
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    assert child.poll() is None, log_path.read_text()
                    code = None
                    try:
                        with http.open(
                            f"http://127.0.0.1:{port}/{path}", timeout=0.5
                        ) as response:
                            code = response.status
                    except urllib.error.HTTPError as error:
                        code = error.code
                    except (OSError, urllib.error.URLError):
                        pass
                    if code == expected:
                        return
                    time.sleep(0.025)
                pytest.fail(
                    f"/{path} did not return {expected}: {log_path.read_text()}"
                )

            wait_status("live", 200)
            wait_status("health", 200)
            child.terminate()
            if engine == "trtllm":
                # TRT-LLM connects inside Worker.start(), whose existing policy
                # waits for start to finish before cleanup. Probes must reflect
                # shutdown immediately while that lifecycle remains in progress.
                wait_status("health", 503)
                wait_status("live", 200)
            else:
                # vLLM/SGLang metadata discovery precedes Worker.start() and is
                # cancelled promptly by the shared sidecar runner.
                assert child.wait(timeout=5) == 0, log_path.read_text()
        finally:
            if connection is not None:
                connection.close()
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


@pytest.mark.parametrize(
    ("engine", "args", "exception", "discovery"),
    [
        ("trtllm", ["--help"], "SystemExit", "invalid-backend"),
        (
            "trtllm",
            ["--model-path", "unused", "--context-length", "0"],
            "ValueError",
            "invalid-backend",
        ),
        ("trtllm", ["--model-path", "unused"], "RuntimeError", "invalid-backend"),
        ("vllm", ["--grpc-startup-deadline-secs", "1"], "ValueError", "mem"),
    ],
)
def test_python_sidecar_preserves_error_categories(
    sidecar_env, engine, args, exception, discovery
):
    # Regression: combining bootstrap and serving must preserve Python exception
    # types, and invalid local arguments must fail before dependency setup.
    sidecar_env["DYN_DISCOVERY_BACKEND"] = discovery
    script = """
import sys
from dynamo._core import backend
try:
    getattr(backend, f"_run_{sys.argv[1]}_sidecar")(sys.argv[3:])
except BaseException as error:
    assert type(error).__name__ == sys.argv[2], repr(error)
else:
    raise AssertionError('launcher unexpectedly succeeded')
"""
    with socket.socket() as engine_listener:
        engine_listener.bind(("127.0.0.1", 0))
        engine_listener.listen()
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                engine,
                exception,
                *args,
                "--grpc-endpoint",
                f"http://127.0.0.1:{engine_listener.getsockname()[1]}",
            ],
            env=sidecar_env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
