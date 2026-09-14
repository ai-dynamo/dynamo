# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native launcher smoke test; requires a locally built Dynamo extension."""

import hashlib
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0]


def write_sources(path: Path, namespaces: list[str]) -> str:
    document = {
        "version": 1,
        "connectionRevision": "connection",
        "sources": [{"namespace": name} for name in sorted(namespaces)],
    }
    revision = hashlib.sha256(
        json.dumps(document, separators=(",", ":")).encode()
    ).hexdigest()
    staged = path.with_suffix(".next")
    staged.write_text(json.dumps(document))
    staged.replace(path)
    return revision


def wait_for_state(
    process: subprocess.Popen, revision: str | None, port: int, *, error: bool = False
) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        assert process.poll() is None, "Relay exited before applying sources"
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/engine/state", timeout=1
            ) as response:
                result = json.load(response)
            if (
                result["sources"]["appliedRevision"] == revision
                and bool(result["sources"]["lastError"]) == error
                and result["ready"] is True
            ):
                return result
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.05)
    raise AssertionError("Relay did not acknowledge sources before timeout")


@pytest.mark.parametrize("mode", ["from-file", "discovery"])
def test_native_launcher_reloads_without_restart(tmp_path: Path, mode: str) -> None:
    pytest.importorskip("dynamo._core", reason="requires the native Dynamo extension")
    # Runtime configuration currently stores the system port as a signed i16.
    for _ in range(100):
        port = random.SystemRandom().randrange(1024, 32768)
        with socket.socket() as reservation:
            try:
                reservation.bind(("127.0.0.1", port))
            except OSError:
                continue
            break
    else:
        raise AssertionError("No free runtime HTTP port")
    path = tmp_path / "sources.json"
    initial = write_sources(path, [])
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NATS_", "DYN_RELAY_"))
    }
    env.update(
        {
            "DYN_DISCOVERY_BACKEND": "mem",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "zmq",
            "DYN_SYSTEM_PORT": str(port),
            "POD_UID": "native-test-pod",
        }
    )
    if mode == "from-file":
        env["DYN_RELAY_CONNECTION_REVISION"] = "connection"
    with (tmp_path / "relay.log").open("w+") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "dynamo.kv_dc_relay",
                "--dc-id",
                "native-test",
                *(["--sources-file", str(path)] if mode == "from-file" else []),
                "--bind",
                "127.0.0.1:0",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            first = wait_for_state(
                process, initial if mode == "from-file" else None, port
            )
            assert first["podUID"] == "native-test-pod"
            assert first["mode"] == mode
            assert first["sources"]["count"] == 0
            for route in ("/scope", "/sources", "/catalog"):
                with pytest.raises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=1)
                assert error.value.code == 404
            if mode == "discovery":
                assert first["sources"]["desiredRevision"] is None
                return
            pid = process.pid
            added = write_sources(path, ["a", "b"])
            assert (
                wait_for_state(process, added, port)["connectionRevision"]
                == "connection"
            )
            path.write_text('{"private-invalid-file":')
            rejected = wait_for_state(process, added, port, error=True)
            assert rejected["sources"]["desiredRevision"] is None
            assert rejected["sources"]["count"] == 2
            assert "private-invalid-file" not in json.dumps(rejected)
            removed = write_sources(path, ["b"])
            wait_for_state(process, removed, port)
            empty = write_sources(path, [])
            wait_for_state(process, empty, port)
            assert process.pid == pid and process.poll() is None
        except BaseException:
            log.flush()
            log.seek(0)
            print(log.read())
            raise
        finally:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.parametrize(
    "document", ["valid", "missing", "invalid", "mismatched-connection"]
)
def test_standalone_file_startup_without_http(tmp_path: Path, document: str) -> None:
    pytest.importorskip("dynamo._core", reason="requires the native Dynamo extension")
    path = tmp_path / "sources.json"
    if document == "valid":
        path.write_text('{"version":1,"sources":[]}')
    elif document == "invalid":
        path.write_text('{"private-invalid-file":')
    elif document == "mismatched-connection":
        path.write_text('{"version":1,"connectionRevision":"other","sources":[]}')
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NATS_", "DYN_RELAY_")) and key != "POD_UID"
    }
    env.update({"DYN_SYSTEM_PORT": "-1", "DYN_EVENT_PLANE": "zmq"})
    script = """
import asyncio, sys
from dynamo.runtime import DistributedRuntime
from dynamo.llm import KvDcRelay
async def main():
    runtime = DistributedRuntime(asyncio.get_running_loop(), "mem", "tcp")
    try:
        relay = KvDcRelay(runtime.endpoint("test.relay.control"), "test",
                          sources_file=sys.argv[1], bind="127.0.0.1:0")
        await relay.start()
        await relay.shutdown()
    finally:
        runtime.shutdown()
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    if document == "valid":
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "private-invalid-file" not in result.stderr
