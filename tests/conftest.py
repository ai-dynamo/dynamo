# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import subprocess
import time

import pytest

from tests.utils import managed_process, find_free_port


def _ensure_killed(pattern: str):
    subprocess.run(["pkill", "-9", "-f", pattern], check=False)
    time.sleep(0.1)


@pytest.fixture(scope="session")
def etcd_server():
    _ensure_killed("etcd")
    client_port = find_free_port()
    peer_port   = find_free_port()

    client_url = f"http://127.0.0.1:{client_port}"
    peer_url   = f"http://127.0.0.1:{peer_port}"

    etcd_env = os.environ.copy()
    etcd_env["ALLOW_NONE_AUTHENTICATION"] = "yes"
    shutil.rmtree("/tmp/etcd-test-data", ignore_errors=True)

    cmd = [
        "etcd",
        "--listen-client-urls", client_url,
        "--advertise-client-urls", client_url,
        "--listen-peer-urls", peer_url,
        "--initial-advertise-peer-urls", peer_url,
        "--initial-cluster",              f"default={peer_url}",
        "--initial-cluster-state",        "new",
        "--data-dir",                     "/tmp/etcd-test-data",
    ]

    with managed_process(cmd, env=etcd_env, check_ports=[client_port], output=True):
        yield client_url


@pytest.fixture(scope="session")
def nats_server():
    _ensure_killed("nats-server")
    client_port = find_free_port()
    http_port   = find_free_port()

    shutil.rmtree("/tmp/nats/jetstream", ignore_errors=True)
    os.makedirs("/tmp/nats/jetstream", exist_ok=True)

    cmd = [
        "nats-server",
        "-js",
        "--trace",
        "--store_dir", "/tmp/nats/jetstream",
        "--port",        str(client_port),
        "--http_port",   str(http_port),
    ]

    with managed_process(cmd, check_ports=[client_port, http_port], output=True):
        time.sleep(0.2)
        yield {
            "client_url":   f"nats://127.0.0.1:{client_port}",
            "http_monitor": f"http://127.0.0.1:{http_port}",
        }


def pytest_sessionfinish(session, exitstatus):
    """Final cleanup: kill stray daemons and reap zombies."""
    subprocess.run(["pkill", "-9", "-f", "etcd|nats-server"], check=False)
    import os
    from time import sleep
    sleep(0.1)
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
        except ChildProcessError:
            break