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

import subprocess
import time

import pytest
from typer.testing import CliRunner

pytestmark = pytest.mark.pre_merge
runner = CliRunner()


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    nats_server = subprocess.Popen(["nats-server", "-js"])
    etcd = subprocess.Popen(["etcd"])
    print("Setting up resources")

    # Start the server using subprocess
    server = subprocess.Popen(
        [
            "dynamo",
            "serve",
            "pipeline:Frontend",
            "--working-dir",
            "deploy/sdk/src/dynamo/sdk/tests",
        ]
    )

    time.sleep(5)  # Wait for server to start

    yield

    # Teardown code
    print("Tearing down resources")
    server.terminate()
    server.wait()
    nats_server.terminate()
    nats_server.wait()
    etcd.terminate()
    etcd.wait()


async def test_ping_healthy(setup_and_teardown):
    """Test ping endpoint when all dependencies are available."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/ping") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data == {"status": "healthy"}
