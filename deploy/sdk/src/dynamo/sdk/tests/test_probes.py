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
import subprocess
import time

import pytest
from typer.testing import CliRunner

pytestmark = pytest.mark.pre_merge
runner = CliRunner()


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup code
    nats_server = subprocess.Popen(["nats-server", "-js"])
    etcd = subprocess.Popen(["etcd"])
    print("Setting up resources")

    # Enable system app for health probes
    os.environ["DYNAMO_SYSTEM_APP_ENABLED"] = "true"
    os.environ["DYNAMO_SYSTEM_APP_PORT"] = "8002"

    # Start the server in standalone mode
    server = subprocess.Popen(
        [
            "dynamo",
            "serve",
            "pipeline:Frontend",
            "--working-dir",
            "deploy/sdk/src/dynamo/sdk/tests",
            "--service-name",
            "Frontend",  # Enable standalone mode
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

    # Clean up environment variables
    os.environ.pop("DYNAMO_SYSTEM_APP_ENABLED", None)
    os.environ.pop("DYNAMO_SYSTEM_APP_PORT", None)


async def test_custom_probes(setup_and_teardown):
    import asyncio

    import aiohttp

    max_retries = 5
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                # Test custom liveness probe
                async with session.get("http://localhost:8002/custom/health") as resp:
                    assert resp.status == 200

                # Test custom readiness probe
                async with session.get("http://localhost:8002/custom/ready") as resp:
                    assert resp.status == 200

                break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed, retrying... {e}")
            await asyncio.sleep(3)
