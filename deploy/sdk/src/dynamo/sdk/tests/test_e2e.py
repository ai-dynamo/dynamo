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

import re
import subprocess
import time

import pytest
from typer.testing import CliRunner

from dynamo.sdk.cli.cli import cli

pytestmark = pytest.mark.pre_merge
runner = CliRunner()


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup code
    nats_server = subprocess.Popen(["nats-server", "-js"])
    etcd = subprocess.Popen(["etcd"])
    print("Setting up resources")

    # Run the serve command in dry-run mode with CLI runner to check it's working
    result = runner.invoke(
        cli,
        [
            "serve",
            "pipeline:Frontend",
            "--working-dir",
            "deploy/sdk/src/dynamo/sdk/tests",
            "--Frontend.model=qwentastic",
            "--Middle.bias=0.5",
            "--dry-run",
        ],
    )

    # Now start the actual server using subprocess for the real integration test
    server = subprocess.Popen(
        [
            "dynamo",
            "serve",
            "pipeline:Frontend",
            "--working-dir",
            "deploy/sdk/src/dynamo/sdk/tests",
            "--Frontend.model=qwentastic",
            "--Middle.bias=0.5",
        ]
    )

    time.sleep(5)

    yield result

    # Teardown code
    print("Tearing down resources")
    server.terminate()
    server.wait()
    nats_server.terminate()
    nats_server.wait()
    etcd.terminate()
    etcd.wait()


async def test_pipeline(setup_and_teardown):
    # Check the CLI command ran successfully
    result = setup_and_teardown
    assert result.exit_code == 0

    # Clean the output to check for expected content
    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "Service Configuration:" in clean_output
    assert '"Frontend": {' in clean_output
    assert '"model": "qwentastic"' in clean_output

    import asyncio

    import aiohttp

    max_retries = 5
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8000/generate",
                    json={"text": "federer-is-the-greatest-tennis-player-of-all-time"},
                    headers={"accept": "text/event-stream"},
                ) as resp:
                    assert resp.status == 200
                    text = await resp.text()
                    assert (
                        "federer-is-the-greatest-tennis-player-of-all-time-mid-back"
                        in text
                    )
                    break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed, retrying... {e}")
            await asyncio.sleep(3)
