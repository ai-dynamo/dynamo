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

import json
import subprocess
import time

import pytest

pytestmark = pytest.mark.pre_merge


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup code
    nats_server = subprocess.Popen(["nats-server", "-js"])
    etcd = subprocess.Popen(["etcd"])
    print("Setting up resources")

    server = subprocess.Popen(
        [
            "dynamo",
            "serve",
            "pipeline_gpu:Frontend",
            "--working-dir",
            "deploy/dynamo/sdk/src/dynamo/sdk/tests",
        ]
    )

    time.sleep(5)

    yield

    # Teardown code
    print("Tearing down resources")
    server.terminate()
    server.wait()
    nats_server.terminate()
    nats_server.wait()
    etcd.terminate()
    etcd.wait()


async def test_pipeline():
    import asyncio

    import aiohttp

    max_retries = 5
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:3000/generate",
                    json={"text": "test"},
                    headers={"accept": "text/event-stream"},
                ) as resp:
                    assert resp.status == 200
                    text = await resp.text()
                    resp_json = json.loads(text)
                    assert set(list(resp_json.items())) == {
                        [0, 1],
                        [2, 3],
                        [4, 5],
                    }
                    break
        except Exception:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed, retrying...")
            await asyncio.sleep(1)
