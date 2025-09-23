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

import asyncio
import subprocess
from time import sleep

import pytest

from dynamo.runtime import DistributedRuntime


@pytest.fixture(scope="module", autouse=True)
def nats_and_etcd():
    # Setup code
    nats_server = subprocess.Popen(["nats-server", "-js"])
    etcd = subprocess.Popen(["etcd"])
    print("Setting up resources")

    sleep(5)  # wait for nats-server and etcd to start
    yield

    # Teardown code
    print("Tearing down resources")
    nats_server.terminate()
    nats_server.wait()
    etcd.terminate()
    etcd.wait()


@pytest.fixture(scope="function", autouse=False)
async def runtime():
    """
    Create a DistributedRuntime for testing.
    DistributedRuntime has singleton requirements, so tests using this fixture should be
    marked with `@pytest.mark.forked` to run in a separate process for isolation.
    """
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, True)
    yield runtime
    runtime.shutdown()
