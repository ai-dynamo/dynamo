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
import random
import string

import pytest

from dynamo._core import Client
from dynamo.runtime import DistributedRuntime, dynamo_worker

# Soak Test
#
# This was a failure case for the distributed runtime. If the Rust Tokio
# runtime is started with a small number of threads, it will starve the
# the GIL + asyncio event loop can starve timeout the ingress handler.
#
# There may still be some blocking operations in the ingress handler that
# could still eventually be a problem.


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    ns = random_string()
    task = asyncio.create_task(server_init(runtime, ns))

    client = await client_init(runtime, ns)

    async def cleanup():
        runtime.shutdown()
        await task

    return client, cleanup


async def client_init(runtime: DistributedRuntime, ns: str) -> Client:
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    # get endpoint
    endpoint = runtime.namespace(ns).component("backend").endpoint("generate")

    # create client
    client = await endpoint.client()

    # wait for an endpoint to be ready
    await client.wait_for_endpoints()

    return client


async def do_one(client):
    stream = await client.generate("hello world")
    async for char in stream:
        pass


async def server_init(runtime: DistributedRuntime, ns: str):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace(ns).component("backend")
    await component.create_service()

    endpoint = component.endpoint("generate")
    handler = RequestHandler()
    print("Started server instance")

    await endpoint.serve_endpoint(handler.generate)


class RequestHandler:
    """
    Request handler for the generate endpoint
    """

    async def generate(self, request):
        for char in request:
            await asyncio.sleep(0.1)
            yield char


def random_string(length=10):
    chars = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    return "".join(random.choices(chars, k=length))


@pytest.fixture(scope="module")
async def worker_fixture():
    """Worker fixture that uses the current event loop"""

    client, cleanup = await worker()
    try:
        yield client
    finally:
        await cleanup()


@pytest.mark.nightly
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("concurrent_requests", [1000, 5000, 10000])
async def test_worker(worker_fixture, concurrent_requests: int):
    print(f"Running test with {concurrent_requests} concurrent requests")
    client = worker_fixture
    tasks = []
    for _ in range(concurrent_requests):
        tasks.append(asyncio.create_task(do_one(client)))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    error_count = sum(1 for result in results if isinstance(result, Exception))
    assert error_count == 0, f"expected 0 errors, got {error_count}"
