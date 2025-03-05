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

import uvloop

from dynemo.runtime import DistributedRuntime, dynemo_worker

# Soak Test
#
# This was a failure case for the distributed runtime. If the Rust Tokio
# runtime is started with a small number of threads, it will starve the
# the GIL + asyncio event loop can starve timeout the ingress handler.
#
# There may still be some blocking operations in the ingress handler that
# could still eventually be a problem.


@dynemo_worker()
async def worker(runtime: DistributedRuntime):
    ns = random_string()
    task = asyncio.create_task(server_init(runtime, ns))
    await client_init(runtime, ns)
    runtime.shutdown()
    await task


async def client_init(runtime: DistributedRuntime, ns: str):
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    # get endpoint
    endpoint = runtime.namespace(ns).component("backend").endpoint("generate")

    # create client
    client = await endpoint.client()

    # wait for an endpoint to be ready
    await client.wait_for_endpoints()

    # Issue many concurrent requests to put load on the server,
    # the task should issue the request and process the response
    tasks = []
    for i in range(20000):
        tasks.append(asyncio.create_task(do_one(client)))

    await asyncio.gather(*tasks)

    # ensure all tasks are done and without errors
    error_count = 0
    for task in tasks:
        if task.exception():
            error_count += 1

    assert error_count == 0, f"expected 0 errors, got {error_count}"


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
    print("Started server instance")
    await endpoint.serve_endpoint(RequestHandler().generate)


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


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
