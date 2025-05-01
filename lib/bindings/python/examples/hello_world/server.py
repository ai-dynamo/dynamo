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

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker


class RequestHandler:
    """
    Request handler for the generate endpoint
    """

    async def generate(self, request):
        print(f"Received request: {request}")
        for char in request:
            await asyncio.sleep(1)
            yield char


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    await init(runtime, "dynamo")


async def init(runtime: DistributedRuntime, ns: str):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace(ns).component("backend")
    lease = await component.create_service_with_custom_lease(ttl=1)
    lease_id = lease.id()
    print(f"Created custom lease with ID: {lease_id}/{lease_id:#x}")

    endpoint = component.endpoint("generate")
    print("Started server instance")

    # the server will gracefully shutdown (i.e., keep opened TCP streams finishes)
    # after the lease is revoked
    await endpoint.serve_endpoint(RequestHandler().generate, lease)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
