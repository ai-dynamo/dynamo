# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


@dynamo_worker(register_shutdown=True)
async def worker(runtime: DistributedRuntime):
    await init(runtime, "dynamo")


async def init(runtime: DistributedRuntime, ns: str):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace(ns).component("backend")

    endpoint = component.endpoint("generate")
    print("Started server instance")

    # the server will gracefully shutdown (i.e., keep opened TCP streams finishes)
    # after the lease is revoked
    await endpoint.serve_endpoint(RequestHandler().generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
