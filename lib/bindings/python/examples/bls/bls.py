# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

uvloop.install()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    foo = (
        await runtime.namespace("examples/bls")
        .component("foo")
        .endpoint("generate")
        .client()
    )
    bar = (
        await runtime.namespace("examples/bls")
        .component("bar")
        .endpoint("generate")
        .client()
    )

    # hello world showed us the client has a .generate, which uses the default load balancer
    # however, you can explicitly opt-in to client side load balancing by using the `round_robin`
    # or `random` methods on client. note - there is a direct method as well, but that is for a
    # router example
    async for char in await foo.round_robin("hello world"):
        # the responses are sse-style responses, so we extract the data key
        async for x in await bar.random(char.get("data")):
            print(x)


asyncio.run(worker())
