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
    """
    # Pipeline Example

    This example demonstrates how to create a pipeline of components:
    - `frontend` call `middle` which calls `backend`
    - each component transforms the request before passing it to the backend
    """
    pipeline = (
        await runtime.namespace("examples/pipeline")
        .component("frontend")
        .endpoint("generate")
        .client()
    )

    async for char in await pipeline.round_robin("hello from"):
        print(char)


asyncio.run(worker())
