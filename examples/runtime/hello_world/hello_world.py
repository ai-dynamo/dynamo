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
import logging

import uvloop

from dynamo.runtime import Context, DistributedRuntime, dynamo_endpoint, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="backend")


@dynamo_endpoint(str, str)
async def content_generator(request: str, context: Context):
    logger.info(f"Received request: {request} with `id={context.id()}`")
    for word in request.split(","):
        await asyncio.sleep(1)
        if context.is_stopped() or context.is_killed():
            print("request got cancelled.")
            return
        yield f"Hello {word}!"


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    namespace_name = "hello_world"
    component_name = "backend"
    endpoint_name = "generate"
    lease_id = runtime.etcd_client().primary_lease_id()

    component = runtime.namespace(namespace_name).component(component_name)
    await component.create_service()

    logger.info(f"Created service {namespace_name}/{component_name}")

    endpoint = component.endpoint(endpoint_name)

    logger.info(f"Serving endpoint {endpoint_name} on lease {lease_id}")
    await endpoint.serve_endpoint(content_generator)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
