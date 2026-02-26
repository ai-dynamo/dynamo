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
import time

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    # Get endpoint (sender -> receiver)
    sender_endpoint = runtime.endpoint("embedding_transfer.sender.generate")
    receiver_endpoint = runtime.endpoint("embedding_transfer.receiver.generate")

    # Create client and wait for service to be ready
    sender_client = await sender_endpoint.client()
    await sender_client.wait_for_instances()
    receiver_client = await receiver_endpoint.client()
    await receiver_client.wait_for_instances()

    client = receiver_client
    # client = sender_client
    num_requests = 100
    try:
        start_time = time.perf_counter()
        streams = [
            await client.round_robin("world,sun,moon,star") for _ in range(num_requests)
        ]
        for stream in streams:
            async for response in stream:
                continue
        end_time = time.perf_counter()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    except Exception as e:
        # Log the exception with context
        print(f"Error in worker: {type(e).__name__}: {e}")
        # Re-raise for graceful shutdown
        raise


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
