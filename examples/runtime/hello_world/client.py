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


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    # Get endpoint
    endpoint = (
        runtime.namespace("hello_world").component("backend").endpoint("generate")
    )

    # Create client and wait for service to be ready
    client = await endpoint.client()
    await client.wait_for_instances()

    idx = 0
    base_delay = 0.1  # Start with 100ms
    max_delay = 5.0  # Max 5 seconds
    current_delay = base_delay

    while True:
        try:
            # Issue request and process the stream
            idx += 1
            stream = await client.generate("world,sun,moon,star")
            async for response in stream:
                print(response.data())
            # Reset backoff on successful iteration
            current_delay = base_delay
            # Sleep for 1 second
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Re-raise for graceful shutdown
            raise
        except Exception as e:
            # Log the exception with context
            print(f"Error in worker iteration {idx}: {type(e).__name__}: {e}")
            # Perform exponential backoff
            print(f"Retrying after {current_delay:.2f} seconds...")
            await asyncio.sleep(current_delay)
            # Double the delay for next time, up to max_delay
            current_delay = min(current_delay * 2, max_delay)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
