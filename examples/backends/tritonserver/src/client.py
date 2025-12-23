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
from typing import List

import numpy as np
import uvloop
from pydantic import BaseModel

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="tritonserver_client", worker_id=0)


class ArrayData(BaseModel):
    data: List
    shape: List[int]
    dtype: str


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    # Get endpoint
    endpoint = (
        runtime.namespace("triton").component("tritonserver").endpoint("generate")
    )

    # Create client and wait for service to be ready
    client = await endpoint.client()
    await client.wait_for_instances()

    idx = 0
    base_delay = 0.1  # Start with 100ms
    max_delay = 5.0  # Max 5 seconds
    current_delay = base_delay

    logger.info("Starting worker")
    while True:
        try:
            # Issue request and process the stream
            idx += 1
            # Create numpy array and serialize to ArrayData
            arr = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
            request = ArrayData(
                data=arr.flatten().tolist(), shape=list(arr.shape), dtype=str(arr.dtype)
            )
            # Convert Pydantic model to dict for dynamo runtime
            stream = await client.generate(request.model_dump())
            async for response in stream:
                # Deserialize response back to numpy array
                response_data = response.data()
                if isinstance(response_data, dict):
                    result_arr = np.array(
                        response_data["data"], dtype=np.dtype(response_data["dtype"])
                    ).reshape(response_data["shape"])
                    print(f"Received: {result_arr}")
                else:
                    print(response_data)
            # Reset backoff on successful iteration
            current_delay = base_delay
            # Sleep for 1 second
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Re-raise for graceful shutdown
            raise
        except Exception as e:
            # Log the exception with context
            logger.error(f"Error in worker iteration {idx}: {type(e).__name__}: {e}")
            # Perform exponential backoff
            logger.error(f"Retrying after {current_delay:.2f} seconds...")
            await asyncio.sleep(current_delay)
            # Double the delay for next time, up to max_delay
            current_delay = min(current_delay * 2, max_delay)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
