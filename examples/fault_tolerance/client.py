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
import time

from pydantic import BaseModel

from dynamo.runtime import DistributedRuntime, dynamo_worker


class RequestType(BaseModel):
    text: str


@dynamo_worker()
async def client(runtime: DistributedRuntime):
    client = (
        await runtime.namespace("FaultTolerance")
        .component("Processor")
        .endpoint("generate")
        .client()
    )
    for i in range(100):
        request = RequestType(text="1 2 3 4 5 6 7 8 9")
        print(f"Sending: {request.text}")
        responses = await client.generate(request.model_dump_json())
        i = 0
        async for response in responses:
            print(f"Received: {response.data()}")
            if response.data() != f'{{"text":"{i + 1}"}}':
                raise RuntimeError(f"Invalid response #{i + 1}: {response.data()}")
            i += 1
        if i != 9:
            raise RuntimeError(f"Expected 9 responses, got {i}")
        time.sleep(1)


if __name__ == "__main__":
    asyncio.run(client())
