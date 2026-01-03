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
import uuid

import uvloop

from dynamo.llm import HttpAsyncEngine, HttpService
from dynamo.runtime import DistributedRuntime, dynamo_worker


class MockEngine:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, request):
        id = f"chat-{uuid.uuid4()}"
        created = int(time.time())
        model = self.model_name
        print(f"{created} | Received request: {request}")

        async def generator():
            num_chunks = 5
            for i in range(num_chunks):
                mock_content = f"chunk{i}"
                finish_reason = "stop" if (i == num_chunks - 1) else None
                chunk = {
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": i,
                            "delta": {"role": None, "content": mock_content},
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield chunk

        return generator()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    model: str = "mock_model"
    served_model_name: str = "mock_model"

    loop = asyncio.get_running_loop()
    python_engine = MockEngine(model)
    engine = HttpAsyncEngine(python_engine.generate, loop)

    host: str = "localhost"
    port: int = 8000
    service: HttpService = HttpService(port=port)
    service.add_chat_completions_model(served_model_name, "mdcsum", engine)

    print("Starting service...")
    shutdown_signal = service.run(runtime.child_token())

    try:
        print(f"Serving endpoint: {host}:{port}/v1/models")
        print(f"Serving endpoint: {host}:{port}/v1/chat/completions")
        print(f"Serving the following models: {service.list_chat_completions_models()}")
        # Block until shutdown signal received
        await shutdown_signal
    except KeyboardInterrupt:
        # TODO: Handle KeyboardInterrupt gracefully in triton_worker
        # TODO: Caught by DistributedRuntime or HttpService, so it's not caught here
        pass
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
    finally:
        print("Shutting down worker...")
        runtime.shutdown()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
