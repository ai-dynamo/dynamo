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
import os
import sys
import time
import uuid

import uvloop

from dynamo.llm import HttpAsyncEngine, HttpError, HttpService
from dynamo.runtime import DistributedRuntime, dynamo_worker

# Import shared memory monitoring
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from memory_monitor import create_monitor, setup_background_monitor

# Create monitor if profiling is enabled
monitor = create_monitor("FRONTEND")


class MockEngine:
    def __init__(self, model_name, pipeline_client):
        self.model_name = model_name
        self.pipeline_client = pipeline_client
        if monitor:
            print("Initialized frontend engine with memory monitoring")
            monitor.log_memory("Initial:")

    def generate(self, request):
        if monitor:
            monitor.increment_request()

        id = f"chat-{uuid.uuid4()}"
        created = int(time.time())
        model = self.model_name
        # print(f"{created} | Received request: {request}")

        async def generator():
            num_chunks = 10
            i = 0
            for _ in range(5):
                try:
                    stream = await self.pipeline_client.random(request)
                    break
                except Exception as e:
                    print(f"Error contacting pipeline: {e}")
                    await asyncio.sleep(0.1)
            else:
                print(f"Failed to contact pipeline after retries for request {id}")
                raise HttpError(401, "Failed to contact pipeline after retries")

            async for output in stream:
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
                            "delta": {"role": "assistant", "content": mock_content},
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                i += 1
                yield chunk

        return generator()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    model: str = "mock_model"
    served_model_name: str = "mock_model"

    pipeline = (
        await runtime.namespace("openai/pipeline")
        .component("proxy_worker")
        .endpoint("generate")
        .client()
    )

    loop = asyncio.get_running_loop()
    python_engine = MockEngine(model, pipeline)
    engine = HttpAsyncEngine(python_engine.generate, loop)
    # test the engine with a dummy request
    async for _ in python_engine.generate(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 2,
        }
    ):
        pass

    host: str = "localhost"
    port: int = 8000
    service: HttpService = HttpService(port=port)
    service.add_chat_completions_model(served_model_name, "mdcsum", engine)
    service.enable_endpoint("chat", True)

    print("Starting service")
    shutdown_signal = service.run(runtime.child_token())

    # Setup background memory monitoring
    monitor_task = setup_background_monitor(monitor)

    try:
        print(f"Serving endpoint: {host}:{port}/v1/models")
        print(f"Serving endpoint: {host}:{port}/v1/chat/completions")
        print(f"Serving the following models: {service.list_chat_completions_models()}")
        # Block until shutdown signal received
        await shutdown_signal
    except asyncio.CancelledError:
        # Handle cancellation during shutdown gracefully
        print("Service shutdown requested...")
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("Keyboard interrupt received, shutting down...")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
    finally:
        if monitor:
            print("\nShutdown - Final memory state:")
            monitor.log_memory("Final:")
        if monitor_task:
            monitor_task.cancel()
        print("Shutting down worker...")


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
