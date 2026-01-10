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

from dynamo.llm import KserveGrpcService, PythonAsyncEngine
from dynamo.runtime import DistributedRuntime, dynamo_worker


class MockCompletionEngine:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, request):
        created = int(time.time())
        request_id = f"cmpl-{uuid.uuid4()}"
        print(f"{created} | Received request: {request}")

        async def generator():
            response = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": "Mock completion response from Dynamo.",
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 5,
                    "total_tokens": 10,
                },
            }
            yield response

        return generator()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    model_name = "mock_model"
    checksum = "mdcsum"

    loop = asyncio.get_running_loop()
    python_engine = MockCompletionEngine(model_name)
    engine = PythonAsyncEngine(python_engine.generate, loop)

    host = "0.0.0.0"
    port = 8787
    service = KserveGrpcService(port=port, host=host)
    service.add_completions_model(model_name, checksum, engine)

    print("Starting KServe gRPC service...")
    shutdown_signal = service.run(runtime.child_token())

    try:
        print(
            f"Serving endpoint: {host}:{port} inference.GRPCInferenceService/ModelInfer"
        )
        print(f"Serving completions models: {service.list_completions_models()}")
        await shutdown_signal
    except KeyboardInterrupt:
        pass
    except Exception as err:  # pragma: no cover - example logging
        print(f"Unexpected error occurred: {err}")
    finally:
        print("Shutting down worker...")
        runtime.shutdown()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
