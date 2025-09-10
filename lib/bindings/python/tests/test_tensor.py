#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.mocker --model-path /data/models/Qwen3-0.6B-Q8_0.gguf --extra-engine-args args.json`

import os

import uvloop

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker

TEST_END_TO_END = os.environ.get("TEST_END_TO_END", 0)


@dynamo_worker(static=False)
async def test_register(runtime: DistributedRuntime):
    component = runtime.namespace("test").component("tensor")
    await component.create_service()

    endpoint = component.endpoint("generate")

    model_config = {
        "name": "tensor",
        "inputs": [{"name": "input", "data_type": "Int32", "shape": [-1]}],
        "outputs": [{"name": "output", "data_type": "Int32", "shape": [-1]}],
    }
    runtime_config = ModelRuntimeConfig()
    runtime_config.set_tensor_model_config(model_config)

    assert model_config == runtime_config.get_tensor_model_config()

    # [gluo FIXME] register_llm will attempt to load a LLM model,
    # which is not well-defined for Tensor yet. Currently provide
    # a valid model name to pass the registration.
    await register_llm(
        ModelInput.Tensor,
        ModelType.Tensor,
        endpoint,
        "Qwen/Qwen3-0.6B",
        "tensor",
        runtime_config=runtime_config,
    )

    if TEST_END_TO_END:
        await endpoint.serve_endpoint(generate)


async def generate(request, context):
    print(f"Received request: {request}")
    yield {"model": request["model"], "tensors": request["tensors"]}


if __name__ == "__main__":
    uvloop.run(test_register())
