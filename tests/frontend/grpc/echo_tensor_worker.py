#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `TEST_END_TO_END=1 python test_tensor.py` to run this worker as tensor based echo worker.


import uvloop

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker


@dynamo_worker(static=False)
async def echo_tensor_worker(runtime: DistributedRuntime):
    component = runtime.namespace("tensor").component("echo")
    await component.create_service()

    endpoint = component.endpoint("generate")

    model_config = {
        "name": "echo",
        "inputs": [
            {"name": "dummy_input", "data_type": "Bytes", "shape": [-1]},
        ],
        "outputs": [{"name": "dummy_output", "data_type": "Bytes", "shape": [-1]}],
    }
    runtime_config = ModelRuntimeConfig()
    runtime_config.set_tensor_model_config(model_config)

    assert model_config == runtime_config.get_tensor_model_config()

    # [gluo FIXME] register_llm will attempt to load a LLM model,
    # which is not well-defined for Tensor yet. Currently provide
    # a valid model name to pass the registration.
    await register_llm(
        ModelInput.Tensor,
        ModelType.TensorBased,
        endpoint,
        "Qwen/Qwen3-0.6B",
        "echo",
        runtime_config=runtime_config,
    )

    await endpoint.serve_endpoint(generate)


async def generate(request, context):
    """Echo tensors and parameters back to the client."""
    print(f"Echoing request: {request}")

    params = {}
    if "parameters" in request:
        params.update(request["parameters"])

    params["processed"] = {"bool": True}

    yield {
        "model": request["model"],
        "tensors": request["tensors"],
        "parameters": params,
    }


if __name__ == "__main__":
    uvloop.run(echo_tensor_worker())
