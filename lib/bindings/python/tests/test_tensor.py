#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `TEST_END_TO_END=1 python test_tensor.py` to run this worker as tensor based echo worker.

import os

import uvloop

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker

TEST_END_TO_END = os.environ.get("TEST_END_TO_END", 0)


@dynamo_worker()
async def test_register(runtime: DistributedRuntime):
    component = runtime.namespace("test").component("tensor")

    endpoint = component.endpoint("generate")

    model_config = {
        "name": "tensor",
        "inputs": [
            {"name": "input_text", "data_type": "Bytes", "shape": [-1]},
            {"name": "custom", "data_type": "Bytes", "shape": [-1]},
            {"name": "streaming", "data_type": "Bool", "shape": [1]},
        ],
        "outputs": [{"name": "output_text", "data_type": "Bytes", "shape": [-1]}],
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
        "tensor",
        runtime_config=runtime_config,
    )

    if TEST_END_TO_END:
        await endpoint.serve_endpoint(generate)


async def generate(request, context):
    print(f"Received request: {request}")
    # Echo input_text in output_text
    output_text = None
    streaming = False
    for tensor in request["tensors"]:
        if tensor["metadata"]["name"] == "input_text":
            input_text_str = "".join(map(chr, tensor["data"]["values"][0]))
            print(f"Input text: {input_text_str}")
            output_text = tensor
            output_text["metadata"]["name"] = "output_text"
        if tensor["metadata"]["name"] == "streaming":
            streaming = tensor["data"]["values"][0]
    if output_text is None:
        raise ValueError("input_text tensor not found in request")
    if streaming:
        for i in range(len(output_text["data"]["values"][0])):
            chunk = {
                "model": request["model"],
                "tensors": [
                    {
                        "metadata": output_text["metadata"],
                        "data": {
                            "data_type": output_text["data"]["data_type"],
                            "values": [[output_text["data"]["values"][0][i]]],
                        },
                    }
                ],
            }
            yield chunk
    else:
        yield {"model": request["model"], "tensors": [output_text]}


if __name__ == "__main__":
    uvloop.run(test_register())
