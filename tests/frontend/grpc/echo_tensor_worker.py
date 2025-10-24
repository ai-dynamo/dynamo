#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `TEST_END_TO_END=1 python test_tensor.py` to run this worker as tensor based echo worker.


from base64 import b64encode

import uvloop
from model_config_pb2 import ModelConfig

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker


@dynamo_worker(static=False)
async def echo_tensor_worker(runtime: DistributedRuntime):
    component = runtime.namespace("tensor").component("echo")
    await component.create_service()

    endpoint = component.endpoint("generate")

    triton_model_config = ModelConfig()
    triton_model_config.name = "echo"
    triton_model_config.platform = "custom"
    input_tensor = triton_model_config.input.add()
    input_tensor.name = "input"
    input_tensor.data_type = "TYPE_STRING"
    input_tensor.dims.extend([-1])
    optional_input_tensor = triton_model_config.input.add()
    optional_input_tensor.name = "optional_input"
    optional_input_tensor.data_type = "TYPE_INT32"
    optional_input_tensor.dims.extend([-1])
    optional_input_tensor.optional = True
    output_tensor = triton_model_config.output.add()
    output_tensor.name = "dummy_output"
    output_tensor.data_type = "TYPE_STRING"
    output_tensor.dims.extend([-1])
    triton_model_config.model_transaction_policy.decoupled = True

    # Serialize and base64-encode the Triton model config
    b64_config = b64encode(triton_model_config.SerializeToString())

    model_config = {
        "name": "",
        "inputs": [],
        "outputs": [],
        "triton_model_config": b64_config,
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
    # [NOTE] gluo: currently there is no frontend side
    # validation between model config and actual request,
    # so any request will reach here and be echoed back.
    print(f"Echoing request: {request}")
    yield {"model": request["model"], "tensors": request["tensors"]}


if __name__ == "__main__":
    uvloop.run(echo_tensor_worker())
