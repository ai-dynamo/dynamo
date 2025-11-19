#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `TEST_END_TO_END=1 python test_tensor.py` to run this worker as tensor based echo worker.


# Knowing the test will be run in environment that has tritonclient installed,
# which contain the generated file equivalent to model_config.proto.
import tritonclient.grpc.model_config_pb2 as mc
import uvloop

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker


@dynamo_worker()
async def echo_tensor_worker(runtime: DistributedRuntime):
    component = runtime.namespace("tensor").component("echo")
    await component.create_service()

    endpoint = component.endpoint("generate")

    triton_model_config = mc.ModelConfig()
    triton_model_config.name = "echo"
    triton_model_config.platform = "custom"
    input_tensor = triton_model_config.input.add()
    input_tensor.name = "input"
    input_tensor.data_type = mc.TYPE_STRING
    input_tensor.dims.extend([-1])
    optional_input_tensor = triton_model_config.input.add()
    optional_input_tensor.name = "optional_input"
    optional_input_tensor.data_type = mc.TYPE_INT32
    optional_input_tensor.dims.extend([-1])
    optional_input_tensor.optional = True
    output_tensor = triton_model_config.output.add()
    output_tensor.name = "dummy_output"
    output_tensor.data_type = mc.TYPE_STRING
    output_tensor.dims.extend([-1])
    triton_model_config.model_transaction_policy.decoupled = True

    model_config = {
        "name": "",
        "inputs": [],
        "outputs": [],
        "triton_model_config": triton_model_config.SerializeToString(),
    }
    runtime_config = ModelRuntimeConfig()
    runtime_config.set_tensor_model_config(model_config)

    # Internally the bytes string will be converted to List of int
    retrieved_model_config = runtime_config.get_tensor_model_config()
    retrieved_model_config["triton_model_config"] = bytes(
        retrieved_model_config["triton_model_config"]
    )
    assert model_config == retrieved_model_config

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
    # [NOTE] gluo: currently there is no frontend side
    # validation between model config and actual request,
    # so any request will reach here and be echoed back.
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
