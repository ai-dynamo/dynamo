#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Simple worker that echoes tensors and parameters for gRPC testing."""

import uvloop

from dynamo.llm import ModelInput, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker


@dynamo_worker(static=False)
async def tensor_params_worker(runtime: DistributedRuntime):
    """Worker that echoes tensors and parameters."""
    component = runtime.namespace("test").component("tensor")
    await component.create_service()
    endpoint = component.endpoint("generate")

    await register_llm(
        ModelInput.Tensor,
        ModelType.TensorBased,
        endpoint,
        "Qwen/Qwen3-0.6B",
        "tensor-params-model",
    )

    await endpoint.serve_endpoint(echo_generate)


async def echo_generate(request, context):
    """Echo tensors and add processed flag to parameters."""
    params = {}

    if "parameters" in request:
        params.update(request["parameters"])

    # Add worker flag in ParameterValue format
    params["processed"] = {"bool": True}

    # Echo tensors
    yield {
        "model": request["model"],
        "tensors": request["tensors"],
        "parameters": params,
    }


if __name__ == "__main__":
    uvloop.run(tensor_params_worker())
