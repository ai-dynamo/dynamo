# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import logging
import os

import numpy as np
import tritonclient.grpc.model_config_pb2 as mc
import tritonserver
import uvloop
from google.protobuf import text_format
from tritonclient.utils import triton_to_np_dtype

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="tritonserver", worker_id=0)

# Mapping from Triton dtype (uppercase) to Dynamo dtype (camelCase)
TRITON_TO_DYNAMO_DTYPE = {
    "BOOL": "Bool",
    "UINT8": "Uint8",
    "UINT16": "Uint16",
    "UINT32": "Uint32",
    "UINT64": "Uint64",
    "INT8": "Int8",
    "INT16": "Int16",
    "INT32": "Int32",
    "INT64": "Int64",
    "FP16": "Float16",
    "FP32": "Float32",
    "FP64": "Float64",
    "BYTES": "Bytes",
}


class RequestHandler:
    def __init__(self, tritonserver: tritonserver.Server, model: tritonserver.Model):
        self.tritonserver = tritonserver
        self.model = model

    async def generate(self, request: dict) -> dict:
        # Deserialize to numpy array
        logger.debug(f"Received request: {request}")

        inference_request = self.model.create_request()
        for tensor in request["tensors"]:
            logger.debug(f"Tensor: {tensor}")
            # Convert Triton dtype string ("INT32") to NumPy dtype (np.int32) for array construction
            np_dtype = triton_to_np_dtype(tensor["metadata"]["data_type"].upper())
            arr = np.array(tensor["data"]["values"], dtype=np_dtype).reshape(
                tensor["metadata"]["shape"]
            )
            inference_request.inputs[tensor["metadata"]["name"]] = arr

        inference_responses = self.model.async_infer(inference_request)
        async for inference_response in inference_responses:
            response_tensors = []
            for output in self.model.metadata()["outputs"]:
                output_data = np.from_dlpack(inference_response.outputs[output["name"]])
                response_arr = output_data
                # Convert Triton dtype (e.g., "INT32") to Dynamo dtype (e.g., "Int32")
                dtype_str = TRITON_TO_DYNAMO_DTYPE.get(
                    output["datatype"], output["datatype"]
                )
                response_tensors.append(
                    {
                        "metadata": {
                            "name": output["name"],
                            "shape": list(response_arr.shape),
                            "data_type": dtype_str,
                        },
                        "data": {
                            "data_type": dtype_str,
                            "values": response_arr.flatten().tolist(),
                        },
                    }
                )

            response = {
                "id": inference_response.request_id,
                "model": inference_response.model.name,
                "tensors": response_tensors,
            }

            yield response


@dynamo_worker()
async def triton_worker(runtime: DistributedRuntime, args: argparse.Namespace):
    logger.info("=" * 60)
    logger.info("Starting Triton Worker for Dynamo")
    logger.info("=" * 60)
    logger.info(
        f"Environment: ETCD_ENDPOINTS={os.environ.get('ETCD_ENDPOINTS', 'NOT SET')}"
    )
    logger.info(f"Environment: NATS_SERVER={os.environ.get('NATS_SERVER', 'NOT SET')}")
    logger.info(
        f"Environment: DYN_STORE_KV={os.environ.get('DYN_STORE_KV', 'NOT SET')}"
    )

    component = runtime.namespace("triton").component("tritonserver")
    logger.info("✓ Created component: triton/tritonserver")

    endpoint = component.endpoint("generate")
    logger.info("✓ Created endpoint: triton/tritonserver/generate")

    model_repository = args.model_repository
    model_name = args.model_name
    backend_dir = args.backend_directory

    logger.info(
        f"Initializing Triton Server with model_repository={model_repository}, backend_dir={backend_dir}"
    )
    server = tritonserver.Server(
        model_repository=model_repository,
        backend_directory=backend_dir,
        log_verbose=args.log_verbose,
        model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
    )
    server.start(wait_until_ready=True)
    logger.info("✓ Triton Server started")

    server.load(model_name)
    model = server.model(model_name)
    logger.info(f"✓ Model '{model_name}' loaded")

    # Read Triton model config from config.pbtxt
    config_path = f"{model_repository}/{model_name}/config.pbtxt"
    with open(config_path, "r") as f:
        triton_model_config = text_format.Parse(f.read(), mc.ModelConfig())

    logger.info(f"Loaded model config from {config_path}")

    # Set up model metadata for KServe frontend
    model_config = {
        "name": "",
        "inputs": [],
        "outputs": [],
        "triton_model_config": triton_model_config.SerializeToString(),
    }

    runtime_config = ModelRuntimeConfig()
    runtime_config.set_tensor_model_config(model_config)

    logger.info("Attempting to register model with Dynamo runtime...")
    # Use register_llm for tensor-based models (skips HuggingFace downloads)
    await register_llm(
        ModelInput.Tensor,
        ModelType.TensorBased,
        endpoint,
        model_name,  # model_path (used as display name for tensor-based models)
        runtime_config=runtime_config,
    )
    logger.info(
        f"✓ Successfully registered model '{model_name}' with endpoint triton/tritonserver/generate"
    )

    # Create handler and serve the endpoint
    handler = RequestHandler(server, model)
    logger.info("Starting to serve the endpoint...")

    await endpoint.serve_endpoint(handler.generate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton worker for Dynamo")
    parser.add_argument(
        "--model-repository",
        type=str,
        default="model_repo",
        help="Model repository directory",
    )
    parser.add_argument("--model-name", type=str, default="identity", help="Model name")
    parser.add_argument(
        "--backend-directory", type=str, default="backends", help="Backend directory"
    )
    parser.add_argument("--log-verbose", type=int, default=6, help="Log verbose level")
    args = parser.parse_args()

    uvloop.install()
    asyncio.run(triton_worker(args))
