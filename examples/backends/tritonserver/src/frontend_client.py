# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test client for the Triton identity model served through Dynamo KServe gRPC frontend.
# The script uses the Triton gRPC client to issue a ModelInfer request.
#
# Usage:
#   1. Start all services: ./start_all.sh
#   2. Wait for services to be ready (~30 seconds)
#   3. Run this client: python src/frontend_client.py
#   4. Check logs: docker compose logs -f

import argparse

import numpy as np
import tritonclient.grpc as triton_grpc
from google.protobuf.json_format import MessageToDict
from tritonclient.utils import InferenceServerException


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a test request to the Triton identity model via KServe gRPC frontend."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host serving the gRPC endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Port of the gRPC endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="identity",
        help="Model name to target (default: %(default)s)",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=[1, 5],
        help="Input tensor shape (default: 1 5)",
    )

    args = parser.parse_args()

    target = f"{args.host}:{args.port}"
    print(f"Connecting to {target}...")
    client = triton_grpc.InferenceServerClient(url=target)

    # Query model metadata
    model_outputs = []

    # try:
    #     metadata = client.get_model_metadata(args.model)
    # except InferenceServerException as err:
    #     raise SystemExit(f"Could not retrieve model metadata: {err}") from err

    metadata = client.get_model_metadata(args.model)

    print(metadata)

    # Store output metadata for later use
    model_outputs = [
        (out.name, out.datatype, list(out.shape)) for out in metadata.outputs
    ]

    # Generate input data based on shape
    shape = args.shape
    input_size = np.prod(shape)
    input_data = np.arange(1, input_size + 1, dtype=np.int32).reshape(shape)

    print("\nSending inference request...")
    print(f"  Input shape: {shape}")
    print(f"  Input data:\n{input_data}")

    input_tensor = triton_grpc.InferInput("INPUT0", shape, "INT32")
    input_tensor.set_data_from_numpy(input_data)

    try:
        response = client.infer(args.model, inputs=[input_tensor])
    except InferenceServerException as err:
        raise SystemExit(f"Inference request failed: {err}") from err

    # Parse response for display
    response_dict = MessageToDict(
        response.get_response(),
        preserving_proto_field_name=True,
    )

    print("\nReceived response:")
    print(f"  Model: {response_dict.get('model_name', 'N/A')}")
    print(f"  ID: {response_dict.get('id', 'N/A')}")

    # Extract output data using metadata or fallback
    output_names = (
        [name for name, _, _ in model_outputs] if model_outputs else ["OUTPUT0"]
    )

    for output_name in output_names:
        try:
            output_data = response.as_numpy(output_name)
            print(f"\n  Output '{output_name}':")
            print(f"    Shape: {output_data.shape}")
            print(f"    Data:\n{output_data}")
        except Exception as err:
            print(f"  Error extracting output '{output_name}': {err}")

    # Show available outputs if extraction failed
    if not model_outputs:
        print(
            f"  Available outputs: {[out.name for out in response.get_response().outputs]}"
        )


if __name__ == "__main__":
    main()
