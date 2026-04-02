#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test client for Triton Server backend via Dynamo KServe gRPC frontend.

Usage:
    # After starting the server with ./launch/identity.sh
    python src/client.py
    python src/client.py --model identity --shape 1 10
"""

import argparse

import numpy as np
import tritonclient.grpc as triton_grpc
from tritonclient.utils import InferenceServerException


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send inference requests to Triton model via Dynamo frontend"
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
        nargs="+",
        default=[1, 5],
        help="Input tensor shape (default: 1 5)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of inference iterations (default: %(default)s)",
    )

    args = parser.parse_args()

    target = f"{args.host}:{args.port}"
    print(f"Connecting to {target}...")

    try:
        client = triton_grpc.InferenceServerClient(url=target)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # Query model metadata
    print(f"\nQuerying model '{args.model}' metadata...")
    try:
        metadata = client.get_model_metadata(args.model)
        print(f"  Name: {metadata.name}")
        print(
            f"  Inputs: {[(i.name, i.datatype, list(i.shape)) for i in metadata.inputs]}"
        )
        print(
            f"  Outputs: {[(o.name, o.datatype, list(o.shape)) for o in metadata.outputs]}"
        )
    except InferenceServerException as e:
        print(f"  Could not get metadata: {e}")
        print("  Proceeding with default INPUT0/OUTPUT0...")

    # Generate input data
    shape = args.shape
    input_size = int(np.prod(shape))
    input_data = np.arange(1, input_size + 1, dtype=np.int32).reshape(shape)

    print(f"\nRunning {args.iterations} inference iteration(s)...")
    for i in range(args.iterations):
        print(f"\n--- Iteration {i + 1} ---")
        print(f"Input shape: {shape}")
        print(f"Input data:\n{input_data}")

        # Create input tensor
        input_tensor = triton_grpc.InferInput("INPUT0", shape, "INT32")
        input_tensor.set_data_from_numpy(input_data)

        try:
            response = client.infer(args.model, inputs=[input_tensor])

            # Extract output
            output_data = response.as_numpy("OUTPUT0")
            print(f"Output shape: {output_data.shape}")
            print(f"Output data:\n{output_data}")

            # Verify identity model (output should equal input)
            if np.array_equal(input_data, output_data):
                print("✓ Identity verification passed")
            else:
                print("✗ Identity verification failed - output differs from input")

        except InferenceServerException as e:
            print(f"Inference failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
