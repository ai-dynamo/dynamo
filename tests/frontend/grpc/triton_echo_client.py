# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys

import numpy as np
import tritonclient.grpc as grpcclient

def run_infer():
    server_url = "localhost:8000"
    try:
        triton_client = grpcclient.InferenceServerClient(url=server_url)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = "echo"

    # Infer
    inputs = []
    inputs.append(grpcclient.InferInput("INPUT0", [16], "INT32"))
    inputs.append(grpcclient.InferInput("INPUT1", [16], "BYTES"))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    input0_data = np.arange(start=0, stop=16, dtype=np.int32).reshape([16])
    input1_data = np.array(
        [str(x).encode("utf-8") for x in input0_data.reshape(input0_data.size)], dtype=np.object_
    ).reshape([16])

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    # Test with outputs
    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs
    )

    # Get the output arrays from the results
    output0_data = results.as_numpy("INPUT0")
    output1_data = results.as_numpy("INPUT1")

    assert np.array_equal(input0_data, output0_data)
    assert np.array_equal(input1_data, output1_data)