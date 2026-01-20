#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Simple test client for the mock KServe gRPC server example. The script uses
# the Triton gRPC client to issue a ModelInfer request against the running
# server.

import argparse

import numpy as np
import tritonclient.grpc as triton_grpc
from google.protobuf.json_format import MessageToDict
from tritonclient.utils import InferenceServerException


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a test request to the KServe gRPC mock server."
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
        default="mock_model",
        help="Model name to target (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        default="Hello from Dynamo!",
        help="Prompt text encoded into the BYTES input tensor.",
    )

    args = parser.parse_args()

    target = f"{args.host}:{args.port}"
    client = triton_grpc.InferenceServerClient(url=target)

    text_input = triton_grpc.InferInput("text_input", [1], "BYTES")
    text_input.set_data_from_numpy(
        np.array([args.prompt.encode("utf-8")], dtype=object)
    )

    try:
        response = client.infer(args.model, inputs=[text_input])
    except (
        InferenceServerException
    ) as err:  # pragma: no cover - informational error path
        raise SystemExit(f"Inference request failed: {err}") from err

    response_dict = MessageToDict(
        response.get_response(),
        preserving_proto_field_name=True,
    )
    print("Received response:")
    print(response_dict)


if __name__ == "__main__":
    main()
