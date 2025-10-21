#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Simple test client for the mock KServe gRPC server example. The script
# generates Python stubs from the repo's proto files at runtime so it can be
# used without pre-generating code.

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import grpc
from google.protobuf.json_format import MessageToDict

try:
    from grpc_tools import protoc
except ImportError as exc:  # pragma: no cover - informational error path
    raise SystemExit(
        "grpcio-tools is required. Install it with "
        "'pip install grpcio grpcio-tools'."
    ) from exc

TEMP_PROTO_DIR: tempfile.TemporaryDirectory[str] | None = None


def _generate_proto_modules() -> Tuple[object, object]:
    """Generate Python modules from the repository's KServe protos."""
    global TEMP_PROTO_DIR
    if TEMP_PROTO_DIR is None:
        TEMP_PROTO_DIR = tempfile.TemporaryDirectory()

        proto_root = (
            Path(__file__).resolve().parents[4] / "llm" / "src" / "grpc" / "protos"
        )
        kserve_proto = proto_root / "kserve.proto"
        model_config_proto = proto_root / "model_config.proto"

        if not kserve_proto.exists():
            raise SystemExit(f"Unable to locate kserve.proto at {kserve_proto}")

        args = [
            "protoc",
            f"-I{proto_root}",
            f"--python_out={TEMP_PROTO_DIR.name}",
            f"--grpc_python_out={TEMP_PROTO_DIR.name}",
            str(kserve_proto),
            str(model_config_proto),
        ]
        if protoc.main(args) != 0:
            raise SystemExit("Failed to generate Python gRPC stubs for kserve.proto")

        sys.path.insert(0, TEMP_PROTO_DIR.name)

    import kserve_pb2  # type: ignore
    import kserve_pb2_grpc  # type: ignore

    return kserve_pb2, kserve_pb2_grpc


def build_request(kserve_pb2, model: str, prompt: str):
    """Construct a ModelInferRequest for a simple BYTES input."""
    tensor_contents = kserve_pb2.InferTensorContents(bytes_contents=[prompt.encode()])
    input_tensor = kserve_pb2.ModelInferRequest.InferInputTensor(
        name="text_input",
        datatype="BYTES",
        shape=[1],
        contents=tensor_contents,
    )

    request = kserve_pb2.ModelInferRequest(model_name=model, inputs=[input_tensor])
    return request


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

    kserve_pb2, kserve_pb2_grpc = _generate_proto_modules()
    request = build_request(kserve_pb2, args.model, args.prompt)

    target = f"{args.host}:{args.port}"
    channel = grpc.insecure_channel(target)
    stub = kserve_pb2_grpc.GRPCInferenceServiceStub(channel)

    response = stub.ModelInfer(request)
    response_dict = MessageToDict(response, preserving_proto_field_name=True)
    print("Received response:")
    print(response_dict)


if __name__ == "__main__":
    main()
