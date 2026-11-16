# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python launcher for Dynamo's native TensorRT-LLM sidecar."""

import sys

from dynamo._core import backend as _backend
from dynamo.runtime.logging import configure_dynamo_logging


def main(argv: list[str] | None = None) -> None:
    """Run the Dynamo sidecar against a TensorRT-LLM gRPC endpoint."""
    configure_dynamo_logging(service_name="dynamo.trtllm.sidecar")
    _backend._run_trtllm_sidecar(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
