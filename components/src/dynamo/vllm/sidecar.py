# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python launcher for Dynamo's native vLLM sidecar."""

import sys

from dynamo._jemalloc import maybe_preload


def main(argv: list[str] | None = None) -> None:
    """Run the Dynamo sidecar against a vLLM gRPC endpoint."""
    from dynamo._core import backend as _backend
    from dynamo.runtime.logging import configure_dynamo_logging

    configure_dynamo_logging(service_name="dynamo.vllm.sidecar")
    _backend._run_vllm_sidecar(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    # Re-execs with the bundled jemalloc, so it must run before dynamo._core is imported.
    maybe_preload()
    main()
