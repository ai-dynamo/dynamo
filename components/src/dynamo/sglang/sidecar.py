# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-managed launcher for Dynamo's native sidecar."""

import sys

from dynamo._jemalloc import maybe_preload


def main(argv: list[str] | None = None) -> None:
    """Run the Dynamo sidecar against SGLang's injected gRPC endpoint."""
    from dynamo._core import backend as _backend
    from dynamo.runtime.logging import configure_dynamo_logging

    configure_dynamo_logging(service_name="dynamo.sglang.sidecar")
    _backend._run_sglang_sidecar(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    # Re-execs with the bundled jemalloc, so it must run before dynamo._core is imported.
    # Not in main(): SGLang's in-process --sidecar mode calls main() directly.
    maybe_preload()
    main()
