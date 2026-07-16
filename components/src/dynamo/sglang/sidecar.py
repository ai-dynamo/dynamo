# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-managed launcher for Dynamo's native sidecar."""

import sys

from dynamo._core import backend as _backend

_run_sglang_sidecar = _backend._run_sglang_sidecar


def main(argv: list[str] | None = None) -> None:
    """Run the Dynamo sidecar against SGLang's injected gRPC endpoint."""
    _run_sglang_sidecar(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
