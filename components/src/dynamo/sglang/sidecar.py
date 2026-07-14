# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-managed launcher for Dynamo's native remote backend."""

import sys

from dynamo._core.backend import _run_sglang_remote


def main(argv: list[str] | None = None) -> None:
    """Run the remote backend against SGLang's injected gRPC endpoint."""
    _run_sglang_remote(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
