# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Pass-through entrypoint: forwards `sys.argv[1:]` to the Rust bridge."""

import sys

from dynamo._core import run_sglang_bridge_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging(service_name="dynamo.sglang_grpc")


def main() -> None:
    run_sglang_bridge_worker(sys.argv[1:])


if __name__ == "__main__":
    main()
