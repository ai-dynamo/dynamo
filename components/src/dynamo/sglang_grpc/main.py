# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Pass-through entrypoint for the SGLang gRPC bridge worker.

Forwards `sys.argv[1:]` to `dynamo._core.run_sglang_bridge_worker`. The
caller is responsible for starting `sglang.launch_server --enable-grpc`
separately and passing its address via `--sglang-grpc-endpoint`.
"""

import logging
import sys

from dynamo._core import run_sglang_bridge_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging(service_name="dynamo.sglang_grpc")
logger = logging.getLogger(__name__)


def main() -> None:
    run_sglang_bridge_worker(sys.argv[1:])


if __name__ == "__main__":
    main()
