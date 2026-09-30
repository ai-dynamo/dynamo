# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess entry for the test GMS RPC server.

Hosting GMSRPCServer in pytest pins its CUDA state (contexts, /dev/nvidia*
fds, driver allocations) to the pytest PID for the rest of the session.
Running it here as a subprocess confines that state to a process that dies
with the test module.
"""

from __future__ import annotations

import argparse
import asyncio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("socket_path")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    # Deferred to keep import cost out of --help and pytest collection.
    from gpu_memory_service.server.rpc import GMSRPCServer

    asyncio.run(GMSRPCServer(args.socket_path, device=args.device).serve())


if __name__ == "__main__":
    main()
