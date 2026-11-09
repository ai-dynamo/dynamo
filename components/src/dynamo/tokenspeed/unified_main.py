# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the TokenSpeed backend.

Usage:
    python -m dynamo.tokenspeed <tokenspeed args>

On a multi-node engine, run the same command on every node with its
``--node-rank``. The first node's process is the Dynamo worker; the others run
only their engine ranks, which dial the first node's handshake.
"""

from dynamo.common.backend.run import run
from dynamo.tokenspeed.args import parse_args
from dynamo.tokenspeed.llm_engine import (
    TokenspeedLLMEngine,
    is_follower_node,
    run_follower_node,
)


def main(argv: list[str] | None = None):
    config = parse_args(argv)
    if is_follower_node(config.server_args):
        run_follower_node(config)
        return

    async def engine_factory(_argv):
        return TokenspeedLLMEngine.from_config(config)

    run(TokenspeedLLMEngine, engine_factory=engine_factory)


if __name__ == "__main__":
    main()
