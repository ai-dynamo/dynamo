# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the vLLM backend.

Usage:
    python -m dynamo.vllm.unified_main <vllm args>

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

from dynamo.common.backend.run import run
from dynamo.vllm.args import parse_args
from dynamo.vllm.headless import run_dynamo_headless
from dynamo.vllm.llm_engine import VllmLLMEngine


def main():
    # Headless secondary nodes (multi-node TP/PP with --data-parallel-backend
    # mp) run vLLM workers only and never touch the DistributedRuntime, so they
    # bypass the Worker/engine lifecycle entirely. Intercept before run() —
    # which would otherwise build the full backend and register an endpoint.
    config = parse_args()
    if config.headless:
        run_dynamo_headless(config)
        return
    # Thread the already-parsed config through so from_args doesn't re-parse.
    run(VllmLLMEngine, config=config)


if __name__ == "__main__":
    main()
