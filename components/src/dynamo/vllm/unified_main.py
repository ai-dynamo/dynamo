# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the vLLM backend.

Usage:
    python -m dynamo.vllm.unified_main <vllm args>

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

from dynamo.common.backend.engine import LLMEngine
from dynamo.common.backend.run import run
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.vllm.args import parse_args
from dynamo.vllm.encode_engine import VllmEncodeEngine
from dynamo.vllm.headless import run_dynamo_headless
from dynamo.vllm.llm_engine import VllmLLMEngine


def main():
    # Headless secondary nodes (multi-node TP/PP with --data-parallel-backend
    # mp) run vLLM workers only and never touch the DistributedRuntime, so they
    # bypass the Worker/engine lifecycle entirely. Intercept before run() —
    # which would otherwise build the full backend and register an endpoint.
    config = parse_args(fpm_trace_relay_supported=False)
    if config.headless:
        run_dynamo_headless(config)
        return

    engine_cls = (
        VllmEncodeEngine
        if config.disaggregation_mode == DisaggregationMode.ENCODE
        else VllmLLMEngine
    )

    async def from_parsed_args(
        argv: list[str] | None = None,
    ) -> tuple[LLMEngine, WorkerConfig]:
        return await engine_cls.from_args(argv, config=config)

    # Construct from the already-parsed config without widening BaseEngine's
    # generic from_args(argv) contract with vLLM-specific arguments.
    run(engine_cls, engine_factory=from_parsed_args)


if __name__ == "__main__":
    main()
