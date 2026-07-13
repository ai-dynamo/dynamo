# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the vLLM backend.

Usage:
    python -m dynamo.vllm.unified_main <vllm args>

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""


def main():
    from dynamo.common.snapshot.restore_context import maybe_run_restore_standby_mode

    maybe_run_restore_standby_mode()

    # Import vLLM only after the restore standby has captured the target pod's
    # environment and replaced itself with an inert process.
    from dynamo.common.backend.run import run
    from dynamo.common.backend.worker import WorkerConfig
    from dynamo.vllm.args import parse_args
    from dynamo.vllm.headless import run_dynamo_headless
    from dynamo.vllm.llm_engine import VllmLLMEngine

    # Headless secondary nodes (multi-node TP/PP with --data-parallel-backend
    # mp) run vLLM workers only and never touch the DistributedRuntime, so they
    # bypass the Worker/engine lifecycle entirely. Intercept before run() —
    # which would otherwise build the full backend and register an endpoint.
    config = parse_args(fpm_trace_relay_supported=False)
    if config.headless:
        run_dynamo_headless(config)
        return

    async def from_parsed_args(
        argv: list[str] | None = None,
    ) -> tuple[VllmLLMEngine, WorkerConfig]:
        return await VllmLLMEngine.from_args(argv, config=config)

    # Construct from the already-parsed config without widening BaseEngine's
    # generic from_args(argv) contract with vLLM-specific arguments.
    run(VllmLLMEngine, engine_factory=from_parsed_args)


if __name__ == "__main__":
    main()
