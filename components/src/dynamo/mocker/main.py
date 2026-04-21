#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.mocker --model-path /data/models/Qwen3-0.6B`
# Now supports vLLM-style individual arguments for MockEngineArgs

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path

import uvloop

os.environ.setdefault("DYN_COMPUTE_THREADS", "0")

from dynamo.common.utils.runtime import create_runtime
from dynamo.llm import EngineType, EntrypointArgs, fetch_model, make_engine, run_input
from dynamo.runtime.logging import configure_dynamo_logging

from .args import parse_args, resolve_planner_profile_data
from .config import (
    apply_worker_engine_args_overrides,
    build_runtime_config,
    load_mocker_engine_args,
)
from .utils.kv_cache import compute_kv_bytes_per_token

configure_dynamo_logging()
logger = logging.getLogger(__name__)


async def graceful_shutdown(runtimes: list):
    """
    Shutdown dynamo distributed runtime instances.
    The endpoints will be immediately invalidated so no new requests will be accepted.
    """
    logger.info("Received shutdown signal, shutting down DistributedRuntime instances")
    for runtime in runtimes:
        runtime.shutdown()
    logger.info("DistributedRuntime shutdown complete")


async def prefetch_model(model_path: str) -> None:
    """Pre-fetch model from HuggingFace to avoid rate limiting with many workers."""

    if Path(model_path).exists():
        logger.info(f"Using local model path: {model_path}")
        return

    logger.info(f"Pre-fetching model from HuggingFace: {model_path}")
    try:
        local_path = await fetch_model(model_path, ignore_weights=True)
        logger.info(f"Model cached at: {local_path}")
    except Exception as e:
        logger.warning(
            f"Failed to pre-fetch model: {e}. "
            "Workers will attempt individual downloads (may cause rate limiting)."
        )


async def worker():
    """Main worker function that launches mocker instances.

    Each mocker gets its own DistributedRuntime instance for true isolation,
    while still sharing the same event loop and tokio runtime.
    """
    args = parse_args()
    # Resolve planner-profile-data: convert profile results dir to NPZ if needed
    profile_data_result = resolve_planner_profile_data(args.planner_profile_data)
    args.planner_profile_data = profile_data_result.npz_path

    try:
        # Pre-fetch model once to avoid HuggingFace rate limiting when launching many workers
        if args.num_workers > 1 and args.model_path:
            await prefetch_model(args.model_path)

        engine_args = load_mocker_engine_args(args)
        logger.info(
            "Loaded MockEngineArgs from JSON file"
            if args.extra_engine_args
            else "Created MockEngineArgs from CLI arguments"
        )

        # Auto-compute kv_bytes_per_token from model config if not explicitly set
        if args.kv_bytes_per_token is None and args.model_path:
            args.kv_bytes_per_token = compute_kv_bytes_per_token(
                args.model_path, args.kv_cache_dtype
            )
        engine_args = apply_worker_engine_args_overrides(
            engine_args, kv_bytes_per_token=args.kv_bytes_per_token
        )

        logger.info(
            f"Launching {args.num_workers} mocker worker(s) with isolated DistributedRuntime instances"
        )
        await launch_workers(args, engine_args)
    finally:
        if profile_data_result is not None:
            del profile_data_result  # Triggers tmpdir cleanup via __del__


def compute_stagger_delay(num_workers: int, stagger_delay: float) -> float:
    """Compute the stagger delay based on worker count to give the frontend time to process registrations.
    Returns the delay in seconds between worker launches.
    """
    if stagger_delay >= 0:
        return stagger_delay

    if stagger_delay != -1:
        raise ValueError(
            f"Invalid --stagger-delay value: {stagger_delay}. "
            "Use -1 for auto mode, 0 to disable, or a positive value for explicit delay."
        )

    # Auto mode: stagger based on worker count
    if num_workers <= 32:
        return 0.0
    elif num_workers <= 128:
        return 0.1
    else:
        return 0.2


async def launch_workers(args: argparse.Namespace, base_engine_args):
    """Launch mocker worker(s) with isolated DistributedRuntime instances.

    Each worker gets its own DistributedRuntime, which means:
    - Separate etcd/NATS connections
    - Separate Component instances (no shared overhead)
    - Independent service registration and stats scraping
    - But still sharing the same tokio runtime (efficient)
    """
    futures = []
    runtimes = []

    stagger_delay = compute_stagger_delay(args.num_workers, args.stagger_delay)
    batch_size = 32
    batch_pause = 2.0

    if stagger_delay > 0:
        total_time = (args.num_workers - 1) * stagger_delay
        if args.num_workers > batch_size:
            num_batches = (args.num_workers + batch_size - 1) // batch_size
            total_time += batch_pause * (num_batches - 1)
        logger.info(
            f"Staggering {args.num_workers} worker launches: "
            f"{stagger_delay}s between workers, {batch_pause}s pause every {batch_size} workers "
            f"(estimated total: {total_time:.1f}s)"
        )

    needs_per_worker_overrides = bool(
        args.bootstrap_ports_list
        or args.zmq_kv_events_ports_list
        or args.zmq_replay_ports_list
    )

    for worker_id in range(args.num_workers):
        logger.info(f"Creating mocker worker {worker_id + 1}/{args.num_workers}")

        # Create a separate DistributedRuntime for this worker (on same event loop)

        runtime, loop = create_runtime(
            args.discovery_backend,
            args.request_plane,
            args.event_plane,
        )
        runtimes.append(runtime)

        if needs_per_worker_overrides:
            worker_engine_args = apply_worker_engine_args_overrides(
                base_engine_args,
                bootstrap_port=(
                    args.bootstrap_ports_list[worker_id]
                    if args.bootstrap_ports_list
                    else None
                ),
                zmq_kv_events_port=(
                    args.zmq_kv_events_ports_list[worker_id]
                    if args.zmq_kv_events_ports_list
                    else None
                ),
                zmq_replay_port=(
                    args.zmq_replay_ports_list[worker_id]
                    if args.zmq_replay_ports_list
                    else None
                ),
            )
        else:
            worker_engine_args = base_engine_args

        kv_cache_block_size, runtime_config = build_runtime_config(worker_engine_args)

        # Create EntrypointArgs for this worker
        entrypoint_args = EntrypointArgs(
            engine_type=EngineType.Mocker,
            model_path=args.model_path,
            model_name=args.model_name,
            endpoint_id=args.endpoint,
            context_length=0,
            extra_engine_args=None,
            mocker_engine_args=worker_engine_args,
            runtime_config=runtime_config,
            kv_cache_block_size=kv_cache_block_size,
            is_prefill=args.is_prefill_worker,
        )

        # Create the engine with this worker's isolated runtime
        engine_config = await make_engine(runtime, entrypoint_args)

        # run_input returns a Rust Future (not a Python coroutine)
        future = run_input(runtime, args.endpoint, engine_config)
        futures.append(future)

        # Stagger worker launches for large deployments
        if stagger_delay > 0 and worker_id < args.num_workers - 1:
            await asyncio.sleep(stagger_delay)
            # Add extra pause between batches to let frontend catch up
            if (worker_id + 1) % batch_size == 0:
                logger.info(
                    f"Batch {(worker_id + 1) // batch_size} complete, "
                    f"pausing {batch_pause}s for frontend to process..."
                )
                await asyncio.sleep(batch_pause)

    logger.info(f"All {args.num_workers} mocker worker(s) created and running")

    # Set up signal handler for graceful shutdown
    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtimes))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logger.info("Signal handlers set up for graceful shutdown")

    try:
        # Wait for all futures to complete
        await asyncio.gather(*futures, return_exceptions=True)
    finally:
        # Clean up runtimes (in case they weren't already shut down by signal handler)
        logger.info("Shutting down DistributedRuntime instances")
        for runtime in runtimes:
            runtime.shutdown()


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
