#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.mocker --model-path /data/models/Qwen3-0.6B`
# Now supports vLLM-style individual arguments for MockEngineArgs

import asyncio
import json
import logging
import os
import signal
import tempfile
from pathlib import Path

import uvloop

os.environ.setdefault("DYN_COMPUTE_THREADS", "0")

from dynamo.llm import EngineType, EntrypointArgs, fetch_llm, make_engine, run_input
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

from .args import create_temp_engine_args_file, parse_args, resolve_planner_profile_data

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
        local_path = await fetch_llm(model_path, ignore_weights=True)
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

    # Handle extra_engine_args: either use provided file or create from CLI args
    if args.extra_engine_args:
        # User provided explicit JSON file
        extra_engine_args_path = args.extra_engine_args
        logger.info(f"Using provided MockEngineArgs from {extra_engine_args_path}")
    else:
        # Create temporary JSON file from CLI arguments
        extra_engine_args_path = create_temp_engine_args_file(args)
        logger.info("Created MockEngineArgs from CLI arguments")

    # Pre-fetch model once to avoid HuggingFace rate limiting when launching many workers
    if args.num_workers > 1 and args.model_path:
        await prefetch_model(args.model_path)

    try:
        logger.info(
            f"Launching {args.num_workers} mocker worker(s) with isolated DistributedRuntime instances"
        )
        await launch_workers(args, extra_engine_args_path)
    finally:
        # Clean up temporary file if we created one
        if not args.extra_engine_args and extra_engine_args_path.exists():
            try:
                extra_engine_args_path.unlink()
                logger.debug(f"Cleaned up temporary file {extra_engine_args_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")

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


async def launch_workers(args, extra_engine_args_path):
    """Launch mocker worker(s) with isolated DistributedRuntime instances.

    Each worker gets its own DistributedRuntime, which means:
    - Separate etcd/NATS connections
    - Separate Component instances (no shared overhead)
    - Independent service registration and stats scraping
    - But still sharing the same tokio runtime (efficient)
    """
    loop = asyncio.get_running_loop()
    futures = []
    runtimes = []
    per_worker_temp_files: list[Path] = []

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

    # Load base engine args if we need to create per-worker files with bootstrap_port
    base_engine_args = None
    if args.bootstrap_ports_list:
        with open(extra_engine_args_path) as f:
            base_engine_args = json.load(f)

    for worker_id in range(args.num_workers):
        logger.info(f"Creating mocker worker {worker_id + 1}/{args.num_workers}")

        # Create a separate DistributedRuntime for this worker (on same event loop)
        runtime = DistributedRuntime(loop, args.store_kv, args.request_plane)
        runtimes.append(runtime)

        # Determine which engine args file to use
        if args.bootstrap_ports_list:
            # Create per-worker temp file with this worker's bootstrap_port
            worker_args = base_engine_args.copy()
            worker_args["bootstrap_port"] = args.bootstrap_ports_list[worker_id]
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(worker_args, f)
                worker_engine_args_path = Path(f.name)
            per_worker_temp_files.append(worker_engine_args_path)
            logger.debug(
                f"Worker {worker_id}: using bootstrap_port {args.bootstrap_ports_list[worker_id]}"
            )
        else:
            worker_engine_args_path = extra_engine_args_path

        # Create EntrypointArgs for this worker
        entrypoint_args = EntrypointArgs(
            engine_type=EngineType.Mocker,
            model_path=args.model_path,
            model_name=args.model_name,
            endpoint_id=args.endpoint,
            extra_engine_args=worker_engine_args_path,
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

        # Clean up per-worker temp files
        for temp_file in per_worker_temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
