#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.mocker --model-path /data/models/Qwen3-0.6B`
# Now supports vLLM-style individual arguments for MockEngineArgs

import asyncio
import logging
import os

import uvloop

os.environ.setdefault("DYN_COMPUTE_THREADS", "0")

from dynamo.llm import EngineType, EntrypointArgs, make_engine, run_input
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

from .args import create_temp_engine_args_file, parse_args

configure_dynamo_logging()
logger = logging.getLogger(__name__)


async def worker():
    """Main worker function that launches mocker instances.

    Each mocker gets its own DistributedRuntime instance for true isolation,
    while still sharing the same event loop and tokio runtime.
    """
    args = parse_args()

    # Handle extra_engine_args: either use provided file or create from CLI args
    if args.extra_engine_args:
        # User provided explicit JSON file
        extra_engine_args_path = args.extra_engine_args
        logger.info(f"Using provided MockEngineArgs from {extra_engine_args_path}")
    else:
        # Create temporary JSON file from CLI arguments
        extra_engine_args_path = create_temp_engine_args_file(args)
        logger.info("Created MockEngineArgs from CLI arguments")

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

    for worker_id in range(args.num_workers):
        logger.info(f"Creating mocker worker {worker_id + 1}/{args.num_workers}")

        # Create a separate DistributedRuntime for this worker (on same event loop)
        runtime = DistributedRuntime(loop, args.store_kv, args.request_plane)
        runtimes.append(runtime)

        # Create EntrypointArgs for this worker
        entrypoint_args = EntrypointArgs(
            engine_type=EngineType.Mocker,
            model_path=args.model_path,
            model_name=args.model_name,
            endpoint_id=args.endpoint,
            extra_engine_args=extra_engine_args_path,
            is_prefill=args.is_prefill_worker,
        )

        # Create the engine with this worker's isolated runtime
        engine_config = await make_engine(runtime, entrypoint_args)

        # run_input returns a Rust Future (not a Python coroutine)
        future = run_input(runtime, args.endpoint, engine_config)
        futures.append(future)

    logger.info(f"All {args.num_workers} mocker worker(s) created and running")

    try:
        # Wait for all futures to complete
        await asyncio.gather(*futures, return_exceptions=True)
    finally:
        # Clean up runtimes
        logger.info("Shutting down DistributedRuntime instances")
        for runtime in runtimes:
            runtime.shutdown()


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
