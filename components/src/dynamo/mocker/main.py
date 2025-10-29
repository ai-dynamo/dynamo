#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.mocker --model-path /data/models/Qwen3-0.6B`
# Now supports vLLM-style individual arguments for MockEngineArgs

import asyncio
import logging

import uvloop

from dynamo.llm import EngineType, EntrypointArgs, make_engine, run_input
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .args import create_temp_engine_args_file, parse_args

configure_dynamo_logging()
logger = logging.getLogger(__name__)


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
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
        # Launch workers (works for both single and multi-worker cases)
        logger.info(
            f"Launching {args.num_workers} mocker worker(s) in the same process"
        )
        await launch_workers(runtime, args, extra_engine_args_path)
    finally:
        # Clean up temporary file if we created one
        if not args.extra_engine_args and extra_engine_args_path.exists():
            try:
                extra_engine_args_path.unlink()
                logger.debug(f"Cleaned up temporary file {extra_engine_args_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")


async def launch_workers(runtime, args, extra_engine_args_path):
    """Launch mocker worker(s) in the same process.

    All workers share the same tokio runtime and thread pool, but each gets
    a unique lease_id from the DistributedRuntime, making them discoverable
    as separate instances on the same endpoint.
    """
    tasks = []

    for worker_id in range(args.num_workers):
        logger.info(f"Creating mocker worker {worker_id + 1}/{args.num_workers}")

        # Create EntrypointArgs for this worker
        # All workers use the same endpoint - they'll be differentiated by lease_id
        entrypoint_args = EntrypointArgs(
            engine_type=EngineType.Mocker,
            model_path=args.model_path,
            model_name=args.model_name,
            endpoint_id=args.endpoint,
            extra_engine_args=extra_engine_args_path,
            is_prefill=args.is_prefill_worker,
        )

        # Create the engine
        engine_config = await make_engine(runtime, entrypoint_args)

        # Create the task for running this worker
        # run_input returns an awaitable that runs until cancelled
        task = asyncio.create_task(
            run_input(runtime, args.endpoint, engine_config),
            name=f"mocker-worker-{worker_id}",
        )
        tasks.append(task)

    logger.info(f"All {args.num_workers} mocker worker(s) created and running")

    # Wait for all tasks to complete (or until cancelled)
    # Using return_exceptions=True to handle individual worker failures gracefully
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Error in worker execution: {e}")
        # Cancel any remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        # Wait for cancellation to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
