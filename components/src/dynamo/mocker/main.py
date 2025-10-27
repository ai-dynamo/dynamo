#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.mocker --model-path /data/models/Qwen3-0.6B`
# Now supports vLLM-style individual arguments for MockEngineArgs

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
        # Create engine configuration
        entrypoint_args = EntrypointArgs(
            engine_type=EngineType.Mocker,
            model_path=args.model_path,
            model_name=args.model_name,
            endpoint_id=args.endpoint,
            extra_engine_args=extra_engine_args_path,
            is_prefill=args.is_prefill_worker,
        )

        # Create and run the engine
        # NOTE: only supports dyn endpoint for now
        engine_config = await make_engine(runtime, entrypoint_args)
        await run_input(runtime, args.endpoint, engine_config)
    finally:
        # Clean up temporary file if we created one
        if not args.extra_engine_args and extra_engine_args_path.exists():
            try:
                extra_engine_args_path.unlink()
                logger.debug(f"Cleaned up temporary file {extra_engine_args_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
