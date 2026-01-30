# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main entry point for TensorRT-LLM Video Diffusion worker."""

import asyncio
import logging
import signal

from dynamo.llm import ModelInput, ModelType, register_llm
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

from dynamo.trtllm_diffusion.args import VideoConfig, parse_args
from dynamo.trtllm_diffusion.engine import WanDiffusionEngine
from dynamo.trtllm_diffusion.request_handlers import VideoGenerationHandler

configure_dynamo_logging()

logger = logging.getLogger(__name__)


async def graceful_shutdown(runtime: DistributedRuntime, shutdown_event: asyncio.Event) -> None:
    """Handle graceful shutdown on signal.

    Args:
        runtime: The Dynamo distributed runtime.
        shutdown_event: Event to signal shutdown to worker tasks.
    """
    logger.info("Received shutdown signal, initiating graceful shutdown")
    shutdown_event.set()
    runtime.shutdown()
    logger.info("Distributed runtime shutdown complete")


async def init_video_worker(
    runtime: DistributedRuntime,
    config: VideoConfig,
    shutdown_event: asyncio.Event,
) -> None:
    """Initialize and run the video generation worker.

    Args:
        runtime: The Dynamo distributed runtime.
        config: Video generation configuration.
        shutdown_event: Event to signal shutdown.
    """
    logger.info(f"Initializing video generation worker with config: {config}")

    # Get the component and endpoint from the runtime
    component = runtime.namespace(config.namespace).component(config.component)
    endpoint = component.endpoint(config.endpoint)

    # Initialize the video generation engine
    engine = WanDiffusionEngine(config)
    await engine.initialize()

    # Create the request handler
    handler = VideoGenerationHandler(component, engine, config)

    # Register the model with Dynamo's discovery system
    model_name = config.served_model_name or config.model_path

    # Use ModelType.Videos - this is available after merging PR #5793
    # If Videos doesn't exist (older Dynamo), fall back to Chat for testing
    model_type = getattr(ModelType, "Videos", ModelType.Chat)
    if model_type == ModelType.Chat:
        logger.warning(
            "ModelType.Videos not available, using ModelType.Chat as fallback. "
            "This may not work correctly with the HTTP frontend."
        )

    logger.info(f"Registering model '{model_name}' with ModelType.{model_type.name}")

    await register_llm(
        ModelInput.Text,
        model_type,
        endpoint,
        config.model_path,
        model_name,
    )

    logger.info(f"Model registered, serving endpoint: {config.endpoint}")

    # Serve the endpoint
    try:
        await endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
        )
    except asyncio.CancelledError:
        logger.info("Endpoint serving cancelled")
    except Exception as e:
        logger.error(f"Error serving endpoint: {e}", exc_info=True)
        raise
    finally:
        handler.cleanup()
        engine.cleanup()


async def worker() -> None:
    """Main worker entry point.

    This function:
    1. Parses command-line arguments
    2. Creates the Dynamo distributed runtime
    3. Sets up signal handlers for graceful shutdown
    4. Initializes and runs the video generation worker
    """
    # Parse configuration
    config = parse_args()
    logger.info(f"Starting TensorRT-LLM Video Diffusion worker")
    logger.info(f"Configuration: {config}")

    # Get the event loop
    loop = asyncio.get_running_loop()

    # Create the distributed runtime
    runtime = DistributedRuntime(
        loop,
        config.store_kv,
        config.request_plane,
        enable_nats=False,
    )

    # Create shutdown event
    shutdown_event = asyncio.Event()

    # Set up signal handlers for graceful shutdown
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(graceful_shutdown(runtime, shutdown_event)),
        )

    logger.info("Signal handlers configured for graceful shutdown")

    # Initialize and run the video worker
    try:
        await init_video_worker(runtime, config, shutdown_event)
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("Worker shutdown complete")


if __name__ == "__main__":
    import uvloop
    uvloop.run(worker())
