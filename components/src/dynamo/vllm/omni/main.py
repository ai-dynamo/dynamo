# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Omni worker entrypoint for python -m dynamo.vllm.omni."""

import asyncio
import logging
import os

import uvloop

from dynamo import prometheus_names
from dynamo.common.config_dump import dump_config
from dynamo.common.storage import get_fs
from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.output_modalities import get_output_modalities
from dynamo.common.utils.runtime import create_runtime
from dynamo.llm import ModelInput, ModelType, fetch_model, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.health_check import VllmOmniHealthCheckPayload
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.tts_utils import ensure_dummy_tokenizer_for_tts

from .args import OmniConfig, parse_omni_args

configure_dynamo_logging()
logger = logging.getLogger(__name__)
shutdown_endpoints: list = []


async def init_omni(
    runtime: DistributedRuntime, config: OmniConfig, shutdown_event: asyncio.Event
):
    """Initialize Omni worker for multi-stage pipeline generation."""
    from dynamo.vllm.omni import OmniHandler

    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    media_fs = (
        get_fs(config.media_output_fs_url) if config.media_output_fs_url else None
    )

    handler = OmniHandler(
        runtime=runtime,
        config=config,
        default_sampling_params={},
        shutdown_event=shutdown_event,
        media_output_fs=media_fs,
        media_output_http_url=config.media_output_http_url,
    )

    logger.info("Omni worker initialized for model: %s", config.model)

    setup_metrics_collection(config, generate_endpoint, logger)

    if config.engine_args.data_parallel_rank:
        logger.info(
            "Non-leader DP rank %d; skipping endpoint registration",
            config.engine_args.data_parallel_rank,
        )
        await shutdown_event.wait()
        return

    model_type = get_output_modalities(config.output_modalities, config.model)
    if model_type is None:
        model_type = ModelType.Images

    # Audio/TTS models (e.g., Qwen3-TTS) don't ship a standard tokenizer.json,
    # which causes register_model to fail when building the ModelDeploymentCard.
    # Create a minimal placeholder so the Rust card loader doesn't bail.
    if "audio" in config.output_modalities:
        ensure_dummy_tokenizer_for_tts(config.model)

    await register_model(
        ModelInput.Text,
        model_type,
        generate_endpoint,
        config.model,
        config.served_model_name,
        kv_cache_block_size=config.engine_args.block_size,
    )

    logger.info("Starting to serve Omni worker endpoint...")

    health_check_payload = (
        await VllmOmniHealthCheckPayload.create(handler.engine_client)
    ).to_dict()

    try:
        await generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=[
                (
                    prometheus_names.labels.MODEL,
                    config.served_model_name or config.model,
                ),
                (
                    prometheus_names.labels.MODEL_NAME,
                    config.served_model_name or config.model,
                ),
            ],
            health_check_payload=health_check_payload,
        )
    except Exception as e:
        logger.error("Failed to serve Omni endpoint: %s", e)
        raise
    finally:
        logger.debug("Cleaning up Omni worker")
        handler.cleanup()


async def worker():
    config = parse_omni_args()

    dump_config(config.dump_config_to, config)

    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model

    if not os.path.exists(config.model):
        await fetch_model(config.model)

    shutdown_event = asyncio.Event()
    runtime, loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
        use_kv_events=False,
    )

    install_signal_handlers(loop, runtime, shutdown_endpoints, shutdown_event)

    await init_omni(runtime, config, shutdown_event)
    logger.debug("Omni worker completed, exiting...")


def main():
    uvloop.run(worker())
