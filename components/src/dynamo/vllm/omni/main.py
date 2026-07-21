# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Omni worker entrypoint for python -m dynamo.vllm.omni."""

import asyncio
import logging
import os

import uvloop

from dynamo import prometheus_names
from dynamo.common.config_dump import dump_config
from dynamo.common.rl import first_endpoint_response
from dynamo.common.storage import get_fs
from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.output_modalities import get_output_modalities
from dynamo.common.utils.runtime import create_runtime
from dynamo.llm import ModelInput, ModelType, WorkerType, fetch_model, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.handlers import get_lora_manager
from dynamo.vllm.health_check import VllmOmniHealthCheckPayload
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.realtime_utils import init_omni_realtime
from dynamo.vllm.omni.stage_router import init_omni_stage_router
from dynamo.vllm.omni.stage_worker import init_omni_stage

from .args import OmniConfig, parse_omni_args

configure_dynamo_logging()
logger = logging.getLogger(__name__)
shutdown_endpoints: list = []


def _register_lora_engine_routes(runtime, handler) -> None:
    route_handlers = {
        "load_lora": handler.load_lora,
        "unload_lora": handler.unload_lora,
        "list_loras": handler.list_loras,
    }

    for route_name, endpoint_handler in route_handlers.items():

        async def _engine_route(
            body: dict,
            endpoint_handler=endpoint_handler,
        ) -> dict:
            return await first_endpoint_response(endpoint_handler, body)

        # Register under update/<name> prefix per unified worker convention
        # to match system status server's call_lora_endpoint lookup
        runtime.register_engine_route(f"update/{route_name}", _engine_route)

    logger.info(
        "Registered LoRA engine routes: %s",
        ", ".join(f"update/{name}" for name in route_handlers),
    )


async def init_omni(
    runtime: DistributedRuntime, config: OmniConfig, shutdown_event: asyncio.Event
):
    """Initialize Omni worker for multi-stage pipeline generation."""
    # Intentional function-local import to avoid omni package import cycles.
    from dynamo.vllm.omni import OmniHandler

    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    engine_lora_enabled = bool(getattr(config.engine_args, "enable_lora", False))
    lora_enabled = engine_lora_enabled and (get_lora_manager() is not None)
    if engine_lora_enabled and not lora_enabled:
        logger.warning(
            "LoRA is enabled in engine args, but LoRA endpoints are disabled "
            "because LoRAManager is unavailable. Ensure DYN_LORA_ENABLED=true "
            "and LoRAManager initialization succeeds."
        )
    load_lora_endpoint = None
    unload_lora_endpoint = None
    list_loras_endpoint = None
    if lora_enabled:
        load_lora_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.load_lora"
        )
        unload_lora_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.unload_lora"
        )
        list_loras_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.list_loras"
        )
        shutdown_endpoints.extend(
            [load_lora_endpoint, unload_lora_endpoint, list_loras_endpoint]
        )

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
        generate_endpoint=generate_endpoint,
    )

    logger.info("Omni worker initialized for model: %s", config.model)

    if lora_enabled:
        _register_lora_engine_routes(runtime, handler)

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

    try:
        await register_model(
            ModelInput.Text,
            model_type,
            generate_endpoint,
            config.model,
            config.served_model_name,
            kv_cache_block_size=config.engine_args.block_size,
            # Omni workers serve the full multi-stage pipeline behind one
            # endpoint; there is no prefill/decode split visible to the
            # frontend, so they register as Aggregated.
            worker_type=WorkerType.Aggregated,
            needs=[],
            max_gpu_lora_count=1 if engine_lora_enabled else None,
        )

        logger.info("Starting to serve Omni worker endpoint...")

        health_check_payload = (
            await VllmOmniHealthCheckPayload.create(handler.engine_client)
        ).to_dict()

        model_metrics_labels = [
            (
                prometheus_names.labels.MODEL,
                config.served_model_name or config.model,
            ),
            (
                prometheus_names.labels.MODEL_NAME,
                config.served_model_name or config.model,
            ),
        ]

        serve_tasks = [
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=model_metrics_labels,
                health_check_payload=health_check_payload,
            )
        ]
        if lora_enabled:
            serve_tasks.extend(
                [
                    load_lora_endpoint.serve_endpoint(
                        handler.load_lora,
                        metrics_labels=model_metrics_labels,
                    ),
                    unload_lora_endpoint.serve_endpoint(
                        handler.unload_lora,
                        metrics_labels=model_metrics_labels,
                    ),
                    list_loras_endpoint.serve_endpoint(
                        handler.list_loras,
                        metrics_labels=model_metrics_labels,
                    ),
                ]
            )

        await asyncio.gather(*serve_tasks)
    except Exception as e:
        logger.error("Omni worker failed: %s", e)
        raise
    finally:
        logger.debug("Cleaning up Omni worker")
        handler.cleanup()


async def worker():
    config = parse_omni_args()

    dump_config(config.dump_config_to, config)

    if getattr(config.engine_args, "enable_lora", False):
        if "DYN_LORA_ENABLED" not in os.environ:
            os.environ["DYN_LORA_ENABLED"] = "true"

    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model

    if not os.path.exists(config.model):
        await fetch_model(config.model)

    shutdown_event = asyncio.Event()
    runtime, loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
    )

    install_signal_handlers(loop, runtime, shutdown_endpoints, shutdown_event)

    if config.stage_id is not None:
        await init_omni_stage(runtime, config, shutdown_endpoints, shutdown_event)
        logger.debug("init_omni_stage completed (stage %d)", config.stage_id)
    elif config.omni_router:
        await init_omni_stage_router(runtime, config, shutdown_endpoints)
        logger.debug("init_omni_stage_router completed")
    elif config.realtime:
        await init_omni_realtime(runtime, config, shutdown_endpoints, shutdown_event)
        logger.debug("init_omni_realtime completed, exiting...")
    else:
        await init_omni(runtime, config, shutdown_event)
        logger.debug("Omni worker completed, exiting...")


def main():
    uvloop.run(worker())
