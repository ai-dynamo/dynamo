# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realtime (bidirectional) Omni worker initialization.

Serves a ``ModelType.Realtime`` model backed by vLLM-Omni's streaming engine
via ``serve_bidirectional_endpoint``. The frontend discovers it and installs a
typed realtime PushRouter; see ``realtime_handler.RealtimeOmniHandler`` for the
event translation.
"""

import asyncio
import logging

from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.speech_to_text.realtime.serving import OpenAIServingRealtime

from dynamo import prometheus_names
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.base_handler import BaseOmniHandler
from dynamo.vllm.omni.realtime_handler import RealtimeOmniHandler

from .args import OmniConfig

logger = logging.getLogger(__name__)


async def init_omni_realtime(
    runtime: DistributedRuntime,
    config: OmniConfig,
    shutdown_endpoints: list,
    shutdown_event: asyncio.Event,
) -> None:
    """Initialize and serve the realtime bidirectional Omni worker."""
    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint}"
    )
    shutdown_endpoints[:] = [generate_endpoint]

    # BaseOmniHandler builds the AsyncOmni engine from the same kwargs the unary
    # Omni worker uses; we only need its engine_client for the realtime bridge.
    base = BaseOmniHandler(
        runtime=runtime,
        config=config,
        default_sampling_params={},
        shutdown_event=shutdown_event,
    )

    sampling_params_list = streaming_sampling_params(base.engine_client)
    streaming_input_factory = build_streaming_input_factory(config, base.engine_client)

    handler = RealtimeOmniHandler(
        engine_client=base.engine_client,
        model_name=config.served_model_name or config.model,
        streaming_input_factory=streaming_input_factory,
        default_sampling_params_list=sampling_params_list,
    )

    logger.info("Realtime Omni worker initialized for model: %s", config.model)

    setup_metrics_collection(config, generate_endpoint, logger)

    if config.engine_args.data_parallel_rank:
        logger.info(
            "Non-leader DP rank %d; skipping endpoint registration",
            config.engine_args.data_parallel_rank,
        )
        await shutdown_event.wait()
        return

    model_label = config.served_model_name or config.model
    try:
        await register_model(
            ModelInput.Text,
            ModelType.Realtime,
            generate_endpoint,
            config.model,
            config.served_model_name,
            kv_cache_block_size=config.engine_args.block_size,
            # The realtime worker serves the full multi-stage pipeline behind one
            # endpoint, so it registers as Aggregated like the unary Omni worker.
            worker_type=WorkerType.Aggregated,
            needs=[],
        )

        logger.info("Starting to serve realtime Omni worker endpoint...")

        # No health_check_payload: serve_bidirectional_endpoint does not yet
        # support canary probes (the bidirectional engine is stateful and needs
        # a session.update-shaped payload); see the Rust binding's doc comment.
        await generate_endpoint.serve_bidirectional_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=[
                (prometheus_names.labels.MODEL, model_label),
                (prometheus_names.labels.MODEL_NAME, model_label),
            ],
        )
    except Exception as e:
        logger.error("Realtime Omni worker failed: %s", e)
        raise
    finally:
        logger.debug("Cleaning up realtime Omni worker")
        base.cleanup()


def build_streaming_input_factory(config: OmniConfig, engine_client):
    """Build the audio -> StreamingInput factory from vLLM utils."""
    model_name = config.served_model_name or config.model
    base_model_paths = [BaseModelPath(name=model_name, model_path=config.model)]
    serving_models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
        lora_modules=None,
    )
    serving_realtime = OpenAIServingRealtime(
        engine_client=engine_client,
        models=serving_models,
        request_logger=None,
    )
    return serving_realtime.transcribe_realtime


def streaming_sampling_params(engine_client) -> list | None:
    """Default per-stage sampling params coerced for streaming generation.

    vLLM-Omni requires streaming requests to emit incremental (delta) outputs;
    ``coerce_param_message_types`` flips the engine defaults accordingly. Falls
    back to ``None`` (engine defaults) if anything is unavailable.
    """
    try:
        from vllm_omni.entrypoints.utils import coerce_param_message_types

        defaults = list(engine_client.default_sampling_params_list or [])
        if not defaults:
            return None
        return coerce_param_message_types(defaults, is_streaming=True)
    except Exception as e:  # noqa: BLE001 - fall back to engine defaults
        logger.warning(
            "Could not coerce streaming sampling params; using engine defaults: %s",
            e,
        )
        return None
