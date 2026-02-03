# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import signal
import sys
from typing import Any, Callable

import sglang as sgl
import uvloop

from dynamo.common.config_dump import dump_config
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.llm import ModelInput, ModelType
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang.args import Config, DisaggregationMode, parse_args
from dynamo.sglang.health_check import (
    SglangHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang.publisher import (
    DynamoSglangPublisher,
    setup_prometheus_registry,
    setup_sgl_metrics,
)
from dynamo.sglang.register import register_llm_with_readiness_gate
from dynamo.sglang.request_handlers import (
    DecodeWorkerHandler,
    DiffusionWorkerHandler,
    EmbeddingWorkerHandler,
    MultimodalEncodeWorkerHandler,
    MultimodalPrefillWorkerHandler,
    MultimodalProcessorHandler,
    MultimodalWorkerHandler,
    PrefillWorkerHandler,
)

configure_dynamo_logging()


async def _handle_non_leader_node(
    engine: sgl.Engine,
    publisher: DynamoSglangPublisher,
    metrics_task: asyncio.Task,
) -> None:
    """
    Handle non-leader node (node_rank >= 1) in multi-node deployments.

    Non-leader nodes run scheduler processes but don't handle requests directly.
    They still need:
    - KV event publishing (subscribe to local DP ranks, forward to NATS)
    - Metrics collection from local schedulers
    - Prometheus metrics exposure

    Args:
        engine: The SGLang engine instance.
        publisher: The DynamoSglangPublisher for metrics and KV events.
        metrics_task: The asyncio task running the metrics loop.
    """
    logging.info(
        f"Non-leader node detected (node_rank={engine.server_args.node_rank}). "
        "Running with metrics and KV event publishing for local DP ranks."
    )

    try:
        # Wait indefinitely - the process will be terminated via signal handlers
        await asyncio.Event().wait()
    finally:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass
        publisher.cleanup()


async def worker():
    config = await parse_args(sys.argv[1:])
    dump_config(config.dynamo_args.dump_config_to, config)

    # Setup GPU Memory Service if --load-format gms is used
    if config.server_args.load_format == "gms":
        from gpu_memory_service.integrations.sglang import setup_gms

        config.server_args.load_format = setup_gms(config.server_args)

    loop = asyncio.get_running_loop()

    # Set DYN_EVENT_PLANE environment variable based on config
    os.environ["DYN_EVENT_PLANE"] = config.dynamo_args.event_plane

    # NATS is needed when:
    # 1. Request plane is NATS, OR
    # 2. Event plane is NATS AND use_kv_events is True
    enable_nats = config.dynamo_args.request_plane == "nats" or (
        config.dynamo_args.event_plane == "nats" and config.dynamo_args.use_kv_events
    )

    runtime = DistributedRuntime(
        loop,
        config.dynamo_args.store_kv,
        config.dynamo_args.request_plane,
        enable_nats,
    )

    # Set up signal handlers using signal module to allow chaining
    shutdown_event = asyncio.Event()
    old_handlers = {}

    def signal_handler(signum, frame):
        """Handle SIGTERM/SIGINT and chain to previous handlers"""
        logging.info(f"Received signal {signum}, initiating graceful shutdown")
        # Schedule shutdown in the event loop from the signal handler context
        loop.call_soon_threadsafe(shutdown_event.set)

        # Chain to the old handler if it exists and is not default/ignore
        try:
            old_handler = old_handlers.get(signum)
            if old_handler and old_handler not in (
                signal.SIG_DFL,
                signal.SIG_IGN,
                signal.default_int_handler,
            ):
                old_handler(signum, frame)
        finally:
            runtime.shutdown()

    # Install signal handlers and save old ones for chaining
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, signal_handler)

    # Override add_signal_handler to prevent SGLang from installing its own signal handlers.
    # This ensures that Dynamo retains full control over the graceful shutdown flow,
    # rather than allowing SGLang (or other libraries) to overwrite or interfere
    # with the signal handlers that Dynamo sets up for SIGTERM/SIGINT.
    _orig_add = loop.add_signal_handler

    def watching_add_signal_handler(sig: int, callback: Callable, *args: Any):
        if sig in [signal.SIGTERM, signal.SIGINT]:
            logging.info(
                "SIGTERM/SIGINThandler changed via loop.add_signal_handler() is being suppressed by Dynamo"
            )
        else:
            return _orig_add(sig, callback, *args)

    loop.add_signal_handler = watching_add_signal_handler  # type: ignore[assignment]

    logging.info("Signal handlers set up for graceful shutdown (with chaining)")

    if config.dynamo_args.embedding_worker:
        await init_embedding(runtime, config, shutdown_event)
    elif config.dynamo_args.multimodal_processor:
        await init_multimodal_processor(runtime, config, shutdown_event)
    elif config.dynamo_args.multimodal_encode_worker:
        await init_multimodal_encode_worker(runtime, config, shutdown_event)
    elif config.dynamo_args.multimodal_worker:
        if config.serving_mode != DisaggregationMode.PREFILL:
            await init_multimodal_worker(runtime, config, shutdown_event)
        else:
            await init_multimodal_prefill_worker(runtime, config, shutdown_event)
    elif config.dynamo_args.diffusion_worker:
        await init_diffusion(runtime, config, shutdown_event)
    elif config.serving_mode != DisaggregationMode.PREFILL:
        await init(runtime, config, shutdown_event)
    else:
        await init_prefill(runtime, config, shutdown_event)


async def init(
    runtime: DistributedRuntime, config: Config, shutdown_event: asyncio.Event
):
    server_args, dynamo_args = config.server_args, config.dynamo_args

    # Prevent SGLang from blocking on non-leader nodes
    if server_args.node_rank >= 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # Setup metrics and KV events for ALL nodes (including non-leader)
    # Non-leader nodes need KV event publishing for their local DP ranks
    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, component, generate_endpoint
    )

    # Register Prometheus metrics callback if enabled
    if engine.server_args.enable_metrics:
        setup_prometheus_registry(engine, generate_endpoint)

    # Handle non-leader nodes (multi-node parallelism)
    # Non-leader nodes run schedulers and publish KV events, but don't serve requests
    if server_args.node_rank >= 1:
        await _handle_non_leader_node(engine, publisher, metrics_task)
        return

    # Readiness gate: requests wait until model is registered
    ready_event = asyncio.Event()

    handler = DecodeWorkerHandler(
        component, engine, config, publisher, generate_endpoint, shutdown_event
    )
    handler.register_engine_routes(runtime)

    health_check_payload = SglangHealthCheckPayload(
        engine, use_text_input=dynamo_args.use_sglang_tokenizer
    ).to_dict()

    logging.info(
        f"Registering model with endpoint types: {dynamo_args.dyn_endpoint_types}"
    )
    if (
        dynamo_args.custom_jinja_template
        and "chat" not in dynamo_args.dyn_endpoint_types
    ):
        logging.warning(
            "Custom Jinja template provided (--custom-jinja-template) but 'chat' not in --dyn-endpoint-types. "
            "The chat template will be loaded but the /v1/chat/completions endpoint will not be available."
        )

    try:
        # Start endpoint immediately and register model concurrently
        # Requests queue until ready_event is set (TODO: Part of new PR)
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
                health_check_payload=health_check_payload,
            ),
            register_llm_with_readiness_gate(
                engine,
                generate_endpoint,
                server_args,
                dynamo_args,
                output_type=parse_endpoint_types(dynamo_args.dyn_endpoint_types),
                readiness_gate=ready_event,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            logging.info("Metrics task succesfully cancelled")
            pass
        handler.cleanup()


async def init_prefill(
    runtime: DistributedRuntime, config: Config, shutdown_event: asyncio.Event
):
    server_args, dynamo_args = config.server_args, config.dynamo_args

    # Prevent SGLang from blocking on non-leader nodes
    if server_args.node_rank >= 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # Setup metrics and KV events for ALL nodes (including non-leader)
    # Non-leader nodes need KV event publishing for their local DP ranks
    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, component, generate_endpoint
    )

    # Register Prometheus metrics callback if enabled
    if engine.server_args.enable_metrics:
        setup_prometheus_registry(engine, generate_endpoint)

    # Handle non-leader nodes (multi-node parallelism)
    # Non-leader nodes run schedulers and publish KV events, but don't serve requests
    if server_args.node_rank >= 1:
        await _handle_non_leader_node(engine, publisher, metrics_task)
        return

    # Perform dummy warmup for prefill worker to avoid initial TTFT hit
    # Only needed on leader node that handles requests
    await _warmup_prefill_engine(engine, server_args)

    handler = PrefillWorkerHandler(
        component, engine, config, publisher, generate_endpoint, shutdown_event
    )
    handler.register_engine_routes(runtime)

    health_check_payload = SglangPrefillHealthCheckPayload(engine).to_dict()

    # Readiness gate: requests wait until model is registered
    ready_event = asyncio.Event()

    try:
        # Start endpoint immediately and register model concurrently
        # Registration publishes runtime_config with bootstrap endpoint for optimization
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
                health_check_payload=health_check_payload,
            ),
            register_llm_with_readiness_gate(
                engine,
                generate_endpoint,
                server_args,
                dynamo_args,
                input_type=ModelInput.Tokens,
                output_type=ModelType.Prefill,
                readiness_gate=ready_event,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            logging.info("Metrics task successfully cancelled")
            pass
        handler.cleanup()


async def init_diffusion(
    runtime: DistributedRuntime, config: Config, shutdown_event: asyncio.Event
):
    """Initialize diffusion language model worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    logging.info(
        f"Initializing diffusion worker with algorithm: {server_args.dllm_algorithm}"
    )
    if server_args.dllm_algorithm_config:
        logging.info(
            f"Using diffusion algorithm config: {server_args.dllm_algorithm_config}"
        )

    # Prevent SGLang from blocking on non-leader nodes
    if server_args.node_rank >= 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # Setup metrics and KV events for ALL nodes (including non-leader)
    # Non-leader nodes need KV event publishing for their local DP ranks
    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, component, generate_endpoint
    )

    # Register Prometheus metrics callback if enabled
    if engine.server_args.enable_metrics:
        setup_prometheus_registry(engine, generate_endpoint)

    # Handle non-leader nodes (multi-node parallelism)
    # Non-leader nodes run schedulers and publish KV events, but don't serve requests
    if server_args.node_rank >= 1:
        await _handle_non_leader_node(engine, publisher, metrics_task)
        return

    # Readiness gate: requests wait until model is registered
    ready_event = asyncio.Event()

    handler = DiffusionWorkerHandler(
        component, engine, config, publisher, generate_endpoint, shutdown_event
    )
    handler.register_engine_routes(runtime)

    health_check_payload = SglangHealthCheckPayload(
        engine, use_text_input=dynamo_args.use_sglang_tokenizer
    ).to_dict()

    logging.info(
        f"Registering diffusion model with endpoint types: {dynamo_args.dyn_endpoint_types}"
    )

    try:
        # Start endpoint and register model
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
                health_check_payload=health_check_payload,
            ),
            register_llm_with_readiness_gate(
                engine,
                generate_endpoint,
                server_args,
                dynamo_args,
                output_type=parse_endpoint_types(dynamo_args.dyn_endpoint_types),
                readiness_gate=ready_event,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve diffusion endpoints: {e}")
        raise
    finally:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            logging.info("Metrics task successfully cancelled")
            pass
        handler.cleanup()


async def init_embedding(
    runtime: DistributedRuntime, config: Config, shutdown_event: asyncio.Event
):
    """Initialize embedding worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # publisher instantiates the metrics and kv event publishers
    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, component, generate_endpoint
    )

    # Register Prometheus metrics callback if enabled
    if engine.server_args.enable_metrics:
        setup_prometheus_registry(engine, generate_endpoint)

    # Readiness gate: requests wait until model is registered
    ready_event = asyncio.Event()

    handler = EmbeddingWorkerHandler(
        component, engine, config, publisher, shutdown_event
    )
    health_check_payload = SglangHealthCheckPayload(
        engine, use_text_input=dynamo_args.use_sglang_tokenizer
    ).to_dict()

    try:
        # Start endpoint immediately and register model concurrently
        # Requests queue until ready_event is set
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
                health_check_payload=health_check_payload,
            ),
            register_llm_with_readiness_gate(
                engine,
                generate_endpoint,
                server_args,
                dynamo_args,
                input_type=ModelInput.Text,
                output_type=ModelType.Embedding,
                readiness_gate=ready_event,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve embedding endpoints: {e}")
        raise
    finally:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            logging.info("Metrics task successfully cancelled")
            pass
        handler.cleanup()


async def init_multimodal_processor(
    runtime: DistributedRuntime, config: Config, shutdown_event: asyncio.Event
):
    """Initialize multimodal processor component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args
    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # For processor, we need to connect to the encode worker
    encode_worker_client = (
        await runtime.namespace(dynamo_args.namespace)
        .component("encoder")
        .endpoint("generate")
        .client()
    )

    ready_event = asyncio.Event()

    handler = MultimodalProcessorHandler(
        component, config, encode_worker_client, shutdown_event
    )

    logging.info("Waiting for Encoder Worker Instances ...")
    await encode_worker_client.wait_for_instances()

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[("model", server_args.served_model_name)],
            ),
            register_llm_with_readiness_gate(
                None,  # engine
                generate_endpoint,
                server_args,
                dynamo_args,
                input_type=ModelInput.Text,
                readiness_gate=ready_event,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def init_multimodal_encode_worker(
    runtime: DistributedRuntime, config: Config, shutdown_event: asyncio.Event
):
    """Initialize multimodal encode worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # For encode worker, we need to connect to the downstream LLM worker
    pd_worker_client = (
        await runtime.namespace(dynamo_args.namespace)
        .component("backend")
        .endpoint("generate")
        .client()
    )

    handler = MultimodalEncodeWorkerHandler(
        component, config, pd_worker_client, shutdown_event
    )
    await handler.async_init(runtime)

    await pd_worker_client.wait_for_instances()

    try:
        # Encode Worker is an internal component, should not register with Frontend
        # Only needs to provide internal service endpoint for Processor to call
        await generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=[("model", server_args.served_model_name)],
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def init_multimodal_worker(
    runtime: DistributedRuntime, config: Config, shutdown_event: asyncio.Event
):
    """Initialize multimodal worker component for aggregated or decode mode"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    engine = sgl.Engine(server_args=server_args)

    if config.serving_mode == DisaggregationMode.DECODE:
        logging.info("Initializing prefill client for multimodal decode worker")
        prefill_client = (
            await runtime.namespace(dynamo_args.namespace)
            .component("prefill")
            .endpoint("generate")
            .client()
        )
        handler = MultimodalWorkerHandler(
            component, engine, config, prefill_client, shutdown_event
        )
    else:
        handler = MultimodalWorkerHandler(
            component, engine, config, None, shutdown_event
        )

    await handler.async_init()

    health_check_payload = SglangHealthCheckPayload(engine).to_dict()
    ready_event = asyncio.Event()

    try:
        if config.serving_mode == DisaggregationMode.DECODE:
            # Decode Worker is an internal component, should not register with Frontend
            # Only needs to provide internal service endpoint for Processor to call
            await generate_endpoint.serve_endpoint(
                handler.generate,
                metrics_labels=[("model", server_args.served_model_name)],
                graceful_shutdown=True,
                health_check_payload=health_check_payload,
            )
        else:
            # In aggregated mode, need to register with Frontend
            await asyncio.gather(
                generate_endpoint.serve_endpoint(
                    handler.generate,
                    metrics_labels=[("model", server_args.served_model_name)],
                    graceful_shutdown=True,
                    health_check_payload=health_check_payload,
                ),
                register_llm_with_readiness_gate(
                    engine,
                    generate_endpoint,
                    server_args,
                    dynamo_args,
                    readiness_gate=ready_event,
                ),
            )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def init_multimodal_prefill_worker(
    runtime: DistributedRuntime, config: Config, shutdown_event: asyncio.Event
):
    """Initialize multimodal prefill worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    handler = MultimodalPrefillWorkerHandler(component, engine, config, shutdown_event)
    await handler.async_init()

    health_check_payload = SglangPrefillHealthCheckPayload(engine).to_dict()

    try:
        # Prefill Worker is an internal component, should not register with Frontend
        # Only needs to provide internal service endpoint for Decode Worker to call
        await generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=[("model", server_args.served_model_name)],
            health_check_payload=health_check_payload,
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def _warmup_prefill_engine(engine: sgl.Engine, server_args) -> None:
    """Perform warmup request for prefill engine to reduce initial TTFT."""
    logging.info("Start of prefill disaggregation warmup ...")
    try:
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
        from sglang.srt.sampling.sampling_params import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.0,
            max_new_tokens=8,
            ignore_eos=True,
        )

        # Timeout: 1800s (30 min) for deep gemm precache
        async def _do_warmup():
            results = await engine.async_generate(
                input_ids=[0, 1, 2, 3],
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=FAKE_BOOTSTRAP_HOST,
                bootstrap_port=server_args.disaggregation_bootstrap_port,
                bootstrap_room=999999,
            )
            # Consume the stream
            async for _ in results:
                pass

        await asyncio.wait_for(_do_warmup(), timeout=1800)
        logging.info("Prefill warmup completed")
    except asyncio.TimeoutError:
        logging.warning("Prefill warmup timed out after 1800s")
    except Exception as e:
        logging.warning(f"Prefill warmup failed: {e}")


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
