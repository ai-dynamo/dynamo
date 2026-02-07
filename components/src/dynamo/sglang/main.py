# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import signal
import sys
from collections import defaultdict
from typing import Any, Callable, DefaultDict

import sglang as sgl
import uvloop

from dynamo.common.config_dump import dump_config
from dynamo.common.storage import get_fs
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.llm import ModelInput, ModelType
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang.args import Config, DisaggregationMode, parse_args
from dynamo.sglang.health_check import (
    ImageDiffusionHealthCheckPayload,
    SglangHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang.publisher import (
    DynamoSglangPublisher,
    setup_prometheus_registry,
    setup_sgl_metrics,
)
from dynamo.sglang.register import (
    register_image_diffusion_model,
    register_llm_with_readiness_gate,
)
from dynamo.sglang.request_handlers import (
    DecodeWorkerHandler,
    DiffusionWorkerHandler,
    EmbeddingWorkerHandler,
    ImageDiffusionWorkerHandler,
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


SignalCallback = Callable[..., Any]


def install_graceful_shutdown(
    loop: asyncio.AbstractEventLoop,
    runtime: Any,
    *,
    signals: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT),
    chain_old_os_handlers: bool = False,
) -> tuple[asyncio.Event, dict[int, list[tuple[SignalCallback, tuple[Any, ...]]]]]:
    """
    Set up graceful shutdown + callback chaining.

    What it does:
      - Owns OS-level SIGTERM/SIGINT via signal.signal(...)
      - Captures (suppresses) loop.add_signal_handler(SIGTERM/SIGINT, ...) registrations
        and runs them during shutdown (sync or async)
      - Calls runtime.shutdown() during shutdown (sync or async)
      - Sets and returns an asyncio.Event you can await to know shutdown was requested

    Returns:
      (shutdown_event, deferred_handlers)
    """
    shutdown_event = asyncio.Event()

    # Deferred handlers registered via loop.add_signal_handler for these signals
    deferred_handlers: DefaultDict[int, list[tuple[SignalCallback, tuple[Any, ...]]]] = defaultdict(list)  # type: ignore[assignment]

    # Previous OS handlers (for optional chaining)
    old_os_handlers: dict[int, Any] = {}

    shutdown_started = False

    async def _shutdown_sequence(signum: int, frame: Any | None) -> None:
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True

        logging.info("Received signal %s, starting graceful shutdown", signum)
        shutdown_event.set()

        # Run deferred callbacks in the loop context
        for cb, args in list(deferred_handlers.get(signum, [])):
            try:
                res = cb(*args)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                logging.exception("Deferred signal callback failed: %r", cb)

        # Call runtime.shutdown() (sync or async)
        try:
            res = runtime.shutdown()
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            logging.exception("runtime.shutdown() failed")

        # Optional: chain old OS handler
        if chain_old_os_handlers:
            old = old_os_handlers.get(signum)
            if old and old not in (
                signal.SIG_DFL,
                signal.SIG_IGN,
                signal.default_int_handler,
            ):
                try:
                    old(signum, frame)
                except Exception:
                    logging.exception("Chained old OS handler failed")

    def _schedule_shutdown(signum: int, frame: Any | None) -> None:
        def _kick() -> None:
            asyncio.create_task(_shutdown_sequence(signum, frame))

        loop.call_soon_threadsafe(_kick)

    def _os_signal_handler(signum: int, frame: Any) -> None:
        # Keep the OS handler tiny; do real work in the loop thread.
        _schedule_shutdown(signum, frame)

    # Install OS-level handlers
    for sig in signals:
        old_os_handlers[sig] = signal.signal(sig, _os_signal_handler)

    # Intercept loop.add_signal_handler for SIGTERM/SIGINT and defer them
    orig_add = loop.add_signal_handler

    def watching_add_signal_handler(sig: int, callback: SignalCallback, *args: Any):
        if sig in signals:
            logging.info(
                "Captured loop.add_signal_handler(%s, %r, ...) (deferred).",
                sig,
                callback,
            )
            deferred_handlers[sig].append((callback, args))
            return None
        return orig_add(sig, callback, *args)

    loop.add_signal_handler = watching_add_signal_handler  # type: ignore[assignment]

    return shutdown_event, deferred_handlers


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
    shutdown_event, _ = install_graceful_shutdown(loop, runtime)
    logging.info("Signal handlers set up for graceful shutdown (with chaining)")

    if config.dynamo_args.image_diffusion_worker:
        await init_image_diffusion(runtime, config)
    elif config.dynamo_args.embedding_worker:
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


async def init_image_diffusion(runtime: DistributedRuntime, config: Config):
    """Initialize image diffusion worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    # Initialize DiffGenerator (not sgl.Engine)
    from sglang.multimodal_gen import DiffGenerator

    if not server_args.model_path:
        raise ValueError("--model is required for diffusion workers")

    # Parallelism configuration
    tp_size = getattr(server_args, "tp_size", 1)
    dp_size = getattr(server_args, "dp_size", 1)
    num_gpus = tp_size * dp_size

    # Distributed configuration
    dist_timeout = getattr(server_args, "dist_timeout", None)

    generator = DiffGenerator.from_pretrained(
        model_path=server_args.model_path,
        # Parallelism configuration
        num_gpus=num_gpus,
        tp_size=tp_size,
        dp_size=dp_size,
        # Distributed configuration
        dist_timeout=dist_timeout,
    )

    # Initialize fsspec filesystems for image storage
    fs_url = dynamo_args.image_diffusion_fs_url

    # Initialize primary filesystem
    if not fs_url:
        raise ValueError("--image-diffusion-fs-url is required for diffusion workers")

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # Image diffusion doesn't have metrics publisher like LLM
    # Could add custom metrics for images/sec, steps/sec later

    handler = ImageDiffusionWorkerHandler(
        component,
        generator,
        config,
        publisher=None,
        fs=get_fs(fs_url),
    )

    # Create proper health check payload that sends a minimal diffusion request
    health_check_payload = ImageDiffusionHealthCheckPayload(
        model_path=server_args.model_path
    ).to_dict()

    ready_event = asyncio.Event()

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[],  # No LLM metrics labels
                health_check_payload=health_check_payload,
            ),
            register_image_diffusion_model(
                generator,
                generate_endpoint,
                server_args,
                readiness_gate=ready_event,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve image diffusion endpoints: {e}")
        raise
    finally:
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
