# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import signal
import sys

import sglang as sgl
import uvloop

from dynamo.common.config_dump import dump_config
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.llm import ModelInput, ModelType, unregister_llm
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang.args import Config, DisaggregationMode, parse_args
from dynamo.sglang.health_check import (
    SglangHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang.publisher import setup_prometheus_registry, setup_sgl_metrics
from dynamo.sglang.register import register_llm_with_readiness_gate
from dynamo.sglang.request_handlers import (
    DecodeWorkerHandler,
    EmbeddingWorkerHandler,
    MultimodalEncodeWorkerHandler,
    MultimodalPrefillWorkerHandler,
    MultimodalProcessorHandler,
    MultimodalWorkerHandler,
    PrefillWorkerHandler,
)

configure_dynamo_logging()
logger = logging.getLogger(__name__)

# Track if GMS has been set up to avoid duplicate setup
_gms_setup_done = False


def _setup_gms_if_needed(config: Config) -> None:
    """Setup GPU Memory Service if --load-format gpu_memory_service is passed.

    This does TWO things:
    1. Sets environment variables for patches (needed in spawned workers)
    2. Applies GMS patches and sets load_format to GPUServiceModelLoader class

    Usage:
        python -m dynamo.sglang --model-path ... \\
            --load-format gpu_memory_service \\
            --model-loader-extra-config '{"gms_socket_path": "/tmp/gms_{device}.sock"}'
    """
    global _gms_setup_done
    if _gms_setup_done:
        return

    load_format = getattr(config.server_args, "load_format", None)
    if load_format != "gpu_memory_service":
        return

    logger.info("[GMS] Setting up GMS integration")

    # Set env var to trigger auto-registration in spawned workers
    os.environ["GMS_SGLANG_AUTO_REGISTER"] = "1"

    # Apply patches in main process
    try:
        from dynamo.sglang.gms_adapters import (
            GPUServiceModelLoader,
            patch_model_runner_for_gms,
        )

        patch_model_runner_for_gms()
        logger.info("[GMS] Applied GMS patches for SGLang")

        # Set load_format to the actual class so SGLang uses our custom loader
        config.server_args.load_format = GPUServiceModelLoader
        logger.info("[GMS] Set load_format=GPUServiceModelLoader")
    except Exception as e:
        logger.error(f"[GMS] Failed to setup GMS: {e}")
        raise

    _gms_setup_done = True


async def _handle_non_leader_node(
    engine: sgl.Engine,
    generate_endpoint,
) -> None:
    """
    Handle non-leader node (node_rank >= 1) in multi-node deployments.

    Non-leader nodes only run scheduler processes and don't handle requests,
    but they should still expose metrics via Dynamo's metrics endpoint.

    Args:
        engine: The SGLang engine instance.
        config: SGLang configuration including server args.
        component: The Dynamo runtime component.
        generate_endpoint: The Dynamo endpoint for generation requests.
    """
    logging.info(
        f"Non-leader node detected (node_rank={engine.server_args.node_rank})."
    )

    # Only setup Prometheus registry to expose SGLang metrics from shared memory
    # Non-leader nodes don't need Dynamo metrics publishing or KV events
    if engine.server_args.enable_metrics:
        setup_prometheus_registry(engine, generate_endpoint)
        logging.info("Prometheus metrics registry configured for non-leader node")

    # Wait indefinitely - the process will be terminated via signal handlers
    await asyncio.Event().wait()


async def worker():
    config = await parse_args(sys.argv[1:])
    dump_config(config.dynamo_args.dump_config_to, config)

    # Setup GMS if using gpu_memory_service load format
    # This must be called before sgl.Engine() is created
    _setup_gms_if_needed(config)

    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(
        loop, config.dynamo_args.store_kv, config.dynamo_args.request_plane
    )

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers will trigger a graceful shutdown of the runtime")

    if config.dynamo_args.embedding_worker:
        await init_embedding(runtime, config)
    elif config.dynamo_args.multimodal_processor:
        await init_multimodal_processor(runtime, config)
    elif config.dynamo_args.multimodal_encode_worker:
        await init_multimodal_encode_worker(runtime, config)
    elif config.dynamo_args.multimodal_worker:
        if config.serving_mode != DisaggregationMode.PREFILL:
            await init_multimodal_worker(runtime, config)
        else:
            await init_multimodal_prefill_worker(runtime, config)
    elif config.serving_mode != DisaggregationMode.PREFILL:
        await init(runtime, config)
    else:
        await init_prefill(runtime, config)


async def init(runtime: DistributedRuntime, config: Config):
    server_args, dynamo_args = config.server_args, config.dynamo_args

    # Prevent SGLang from blocking on non-leader nodes
    if server_args.node_rank >= 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # Handle non-leader nodes (multi-node parallelism)
    # Non-leader nodes only run scheduler processes and expose metrics
    if server_args.node_rank >= 1:
        await _handle_non_leader_node(engine, generate_endpoint)
        return

    # Register engine routes for profiling
    async def start_profile_handler(body: dict) -> dict:
        """Handle /engine/start_profile requests"""
        await engine.tokenizer_manager.start_profile(**body)
        return {"status": "ok", "message": "Profiling started"}

    async def stop_profile_handler(body: dict) -> dict:
        """Handle /engine/stop_profile requests"""
        await engine.tokenizer_manager.stop_profile()
        return {"status": "ok", "message": "Profiling stopped"}

    runtime.register_engine_route("start_profile", start_profile_handler)
    runtime.register_engine_route("stop_profile", stop_profile_handler)

    # Register engine routes for pause/resume/status
    async def pause_handler(body: dict) -> dict:
        """Pause the engine to release GPU memory and unregister from discovery.

        Args:
            tag: Memory tag to pause (default: "weights")

        With GMS enabled, torch_memory_saver handles VA-stable pause for weights.
        After pausing, unregisters from etcd so frontend stops routing to this worker.
        """
        tag = body.get("tag", "weights")
        try:
            from torch_memory_saver import torch_memory_saver

            torch_memory_saver.pause(tag)

            # Unregister from discovery so frontend stops routing to us
            try:
                await unregister_llm(generate_endpoint)
                logging.info(
                    "[Pause] Unregistered model from discovery - frontend will stop routing here"
                )
            except Exception as unreg_err:
                logging.warning(
                    f"[Pause] Failed to unregister from discovery: {unreg_err}"
                )

            return {"status": "ok", "message": f"Engine paused (tag={tag})"}
        except Exception as e:
            logging.error(f"Failed to pause engine: {e}")
            return {"status": "error", "message": str(e)}

    async def resume_handler(body: dict) -> dict:
        """Resume the engine to restore GPU memory and re-register to discovery.

        Args:
            tag: Memory tag to resume (default: "weights")

        With GMS enabled, torch_memory_saver handles VA-stable resume for weights.
        After resuming, re-registers to etcd so frontend can route to this worker again.
        """
        tag = body.get("tag", "weights")
        try:
            from torch_memory_saver import torch_memory_saver

            torch_memory_saver.resume(tag)

            # Re-register to discovery so frontend can route to us again
            try:
                from dynamo.sglang.register import _register_llm_with_runtime_config

                model_type = parse_endpoint_types(dynamo_args.dyn_endpoint_types)
                await _register_llm_with_runtime_config(
                    engine,
                    generate_endpoint,
                    server_args,
                    dynamo_args,
                    output_type=model_type,
                )
                logging.info(
                    "[Resume] Re-registered model to discovery - frontend can route here again"
                )
            except Exception as reg_err:
                logging.warning(
                    f"[Resume] Failed to re-register to discovery: {reg_err}"
                )

            return {"status": "ok", "message": f"Engine resumed (tag={tag})"}
        except Exception as e:
            logging.error(f"Failed to resume engine: {e}")
            return {"status": "error", "message": str(e)}

    async def status_handler(body: dict) -> dict:
        """Get engine pause/resume status."""
        try:
            from dynamo.sglang.gms_adapters import _get_gms_allocator

            allocator = _get_gms_allocator()
            if allocator is not None:
                return {
                    "status": "ok",
                    "is_paused": allocator.is_sleeping,
                    "gms_enabled": True,
                    "model": server_args.served_model_name,
                }
            else:
                # Non-GMS mode - torch_memory_saver doesn't expose pause state directly
                return {
                    "status": "ok",
                    "gms_enabled": False,
                    "model": server_args.served_model_name,
                }
        except Exception as e:
            logging.error(f"Failed to get engine status: {e}")
            return {"status": "error", "message": str(e)}

    runtime.register_engine_route("pause", pause_handler)
    runtime.register_engine_route("resume", resume_handler)
    runtime.register_engine_route("status", status_handler)
    logging.info(
        "Registered engine routes: /engine/start_profile, /engine/stop_profile, "
        "/engine/pause, /engine/resume, /engine/status"
    )

    # publisher instantiates the metrics and kv event publishers
    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, component, generate_endpoint
    )

    # Register Prometheus metrics callback if enabled
    if engine.server_args.enable_metrics:
        setup_prometheus_registry(engine, generate_endpoint)

    # Readiness gate: requests wait until model is registered
    ready_event = asyncio.Event()

    handler = DecodeWorkerHandler(component, engine, config, publisher)
    print(f"Config: {config}")
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


async def init_prefill(runtime: DistributedRuntime, config: Config):
    server_args, dynamo_args = config.server_args, config.dynamo_args

    # Prevent SGLang from blocking on non-leader nodes
    if server_args.node_rank >= 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # Handle non-leader nodes (multi-node tensor parallelism)
    # Non-leader nodes only run scheduler processes and expose metrics
    if server_args.node_rank >= 1:
        await _handle_non_leader_node(engine, generate_endpoint)
        return

    # Register engine routes for profiling
    async def start_profile_handler(body: dict) -> dict:
        """Handle /engine/start_profile requests"""
        await engine.tokenizer_manager.start_profile(**body)
        return {"status": "ok", "message": "Profiling started"}

    async def stop_profile_handler(body: dict) -> dict:
        """Handle /engine/stop_profile requests"""
        await engine.tokenizer_manager.stop_profile()
        return {"status": "ok", "message": "Profiling stopped"}

    runtime.register_engine_route("start_profile", start_profile_handler)
    runtime.register_engine_route("stop_profile", stop_profile_handler)

    # Register engine routes for pause/resume/status
    async def pause_handler(body: dict) -> dict:
        """Pause the engine to release GPU memory and unregister from discovery.

        Args:
            tag: Memory tag to pause (default: "weights")

        With GMS enabled, torch_memory_saver handles VA-stable pause for weights.
        After pausing, unregisters from etcd so frontend stops routing to this worker.
        """
        tag = body.get("tag", "weights")
        try:
            from torch_memory_saver import torch_memory_saver

            torch_memory_saver.pause(tag)

            # Unregister from discovery so frontend stops routing to us
            try:
                await unregister_llm(generate_endpoint)
                logging.info(
                    "[Pause] Unregistered model from discovery - frontend will stop routing here"
                )
            except Exception as unreg_err:
                logging.warning(
                    f"[Pause] Failed to unregister from discovery: {unreg_err}"
                )

            return {"status": "ok", "message": f"Engine paused (tag={tag})"}
        except Exception as e:
            logging.error(f"Failed to pause engine: {e}")
            return {"status": "error", "message": str(e)}

    async def resume_handler(body: dict) -> dict:
        """Resume the engine to restore GPU memory and re-register to discovery.

        Args:
            tag: Memory tag to resume (default: "weights")

        With GMS enabled, torch_memory_saver handles VA-stable resume for weights.
        After resuming, re-registers to etcd so frontend can route to this worker again.
        """
        tag = body.get("tag", "weights")
        try:
            from torch_memory_saver import torch_memory_saver

            torch_memory_saver.resume(tag)

            # Re-register to discovery so frontend can route to us again
            try:
                from dynamo.sglang.register import _register_llm_with_runtime_config

                await _register_llm_with_runtime_config(
                    engine,
                    generate_endpoint,
                    server_args,
                    dynamo_args,
                    input_type=ModelInput.Tokens,
                    output_type=ModelType.Prefill,
                )
                logging.info(
                    "[Resume] Re-registered model to discovery - frontend can route here again"
                )
            except Exception as reg_err:
                logging.warning(
                    f"[Resume] Failed to re-register to discovery: {reg_err}"
                )

            return {"status": "ok", "message": f"Engine resumed (tag={tag})"}
        except Exception as e:
            logging.error(f"Failed to resume engine: {e}")
            return {"status": "error", "message": str(e)}

    async def status_handler(body: dict) -> dict:
        """Get engine pause/resume status."""
        try:
            from dynamo.sglang.gms_adapters import _get_gms_allocator

            allocator = _get_gms_allocator()
            if allocator is not None:
                return {
                    "status": "ok",
                    "is_paused": allocator.is_sleeping,
                    "gms_enabled": True,
                    "model": server_args.served_model_name,
                }
            else:
                # Non-GMS mode - torch_memory_saver doesn't expose pause state directly
                return {
                    "status": "ok",
                    "gms_enabled": False,
                    "model": server_args.served_model_name,
                }
        except Exception as e:
            logging.error(f"Failed to get engine status: {e}")
            return {"status": "error", "message": str(e)}

    runtime.register_engine_route("pause", pause_handler)
    runtime.register_engine_route("resume", resume_handler)
    runtime.register_engine_route("status", status_handler)
    logging.info(
        "Registered engine routes: /engine/start_profile, /engine/stop_profile, "
        "/engine/pause, /engine/resume, /engine/status"
    )

    # Perform dummy warmup for prefill worker to avoid initial TTFT hit
    # Only needed on leader node that handles requests
    await _warmup_prefill_engine(engine, server_args)

    # publisher instantiates the metrics and kv event publishers
    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, component, generate_endpoint
    )

    # Register Prometheus metrics callback if enabled
    if engine.server_args.enable_metrics:
        setup_prometheus_registry(engine, generate_endpoint)

    handler = PrefillWorkerHandler(component, engine, config, publisher)

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


async def init_embedding(runtime: DistributedRuntime, config: Config):
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

    handler = EmbeddingWorkerHandler(component, engine, config, publisher)
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


async def init_multimodal_processor(runtime: DistributedRuntime, config: Config):
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

    handler = MultimodalProcessorHandler(component, config, encode_worker_client)

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


async def init_multimodal_encode_worker(runtime: DistributedRuntime, config: Config):
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

    handler = MultimodalEncodeWorkerHandler(component, config, pd_worker_client)
    await handler.async_init(runtime)

    await pd_worker_client.wait_for_instances()

    ready_event = asyncio.Event()

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[("model", server_args.served_model_name)],
            ),
            register_llm_with_readiness_gate(
                None,  # encode worker doesn't have engine
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


async def init_multimodal_worker(runtime: DistributedRuntime, config: Config):
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
        handler = MultimodalWorkerHandler(component, engine, config, prefill_client)
    else:
        handler = MultimodalWorkerHandler(component, engine, config)

    await handler.async_init()

    health_check_payload = SglangHealthCheckPayload(engine).to_dict()
    ready_event = asyncio.Event()

    try:
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


async def init_multimodal_prefill_worker(runtime: DistributedRuntime, config: Config):
    """Initialize multimodal prefill worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    handler = MultimodalPrefillWorkerHandler(component, engine, config)
    await handler.async_init()

    health_check_payload = SglangPrefillHealthCheckPayload(engine).to_dict()
    ready_event = asyncio.Event()

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[("model", server_args.served_model_name)],
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


async def graceful_shutdown(runtime):
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
