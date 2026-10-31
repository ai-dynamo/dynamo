# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import logging
import os
import time
from typing import Awaitable, Callable, Optional

import sglang as sgl
from sglang.srt.observability.trace import set_global_trace_level

from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.env import env_bool
from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.gateway import (
    attached_engine_load_time,
    gateway_worker_count,
    serve_via_gateway_children,
)
from dynamo.sglang.health_check import (
    SglangDisaggHealthCheckPayload,
    SglangHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang.publisher import (
    finish_worker_teardown,
    handle_non_leader_node,
    set_forward_pass_metrics_worker_id,
    setup_sgl_metrics,
)
from dynamo.sglang.register import register_model_with_readiness_gate
from dynamo.sglang.request_handlers import DecodeWorkerHandler, PrefillWorkerHandler
from dynamo.sglang.state_agent import StateAgentLifecycle


async def _warmup_prefill_engine(
    engine: sgl.Engine, server_args, metrics_task: asyncio.Task
) -> None:
    """Warm the prefill engine and stop metrics if warmup fails.

    Raises on failure so the worker never registers with a broken engine.
    """
    from dynamo.sglang._disagg import warmup_prefill_engine

    try:
        await warmup_prefill_engine(engine, server_args.disaggregation_bootstrap_port)
    except asyncio.CancelledError:
        await finish_worker_teardown(metrics_task, lambda: None, body_failed=True)
        raise
    except asyncio.TimeoutError as exc:
        await finish_worker_teardown(metrics_task, lambda: None, body_failed=True)
        logging.error("Prefill warmup timed out after 1800s — aborting worker startup")
        raise RuntimeError(
            "Prefill warmup timed out; worker cannot serve requests"
        ) from exc
    except Exception as exc:
        await finish_worker_teardown(metrics_task, lambda: None, body_failed=True)
        logging.error("Prefill warmup failed: %s — aborting worker startup", exc)
        raise RuntimeError(f"Prefill warmup failed: {exc}") from exc


def _freeze_gc_after_init(enabled: bool) -> None:
    if enabled:
        collected = gc.collect()
        gc.freeze()
        logging.info("Froze SGLang worker GC objects after collecting %d", collected)


async def _arm_attach_gate(handler, engine, config, metrics_task, run_deferred):
    """Create the lifecycle and close the handler's attach gate.

    Misconfiguration raises here, before the serving try block, so tear down
    the same way the serving failure path does.
    """
    state_agent = StateAgentLifecycle()
    try:
        state_agent.arm(engine, config)
    except Exception as e:
        logging.error(f"Failed to arm the KV state agent attach gate: {e}")
        await finish_worker_teardown(
            metrics_task, handler.cleanup, run_deferred, body_failed=True
        )
        raise
    handler.attach_gate = state_agent
    return state_agent


def _guard_health_check_payload(state_agent, payload):
    # The canary needs generate to work; until the worker is attached its
    # generate path is gated, so a canary would deadlock readiness.
    # With health checks off the runtime ignores the canary but a payload-less
    # endpoint would never turn Ready, so keep the payload then.
    if not state_agent.requires_attachment or not env_bool("DYN_HEALTH_CHECK_ENABLED"):
        return payload
    logging.warning(
        "Generate health-check canary is disabled for KV state-agent workers because "
        "readiness must not depend on generation before attachment "
        "(DYN_HEALTH_CHECK_ENABLED is ignored for generate)"
    )
    return None


async def _serve_attach_register(state_agent, serve_tasks, register_model):
    """Serve endpoints, then (Guard) await attachment, then register the model.

    Under Kubernetes discovery the worker's attachment intent is visible only
    once it is Ready, which needs the endpoints serving; the model card (what
    the router routes by) is published only after the agent attached.
    """
    if not state_agent.requires_attachment:
        await asyncio.gather(*serve_tasks, register_model())
        return

    async def attach_then_register():
        await state_agent.await_attached()
        await register_model()

    # Drain every branch before an error reaches the teardown that withdraws
    # the owner: a late ack must not open the gate or register the model then.
    tasks = [asyncio.ensure_future(t) for t in serve_tasks]
    tasks.append(asyncio.ensure_future(attach_then_register()))
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def init_decode(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
    snapshot_engine: Optional[sgl.Engine] = None,
    attached_engine: Optional[object] = None,
) -> None:
    server_args, dynamo_args = config.server_args, config.dynamo_args

    if server_args.node_rank >= 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )
    clear_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.clear_kv_blocks"
    )

    # Use pre-created engine if provided (snapshot mode)
    load_time: Optional[float]
    if snapshot_engine is not None:
        engine = snapshot_engine
        load_time = 0.0
        if getattr(server_args, "enable_forward_pass_metrics", False):
            raise RuntimeError(
                "Snapshot ServerArgs must disable forward-pass metrics before "
                "engine creation"
            )
    elif attached_engine is not None:
        # Gateway child: the parent owns the engine, this process only holds a
        # TokenizerWorker registered with its router.
        engine = attached_engine
        load_time = attached_engine_load_time()
    else:
        set_forward_pass_metrics_worker_id(server_args, generate_endpoint)
        start_time = time.time()
        engine = sgl.Engine(server_args=server_args)
        load_time = time.time() - start_time

    server_args = config.use_resolved_server_args(engine.server_args)
    gateway_count = gateway_worker_count(server_args, dynamo_args)
    if gateway_count > 1:
        # engine.tokenizer_manager is SGLang's MultiTokenizerRouter here and cannot
        # serve requests; gateway children do, this process keeps the engine alive.
        try:
            _freeze_gc_after_init(dynamo_args.freeze_gc_after_init)
            await serve_via_gateway_children(
                engine, gateway_count, shutdown_event, load_time=load_time
            )
        finally:
            engine.shutdown()
            if run_deferred_handlers is not None:
                await run_deferred_handlers()
        return

    if server_args.enable_trace:
        set_global_trace_level(dynamo_args.sglang_trace_level)

    load_lora_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.load_lora"
    )
    unload_lora_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.unload_lora"
    )
    list_loras_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.list_loras"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, generate_endpoint
    )
    # ``setup_sgl_metrics`` only returns ``None`` for embedding workers,
    # which take a different init path entirely. Narrow for mypy.
    assert publisher is not None, "setup_sgl_metrics returned None on chat path"

    if load_time is not None:
        publisher.component_gauges.set_model_load_time(load_time)
        logging.debug(f"SGLang model load time: {load_time:.2f}s")

    if server_args.node_rank >= 1:
        await handle_non_leader_node(engine, publisher, metrics_task)
        return

    ready_event = asyncio.Event()

    # Worker type and needs, derived from serving_mode.
    if config.serving_mode == DisaggregationMode.DECODE:
        decode_worker_type = WorkerType.Decode
        decode_needs: list[list[WorkerType]] = [[WorkerType.Prefill]]
    else:
        decode_worker_type = WorkerType.Aggregated
        decode_needs = []

    first_token_source = await generate_endpoint.first_token_source(decode_worker_type)

    handler = DecodeWorkerHandler(
        engine,
        config,
        publisher,
        generate_endpoint,
        shutdown_event,
        enable_frontend_decoding=dynamo_args.frontend_decoding,
        first_token_source=first_token_source,
    )
    # Close the attach gate before any route is registered: the system server
    # is already up and routes must not act on an unattached Guard worker.
    state_agent = await _arm_attach_gate(
        handler, engine, config, metrics_task, run_deferred_handlers
    )
    handler.register_engine_routes(runtime)
    if attached_engine is not None:
        handler.follow_shared_pause_state()

    if config.serving_mode == DisaggregationMode.DECODE:
        health_check_payload = SglangDisaggHealthCheckPayload(
            engine, use_text_input=dynamo_args.use_sglang_tokenizer
        ).to_dict()
    else:
        health_check_payload = SglangHealthCheckPayload(
            engine, use_text_input=dynamo_args.use_sglang_tokenizer
        ).to_dict()

    _freeze_gc_after_init(dynamo_args.freeze_gc_after_init)

    logging.info(f"Registering model with endpoint types: {dynamo_args.endpoint_types}")
    if dynamo_args.custom_jinja_template and "chat" not in dynamo_args.endpoint_types:
        logging.warning(
            "Custom Jinja template provided (--custom-jinja-template) but 'chat' not in --dyn-endpoint-types. "
            "The chat template will be loaded but the /v1/chat/completions endpoint will not be available."
        )

    body_failed = True
    try:
        await state_agent.start(engine, config, generate_endpoint)
        health_check_payload = _guard_health_check_payload(
            state_agent, health_check_payload
        )
        serve_tasks = [
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
                health_check_payload=health_check_payload,
            ),
            load_lora_endpoint.serve_endpoint(
                handler.load_lora,
                metrics_labels=metrics_labels,
            ),
            unload_lora_endpoint.serve_endpoint(
                handler.unload_lora,
                metrics_labels=metrics_labels,
            ),
            list_loras_endpoint.serve_endpoint(
                handler.list_loras,
                metrics_labels=metrics_labels,
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks,
                metrics_labels=metrics_labels,
            ),
        ]
        await _serve_attach_register(
            state_agent,
            serve_tasks,
            lambda: register_model_with_readiness_gate(
                engine,
                generate_endpoint,
                server_args,
                dynamo_args,
                output_type=parse_endpoint_types(dynamo_args.endpoint_types),
                readiness_gate=ready_event,
                worker_type=decode_worker_type,
                needs=decode_needs,
                # Decode workers serve the LoRA load endpoints, so they may advertise capacity.
                serves_lora_load=True,
                forwards_kv_hints=True,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints or attach to the KV state agent: {e}")
        raise
    else:
        body_failed = False
    finally:
        try:
            await state_agent.close()
        finally:
            await finish_worker_teardown(
                metrics_task,
                handler.cleanup,
                run_deferred_handlers,
                body_failed=body_failed,
            )


async def init_prefill(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
    snapshot_engine: Optional[sgl.Engine] = None,
    attached_engine: Optional[object] = None,
) -> None:
    server_args, dynamo_args = config.server_args, config.dynamo_args

    if server_args.node_rank >= 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )
    clear_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.clear_kv_blocks"
    )

    # Use pre-created engine if provided (snapshot mode)
    load_time: Optional[float]
    if snapshot_engine is not None:
        engine = snapshot_engine
        load_time = 0.0
        if getattr(server_args, "enable_forward_pass_metrics", False):
            raise RuntimeError(
                "Snapshot ServerArgs must disable forward-pass metrics before "
                "engine creation"
            )
    elif attached_engine is not None:
        # Gateway child: the parent owns the engine, this process only holds a
        # TokenizerWorker registered with its router.
        engine = attached_engine
        load_time = attached_engine_load_time()
    else:
        set_forward_pass_metrics_worker_id(server_args, generate_endpoint)
        start_time = time.time()
        engine = sgl.Engine(server_args=server_args)
        load_time = time.time() - start_time

    server_args = config.use_resolved_server_args(engine.server_args)
    gateway_count = gateway_worker_count(server_args, dynamo_args)
    if gateway_count > 1:
        # engine.tokenizer_manager is SGLang's MultiTokenizerRouter here and cannot
        # serve requests; gateway children do, this process keeps the engine alive.
        try:
            await serve_via_gateway_children(
                engine, gateway_count, shutdown_event, load_time=load_time
            )
        finally:
            engine.shutdown()
            if run_deferred_handlers is not None:
                await run_deferred_handlers()
        return

    if server_args.enable_trace:
        set_global_trace_level(dynamo_args.sglang_trace_level)

    load_lora_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.load_lora"
    )
    unload_lora_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.unload_lora"
    )
    list_loras_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.list_loras"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, generate_endpoint
    )
    # ``setup_sgl_metrics`` only returns ``None`` for embedding workers,
    # which take a different init path entirely. Narrow for mypy.
    assert publisher is not None, "setup_sgl_metrics returned None on chat path"

    if load_time is not None:
        publisher.component_gauges.set_model_load_time(load_time)

    if server_args.node_rank >= 1:
        await handle_non_leader_node(engine, publisher, metrics_task)
        return

    await _warmup_prefill_engine(engine, server_args, metrics_task)

    handler = PrefillWorkerHandler(
        engine, config, publisher, generate_endpoint, shutdown_event
    )
    # Close the attach gate before any route is registered: the system server
    # is already up and routes must not act on an unattached Guard worker.
    state_agent = await _arm_attach_gate(
        handler, engine, config, metrics_task, run_deferred_handlers
    )
    handler.register_engine_routes(runtime)
    if attached_engine is not None:
        handler.follow_shared_pause_state()

    health_check_payload = SglangPrefillHealthCheckPayload(engine).to_dict()

    ready_event = asyncio.Event()

    body_failed = True
    try:
        await state_agent.start(engine, config, generate_endpoint)
        health_check_payload = _guard_health_check_payload(
            state_agent, health_check_payload
        )
        serve_tasks = [
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
                health_check_payload=health_check_payload,
            ),
            load_lora_endpoint.serve_endpoint(
                handler.load_lora,
                metrics_labels=metrics_labels,
            ),
            unload_lora_endpoint.serve_endpoint(
                handler.unload_lora,
                metrics_labels=metrics_labels,
            ),
            list_loras_endpoint.serve_endpoint(
                handler.list_loras,
                metrics_labels=metrics_labels,
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks,
                metrics_labels=metrics_labels,
            ),
        ]
        await _serve_attach_register(
            state_agent,
            serve_tasks,
            lambda: register_model_with_readiness_gate(
                engine,
                generate_endpoint,
                server_args,
                dynamo_args,
                input_type=ModelInput.Tokens,
                # Prefill workers have no OpenAI surface — the role is carried
                # by `worker_type=Prefill` below. We register the legacy
                # `ModelType.Prefill` marker bit (not a surface) so an OLD
                # frontend, which detects prefill via that bit, still routes
                # disaggregated traffic during the cross-version rollout. A new
                # frontend ignores it and dispatches off `worker_type`.
                output_type=ModelType.Prefill,
                readiness_gate=ready_event,
                worker_type=WorkerType.Prefill,
                needs=[[WorkerType.Decode]],
                # Prefill workers also serve the LoRA load endpoints (init_prefill), so they may
                # advertise capacity.
                serves_lora_load=True,
                forwards_kv_hints=True,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints or attach to the KV state agent: {e}")
        raise
    else:
        body_failed = False
    finally:
        try:
            await state_agent.close()
        finally:
            await finish_worker_teardown(
                metrics_task,
                handler.cleanup,
                run_deferred_handlers,
                body_failed=body_failed,
            )
