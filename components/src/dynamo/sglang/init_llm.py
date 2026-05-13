# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import time
from typing import Awaitable, Callable, Optional

import sglang as sgl
from sglang.srt.observability.trace import set_global_trace_level

from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.llm import ModelInput, ModelType
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.health_check import (
    SglangDisaggHealthCheckPayload,
    SglangHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang.publisher import (
    handle_non_leader_node,
    set_forward_pass_metrics_worker_id,
    setup_sgl_metrics,
)
from dynamo.sglang.register import register_model_with_readiness_gate
from dynamo.sglang.request_handlers import DecodeWorkerHandler, PrefillWorkerHandler


async def _warmup_prefill_engine(engine: sgl.Engine, server_args) -> None:
    """Perform warmup request for prefill engine to reduce initial TTFT.

    Raises on failure so the caller can prevent the worker from registering
    with a broken engine (silent request drops). Shared with the unified
    backend (`dynamo.sglang.llm_engine`) via `_disagg.warmup_prefill_engine`.
    """
    from dynamo.sglang._disagg import warmup_prefill_engine

    await warmup_prefill_engine(engine, server_args.disaggregation_bootstrap_port)


async def init_decode(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
    snapshot_engine: Optional[sgl.Engine] = None,
) -> None:
    server_args, dynamo_args = config.server_args, config.dynamo_args

    if server_args.node_rank >= 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    # Use pre-created engine if provided (snapshot mode)
    if snapshot_engine is not None:
        engine = snapshot_engine
        load_time = 0.0
        if getattr(server_args, "enable_forward_pass_metrics", False):
            logging.warning(
                "Forward pass metrics disabled in snapshot mode: the engine was "
                "created before the endpoint existed, so its FPM publisher bound "
                "a different IPC path than the relay would subscribe to."
            )
            server_args.enable_forward_pass_metrics = False
    else:
        set_forward_pass_metrics_worker_id(server_args, generate_endpoint)
        start_time = time.time()
        engine = sgl.Engine(server_args=server_args)
        load_time = time.time() - start_time

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

    publisher.component_gauges.set_model_load_time(load_time)
    logging.debug(f"SGLang model load time: {load_time:.2f}s")

    if server_args.node_rank >= 1:
        await handle_non_leader_node(engine, publisher, metrics_task)
        return

    ready_event = asyncio.Event()

    handler = DecodeWorkerHandler(
        engine,
        config,
        publisher,
        generate_endpoint,
        shutdown_event,
        enable_frontend_decoding=dynamo_args.frontend_decoding,
    )
    handler.register_engine_routes(runtime)

    if config.serving_mode == DisaggregationMode.DECODE:
        health_check_payload = SglangDisaggHealthCheckPayload(
            engine, use_text_input=dynamo_args.use_sglang_tokenizer
        ).to_dict()
    else:
        health_check_payload = SglangHealthCheckPayload(
            engine, use_text_input=dynamo_args.use_sglang_tokenizer
        ).to_dict()

    logging.info(f"Registering model with endpoint types: {dynamo_args.endpoint_types}")
    if dynamo_args.custom_jinja_template and "chat" not in dynamo_args.endpoint_types:
        logging.warning(
            "Custom Jinja template provided (--custom-jinja-template) but 'chat' not in --dyn-endpoint-types. "
            "The chat template will be loaded but the /v1/chat/completions endpoint will not be available."
        )

    # Only serve session_control when streaming sessions are enabled.
    if getattr(server_args, "enable_streaming_session", False):
        session_control_endpoint = runtime.endpoint(
            f"{dynamo_args.namespace}.{dynamo_args.component}.session_control"
        )
        shutdown_endpoints.append(session_control_endpoint)

    try:
        gather_tasks = [
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
            register_model_with_readiness_gate(
                engine,
                generate_endpoint,
                server_args,
                dynamo_args,
                output_type=parse_endpoint_types(dynamo_args.endpoint_types),
                readiness_gate=ready_event,
            ),
        ]
        if getattr(server_args, "enable_streaming_session", False):
            gather_tasks.append(
                session_control_endpoint.serve_endpoint(handler.session_control)
            )
        await asyncio.gather(*gather_tasks)
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
        if run_deferred_handlers is not None:
            logging.info("Running deferred handlers")
            await run_deferred_handlers()


async def init_prefill(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
    snapshot_engine: Optional[sgl.Engine] = None,
) -> None:
    server_args, dynamo_args = config.server_args, config.dynamo_args

    if server_args.node_rank >= 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    # Use pre-created engine if provided (snapshot mode)
    if snapshot_engine is not None:
        engine = snapshot_engine
        load_time = 0.0
        if getattr(server_args, "enable_forward_pass_metrics", False):
            logging.warning(
                "Forward pass metrics disabled in snapshot mode: the engine was "
                "created before the endpoint existed, so its FPM publisher bound "
                "a different IPC path than the relay would subscribe to."
            )
            server_args.enable_forward_pass_metrics = False
    else:
        set_forward_pass_metrics_worker_id(server_args, generate_endpoint)
        start_time = time.time()
        engine = sgl.Engine(server_args=server_args)
        load_time = time.time() - start_time

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

    publisher.component_gauges.set_model_load_time(load_time)

    if server_args.node_rank >= 1:
        await handle_non_leader_node(engine, publisher, metrics_task)
        return

    try:
        await _warmup_prefill_engine(engine, server_args)
    except asyncio.TimeoutError as e:
        logging.error("Prefill warmup timed out after 1800s — aborting worker startup")
        raise RuntimeError(
            "Prefill warmup timed out; worker cannot serve requests"
        ) from e
    except Exception as e:
        logging.error(f"Prefill warmup failed: {e} — aborting worker startup")
        raise RuntimeError(f"Prefill warmup failed: {e}") from e

    handler = PrefillWorkerHandler(
        engine, config, publisher, generate_endpoint, shutdown_event
    )
    handler.register_engine_routes(runtime)

    health_check_payload = SglangPrefillHealthCheckPayload(engine).to_dict()

    ready_event = asyncio.Event()

    try:
        await asyncio.gather(
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
            register_model_with_readiness_gate(
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
        if run_deferred_handlers is not None:
            logging.info("Running deferred handlers")
            await run_deferred_handlers()
