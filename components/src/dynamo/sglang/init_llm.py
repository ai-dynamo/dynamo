# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import fcntl
import logging
import os
import time
from typing import Awaitable, Callable, Optional

import sglang as sgl
from sglang.srt.observability.trace import set_global_trace_level

from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.health_check import (
    SglangDisaggHealthCheckPayload,
    SglangHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang.pause import SGLangEnginePauseController
from dynamo.sglang.publisher import (
    handle_non_leader_node,
    set_forward_pass_metrics_worker_id,
    setup_sgl_metrics,
)
from dynamo.sglang.register import register_model_with_readiness_gate
from dynamo.sglang.request_handlers import DecodeWorkerHandler, PrefillWorkerHandler

_GMS_FAILOVER_TAGS = ["weights", "kv_cache"]
_GMS_FAILOVER_LOCK_FD: int | None = None


async def _claim_initial_failover_lock(config: Config) -> None:
    if (
        not config.dynamo_args.gms_shadow_mode
        or config.server_args.node_rank >= 1
        or os.environ.get("ENGINE_ID", "0") != "0"
    ):
        return

    global _GMS_FAILOVER_LOCK_FD
    lock_path = os.environ.get("FAILOVER_LOCK_PATH", "/shared/failover.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        logging.info("[Shadow] Initial engine found failover lock already held")
        return
    except Exception:
        os.close(fd)
        raise

    _GMS_FAILOVER_LOCK_FD = fd
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, b"engine-0")
    logging.info("[Shadow] Initial engine acquired failover lock")


async def _warmup_prefill_engine(engine: sgl.Engine, server_args) -> None:
    """Perform warmup request for prefill engine to reduce initial TTFT.

    Raises on failure so the caller can prevent the worker from registering
    with a broken engine (silent request drops). Shared with the unified
    backend (`dynamo.sglang.llm_engine`) via `_disagg.warmup_prefill_engine`.
    """
    from dynamo.sglang._disagg import warmup_prefill_engine

    await warmup_prefill_engine(engine, server_args.disaggregation_bootstrap_port)


async def _maybe_wait_for_failover_lock(
    engine: sgl.Engine,
    runtime: DistributedRuntime,
    config: Config,
) -> None:
    if not config.dynamo_args.gms_shadow_mode:
        return

    pause_controller = SGLangEnginePauseController(engine)
    if await pause_controller.pause(_GMS_FAILOVER_TAGS):
        runtime.set_health_status(True)
        logging.info(
            "[Shadow] Engine sleeping, startup probe now passing, waiting for lock"
        )

        from gpu_memory_service.failover_lock.flock import FlockFailoverLock

        lock_path = os.environ.get("FAILOVER_LOCK_PATH", "/shared/failover.lock")
        engine_id = os.environ.get("ENGINE_ID", "0")
        if engine_id == "0" and _GMS_FAILOVER_LOCK_FD is not None:
            logging.info("[Shadow] Initial engine already owns failover lock")
        else:
            lock = FlockFailoverLock(lock_path)
            await lock.acquire(engine_id=f"engine-{engine_id}")
            logging.info("[Shadow] Lock acquired, waking engine")

        await pause_controller.resume(_GMS_FAILOVER_TAGS)
        pause_controller.mark_resumed()
        logging.info("[Shadow] Engine awake, registering service")


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

    await _claim_initial_failover_lock(config)

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
    # ``setup_sgl_metrics`` only returns ``None`` for embedding workers,
    # which take a different init path entirely. Narrow for mypy.
    assert publisher is not None, "setup_sgl_metrics returned None on chat path"

    publisher.component_gauges.set_model_load_time(load_time)
    logging.debug(f"SGLang model load time: {load_time:.2f}s")

    if server_args.node_rank >= 1:
        await handle_non_leader_node(engine, publisher, metrics_task)
        return

    await _maybe_wait_for_failover_lock(engine, runtime, config)

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

    # Worker type and needs, derived from serving_mode.
    if config.serving_mode == DisaggregationMode.DECODE:
        decode_worker_type = WorkerType.Decode
        decode_needs: list[list[WorkerType]] = [[WorkerType.Prefill]]
    else:
        decode_worker_type = WorkerType.Aggregated
        decode_needs = []

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
                worker_type=decode_worker_type,
                needs=decode_needs,
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

    await _claim_initial_failover_lock(config)

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
    # ``setup_sgl_metrics`` only returns ``None`` for embedding workers,
    # which take a different init path entirely. Narrow for mypy.
    assert publisher is not None, "setup_sgl_metrics returned None on chat path"

    publisher.component_gauges.set_model_load_time(load_time)

    if server_args.node_rank >= 1:
        await handle_non_leader_node(engine, publisher, metrics_task)
        return

    await _maybe_wait_for_failover_lock(engine, runtime, config)

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
                worker_type=WorkerType.Prefill,
                needs=[[WorkerType.Decode]],
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
