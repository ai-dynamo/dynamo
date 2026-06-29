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
from dynamo.common.gms_failover import (
    acquire_gms_failover_lock_before_init,
    prepare_gms_failover,
    run_gms_failover_promotion_warmup,
)
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.failover_watchdog import (
    maybe_start_gms_failover_child_watchdog,
    maybe_start_rank_liveness,
)
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
from dynamo.sglang.request_handlers.handler_base import SGLangEnginePauseController


def _truthy_env(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _private_bootstrap_shadow_prewarm_enabled() -> bool:
    return _truthy_env("DYN_SGLANG_GMS_SHADOW_PREWARM", default=False)


def _is_private_bootstrap_shadow() -> bool:
    if not _truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE"):
        return False
    if not _truthy_env("DYN_SGLANG_GMS_PRIVATE_BOOTSTRAP_KV"):
        return False
    engine_id = os.environ.get("ENGINE_ID", "0")
    primary_engine_id = os.environ.get("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    return engine_id != primary_engine_id


def _rank_liveness_leader_host(server_args) -> Optional[str]:
    """Leader host that worker nodes heartbeat — derived from --dist-init-addr."""
    addr = getattr(server_args, "dist_init_addr", None)
    if not addr:
        return None
    # "host:port" or "[ipv6]:port" -> host
    return addr.rsplit(":", 1)[0].strip("[]")


class _NonLeaderFailoverController:
    """Rank-local failover controller for SGLang non-leader nodes."""

    def __init__(self, engine: sgl.Engine) -> None:
        self._delegate = (
            SGLangEnginePauseController(engine)
            if getattr(engine, "tokenizer_manager", None) is not None
            else None
        )
        self._is_quiesced = False

    async def quiesce(self, tags: Optional[list[str]] = None) -> bool:
        # The delegate only handles quiesce when it actually implements it (the
        # leader rank's request handler). A standby whose delegate has no quiesce
        # (e.g. a DecodeWorkerHandler) falls through to the lock-only path.
        if self._delegate is not None and hasattr(self._delegate, "quiesce"):
            return await self._delegate.quiesce(tags)
        if self._is_quiesced:
            return False
        logging.info(
            "[GMS failover] sglang rank has no quiesce-capable handler; "
            "using lock-only quiesce"
        )
        self._is_quiesced = True
        return True

    async def resume(self, tags: Optional[list[str]] = None) -> bool:
        if self._delegate is not None and hasattr(self._delegate, "resume"):
            return await self._delegate.resume(tags)
        if not self._is_quiesced:
            return False
        self._is_quiesced = False
        return True

    def mark_resumed(self) -> None:
        if self._delegate is not None and hasattr(self._delegate, "mark_resumed"):
            self._delegate.mark_resumed()
        self._is_quiesced = False


class _NonLeaderFailoverOwner:
    """Failover owner for SGLang multinode ranks that do not publish endpoints."""

    def __init__(self, engine: sgl.Engine) -> None:
        self._quiesce_controller = _NonLeaderFailoverController(engine)


async def _prepare_non_leader_failover(
    engine: sgl.Engine,
    runtime: DistributedRuntime,
    early_failover_activation,
) -> _NonLeaderFailoverOwner | None:
    owner = _NonLeaderFailoverOwner(engine)
    if early_failover_activation is not None and early_failover_activation.enabled:
        early_failover_activation.attach_to(owner)
        maybe_start_gms_failover_child_watchdog(owner, engine)
        return owner

    failover_activation = await prepare_gms_failover(
        owner,
        runtime,
        backend_name="sglang",
        promotion_warmup=None,
    )
    if not failover_activation.enabled:
        return None
    failover_activation.attach_to(owner)
    maybe_start_gms_failover_child_watchdog(owner, engine)
    return owner


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
    clear_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.clear_kv_blocks"
    )

    early_failover_activation = None
    lock_before_init = os.environ.get("DYN_SGLANG_GMS_LOCK_BEFORE_INIT", "1").lower()
    if lock_before_init not in {"0", "false", "no", "off"}:
        early_failover_activation = await acquire_gms_failover_lock_before_init(
            backend_name="sglang"
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
        non_leader_failover_owner = await _prepare_non_leader_failover(
            engine, runtime, early_failover_activation
        )
        # Heartbeat the leader over ZMQ so a crash of this worker node is detected
        # in ~one heartbeat-timeout instead of via the NCCL collective timeout.
        maybe_start_rank_liveness(
            non_leader_failover_owner,
            engine,
            node_rank=server_args.node_rank,
            leader_host=_rank_liveness_leader_host(server_args),
        )
        # Keep the owner alive for the non-leader loop. Its attached lock fd is
        # the local primary/shadow fencing token.
        _ = non_leader_failover_owner
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

    async def promotion_warmup() -> None:
        await run_gms_failover_promotion_warmup(
            handler.generate, health_check_payload, backend_name="sglang"
        )

    if early_failover_activation is not None and early_failover_activation.enabled:
        early_failover_activation.attach_to(handler)
        await promotion_warmup()
    else:
        shadow_prewarmed = False
        if (
            _private_bootstrap_shadow_prewarm_enabled()
            and _is_private_bootstrap_shadow()
        ):
            logging.info(
                "[GMS failover] sglang private-bootstrap shadow prewarm starting"
            )
            await promotion_warmup()
            shadow_prewarmed = True
            logging.info(
                "[GMS failover] sglang private-bootstrap shadow prewarm complete"
            )
        # Give the serving handler a quiesce-capable failover controller so a
        # shadow can quiesce (pause/release memory) before discovery; without it
        # gms_failover falls back to the raw handler, which has no quiesce.
        if getattr(handler, "_quiesce_controller", None) is None:
            handler._quiesce_controller = SGLangEnginePauseController(engine)
        failover_activation = await prepare_gms_failover(
            handler,
            runtime,
            backend_name="sglang",
            promotion_warmup=None if shadow_prewarmed else promotion_warmup,
        )
        failover_activation.attach_to(handler)
    maybe_start_gms_failover_child_watchdog(handler, engine)
    # Leader side of the cross-node liveness channel: watch worker heartbeats and,
    # on a worker crash, reuse the fence+release path to fail over to the warm shadow.
    if server_args.nnodes and server_args.nnodes > 1:
        maybe_start_rank_liveness(
            handler,
            engine,
            node_rank=0,
            leader_host=None,
            expected_ranks=range(1, server_args.nnodes),
        )

    logging.info(f"Registering model with endpoint types: {dynamo_args.endpoint_types}")
    if dynamo_args.custom_jinja_template and "chat" not in dynamo_args.endpoint_types:
        logging.warning(
            "Custom Jinja template provided (--custom-jinja-template) but 'chat' not in --dyn-endpoint-types. "
            "The chat template will be loaded but the /v1/chat/completions endpoint will not be available."
        )

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
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks,
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
                # Decode workers serve the LoRA load endpoints, so they may advertise capacity.
                serves_lora_load=True,
            ),
        ]
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
    clear_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.clear_kv_blocks"
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
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks,
                metrics_labels=metrics_labels,
            ),
            register_model_with_readiness_gate(
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
