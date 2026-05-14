# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone ThunderAgent router service.

Usage:
    python -m dynamo.thunderagent_router \\
        --endpoint dynamo.vllm.generate \\
        --router-block-size 64

The service serves ``{namespace}.thunderagent_router.generate`` as a
model-handler endpoint; the frontend's discovery picks it up the same way
it would any other worker endpoint and dispatches LLM requests to it.

Reads ``request["agent_context"]["trajectory_id"]`` to attach lifecycle decisions
to programs (Dynamo's ``trajectory_id`` is the analogue of ThunderAgent's
``program_id``). Requests that lack ``agent_context`` are routed normally with
no pause/resume gating -- the v0 contract is "opt-in via nvext.agent_context".
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import uvloop

from dynamo.llm import KvRouter, ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.thunderagent_router.args import (
    ThunderAgentRouterConfig,
    build_aic_perf_config,
    build_kv_router_config,
    parse_args,
)
from dynamo.thunderagent_router.capacity import FpmCapacityProvider
from dynamo.thunderagent_router.router import ThunderAgentScheduler

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def _extract_program_id(request: dict[str, Any]) -> Optional[str]:
    """Pull the scheduling program_id off ``nvext.agent_context.trajectory_id``."""
    ctx = request.get("agent_context")
    if not isinstance(ctx, dict):
        return None
    pid = ctx.get("trajectory_id")
    if isinstance(pid, str) and pid:
        return pid
    return None


def _wrap_preprocessed_request(request: dict[str, Any]) -> dict[str, Any]:
    """Build the PreprocessedRequest dict KvRouter.generate_from_request expects.

    Duplicated from ``dynamo.router/__main__.py`` since neither package exports
    it. Keep the field list in sync if the upstream wrapper grows new keys.
    """
    routing = request.get("routing")
    dp_rank = request.get("dp_rank")
    if routing is None and dp_rank is not None:
        routing = {"dp_rank": dp_rank}

    return {
        "model": request.get("model", "unknown"),
        "token_ids": request["token_ids"],
        "stop_conditions": request.get("stop_conditions", {}),
        "sampling_options": request.get("sampling_options", {}),
        "output_options": request.get("output_options", {}),
        "eos_token_ids": request.get("eos_token_ids", []),
        "annotations": request.get("annotations", []),
        "routing": routing,
        "router_config_override": request.get("router_config_override"),
        "prefill_result": request.get("prefill_result"),
        "bootstrap_info": request.get("bootstrap_info"),
        "extra_args": request.get("extra_args"),
        "mm_processor_kwargs": request.get("mm_processor_kwargs"),
        "agent_context": request.get("agent_context"),
        "request_timestamp_ms": request.get("request_timestamp_ms"),
    }


class ThunderAgentRouterHandler:
    """Glue between the Dynamo endpoint runtime and ThunderAgentScheduler."""

    def __init__(
        self,
        runtime: DistributedRuntime,
        config: ThunderAgentRouterConfig,
    ) -> None:
        self._runtime = runtime
        self._config = config
        self._kv_router: Optional[KvRouter] = None
        self._capacity: Optional[FpmCapacityProvider] = None
        self._scheduler: Optional[ThunderAgentScheduler] = None

    async def initialize(self) -> None:
        parts = self._config.endpoint.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --endpoint {self._config.endpoint!r}; "
                "expected namespace.component.endpoint"
            )

        worker_endpoint = self._runtime.endpoint(self._config.endpoint)

        self._kv_router = KvRouter(
            endpoint=worker_endpoint,
            block_size=self._config.router_block_size,
            kv_router_config=build_kv_router_config(self._config),
            aic_perf_config=build_aic_perf_config(self._config),
        )

        self._capacity = FpmCapacityProvider(worker_endpoint)
        self._capacity.start()

        self._scheduler = ThunderAgentScheduler(
            capacity=self._capacity,
            config=self._config.to_thunderagent_config(),
        )
        self._scheduler.start()
        logger.info(
            "ThunderAgentRouterHandler initialized for %s "
            "(block_size=%d, pause=%.2f, soft=%.2f)",
            self._config.endpoint,
            self._config.router_block_size,
            self._config.pause_threshold,
            self._config.soft_demote_threshold,
        )

    async def shutdown(self) -> None:
        if self._scheduler is not None:
            await self._scheduler.stop()
        if self._capacity is not None:
            self._capacity.stop()

    async def generate(self, request: dict[str, Any]):
        """Wrap KvRouter.generate_from_request with ThunderAgent admission,
        sticky-worker pinning, and real-token accounting."""
        program_id = _extract_program_id(request)

        # Path A: no program_id -> behave like the standalone router (no
        # admission, no lifecycle). Backward compat for clients that don't
        # send agent_context.
        if program_id is None:
            preprocessed = _wrap_preprocessed_request(request)
            async for chunk in await self._kv_router.generate_from_request(
                preprocessed  # type: ignore[arg-type]
            ):
                yield chunk
            return

        # Path B: program lifecycle.
        token_ids = request["token_ids"]
        estimated_prompt_tokens = len(token_ids) if isinstance(token_ids, list) else 0

        # When cache-aware admission is on, query the KvRouter for per-worker
        # prefix-cache overlap. Used by the scheduler only on new-program
        # admission; subsequent turns are pinned to a worker already.
        worker_cached_tokens: Optional[dict[int, int]] = None
        if self._config.cache_aware_admission and isinstance(token_ids, list):
            try:
                loads = await self._kv_router.get_potential_loads(token_ids)
            except Exception as exc:
                logger.debug("get_potential_loads failed: %s", exc)
                loads = []
            worker_cached_tokens = {}
            for entry in loads:
                worker_id = entry.get("worker_id") if isinstance(entry, dict) else None
                potential_prefill = (
                    entry.get("potential_prefill_tokens")
                    if isinstance(entry, dict)
                    else None
                )
                if isinstance(worker_id, int) and isinstance(potential_prefill, int):
                    cached = max(0, estimated_prompt_tokens - potential_prefill)
                    # Multiple dp_ranks per worker_id collapse: keep the max.
                    worker_cached_tokens[worker_id] = max(
                        worker_cached_tokens.get(worker_id, 0), cached
                    )

        decision = await self._scheduler.before_request(
            program_id,
            estimated_prompt_tokens=estimated_prompt_tokens,
            worker_cached_tokens=worker_cached_tokens,
        )
        worker_pin = decision.assigned_worker_hint

        preprocessed = _wrap_preprocessed_request(request)
        if decision.priority_jump != 0.0:
            routing = preprocessed.get("routing") or {}
            existing = routing.get("priority_jump") or 0.0
            routing["priority_jump"] = float(existing) + decision.priority_jump
            preprocessed["routing"] = routing

        if worker_pin is not None:
            routing = preprocessed.get("routing") or {}
            routing["backend_instance_id"] = worker_pin
            preprocessed["routing"] = routing

        prompt_tokens_seen = 0
        completion_tokens_seen = 0
        cached_tokens_seen = 0
        first_chunk = True
        try:
            async for chunk in await self._kv_router.generate_from_request(
                preprocessed  # type: ignore[arg-type]
            ):
                if first_chunk and worker_pin is None:
                    first_chunk = False
                    selected_worker = _worker_id_from_chunk(chunk)
                    if selected_worker is not None:
                        self._scheduler.assign_worker(program_id, selected_worker)

                usage = (
                    chunk.get("completion_usage") if isinstance(chunk, dict) else None
                )
                if isinstance(usage, dict):
                    prompt_tokens_seen = int(
                        usage.get("prompt_tokens", prompt_tokens_seen)
                    )
                    completion_tokens_seen = int(
                        usage.get("completion_tokens", completion_tokens_seen)
                    )
                    details = usage.get("prompt_tokens_details")
                    if isinstance(details, dict):
                        cached = details.get("cached_tokens")
                        if isinstance(cached, int):
                            cached_tokens_seen = cached
                token_ids_out = (
                    chunk.get("token_ids", []) if isinstance(chunk, dict) else []
                )
                if isinstance(token_ids_out, list) and token_ids_out:
                    completion_tokens_seen += len(token_ids_out)
                    self._scheduler.record_output_tokens(program_id, len(token_ids_out))

                yield chunk
        finally:
            # Fall back to len(token_ids) if the engine didn't report usage,
            # which is still better than upstream's chars/5 estimator.
            if prompt_tokens_seen == 0 and isinstance(token_ids, list):
                prompt_tokens_seen = len(token_ids)
            # Update per-worker prefix-cache hit-rate EMA from this turn.
            program = self._scheduler._table.programs.get(program_id)
            if program is not None and program.assigned_worker_id is not None:
                self._scheduler.update_cache_hit_rate(
                    program.assigned_worker_id,
                    cached_tokens_seen,
                    prompt_tokens_seen,
                )
            await self._scheduler.after_request(
                program_id,
                prompt_tokens_seen,
                completion_tokens_seen,
            )


def _worker_id_from_chunk(chunk: Any) -> Optional[int]:
    """Extract worker_id set by ``inject_worker_id_from_tracker`` in the
    Python bindings."""
    if not isinstance(chunk, dict):
        return None
    dp = chunk.get("disaggregated_params")
    if not isinstance(dp, dict):
        return None
    info = dp.get("worker_id")
    if isinstance(info, dict):
        worker_id = info.get("worker_id")
        if isinstance(worker_id, int):
            return worker_id
    return None


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    config = parse_args()
    logger.info(
        "Starting ThunderAgent Router (endpoint=%s, namespace=%s)",
        config.endpoint,
        config.namespace,
    )

    handler = ThunderAgentRouterHandler(runtime, config)
    await handler.initialize()

    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.thunderagent_router.generate"
    )

    if config.model_name:
        model_path = config.model_path or config.model_name
        logger.info(
            "Registering thunderagent_router as model handler for %s "
            "(model_path=%s)",
            config.model_name,
            model_path,
        )
        await register_model(
            model_input=ModelInput.Tokens,
            model_type=ModelType.Chat | ModelType.Completions,
            endpoint=generate_endpoint,
            model_path=model_path,
            model_name=config.model_name,
        )

    try:
        await generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=[("service", "thunderagent_router")],
        )
    finally:
        await handler.shutdown()
        logger.info("ThunderAgent Router shutting down")


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
