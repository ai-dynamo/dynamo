# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone ThunderAgent router service.

Usage:
    python -m dynamo.thunderagent_router \\
        --endpoint dynamo.vllm.generate \\
        --router-block-size 64

Mirrors the integration shape of ``dynamo.thompson_router`` (PR #8522). The
service serves ``{namespace}.thunderagent_router.generate`` as a model-handler
endpoint; the frontend's discovery picks it up the same way it would any other
worker endpoint and dispatches LLM requests to it.

Reads ``request["agent_context"]["trajectory_id"]`` to attach lifecycle decisions
to programs (Dynamo's ``trajectory_id`` is the analogue of ThunderAgent's
``program_id``). Requests that lack ``agent_context`` are routed normally with
no pause/resume gating -- the v0 contract is "opt-in via nvext.agent_context".
"""

from __future__ import annotations

import asyncio
import logging
import time
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
from dynamo.thunderagent_router.router import KvThunderAgentRouter

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def _extract_program_id(request: dict[str, Any]) -> Optional[str]:
    """Pull the scheduling program_id off ``nvext.agent_context.trajectory_id``.

    The frontend preprocessor parses ``nvext.agent_context`` into a top-level
    ``agent_context`` field on PreprocessedRequest, which depythonizes into the
    request dict we receive here. Dynamo's ``trajectory_id`` is the
    schedulable identifier and maps 1:1 to ThunderAgent's ``program_id``.
    """
    ctx = request.get("agent_context")
    if not isinstance(ctx, dict):
        return None
    pid = ctx.get("trajectory_id")
    if isinstance(pid, str) and pid:
        return pid
    return None


def _wrap_preprocessed_request(request: dict[str, Any]) -> dict[str, Any]:
    """Build the PreprocessedRequest dict KvRouter.generate_from_request expects.

    Mirrors the wrapping done in ``dynamo.router/__main__.py``.
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
    """Glue between the Dynamo endpoint runtime and KvThunderAgentRouter."""

    def __init__(
        self,
        runtime: DistributedRuntime,
        config: ThunderAgentRouterConfig,
    ) -> None:
        self._runtime = runtime
        self._config = config
        self._kv_router: Optional[KvRouter] = None
        self._capacity: Optional[FpmCapacityProvider] = None
        self._scheduler: Optional[KvThunderAgentRouter] = None

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

        self._scheduler = KvThunderAgentRouter(
            kv_router=self._kv_router,
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
        """Wrap KvRouter.generate_from_request with ThunderAgent admission +
        KV-aware resume placement + token accounting."""
        if self._kv_router is None or self._scheduler is None:
            raise RuntimeError("ThunderAgentRouterHandler not initialized")

        if not getattr(self, "_logged_first_req", False):
            self._logged_first_req = True
            try:
                tids = request.get("token_ids")
                ac = request.get("agent_context")
                logger.info(
                    "first request shape: keys=%s token_ids_type=%s "
                    "token_ids_len=%s agent_context=%s",
                    sorted(list(request.keys()))
                    if isinstance(request, dict)
                    else type(request),
                    type(tids).__name__ if tids is not None else "missing",
                    len(tids) if isinstance(tids, list) else "n/a",
                    ac,
                )
            except Exception as exc:
                logger.warning("first-request diagnostic failed: %s", exc)

        program_id = _extract_program_id(request)

        # Path A: no program_id -> behave like the standalone router (no
        # admission, no lifecycle). Keeps backward compatibility for clients
        # that don't send agent_context.
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
        decision = await self._scheduler.before_request(
            program_id, estimated_prompt_tokens=estimated_prompt_tokens
        )

        # Item 2: KV-aware resume placement. If the scheduler thinks this
        # turn is a resume, ask the KV router for the warmest-prefix worker.
        worker_pin = await self._scheduler.select_worker(
            program_id, token_ids, decision.was_paused
        )
        if decision.was_paused or decision.waited_seconds > 0.010:
            logger.info(
                "admission decision program=%s waited_seconds=%.3f "
                "was_paused=%s priority_jump=%.3f worker_pin=%s",
                program_id,
                decision.waited_seconds,
                decision.was_paused,
                decision.priority_jump,
                worker_pin,
            )

        # Fold decision.priority_jump into the request before it reaches the
        # KV router queue. The frontend may have already populated routing
        # with client-supplied priority_jump; we add to it.
        preprocessed = _wrap_preprocessed_request(request)
        if decision.priority_jump != 0.0:
            routing = preprocessed.get("routing") or {}
            existing = routing.get("priority_jump") or 0.0
            routing["priority_jump"] = float(existing) + decision.priority_jump
            preprocessed["routing"] = routing

        # Apply soft worker pin only when KV-aware resume returned a target.
        # Otherwise let the KV router pick freely from the indexer.
        if worker_pin is not None:
            routing = preprocessed.get("routing") or {}
            routing["backend_instance_id"] = worker_pin
            preprocessed["routing"] = routing
            self._scheduler.assign_worker(program_id, worker_pin)

        prompt_tokens_seen = 0
        completion_tokens_seen = 0
        first_chunk_with_worker = True
        try:
            async for chunk in await self._kv_router.generate_from_request(
                preprocessed  # type: ignore[arg-type]
            ):
                if first_chunk_with_worker:
                    first_chunk_with_worker = False
                    selected_worker = _worker_id_from_chunk(chunk)
                    if selected_worker is not None:
                        self._scheduler.assign_worker(program_id, selected_worker)

                # Item 1: real token accounting from the response usage.
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
                token_ids_out = (
                    chunk.get("token_ids", []) if isinstance(chunk, dict) else []
                )
                if isinstance(token_ids_out, list) and token_ids_out:
                    completion_tokens_seen += len(token_ids_out)
                    self._scheduler.record_output_tokens(program_id, len(token_ids_out))

                yield chunk
        finally:
            # Item 1: capture real prompt_tokens; if no usage was reported,
            # fall back to len(token_ids) which is still better than chars/5.
            if prompt_tokens_seen == 0 and isinstance(token_ids, list):
                prompt_tokens_seen = len(token_ids)
            if not getattr(self, "_logged_first_after", False):
                self._logged_first_after = True
                logger.info(
                    "first after_request: pid=%s prompt=%d completion=%d "
                    "token_ids_type=%s token_ids_len=%s",
                    program_id,
                    prompt_tokens_seen,
                    completion_tokens_seen,
                    type(token_ids).__name__,
                    len(token_ids) if isinstance(token_ids, list) else "n/a",
                )
            await self._scheduler.after_request(
                program_id,
                prompt_tokens_seen,
                completion_tokens_seen,
                last_prefix_token_ids=token_ids
                if isinstance(token_ids, list)
                else None,
            )

    async def stats(self, _request: Any):
        """Diagnostic endpoint -- returns scheduler counters as JSON."""
        if self._scheduler is None:
            yield {"error": "not initialized"}
            return
        yield {
            "ts": time.time(),
            "scheduler": self._scheduler.stats(),
        }


def _worker_id_from_chunk(chunk: Any) -> Optional[int]:
    """Extract worker_id from KvRouter response disaggregated_params (set by
    ``inject_worker_id_from_tracker`` in the Python bindings)."""
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
    stats_endpoint = runtime.endpoint(f"{config.namespace}.thunderagent_router.stats")

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
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[("service", "thunderagent_router")],
            ),
            stats_endpoint.serve_endpoint(
                handler.stats,
                graceful_shutdown=True,
                metrics_labels=[("service", "thunderagent_router")],
            ),
        )
    finally:
        await handler.shutdown()
        logger.info("ThunderAgent Router shutting down")


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
