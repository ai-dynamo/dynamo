# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone ThunderAgent router service.

Usage:
    python -m dynamo.thunderagent_router \\
        --endpoint dynamo.vllm.generate \\
        --router-block-size 64

Serves ``{namespace}.thunderagent_router.generate``. Pause/resume is
opt-in per-request via ``nvext.agent_context.trajectory_id``; requests
without it are routed via plain KvRouter with no lifecycle.
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
from dynamo.thunderagent_router.capacity import WorkerCapacityProvider
from dynamo.thunderagent_router.router import ThunderAgentScheduler

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def _extract_program_id(request: dict[str, Any]) -> Optional[str]:
    ctx = request.get("agent_context")
    if not isinstance(ctx, dict):
        return None
    pid = ctx.get("trajectory_id")
    if isinstance(pid, str) and pid:
        return pid
    return None


def _wrap_preprocessed_request(request: dict[str, Any]) -> dict[str, Any]:
    # Duplicated from dynamo.router/__main__.py since neither package exports
    # it. TODO(idhanani): file follow-up to lift this into dynamo.router as a
    # shared helper before the field list drifts.
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
    def __init__(
        self,
        runtime: DistributedRuntime,
        config: ThunderAgentRouterConfig,
    ) -> None:
        self._runtime = runtime
        self._config = config
        self._kv_router: Optional[KvRouter] = None
        self._capacity: Optional[WorkerCapacityProvider] = None
        self._scheduler: Optional[ThunderAgentScheduler] = None
        # First-chunk binding-shape sanity, logged once if the
        # disaggregated_params.worker_id shape changes upstream.
        self._worker_id_extract_warned = False

    async def initialize(self) -> None:
        if self._config.endpoint.count(".") != 2:
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

        self._capacity = WorkerCapacityProvider(worker_endpoint)
        self._capacity.start()

        self._scheduler = ThunderAgentScheduler(
            capacity=self._capacity,
            config=self._config.to_thunderagent_config(),
        )
        self._scheduler.start()

    async def shutdown(self) -> None:
        if self._scheduler is not None:
            await self._scheduler.stop()
        if self._capacity is not None:
            self._capacity.stop()

    async def generate(self, request: dict[str, Any]):
        assert self._scheduler is not None and self._kv_router is not None
        program_id = _extract_program_id(request)

        # Path A: no program_id -> behave like the standalone router.
        # Backward compat for clients that don't send agent_context.
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
            program_id,
            estimated_prompt_tokens=estimated_prompt_tokens,
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
        first_chunk = True
        try:
            async for chunk in await self._kv_router.generate_from_request(
                preprocessed  # type: ignore[arg-type]
            ):
                if first_chunk and worker_pin is None:
                    first_chunk = False
                    selected_worker = self._extract_worker_id(chunk)
                    if selected_worker is not None:
                        await self._scheduler.assign_worker(program_id, selected_worker)

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
            # Fall back to len(token_ids) if the engine didn't report usage --
            # still better than upstream's chars/5 estimator.
            if prompt_tokens_seen == 0 and isinstance(token_ids, list):
                prompt_tokens_seen = len(token_ids)
            await self._scheduler.after_request(
                program_id,
                prompt_tokens_seen,
                completion_tokens_seen,
            )

    def _extract_worker_id(self, chunk: Any) -> Optional[int]:
        # Expects the shape set by ``inject_worker_id_from_tracker`` in the
        # Python bindings. Log once if the shape no longer matches; silent
        # extraction failure here means we lose worker-affinity on pin.
        if not isinstance(chunk, dict):
            self._warn_unexpected_chunk_shape("not a dict")
            return None
        dp = chunk.get("disaggregated_params")
        if not isinstance(dp, dict):
            self._warn_unexpected_chunk_shape("no disaggregated_params dict")
            return None
        info = dp.get("worker_id")
        if isinstance(info, dict):
            worker_id = info.get("worker_id")
            if isinstance(worker_id, int):
                return worker_id
        self._warn_unexpected_chunk_shape("worker_id payload shape changed")
        return None

    def _warn_unexpected_chunk_shape(self, reason: str) -> None:
        if self._worker_id_extract_warned:
            return
        self._worker_id_extract_warned = True
        logger.warning(
            "ThunderAgent worker-id extraction failed (%s); sticky pinning is "
            "off for this request. The disaggregated_params binding shape "
            "may have changed.",
            reason,
        )


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    config = parse_args()
    logger.info(
        "ThunderAgent Router starting (endpoint=%s, namespace=%s)",
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


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
