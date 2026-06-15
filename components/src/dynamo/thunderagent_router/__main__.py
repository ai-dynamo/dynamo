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

from dynamo.llm import (
    KvRouter,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
)
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


def _is_trajectory_final(request: dict[str, Any]) -> bool:
    """``nvext.agent_context.trajectory_final`` marks a trajectory's
    last turn. The router releases the program and short-circuits -- the request is NOT
    forwarded to the engine (an empty completion returns), so producers send it as a
    dedicated minimal request (e.g. ``max_tokens=1``; the body is just a carrier).
    It is a separate close ping rather than a flag on the last real turn because a
    reactive agent loop only learns a turn was terminal from its response -- so a run's
    end is typically known only after its last real turn already returned (e.g.
    pi-dynamo-provider fires it on ``agent_end``)."""
    ctx = request.get("agent_context")
    return isinstance(ctx, dict) and bool(ctx.get("trajectory_final"))


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
        self._worker_id_extract_warned = False

    async def initialize(self) -> None:
        # Endpoint shape was validated by ThunderAgentRouterConfig.validate()
        # in parse_args; it also populates ``config.namespace``.
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
        if self._scheduler is None or self._kv_router is None:
            raise RuntimeError(
                "ThunderAgentRouterHandler used before initialize() was called"
            )
        program_id = _extract_program_id(request)

        # A request marked trajectory_final just releases the program from the
        # table and is NOT forwarded to the engine (short-circuit).
        if program_id is not None and _is_trajectory_final(request):
            await self._scheduler.end_program(program_id)
            return

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
        usage_completion_seen = False
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
                    if isinstance(usage.get("completion_tokens"), int):
                        completion_tokens_seen = int(usage["completion_tokens"])
                        usage_completion_seen = True
                token_ids_out = (
                    chunk.get("token_ids", []) if isinstance(chunk, dict) else []
                )
                if isinstance(token_ids_out, list) and token_ids_out:
                    # Engine usage is authoritative if present; only the
                    # token-id fallback path increments completion_tokens_seen.
                    if not usage_completion_seen:
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
        # Expects the shape set by ``inject_worker_id_from_tracker`` in the Python
        # bindings: worker attribution rides ``routing_data.worker_id``. Log once if the
        # shape no longer matches; silent extraction failure here means we lose
        # worker-affinity on pin.
        if not isinstance(chunk, dict):
            self._warn_unexpected_chunk_shape("not a dict")
            return None
        routing_data = chunk.get("routing_data")
        if not isinstance(routing_data, dict):
            self._warn_unexpected_chunk_shape("no routing_data dict")
            return None
        info = routing_data.get("worker_id")
        if isinstance(info, dict):
            # ``WorkerIdInfo`` carries prefill/decode IDs (and DP ranks); there is no
            # nested ``worker_id`` key. The sticky pin is applied as
            # ``backend_instance_id``, which the frontend resolves to the decode/backend
            # worker, so prefer ``decode_worker_id`` and fall back to ``prefill_worker_id``
            # (identical in aggregated mode).
            worker_id = info.get("decode_worker_id")
            if not isinstance(worker_id, int):
                worker_id = info.get("prefill_worker_id")
            if isinstance(worker_id, int):
                return worker_id
        self._warn_unexpected_chunk_shape("worker_id payload shape changed")
        return None

    def _warn_unexpected_chunk_shape(self, reason: str) -> None:
        if self._worker_id_extract_warned:
            return
        self._worker_id_extract_warned = True
        logger.warning(
            "ThunderAgent worker-id extraction failed (%s); subsequent "
            "requests will lose sticky pinning until the binding shape is "
            "fixed.",
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
        # Thread the tool_call/reasoning parsers into register_model so the
        # frontend's response path can translate model-native tool calls (e.g.
        # MiniMax's <minimax:tool_call> XML, Qwen's hermes) into OpenAI
        # tool_calls before pi / openhands / other agents see them. These use
        # the same --dyn-tool-call-parser / --dyn-reasoning-parser flag names
        # (and DYN_TOOL_CALL_PARSER / DYN_REASONING_PARSER env vars) as the
        # standalone dynamo.vllm worker.
        runtime_cfg = ModelRuntimeConfig()
        if config.tool_call_parser:
            runtime_cfg.tool_call_parser = config.tool_call_parser
        if config.reasoning_parser:
            runtime_cfg.reasoning_parser = config.reasoning_parser
        await register_model(
            model_input=ModelInput.Tokens,
            model_type=ModelType.Chat | ModelType.Completions,
            endpoint=generate_endpoint,
            model_path=model_path,
            model_name=config.model_name,
            runtime_config=runtime_cfg,
            # The router is the serving entry point (front door) exposing the
            # OpenAI surface; it has no mandatory peer-role dependency.
            worker_type=WorkerType.Aggregated,
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
