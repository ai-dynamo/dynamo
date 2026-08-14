# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone ThunderAgent router service.

Usage:
    python -m dynamo.thunderagent_router \\
        --endpoint dynamo.vllm.generate \\
        --router-block-size 64

Serves ``{namespace}.thunderagent_router.generate``. Pause/resume is
opt-in per-request via header-derived ``session_id``; requests without it
are routed via plain KvRouter with no lifecycle.
"""

from __future__ import annotations

import asyncio
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
    pid = ctx.get("session_id")
    if isinstance(pid, str) and pid:
        return pid
    return None


def _is_session_final(request: dict[str, Any]) -> bool:
    """``x-dynamo-session-final`` marks a session's last turn internally."""
    ctx = request.get("agent_context")
    return isinstance(ctx, dict) and bool(ctx.get("session_final"))


def _nvext_extra_field_requested(request: dict[str, Any], field: str) -> bool:
    """Return whether the raw or preprocessed request opted into an nvext field."""
    sources: list[Any] = [request.get("nvext")]
    extra_args = request.get("extra_args")
    if isinstance(extra_args, dict):
        sources.append(extra_args.get("nvext"))

    for source in sources:
        if not isinstance(source, dict):
            continue
        extra_fields = source.get("extra_fields")
        if isinstance(extra_fields, list) and field in extra_fields:
            return True
    return False


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


def _inject_thunderagent_route_proof(
    chunk: Any,
    proof: dict[str, Any],
) -> None:
    if not isinstance(chunk, dict):
        return

    engine_data = chunk.get("engine_data")
    if engine_data is None:
        engine_data = {}
    elif not isinstance(engine_data, dict):
        engine_data = {"backend_engine_data": engine_data}

    engine_data["thunderagent"] = dict(proof)
    chunk["engine_data"] = engine_data


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
        self._stat_requests_total = 0
        self._stat_program_requests = 0
        self._stat_passthrough_requests = 0
        self._stat_session_final_requests = 0

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

        worker_client = await worker_endpoint.client()
        self._capacity = WorkerCapacityProvider(worker_endpoint, worker_client)
        self._capacity.start()

        self._scheduler = ThunderAgentScheduler(
            capacity=self._capacity,
            config=self._config.to_thunderagent_config(),
        )
        self._scheduler.start()
        logger.info(
            "ThunderAgent Router initialized (worker_endpoint=%s, block_size=%s)",
            self._config.endpoint,
            self._config.router_block_size,
        )

    async def shutdown(self) -> None:
        if self._scheduler is not None:
            await self._scheduler.stop()
        if self._capacity is not None:
            self._capacity.stop()
        logger.info("ThunderAgent Router shutdown complete")

    async def generate(self, request: dict[str, Any]):
        if self._scheduler is None or self._kv_router is None:
            raise RuntimeError(
                "ThunderAgentRouterHandler used before initialize() was called"
            )
        program_id = _extract_program_id(request)
        want_route_proof = _nvext_extra_field_requested(request, "engine_data")
        self._stat_requests_total += 1

        # A request marked session_final just releases the program from the
        # table and is NOT forwarded to the engine (short-circuit).
        if program_id is not None and _is_session_final(request):
            self._stat_session_final_requests += 1
            released = await self._scheduler.end_program(program_id)
            logger.info(
                "thunderagent.route path=session_final program=%s released=%s",
                program_id,
                released,
            )
            return

        # Path A: no program_id -> behave like the standalone router.
        if program_id is None:
            self._stat_passthrough_requests += 1
            logger.debug(
                "thunderagent.route path=passthrough model=%s", request.get("model")
            )
            preprocessed = _wrap_preprocessed_request(request)
            proof: Optional[dict[str, Any]] = (
                {
                    "handled_by": "thunderagent_router",
                    "path": "passthrough",
                    "program_id": None,
                    "session_final": False,
                }
                if want_route_proof
                else None
            )
            first_chunk = True
            async for chunk in await self._kv_router.generate_from_request(
                preprocessed  # type: ignore[arg-type]
            ):
                if proof is not None:
                    if first_chunk:
                        first_chunk = False
                        selected_worker = self._extract_worker_id(chunk)
                        if selected_worker is not None:
                            proof["selected_worker_id"] = selected_worker
                    _inject_thunderagent_route_proof(chunk, proof)
                yield chunk
            return

        # Path B: program lifecycle.
        self._stat_program_requests += 1
        token_ids = request["token_ids"]
        estimated_prompt_tokens = len(token_ids) if isinstance(token_ids, list) else 0

        decision = await self._scheduler.before_request(
            program_id,
            estimated_prompt_tokens=estimated_prompt_tokens,
        )
        worker_pin = decision.assigned_worker_hint
        logger.debug(
            "thunderagent.route path=program program=%s prompt_tokens=%d "
            "worker_hint=%s waited_seconds=%.4f was_paused=%s "
            "soft_demoted=%s priority_jump=%.3f",
            program_id,
            estimated_prompt_tokens,
            worker_pin,
            decision.waited_seconds,
            decision.was_paused,
            decision.was_soft_demoted,
            decision.priority_jump,
        )

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
        selected_worker_id = None
        proof = (
            {
                "handled_by": "thunderagent_router",
                "path": "program",
                "program_id": program_id,
                "session_final": False,
                "was_paused": decision.was_paused,
                "was_soft_demoted": decision.was_soft_demoted,
                "waited_seconds": decision.waited_seconds,
                "priority_jump": decision.priority_jump,
                "assigned_worker_hint": worker_pin,
            }
            if want_route_proof
            else None
        )
        try:
            async for chunk in await self._kv_router.generate_from_request(
                preprocessed  # type: ignore[arg-type]
            ):
                if first_chunk and worker_pin is None:
                    first_chunk = False
                    selected_worker = self._extract_worker_id(chunk)
                    if selected_worker is not None:
                        await self._scheduler.assign_worker(program_id, selected_worker)
                        selected_worker_id = selected_worker
                        if proof is not None:
                            proof["selected_worker_id"] = selected_worker
                        logger.debug(
                            "thunderagent.route_selected program=%s worker=%s "
                            "source=first_chunk",
                            program_id,
                            selected_worker,
                        )

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

                if proof is not None:
                    _inject_thunderagent_route_proof(chunk, proof)
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
            logger.debug(
                "thunderagent.request_complete program=%s prompt_tokens=%d "
                "completion_tokens=%d worker_hint=%s selected_worker=%s",
                program_id,
                prompt_tokens_seen,
                completion_tokens_seen,
                worker_pin,
                selected_worker_id,
            )

    async def status(self, request: Optional[dict[str, Any]] = None):
        scheduler_status = (
            await self._scheduler.status_snapshot()
            if self._scheduler is not None
            else None
        )
        yield {
            "status": "ready" if self._scheduler is not None else "starting",
            "component": "thunderagent_router",
            "namespace": self._config.namespace,
            "worker_endpoint": self._config.endpoint,
            "scheduler": scheduler_status,
            "requests": {
                "total": self._stat_requests_total,
                "program": self._stat_program_requests,
                "passthrough": self._stat_passthrough_requests,
                "session_final": self._stat_session_final_requests,
            },
        }

    async def metrics(self, request: Optional[dict[str, Any]] = None):
        scheduler_metrics = (
            await self._scheduler.metrics_snapshot()
            if self._scheduler is not None
            else {"counters": {}, "gauges": {}, "workers": {}}
        )
        counters = {
            **scheduler_metrics["counters"],
            "requests_total": self._stat_requests_total,
            "program_requests_total": self._stat_program_requests,
            "passthrough_requests_total": self._stat_passthrough_requests,
            "session_final_requests_total": self._stat_session_final_requests,
        }
        yield {
            "component": "thunderagent_router",
            "namespace": self._config.namespace,
            "counters": counters,
            "gauges": scheduler_metrics["gauges"],
            "workers": scheduler_metrics["workers"],
        }

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
            # worker, so prefer ``decode_worker_id`` and fall back to
            # ``prefill_worker_id`` (identical in aggregated mode).
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

    status_endpoint = runtime.endpoint(f"{config.namespace}.thunderagent_router.status")
    metrics_endpoint = runtime.endpoint(
        f"{config.namespace}.thunderagent_router.metrics"
    )

    logger.info(
        "ThunderAgent Router serving endpoints: generate=%s status=%s metrics=%s",
        f"{config.namespace}.thunderagent_router.generate",
        f"{config.namespace}.thunderagent_router.status",
        f"{config.namespace}.thunderagent_router.metrics",
    )

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[("service", "thunderagent_router")],
            ),
            status_endpoint.serve_endpoint(
                handler.status,
                graceful_shutdown=True,
                metrics_labels=[("service", "thunderagent_router")],
            ),
            metrics_endpoint.serve_endpoint(
                handler.metrics,
                graceful_shutdown=True,
                metrics_labels=[("service", "thunderagent_router")],
            ),
        )
    finally:
        await handler.shutdown()


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
