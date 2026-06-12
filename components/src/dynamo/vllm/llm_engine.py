# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM LLMEngine implementation for the unified backend.

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Optional, cast

from vllm.config import VllmConfig
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.inputs import TokensPrompt
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

from dynamo._core import Context
from dynamo.common.backend import logprobs as _shared_logprobs
from dynamo.common.backend import telemetry
from dynamo.common.backend.disagg import require_prefill_result
from dynamo.common.backend.dp_rank import forced_dp_rank, validate_global_dp_rank
from dynamo.common.backend.engine import (
    DYN_ENABLE_TEST_LOGITS_PROCESSOR,
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    LlmRegistration,
    LogitsProcessorSpec,
    is_generation_stage,
    logits_processors_for_request,
    resolve_test_logits_processor_spec,
)
from dynamo.common.backend.health_check import (
    bos_token_id_or,
    build_health_check_payload,
)
from dynamo.common.backend.metrics import (
    ensure_prometheus_multiproc_dir,
    register_global_registry,
)
from dynamo.common.backend.publisher import ComponentSnapshot, KvEventSource, ZmqSource
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput
from dynamo.vllm.args import configure_rl_logprobs_mode, parse_args
from dynamo.vllm.cache_info import (
    configure_kv_event_block_size,
    get_configured_kv_event_block_size,
)
from dynamo.vllm.capacity import per_rank_kv_blocks

from .handlers import (
    VllmEnginePauseController,
    build_sampling_params,
    get_dp_range_for_worker,
)
from .logits_processing import (
    activate_logits_processors,
    register_dynamo_logits_processor,
)

if TYPE_CHECKING:
    from dynamo._core.backend import EngineMetrics  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class _UnifiedStatLogger(StatLoggerBase):
    """vLLM stat-logger that writes a :class:`ComponentSnapshot` into the
    factory's shared dict on every iteration. The framework's poll task
    reads the dict and drives both the router-input signal and the
    ``dynamo_component_*`` gauges."""

    def __init__(self, factory: _UnifiedStatLoggerFactory, dp_rank: int) -> None:
        self._factory = factory
        self.dp_rank = dp_rank

    def record(
        self,
        scheduler_stats: Optional[SchedulerStats],
        iteration_stats: Optional[IterationStats],
        mm_cache_stats: object = None,
        engine_idx: int = 0,
        *args: object,
        **kwargs: object,
    ) -> None:
        if scheduler_stats is None:
            return
        # `num_gpu_blocks` is patched on the factory after AsyncLLM finishes
        # KV profiling. Guard with max(1, ...) so the int-cast is sensible
        # on the few iterations between AsyncLLM startup and the patch.
        total = max(1, self._factory.num_gpu_blocks)
        usage = scheduler_stats.kv_cache_usage
        # vLLM's `prefix_cache_stats` exposes cumulative hits/queries; older
        # versions and configurations without prefix caching omit the field.
        # Treat absent as "no data yet" rather than reporting a 0.0 hit rate.
        hit_rate: Optional[float] = None
        pcs = getattr(scheduler_stats, "prefix_cache_stats", None)
        if pcs is not None:
            hits = getattr(pcs, "hits", 0) or 0
            queries = getattr(pcs, "queries", 0) or 0
            if queries > 0:
                hit_rate = float(hits) / float(queries)
        publisher = self._factory.snapshot_publisher
        if publisher is not None:
            publisher.publish(
                self.dp_rank,
                ComponentSnapshot(
                    kv_used_blocks=int(total * usage),
                    kv_total_blocks=total,
                    gpu_cache_usage=usage,
                    kv_cache_hit_rate=hit_rate,
                    dp_rank=self.dp_rank,
                ),
            )

    def log_engine_initialized(self) -> None:
        pass


class _UnifiedStatLoggerFactory:
    """Shared state for all per-rank stat loggers. ``snapshot_publisher``
    is the Rust-owned :class:`SnapshotPublisher` handed to the engine via
    :meth:`attach_snapshot_publisher`; each per-rank logger calls
    ``publisher.publish(rank, snap)`` from its ``record`` callback.
    ``num_gpu_blocks`` is patched after AsyncLLM finishes KV profiling."""

    def __init__(self) -> None:
        self.snapshot_publisher: Optional[Any] = None
        self.num_gpu_blocks: int = 0

    def __call__(self, vllm_config: VllmConfig, dp_rank: int) -> StatLoggerBase:
        return _UnifiedStatLogger(self, dp_rank)


class VllmLLMEngine(LLMEngine):
    # Class-level default so ``__new__``-built instances (tests skipping
    # ``__init__``) still expose what ``generate()`` reads; ``start()`` sets it.
    _logits_processor_spec: "LogitsProcessorSpec | None" = None

    def __init__(
        self,
        engine_args,
        disaggregation_mode: DisaggregationMode,
        served_model_name: str,
        component: str,
        enable_rl: bool = False,
    ):
        self.engine_args = engine_args
        self.disaggregation_mode = disaggregation_mode
        self._served_model_name = served_model_name
        self._component = component
        self.enable_rl = enable_rl
        self.engine_client: AsyncLLM | None = None
        self._vllm_config: Any = None
        self._default_sampling_params: Any = None
        self._prometheus_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._model_max_len: int | None = None
        self._dp_range: Optional[tuple[int, int]] = None
        # Constructed in start() before AsyncLLM init so vLLM's stat-logger
        # factory call sees a valid object. `num_gpu_blocks` is patched
        # after KV profiling finishes.
        self._stat_logger_factory: Optional[_UnifiedStatLoggerFactory] = None
        self._logits_processor_spec: LogitsProcessorSpec | None = None
        self._pause_controller: VllmEnginePauseController | None = None
        self._pause_lock = asyncio.Lock()
        self._scale_ep_lock = asyncio.Lock()
        self._scale_ep_in_progress = False

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[VllmLLMEngine, WorkerConfig]:
        config = parse_args(argv)

        if config.disaggregation_mode == DisaggregationMode.ENCODE:
            raise NotImplementedError(
                "ENCODE is not supported by the unified vLLM entry point; "
                "use `python -m dynamo.vllm` for multimodal encode workers"
            )

        if not config.served_model_name:
            config.served_model_name = (
                config.engine_args.served_model_name
            ) = config.model

        configure_rl_logprobs_mode(config)

        # _resolve_disaggregation_mode() in DynamoVllmConfig has already
        # promoted the field to a DisaggregationMode enum; the field type
        # is still the input union, so narrow it here for mypy (cast
        # rather than assert so `-O` builds don't drop the narrowing).
        mode = cast(DisaggregationMode, config.disaggregation_mode)
        engine = cls(
            config.engine_args,
            mode,
            served_model_name=config.served_model_name or config.model,
            component=config.component,
            enable_rl=config.enable_rl,
        )
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id  # vLLM's NixlConnector handles its own per-worker IDs
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        # Register the engine-loaded adapter before the engine config is built
        # so vLLM instantiates it. vLLM defaults to tokenizer init, so there is
        # no skip_tokenizer_init flag to flip here (unlike TRT-LLM/SGLang).
        if os.getenv(DYN_ENABLE_TEST_LOGITS_PROCESSOR) == "1" and is_generation_stage(
            self.disaggregation_mode
        ):
            register_dynamo_logits_processor(self.engine_args)

        self._prometheus_temp_dir = ensure_prometheus_multiproc_dir("vllm_prometheus_")

        vllm_config = self.engine_args.create_engine_config(
            usage_context=UsageContext.OPENAI_API_SERVER
        )
        self._vllm_config = vllm_config
        self._default_sampling_params = (
            vllm_config.model_config.get_diff_sampling_param()
        )

        self._dp_range = get_dp_range_for_worker(vllm_config)
        self._stat_logger_factory = _UnifiedStatLoggerFactory()

        self.engine_client = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=UsageContext.OPENAI_API_SERVER,
            stat_loggers=[self._stat_logger_factory],
        )
        # Resolve once the tokenizer is available (see logits_processor_spec()).
        self._logits_processor_spec = await self.logits_processor_spec()
        self._pause_controller = VllmEnginePauseController(self.engine_client)
        num_gpu_blocks = self.engine_client.vllm_config.cache_config.num_gpu_blocks or 0
        per_rank_num_gpu_blocks = per_rank_kv_blocks(num_gpu_blocks, self._dp_range[1])
        if per_rank_num_gpu_blocks is None:
            raise RuntimeError("per-rank KV block count is not set")
        self._stat_logger_factory.num_gpu_blocks = per_rank_num_gpu_blocks
        self._model_max_len = getattr(
            getattr(vllm_config, "model_config", None), "max_model_len", None
        )

        # The KV-event block size can differ from `cache_config.block_size`
        # (the main-attention block size lives in cache-group metadata).
        # Populate `additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY]` so
        # `get_configured_kv_event_block_size` returns the right value
        # both to the runtime via `EngineConfig` and to any future readers.
        await configure_kv_event_block_size(self.engine_client, vllm_config)
        block_size = get_configured_kv_event_block_size(vllm_config)

        return EngineConfig(
            model=self.engine_args.model,
            served_model_name=self.engine_args.served_model_name,
            llm=LlmRegistration(
                context_length=self._model_max_len,
                kv_cache_block_size=block_size,
                total_kv_blocks=per_rank_num_gpu_blocks,
                max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
                max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
                # Router needs the rank range to enumerate per-rank load.
                data_parallel_start_rank=self._dp_range[0],
                data_parallel_size=self._dp_range[1],
            ),
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if self.engine_client is None:
            raise RuntimeError("Engine not initialized")
        if self._default_sampling_params is None:
            raise RuntimeError("Engine not initialized")

        request_id = context.id()

        token_ids = request.get("token_ids", [])
        prompt = TokensPrompt(prompt_token_ids=token_ids)

        # TODO: remove dict() once build_sampling_params accepts GenerateRequest
        sampling_params = build_sampling_params(
            dict(request),
            self._default_sampling_params,
            self._model_max_len,
            enable_rl=self.enable_rl,
        )

        # vLLM's KV transfer is internal to NixlConnector
        # (--kv-transfer-config). Dispatch only sets connector hints and
        # forwards the prefill→decode handoff payload.
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            # `do_remote_decode` is prefill's to own; merge caller-supplied
            # values on top of the defaults so explicit overrides win.
            kv_defaults = {
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            }
            caller_kv = sampling_params.extra_args.get("kv_transfer_params", {})
            sampling_params.extra_args["kv_transfer_params"] = {
                **kv_defaults,
                **caller_kv,
                "do_remote_decode": True,
            }
            sampling_params.max_tokens = 1
            sampling_params.min_tokens = 1
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            prefill_result = require_prefill_result(request, self.disaggregation_mode)
            # `disaggregated_params` may be present-but-None (prefill error path
            # / _build_disaggregated_params returning None), so use `or {}` — a
            # .get default only applies when the key is absent. A None value then
            # falls through to the kv_params ValueError below instead of raising
            # AttributeError on None.get(...).
            disaggregated_params = prefill_result.get("disaggregated_params") or {}
            kv_params = disaggregated_params.get("kv_transfer_params")
            if kv_params is None:
                raise ValueError(
                    "decode worker received prefill_result without "
                    "kv_transfer_params; the prefill peer must populate "
                    "this for vLLM's NixlConnector to pull KV blocks"
                )
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args["kv_transfer_params"] = kv_params

        # Shared gating returns [] for PREFILL / hook-off, so this is a no-op
        # unless the hook is on and this is a generation worker.
        entries = logits_processors_for_request(
            self._logits_processor_spec,
            disaggregation_mode=self.disaggregation_mode,
        )
        activate_logits_processors(sampling_params, entries)

        # Honour the router's DP rank decision; without it vLLM picks
        # its own rank and KV events land on the wrong publisher. vLLM
        # expects a local rank, so subtract this worker's `dp_start`.
        local_dp_rank: Optional[int] = None
        if self._dp_range is not None:
            dp_start, dp_size = self._dp_range
            rank = validate_global_dp_rank(
                forced_dp_rank(request), dp_start, dp_size, "vLLM"
            )
            local_dp_rank = None if rank is None else rank - dp_start

        gen = self.engine_client.generate(
            prompt,
            sampling_params,
            request_id,
            data_parallel_rank=local_dp_rank,
            **telemetry.engine_trace_kwargs(context),
        )

        is_prefill = self.disaggregation_mode == DisaggregationMode.PREFILL
        tokenizer = getattr(self.engine_client, "tokenizer", None)
        # vLLM emits a selected-token logprob dict even at `logprobs=0`,
        # so the top-k suppression happens below, not at the engine.
        (
            requested_logprobs_count,
            requested_prompt_logprobs_count,
        ) = _shared_logprobs.parse_logprob_options(
            request.get("output_options", {}) or {}
        )

        total_output_tokens_by_index: dict[int, int] = {}
        async for res in gen:
            if not res.outputs:
                yield {
                    "finish_reason": "error: No outputs from vLLM engine",
                    "index": 0,
                    "token_ids": [],
                }
                break

            prepared_outputs = []
            for output in res.outputs:
                output_idx = getattr(output, "index", 0) or 0
                token_ids = list(output.token_ids or [])
                total_output_tokens_by_index[
                    output_idx
                ] = total_output_tokens_by_index.get(output_idx, 0) + len(token_ids)
                finish_reason = getattr(output, "finish_reason", None)
                if not token_ids and not finish_reason:
                    continue
                prepared_outputs.append((output, output_idx, token_ids, finish_reason))

            for output, output_idx, token_ids, finish_reason in prepared_outputs:
                out: GenerateChunk = {
                    "index": output_idx,
                    "token_ids": token_ids,
                }

                # `build_sampling_params` forces DELTA output → offset 0.
                # `fallback_to_first_on_missing=True` matches legacy
                # vLLM handler: always emit when vLLM returned a dict.
                (
                    log_probs,
                    top_logprobs,
                ) = _shared_logprobs.extract_from_completion_output(
                    output,
                    0,
                    tokenizer=tokenizer,
                    fallback_to_first_on_missing=True,
                    include_bytes=True,
                )
                if log_probs is not None:
                    out["log_probs"] = log_probs
                if (
                    top_logprobs is not None
                    and requested_logprobs_count is not None
                    and requested_logprobs_count > 0
                ):
                    out["top_logprobs"] = top_logprobs

                if finish_reason:
                    out["finish_reason"] = str(finish_reason)
                    # vLLM hangs prompt_logprobs off `RequestOutput`, not
                    # `CompletionOutput` — read from `res`.
                    if requested_prompt_logprobs_count is not None:
                        prompt_payload = _shared_logprobs.extract_prompt_logprobs_from_completion_output(
                            res, tokenizer=tokenizer
                        )
                        if prompt_payload is not None:
                            out["engine_data"] = {"prompt_logprobs": prompt_payload}
                    prompt_tokens = (
                        len(res.prompt_token_ids) if res.prompt_token_ids else 0
                    )
                    completion_tokens = sum(total_output_tokens_by_index.values())
                    out["completion_usage"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                    # Stamp the connector's transfer handle on the
                    # prefill terminal so PrefillRouter can forward it.
                    if is_prefill:
                        kv_transfer_params = getattr(res, "kv_transfer_params", None)
                        if kv_transfer_params is not None:
                            out["disaggregated_params"] = {
                                "kv_transfer_params": kv_transfer_params,
                            }

                yield out

    def _logits_tokenizer(self) -> Any:
        """Tokenizer the smoke hook tokenizes ``"Hello world!"`` with.

        Accessed lazily (only when the env hook is on, inside the resolver).
        vLLM's ``AsyncLLM`` exposes the HF tokenizer as ``.tokenizer``; if a
        future version moves it, this is the one place to adjust.
        """
        if self.engine_client is None:
            raise RuntimeError("Engine not initialized")
        tokenizer = getattr(self.engine_client, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "vLLM engine exposes no tokenizer; "
                f"{DYN_ENABLE_TEST_LOGITS_PROCESSOR} requires tokenizer init"
            )
        return tokenizer

    async def logits_processor_spec(self) -> LogitsProcessorSpec | None:
        # Only generation roles ever attach (PREFILL gates out per request),
        # so skip spec resolution — and the tokenizer it needs — otherwise.
        if not is_generation_stage(self.disaggregation_mode):
            return None
        return resolve_test_logits_processor_spec(self._logits_tokenizer)

    def _kv_routing_enabled(self) -> bool:
        # Matches the legacy `setup_kv_event_publisher` gate.
        if not self.engine_args.enable_prefix_caching:
            return False
        kv_events_config = self.engine_args.kv_events_config
        if kv_events_config is None or not kv_events_config.enable_kv_cache_events:
            return False
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            return False
        return True

    async def kv_event_sources(self) -> list[KvEventSource]:
        if not self._kv_routing_enabled():
            return []
        if self._vllm_config is None:
            raise RuntimeError("Engine not initialized")
        if self._dp_range is None:
            raise RuntimeError("Engine not initialized")
        kv_events_config = self.engine_args.kv_events_config
        dp_start, dp_size = self._dp_range
        return [
            ZmqSource(
                endpoint=ZmqEventPublisher.offset_endpoint_port(
                    kv_events_config.endpoint,
                    data_parallel_rank=rank,
                ).replace("*", "127.0.0.1"),
                dp_rank=rank,
            )
            for rank in range(dp_start, dp_start + dp_size)
        ]

    def component_metrics_dp_ranks(self) -> list[int]:
        # Gauges are observability data — emit regardless of router state.
        if self._dp_range is None:
            return []
        dp_start, dp_size = self._dp_range
        return list(range(dp_start, dp_start + dp_size))

    def attach_snapshot_publisher(self, publisher) -> None:
        # Stash on the factory so each per-rank _UnifiedStatLogger's
        # `record` callback (running on the engine iteration thread)
        # can push snapshots inline. Push is event-driven — no poll.
        if self._stat_logger_factory is not None:
            self._stat_logger_factory.snapshot_publisher = publisher

    async def register_prometheus(self, metrics: "EngineMetrics") -> None:
        # Framework owns the dynamo_component_* registry; we just bridge
        # vLLM's global REGISTRY (vllm: + lmcache:) for /metrics passthrough.
        # The helper handles the K8s MultiProcessCollector conflict case.
        if not self.engine_args.disable_log_stats:
            register_global_registry(
                metrics,
                engine_prefix="vllm:",
                multiproc_only_prefixes=["lmcache:"],
            )

    def supported_controls(self) -> set[str]:
        controls = {"start_profile", "stop_profile", "sleep", "wake_up"}
        if self.engine_client is not None and hasattr(
            self.engine_client, "scale_elastic_ep"
        ):
            controls.add("scale_elastic_ep")
        return controls

    async def engine_control(self, control: str, body: dict) -> dict:
        handlers = {
            "start_profile": self.start_profile,
            "stop_profile": self.stop_profile,
            "sleep": self.sleep,
            "wake_up": self.wake_up,
        }
        if self.engine_client is not None and hasattr(
            self.engine_client, "scale_elastic_ep"
        ):
            handlers["scale_elastic_ep"] = self.scale_elastic_ep

        handler = handlers.get(control)
        if handler is None:
            return {
                "status": "error",
                "message": f"unsupported engine control: {control}",
            }
        return await handler(body or {})

    async def sleep(self, body: dict) -> dict:
        body = body or {}
        level = body.get("level", 1)
        controller = self._pause_controller
        if controller is None:
            return {"status": "error", "message": "engine is not initialized"}

        async with self._pause_lock:
            if controller.is_paused:
                return {"status": "ok", "message": "Engine already sleeping"}
            if controller.needs_resume_recovery:
                return {
                    "status": "error",
                    "message": "wake_up required before retrying sleep",
                }
            try:
                if not await controller.pause(level):
                    return {"status": "ok", "message": "Engine already sleeping"}
                return {"status": "ok", "message": f"Engine slept (level={level})"}
            except Exception as e:
                logger.error("Failed to sleep engine: %s", e)
                return {"status": "error", "message": str(e)}

    async def wake_up(self, body: dict) -> dict:
        body = body or {}
        tags = body.get("tags")
        controller = self._pause_controller
        if controller is None:
            return {"status": "error", "message": "engine is not initialized"}

        async with self._pause_lock:
            needs_recovery = controller.needs_resume_recovery
            if not controller.is_paused and not needs_recovery:
                return {"status": "ok", "message": "Engine already awake"}
            try:
                await controller.resume(tags)
                controller.mark_resumed()
                return {"status": "ok", "message": "Engine woke"}
            except Exception as e:
                logger.error("Failed to wake up engine: %s", e)
                return {"status": "error", "message": str(e)}

    async def start_profile(self, body: dict) -> dict:
        body = body or {}
        if self.engine_client is None:
            return {"status": "error", "message": "engine is not initialized"}
        profile_prefix = body.get("profile_prefix")
        try:
            await self.engine_client.start_profile(profile_prefix=profile_prefix)
            return {"status": "ok", "message": "Profiling started"}
        except Exception as e:
            logger.error("Failed to start profiling: %s", e)
            return {"status": "error", "message": str(e)}

    async def stop_profile(self, body: dict) -> dict:
        if self.engine_client is None:
            return {"status": "error", "message": "engine is not initialized"}
        try:
            await self.engine_client.stop_profile()
            return {"status": "ok", "message": "Profiling stopped"}
        except Exception as e:
            logger.error("Failed to stop profiling: %s", e)
            return {"status": "error", "message": str(e)}

    async def scale_elastic_ep(self, body: dict) -> dict:
        body = body or {}
        if self.engine_client is None:
            return {"status": "error", "message": "engine is not initialized"}
        new_dp_size = body.get("new_data_parallel_size")
        if new_dp_size is None:
            return {
                "status": "error",
                "message": "Missing required field: new_data_parallel_size",
            }
        try:
            new_dp_size = int(new_dp_size)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "message": f"new_data_parallel_size must be an integer, got: {new_dp_size!r}",
            }
        if new_dp_size < 2:
            return {
                "status": "error",
                "message": "new_data_parallel_size must be >= 2 when elastic EP/ePLB is enabled",
            }

        async with self._scale_ep_lock:
            if self._scale_ep_in_progress:
                msg = (
                    "A scale_elastic_ep operation is already in progress; "
                    f"rejecting concurrent request for new_data_parallel_size={new_dp_size}"
                )
                logger.warning("[ElasticEP] %s", msg)
                return {"status": "error", "message": msg}
            self._scale_ep_in_progress = True

        try:
            import ray
            import ray.util.state as _ray_util_state

            class _NodeInfo:
                __slots__ = ("node_id", "node_ip")

                def __init__(self, d: dict) -> None:
                    self.node_ip: str = d["NodeManagerAddress"]
                    self.node_id: str = d["NodeID"]

            original_list_nodes = _ray_util_state.list_nodes
            try:
                _ray_util_state.list_nodes = lambda **kw: [
                    _NodeInfo(n) for n in ray.nodes() if n.get("Alive", False)
                ]
                await self.engine_client.scale_elastic_ep(new_dp_size)
            finally:
                _ray_util_state.list_nodes = original_list_nodes

            return {
                "status": "ok",
                "message": f"Scaled to data_parallel_size={new_dp_size}",
                "new_data_parallel_size": new_dp_size,
            }
        except Exception as e:
            logger.error("[ElasticEP] Scaling failed: %s", e)
            return {"status": "error", "message": str(e)}
        finally:
            async with self._scale_ep_lock:
                self._scale_ep_in_progress = False

    async def abort(self, context: Context) -> None:
        request_id = context.id()
        if self.engine_client is not None and request_id is not None:
            await self.engine_client.abort(request_id)
            logger.debug("Aborted request %s", request_id)

    async def health_check_payload(self) -> Optional[dict[str, Any]]:
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            logger.warning(
                "DECODE worker: health-check canary disabled — "
                "NixlConnector has no verified local-only bypass. "
                "Endpoint readiness will rely on real request traffic."
            )
            return None
        bos = bos_token_id_or(getattr(self.engine_client, "tokenizer", None))
        return build_health_check_payload(bos_token_id=bos)

    async def cleanup(self) -> None:
        try:
            if self.engine_client is not None:
                self.engine_client.shutdown()
        finally:
            self.engine_client = None
            self._pause_controller = None
            if self._prometheus_temp_dir is not None:
                if (
                    os.environ.get("PROMETHEUS_MULTIPROC_DIR")
                    == self._prometheus_temp_dir.name
                ):
                    os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
                self._prometheus_temp_dir.cleanup()
                self._prometheus_temp_dir = None
            logger.info("vLLM engine shutdown")
