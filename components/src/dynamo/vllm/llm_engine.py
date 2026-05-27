# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM LLMEngine implementation for the unified backend.

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

from __future__ import annotations

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
from dynamo.common.backend import telemetry
from dynamo.common.backend.disagg import require_prefill_result
from dynamo.common.backend.dp_rank import forced_dp_rank, validate_global_dp_rank
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
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
from dynamo.vllm.args import parse_args
from dynamo.vllm.cache_info import (
    configure_kv_event_block_size,
    get_configured_kv_event_block_size,
)
from dynamo.vllm.capacity import per_rank_kv_blocks

from .handlers import build_sampling_params, extract_logprobs, get_dp_range_for_worker

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
    def __init__(
        self,
        engine_args,
        disaggregation_mode: DisaggregationMode,
        served_model_name: str,
        component: str,
    ):
        self.engine_args = engine_args
        self.disaggregation_mode = disaggregation_mode
        self._served_model_name = served_model_name
        self._component = component
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

        self._prometheus_temp_dir = ensure_prometheus_multiproc_dir("vllm_prometheus_")

        self._default_sampling_params = (
            self.engine_args.create_model_config().get_diff_sampling_param()
        )

        vllm_config = self.engine_args.create_engine_config(
            usage_context=UsageContext.OPENAI_API_SERVER
        )
        self._vllm_config = vllm_config

        self._dp_range = get_dp_range_for_worker(vllm_config)
        self._stat_logger_factory = _UnifiedStatLoggerFactory()

        self.engine_client = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=UsageContext.OPENAI_API_SERVER,
            stat_loggers=[self._stat_logger_factory],
        )
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
            context_length=self._model_max_len,
            kv_cache_block_size=block_size,
            total_kv_blocks=per_rank_num_gpu_blocks,
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
            max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
            # Router needs the rank range to enumerate per-rank load.
            data_parallel_start_rank=self._dp_range[0],
            data_parallel_size=self._dp_range[1],
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
            dict(request), self._default_sampling_params, self._model_max_len
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
            kv_params = prefill_result.get("disaggregated_params", {}).get(
                "kv_transfer_params"
            )
            if kv_params is None:
                raise ValueError(
                    "decode worker received prefill_result without "
                    "kv_transfer_params; the prefill peer must populate "
                    "this for vLLM's NixlConnector to pull KV blocks"
                )
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args["kv_transfer_params"] = kv_params

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
                total_output_tokens_by_index[output_idx] = (
                    total_output_tokens_by_index.get(output_idx, 0) + len(token_ids)
                )
                finish_reason = getattr(output, "finish_reason", None)
                if not token_ids and not finish_reason:
                    continue
                # vLLM streams in DELTA mode (see build_sampling_params), so
                # output.logprobs/token_ids carry only the per-chunk delta;
                # pass offset=0 rather than the running cumulative tally.
                log_probs, top_logprobs = extract_logprobs(output, 0)
                # Pre-extract `cumulative_logprob` so the emit-loop below
                # doesn't reach back into the engine-side type.
                cum_logprob = getattr(output, "cumulative_logprob", None)
                prepared_outputs.append(
                    (
                        output_idx,
                        token_ids,
                        finish_reason,
                        log_probs,
                        top_logprobs,
                        cum_logprob,
                    )
                )

            for (
                output_idx,
                token_ids,
                finish_reason,
                log_probs,
                top_logprobs,
                cum_logprob,
            ) in prepared_outputs:
                out: GenerateChunk = {
                    "index": output_idx,
                    "token_ids": token_ids,
                }
                if log_probs is not None:
                    out["log_probs"] = log_probs
                if top_logprobs is not None:
                    out["top_logprobs"] = top_logprobs

                if finish_reason:
                    out["finish_reason"] = str(finish_reason)
                    # `cumulative_logprob` is a single per-completion summary;
                    # emit it on the terminal chunk only (mirrors how
                    # `completion_usage` is attached below).
                    if cum_logprob is not None:
                        out["cum_log_probs"] = float(cum_logprob)
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
            if self._prometheus_temp_dir is not None:
                if (
                    os.environ.get("PROMETHEUS_MULTIPROC_DIR")
                    == self._prometheus_temp_dir.name
                ):
                    os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
                self._prometheus_temp_dir.cleanup()
                self._prometheus_temp_dir = None
            logger.info("vLLM engine shutdown")
