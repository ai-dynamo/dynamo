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
from collections.abc import AsyncGenerator, Callable
from typing import Any, Optional, cast

from vllm.config import VllmConfig
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.inputs import TokensPrompt
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

from dynamo._core import Context
from dynamo.common.backend.disagg import require_prefill_result
from dynamo.common.backend.dp_rank import forced_dp_rank, validate_global_dp_rank
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.publisher import (
    KvEventSource,
    Metrics,
    SnapshotSource,
    ZmqSource,
)
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput
from dynamo.vllm.args import parse_args
from dynamo.vllm.cache_info import (
    configure_kv_event_block_size,
    get_configured_kv_event_block_size,
)

from .handlers import build_sampling_params, get_dp_range_for_worker

logger = logging.getLogger(__name__)


class _DpRankMetricsCache:
    """Thread-safe (single-writer-per-rank) cache of the latest per-rank
    ``Metrics`` snapshot.

    Written by vLLM's ``StatLoggerBase.record`` callback (one logger per
    dp_rank) and read by ``LLMEngine.metrics_sources`` on a fixed interval
    from the runtime side. Reads return ``None`` until the engine has
    emitted its first scheduler iteration for that rank.
    """

    def __init__(self, num_gpu_blocks: int) -> None:
        # vLLM's `SchedulerStats.kv_cache_usage` is fractional; multiply by
        # total blocks to get the absolute block count the router expects.
        self._num_gpu_blocks = max(1, num_gpu_blocks)
        self._by_rank: dict[int, Metrics] = {}

    def set_num_gpu_blocks(self, num_gpu_blocks: int) -> None:
        self._num_gpu_blocks = max(1, num_gpu_blocks)

    def update(self, dp_rank: int, scheduler_stats: SchedulerStats) -> None:
        kv_used = int(self._num_gpu_blocks * scheduler_stats.kv_cache_usage)
        self._by_rank[dp_rank] = Metrics(kv_used_blocks=kv_used)

    def snapshot(self, dp_rank: int) -> Optional[Metrics]:
        return self._by_rank.get(dp_rank)


class _UnifiedStatLogger(StatLoggerBase):
    """vLLM-side hook that just refreshes the per-rank metrics cache.

    Replaces the legacy ``DynamoStatLoggerPublisher`` (which owned its own
    ``WorkerMetricsPublisher`` and NATS endpoint). Under the unified path
    the Rust ``Worker`` owns publishers, so this class only needs to
    update an in-memory snapshot that ``metrics_sources()`` reads.
    """

    def __init__(self, cache: _DpRankMetricsCache, dp_rank: int) -> None:
        self._cache = cache
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
        self._cache.update(self.dp_rank, scheduler_stats)

    def log_engine_initialized(self) -> None:
        pass


class _UnifiedStatLoggerFactory:
    """vLLM calls this once per dp_rank during engine initialization."""

    def __init__(self, cache: _DpRankMetricsCache) -> None:
        self._cache = cache

    def __call__(self, vllm_config: VllmConfig, dp_rank: int) -> StatLoggerBase:
        return _UnifiedStatLogger(self._cache, dp_rank)


class VllmLLMEngine(LLMEngine):
    def __init__(self, engine_args, disaggregation_mode: DisaggregationMode):
        self.engine_args = engine_args
        self.disaggregation_mode = disaggregation_mode
        self.engine_client: AsyncLLM | None = None
        self._vllm_config: Any = None
        self._default_sampling_params: Any = None
        self._prometheus_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._model_max_len: int | None = None
        # Resolved in `start()`; the cache is patched once vLLM finishes
        # KV profiling and reports `num_gpu_blocks`.
        self._dp_range: Optional[tuple[int, int]] = None
        self._metrics_cache: Optional[_DpRankMetricsCache] = None

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
        engine = cls(config.engine_args, mode)
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

        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            self._prometheus_temp_dir = tempfile.TemporaryDirectory(
                prefix="vllm_prometheus_"
            )
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = self._prometheus_temp_dir.name

        self._default_sampling_params = (
            self.engine_args.create_model_config().get_diff_sampling_param()
        )

        vllm_config = self.engine_args.create_engine_config(
            usage_context=UsageContext.OPENAI_API_SERVER
        )
        self._vllm_config = vllm_config

        self._dp_range = get_dp_range_for_worker(vllm_config)
        # Constructed before init so the stat-logger factory can bind to
        # it; `num_gpu_blocks` is patched below once KV profiling finishes.
        self._metrics_cache = _DpRankMetricsCache(num_gpu_blocks=0)

        self.engine_client = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=UsageContext.OPENAI_API_SERVER,
            stat_loggers=[_UnifiedStatLoggerFactory(self._metrics_cache)],
        )
        num_gpu_blocks = self.engine_client.vllm_config.cache_config.num_gpu_blocks or 0
        self._metrics_cache.set_num_gpu_blocks(num_gpu_blocks)
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
            total_kv_blocks=num_gpu_blocks,
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
            max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
            # Router needs the rank range to enumerate per-rank load.
            data_parallel_start_rank=self._dp_range[0],
            data_parallel_size=self._dp_range[1],
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        assert self.engine_client is not None, "Engine not initialized"
        assert self._default_sampling_params is not None, "Engine not initialized"

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
            prompt, sampling_params, request_id, data_parallel_rank=local_dp_rank
        )

        is_prefill = self.disaggregation_mode == DisaggregationMode.PREFILL

        num_output_tokens_so_far: dict[int, int] = {}
        async for res in gen:
            if not res.outputs:
                yield {
                    "finish_reason": "error: No outputs from vLLM engine",
                    "index": 0,
                    "token_ids": [],
                }
                break

            for output in res.outputs:
                output_idx = getattr(output, "index", 0) or 0
                previous_total = num_output_tokens_so_far.get(output_idx, 0)
                next_total = len(output.token_ids)
                out: GenerateChunk = {
                    "index": output_idx,
                    "token_ids": output.token_ids[previous_total:],
                }

                if output.finish_reason:
                    out["finish_reason"] = str(output.finish_reason)
                    prompt_tokens = (
                        len(res.prompt_token_ids) if res.prompt_token_ids else 0
                    )
                    completion_tokens = sum(
                        len(choice.token_ids) for choice in res.outputs
                    )
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
                num_output_tokens_so_far[output_idx] = next_total

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
        assert self._vllm_config is not None
        assert self._dp_range is not None
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

    async def metrics_sources(self) -> list[SnapshotSource]:
        # Same opt-in gating as kv_event_sources — if this worker isn't
        # publishing KV events, it shouldn't publish load metrics either,
        # or the router will score against partial worker state.
        if not self._kv_routing_enabled():
            return []
        if self._dp_range is None or self._metrics_cache is None:
            return []
        cache = self._metrics_cache
        dp_start, dp_size = self._dp_range

        def snapshot_for(r: int) -> Callable[[], Optional[Metrics]]:
            return lambda: cache.snapshot(r)

        return [
            SnapshotSource(snapshot=snapshot_for(rank), dp_rank=rank)
            for rank in range(dp_start, dp_start + dp_size)
        ]

    async def abort(self, context: Context) -> None:
        request_id = context.id()
        if self.engine_client is not None and request_id is not None:
            await self.engine_client.abort(request_id)
            logger.debug("Aborted request %s", request_id)

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
