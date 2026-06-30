# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM LLMEngine implementation for the unified backend.

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import sys
import threading
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional

from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi import KvCacheConfig, SchedulerConfig
from tensorrt_llm.llmapi.disagg_utils import get_global_disagg_request_id
from tensorrt_llm.llmapi.llm import SamplingParams
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.sampling_params import GuidedDecodingParams
from tensorrt_llm.scheduling_params import SchedulingParams
from torch.cuda import device_count

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
from dynamo.common.backend.metrics import register_global_registry
from dynamo.common.backend.publisher import ComponentSnapshot, KvEventSource, PushSource
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode as CommonDisaggregationMode
from dynamo.common.utils.structural_tag import serialize_structural_tag
from dynamo.llm import KvEventPublisher, ModelInput
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.engine import Backend, TensorRTLLMEngine
from dynamo.trtllm.logits_processing.adapter import attach_logits_processors
from dynamo.trtllm.request_handlers.handler_base import TRTLLMEnginePauseController
from dynamo.trtllm.utils.disagg_utils import (
    DisaggregatedParams,
    DisaggregatedParamsCodec,
)
from dynamo.trtllm.utils.request_utils import (
    request_cache_salt,
    stored_event_cache_salt,
)
from dynamo.trtllm.utils.trtllm_utils import deep_update, warn_override_collisions

if TYPE_CHECKING:
    from tensorrt_llm.metrics import MetricsCollector

    from dynamo._core.backend import EngineMetrics  # type: ignore[import-not-found]
    from dynamo.trtllm.metrics import AdditionalMetricsCollector

logger = logging.getLogger(__name__)

# 1021 is the largest 10-bit prime — spreads machine_ids more evenly
# under modulo than 1024 would. Matches legacy
# `workers/llm_worker.py: connection_id() % 1021`.
_DISAGG_MACHINE_ID_MAX = 1021

# Server-side wait per poll; idle sleep bounds CPU when the engine is quiet.
_KV_EVENTS_POLL_TIMEOUT_S = 0.2
_STATS_POLL_TIMEOUT_S = 0.2
_IDLE_SLEEP_S = 0.01

# Mirror the legacy `dynamo.trtllm` worker — required for TRT-LLM to actually
# publish KV cache events. Without this, `get_kv_cache_events` returns empty.
_DEFAULT_KV_EVENT_BUFFER_MAX_SIZE = 100_000

# Note: `metrics_dict` is set per-instance on `GenerationResult` (only
# when TRT-LLM's perf-stats collection is enabled and the request
# finished). Check `hasattr(res, "metrics_dict")` on each instance —
# a class-level guard would skip finished results that DO carry it.


# Bridges trtllm's local enum into the common one. ENCODE absent —
# rejected up front in from_args().
_TRTLLM_TO_COMMON_DISAGG = {
    DisaggregationMode.AGGREGATED: CommonDisaggregationMode.AGGREGATED,
    DisaggregationMode.PREFILL: CommonDisaggregationMode.PREFILL,
    DisaggregationMode.DECODE: CommonDisaggregationMode.DECODE,
}


def _to_signed_i64(value: int | None) -> int | None:
    """Two's-complement cast of a Python int into the signed 64-bit range."""
    if value is None:
        return None
    if value >= 2**63:
        return value - 2**64
    if value < -(2**63):
        return ((value + 2**63) % 2**64) - 2**63
    return value


class TrtllmLLMEngine(LLMEngine):
    def __init__(
        self,
        engine_args: dict[str, Any],
        model_name: str,
        served_model_name: str | None = None,
        max_seq_len: int | None = None,
        max_batch_size: int | None = None,
        max_num_tokens: int | None = None,
        kv_block_size: int = 32,
        disaggregation_mode: DisaggregationMode = DisaggregationMode.AGGREGATED,
        component: str = "backend",
        publish_events_and_metrics: bool = False,
    ):
        self.engine_args = engine_args
        self.model_name = model_name
        self.served_model_name = served_model_name
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.kv_block_size = kv_block_size
        # Drives context_only / generation_only branching in generate().
        self.disaggregation_mode = disaggregation_mode
        # Gates the KV event publication path (engine event buffer + the
        # `_kv_events_thread`). Component metrics + native `trtllm_*` metrics
        # emit unconditionally.
        self.publish_events_and_metrics = publish_events_and_metrics
        self._component = component
        self._additional_metrics: Optional["AdditionalMetricsCollector"] = None
        kv_cache_config = self.engine_args.get("kv_cache_config", {})
        if isinstance(kv_cache_config, dict):
            event_buffer_max_size = kv_cache_config.get("event_buffer_max_size", 0)
        elif isinstance(kv_cache_config, KvCacheConfig):
            event_buffer_max_size = kv_cache_config.event_buffer_max_size
        else:
            raise TypeError(
                "kv_cache_config must be a dict or KvCacheConfig, "
                f"got {type(kv_cache_config).__name__}"
            )
        self._kv_event_buffer_max_size = int(event_buffer_max_size or 0)
        self._trtllm_metrics_collector: Optional["MetricsCollector"] = None
        # Resolved once at construction so the hot poll loop doesn't run
        # `hasattr` per iteration; same for the per-request log method
        # which varies by upstream TRT-LLM version.
        self._log_iteration_stats: Optional[Callable[[dict], None]] = None
        self._log_request_metrics: Optional[Callable[[dict], None]] = None
        self._engine: TensorRTLLMEngine | None = None
        self._default_sampling_params = SamplingParams(detokenize=False)
        # Resolved once in `start()`; None when the smoke hook is off.
        self._logits_processor_spec: LogitsProcessorSpec | None = None
        # Per-request GenerationResult handles for `abort()` lookup. TRT-LLM's
        # abort API is `GenerationResult.abort()` (not by request-id), so we
        # need the handle. Other engines (vllm, sglang) abort by id and
        # don't keep this map.
        self._active_requests: dict[str, GenerationResult] = {}
        # Set in start() from worker_id. 10-bit field is a TRT-LLM API
        # constraint; collisions possible at scale (~30 replicas).
        self._disagg_machine_id: int = 0
        self._publish_stop = threading.Event()
        self._metrics_thread: Optional[threading.Thread] = None
        self._kv_events_thread: Optional[threading.Thread] = None
        self._attention_dp_size: int = 1
        # Worker invokes on_ready callbacks serially during setup (see
        # `setup_kv_publishers` in lib/backend-common/src/publisher.rs); the
        # dict is fully populated before `_kv_events_thread` starts and
        # read-only thereafter.
        self._kv_publishers: dict[int, KvEventPublisher] = {}
        # Set by attach_snapshot_publisher. `_metrics_poll_loop` pushes
        # ComponentSnapshots into it on every TRT-LLM stats tick — event-
        # driven, no framework polling on the reader side.
        self._snapshot_publisher: Optional[Any] = None
        # Per-rank hashes of partial blocks; their later "removed" events
        # must not reach the router (which never saw them stored). Scoping
        # by rank prevents a partial block on one rank from suppressing a
        # legitimate `removed` on another rank.
        self._partial_block_hashes_by_rank: dict[int, set[int]] = {}
        self._last_event_id_by_rank: dict[int, int] = {}
        # One-shot guards so a misbehaving engine doesn't flood logs.
        self._warned_dispatch_failed = False
        self._warned_unknown_dp_rank = False
        self._pause_controller: TRTLLMEnginePauseController | None = None
        self._pause_lock = asyncio.Lock()
        self._inflight_lock = asyncio.Lock()
        self._inflight_requests = 0
        self._no_inflight_requests = asyncio.Event()
        self._no_inflight_requests.set()
        self._reject_new_requests = False
        self._resume_recovery_required = False

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[TrtllmLLMEngine, WorkerConfig]:
        config = parse_args(argv)

        if config.disaggregation_mode == DisaggregationMode.ENCODE:
            raise NotImplementedError(
                "ENCODE is not supported by the unified TRT-LLM entry point; "
                "use `python -m dynamo.trtllm` for multimodal encode workers"
            )

        gpus_per_node = config.gpus_per_node or device_count()

        engine_args = {
            "model": str(config.model),
            "scheduler_config": SchedulerConfig(),
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "moe_expert_parallel_size": config.expert_parallel_size,
            # Required for per-rank KV events under attention-DP; without
            # it `get_attention_dp_size()` collapses to 1 and only rank
            # 0's publisher is created.
            "enable_attention_dp": config.enable_attention_dp,
            "backend": Backend.PYTORCH,
            "kv_cache_config": KvCacheConfig(
                free_gpu_memory_fraction=config.free_gpu_memory_fraction,
            ),
            "gpus_per_node": gpus_per_node,
            "max_num_tokens": config.max_num_tokens,
            "max_seq_len": config.max_seq_len,
            "max_beam_width": config.max_beam_width,
            "max_batch_size": config.max_batch_size,
            # Always on — drives `get_stats()` and `request_perf_metrics`
            # which feed the framework's component-metrics snapshot and the
            # native `trtllm_*` MetricsCollector. KV-event publication is
            # gated separately on `publish_events_and_metrics` (only
            # `event_buffer_max_size` + the events thread care).
            "return_perf_metrics": True,
            "enable_iter_perf_stats": True,
        }

        # Apply --extra-engine-args / --override-engine-args. Match the
        # legacy `dynamo.trtllm` path so profiler/parallel-scheduler
        # overrides behave the same way.
        if config.extra_engine_args:
            engine_args = update_llm_args_with_extra_options(
                engine_args, config.extra_engine_args
            )
        if config.override_engine_args:
            try:
                overrides = json.loads(config.override_engine_args)
            except json.JSONDecodeError as e:
                logging.error("Failed to parse override_engine_args as JSON: %s", e)
                sys.exit(1)
            if not isinstance(overrides, dict):
                logging.error(
                    "override_engine_args must be a JSON object, got %s",
                    type(overrides).__name__,
                )
                sys.exit(1)
            logging.info("Applying engine arg overrides: %s", overrides)
            warn_override_collisions(engine_args, overrides)
            deep_update(engine_args, overrides)

        # Apply event_buffer_max_size AFTER overrides so a user override
        # that strips the field can't disable KV-event publishing. Mirrors
        # the legacy `llm_worker` publish_events_and_metrics block; like
        # legacy, treats `0` as "unset" (TRT-LLM's disabled value).
        if config.publish_events_and_metrics:
            kv_cfg = engine_args["kv_cache_config"]
            if isinstance(kv_cfg, KvCacheConfig):
                kv_cfg = kv_cfg.model_dump(exclude_none=True)
            if not kv_cfg.get("event_buffer_max_size"):
                kv_cfg["event_buffer_max_size"] = _DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
            engine_args["kv_cache_config"] = kv_cfg

        # Force tokenizer init for the smoke hook, after all overrides so an
        # explicit user `skip_tokenizer_init=True` can't starve the processor.
        # Gated to generation roles for the same reason as the spec resolution
        # below. The flag is TRT-LLM-shaped; each backend sets its own.
        if os.getenv(DYN_ENABLE_TEST_LOGITS_PROCESSOR) == "1" and is_generation_stage(
            _TRTLLM_TO_COMMON_DISAGG[config.disaggregation_mode]
        ):
            engine_args["skip_tokenizer_init"] = False

        # Use post-override engine_args so EngineConfig matches what the
        # actual TRT-LLM engine got.
        engine = cls(
            engine_args=engine_args,
            model_name=config.model,
            served_model_name=config.served_model_name,
            max_seq_len=engine_args.get("max_seq_len", config.max_seq_len),
            max_batch_size=engine_args.get("max_batch_size", config.max_batch_size),
            max_num_tokens=engine_args.get("max_num_tokens", config.max_num_tokens),
            kv_block_size=config.kv_block_size,
            disaggregation_mode=config.disaggregation_mode,
            component=config.component,
            publish_events_and_metrics=config.publish_events_and_metrics,
        )
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
            disaggregation_mode=_TRTLLM_TO_COMMON_DISAGG[config.disaggregation_mode],
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        # disagg_request_id is the cluster-wide prefill→decode match
        # key. Derive from runtime-unique worker_id so two replicas
        # cannot mint colliding IDs.
        self._disagg_machine_id = worker_id % _DISAGG_MACHINE_ID_MAX
        logger.info(
            "TRT-LLM disagg_machine_id=%d (worker_id=%d)",
            self._disagg_machine_id,
            worker_id,
        )

        self._engine = TensorRTLLMEngine(self.engine_args, self.disaggregation_mode)
        await self._engine.initialize()
        self._pause_controller = TRTLLMEnginePauseController(self._engine)

        # Resolve the engine-declared spec now the engine (and its tokenizer)
        # is initialized; see `logits_processor_spec()`.
        self._logits_processor_spec = await self.logits_processor_spec()
        # TODO: Thread runtime and shutdown_event through unified LLMEngine
        # startup so the TRT-LLM monitor can match the legacy shutdown path.
        self._engine.start_health_monitor()

        from tensorrt_llm.metrics import MetricsCollector

        from dynamo.trtllm.metrics import AdditionalMetricsCollector

        gauge_model_name = self.served_model_name or self.model_name
        self._additional_metrics = AdditionalMetricsCollector(
            labels={
                "model_name": gauge_model_name,
                "disaggregation_mode": self.disaggregation_mode.value,
                "engine_type": "trtllm",
            },
        )
        self._additional_metrics.set_kv_event_buffer_capacity(
            self._kv_event_buffer_max_size
        )
        self._trtllm_metrics_collector = MetricsCollector(
            {"model_name": gauge_model_name, "engine_type": "trtllm"}
        )
        # Resolve per-version method names once; TRT-LLM renamed
        # `log_metrics_dict` -> `log_request_metrics_dict` mid-cycle.
        self._log_iteration_stats = getattr(
            self._trtllm_metrics_collector, "log_iteration_stats", None
        )
        self._log_request_metrics = getattr(
            self._trtllm_metrics_collector, "log_request_metrics_dict", None
        ) or getattr(self._trtllm_metrics_collector, "log_metrics_dict", None)

        self._attention_dp_size = self._engine.get_attention_dp_size()
        # Always start the metrics-poll thread: it pushes the latest
        # ComponentSnapshot into the framework's SnapshotPublisher and
        # forwards each snap to `_log_iteration_stats` for `trtllm_kv_cache_*`.
        self._metrics_thread = threading.Thread(
            target=self._metrics_poll_loop,
            daemon=True,
            name="trtllm-metrics-poll",
        )
        self._metrics_thread.start()

        return EngineConfig(
            model=self.model_name,
            served_model_name=self.served_model_name,
            llm=LlmRegistration(
                context_length=self.max_seq_len,
                kv_cache_block_size=self.kv_block_size,
                max_num_seqs=self.max_batch_size,
                max_num_batched_tokens=self.max_num_tokens,
                data_parallel_size=self._attention_dp_size,
            ),
        )

    # TRT-LLM's `get_kv_cache_events` / `get_stats` block the calling
    # thread, so we drive them from dedicated worker threads rather than
    # the asyncio event loop.

    def _kv_routing_enabled(self) -> bool:
        # Matches the legacy `workers/llm_worker.py` gate. Decode workers
        # publish too — legacy only flips `enable_local_indexer` off.
        return self.publish_events_and_metrics

    async def kv_event_sources(self) -> list[KvEventSource]:
        if not self._kv_routing_enabled():
            return []
        return [
            PushSource(
                on_ready=self._make_on_publisher_ready(rank),
                dp_rank=rank,
            )
            for rank in range(self._attention_dp_size)
        ]

    async def logits_processor_spec(self) -> LogitsProcessorSpec | None:
        # Only generation roles ever attach (PREFILL/ENCODE gate out per
        # request), so skip spec resolution — and the tokenizer it needs —
        # entirely for the other roles.
        if not is_generation_stage(_TRTLLM_TO_COMMON_DISAGG[self.disaggregation_mode]):
            return None
        # Bind the env-gated smoke hook to TRT-LLM's tokenizer; the lambda is
        # dereferenced lazily, only when the hook is on (see the resolver).
        assert self._engine is not None
        engine = self._engine  # narrow for the closure
        return resolve_test_logits_processor_spec(lambda: engine.llm.tokenizer)

    def component_metrics_dp_ranks(self) -> list[int]:
        return list(range(self._attention_dp_size))

    def attach_snapshot_publisher(self, publisher) -> None:
        self._snapshot_publisher = publisher

    def _make_on_publisher_ready(self, rank: int):
        # Worker invokes on_ready serially during setup; the call that
        # completes the publisher set starts the dispatch thread.
        def on_ready(publisher: KvEventPublisher) -> None:
            self._kv_publishers[rank] = publisher
            if (
                len(self._kv_publishers) == self._attention_dp_size
                and self._kv_events_thread is None
            ):
                self._kv_events_thread = threading.Thread(
                    target=self._kv_events_poll_loop,
                    daemon=True,
                    name="trtllm-kv-events-poll",
                )
                self._kv_events_thread.start()

        return on_ready

    # Stats payloads use camelCase keys (`attentionDpRank`, `kvCacheStats`);
    # the KV-event payloads in `_dispatch_kv_event` use snake_case
    # (`attention_dp_rank`). Both match TRT-LLM's upstream conventions.
    def _metrics_poll_loop(self) -> None:
        assert self._engine is not None
        while not self._publish_stop.is_set():
            try:
                snaps = self._engine.llm.get_stats(timeout=_STATS_POLL_TIMEOUT_S)
            except Exception as e:
                logger.debug("trtllm get_stats raised: %s", e)
                time.sleep(_IDLE_SLEEP_S)
                continue
            if not snaps:
                time.sleep(_IDLE_SLEEP_S)
                continue
            # Per-rank latest snapshot. Stats from different ranks are
            # interleaved; keep the freshest per `attentionDpRank`.
            for snap in snaps:
                kv_stats = snap.get("kvCacheStats", {})
                kv_used = kv_stats.get("usedNumBlocks")
                if kv_used is None:
                    continue
                rank = int(snap.get("attentionDpRank", 0))
                kv_total = int(kv_stats.get("maxNumBlocks") or 0)
                # cacheHitRate is on the kvCacheStats payload in recent
                # TRT-LLM; absent on older releases. None means "no data."
                hit_rate = kv_stats.get("cacheHitRate")
                if self._snapshot_publisher is not None:
                    self._snapshot_publisher.publish(
                        rank,
                        ComponentSnapshot(
                            kv_used_blocks=int(kv_used),
                            kv_total_blocks=kv_total,
                            gpu_cache_usage=(kv_used / kv_total) if kv_total else 0.0,
                            kv_cache_hit_rate=(
                                float(hit_rate) if hit_rate is not None else None
                            ),
                            dp_rank=rank,
                        ),
                    )

                # TRT-LLM-native trtllm_kv_cache_* gauges (PR #11243).
                if self._log_iteration_stats is not None:
                    try:
                        self._log_iteration_stats(snap)
                    except (AttributeError, KeyError, TypeError) as e:
                        logger.debug("TRT-LLM log_iteration_stats failed: %s", e)

    def _kv_events_poll_loop(self) -> None:
        assert self._engine is not None
        while not self._publish_stop.is_set():
            try:
                events = self._engine.llm.get_kv_cache_events(
                    timeout=_KV_EVENTS_POLL_TIMEOUT_S
                )
            except Exception as e:
                logger.debug("trtllm get_kv_cache_events raised: %s", e)
                time.sleep(_IDLE_SLEEP_S)
                continue
            if not events:
                time.sleep(_IDLE_SLEEP_S)
                continue
            if self._additional_metrics is not None:
                self._additional_metrics.record_kv_event_drain_batch(len(events))
            for event in events:
                try:
                    self._dispatch_kv_event(event)
                except Exception as e:
                    if not self._warned_dispatch_failed:
                        self._warned_dispatch_failed = True
                        logger.exception(
                            "Failed to dispatch KV event; suppressing further "
                            "tracebacks (last error: %s)",
                            e,
                        )

    def _dispatch_kv_event(self, event: dict[str, Any]) -> None:
        """Forward stored / removed events to the right publisher. Other
        event types are dropped — the Python publisher has no path for them."""
        rank = int(event.get("attention_dp_rank", 0))
        event_id = event.get("event_id")
        if event_id is not None:
            last = self._last_event_id_by_rank.get(rank)
            if last is not None and event_id != last + 1:
                logger.warning(
                    "Non-consecutive engine event_id on rank=%d: expected %d, got %d",
                    rank,
                    last + 1,
                    event_id,
                )
                if self._additional_metrics is not None:
                    self._additional_metrics.record_kv_event_id_gap(
                        max(0, event_id - (last + 1))
                    )
            self._last_event_id_by_rank[rank] = event_id
        publisher = self._kv_publishers.get(rank)
        if publisher is None:
            if not self._warned_unknown_dp_rank:
                self._warned_unknown_dp_rank = True
                logger.warning(
                    "Dropping KV event for unknown attention_dp_rank=%d "
                    "(have %s); suppressing further warnings",
                    rank,
                    sorted(self._kv_publishers.keys()),
                )
            return
        data = event.get("data") or {}
        kind = data.get("type")
        if kind == "stored":
            parent_hash = _to_signed_i64(data.get("parent_hash"))
            token_ids: list[int] = []
            num_block_tokens: list[int] = []
            block_hashes: list[int] = []
            kv_block_size = self.kv_block_size
            for block in data.get("blocks", []):
                block_tokens = block.get("tokens") or []
                token_num = len(block_tokens)
                if token_num > kv_block_size:
                    logger.error(
                        "Block contains %d tokens > kv_block_size %d",
                        token_num,
                        kv_block_size,
                    )
                    return
                block_hash = _to_signed_i64(block.get("block_hash"))
                if block_hash is None:
                    continue
                if token_num < kv_block_size:
                    self._partial_block_hashes_by_rank.setdefault(rank, set()).add(
                        block_hash
                    )
                    break
                num_block_tokens.append(token_num)
                block_hashes.append(block_hash)
                token_ids.extend(int(t["token_id"]) for t in block_tokens)
            if not block_hashes:
                return
            publisher.publish_stored(
                token_ids,
                num_block_tokens,
                block_hashes,
                parent_hash,
                lora_name=data.get("lora_name"),
                cache_salt=stored_event_cache_salt(data),
            )
        elif kind == "removed":
            partial = self._partial_block_hashes_by_rank.get(rank)
            removed: list[int] = []
            for raw in data.get("block_hashes", []):
                block_hash = _to_signed_i64(raw)
                if block_hash is None:
                    continue
                if partial is not None and block_hash in partial:
                    partial.remove(block_hash)
                    continue
                removed.append(block_hash)
            if removed:
                publisher.publish_removed(removed)

    def supported_controls(self) -> set[str]:
        return {"release_memory_occupation", "resume_memory_occupation"}

    async def engine_control(self, control: str, body: dict) -> dict:
        handlers = {
            "release_memory_occupation": self.release_memory_occupation,
            "resume_memory_occupation": self.resume_memory_occupation,
        }
        handler = handlers.get(control)
        if handler is None:
            return {
                "status": "error",
                "message": f"unsupported engine control: {control}",
            }
        return await handler(body or {})

    async def _set_reject_new_requests(self, reject: bool) -> None:
        async with self._inflight_lock:
            self._reject_new_requests = reject

    async def _mark_request_started(self) -> bool:
        async with self._inflight_lock:
            if self._reject_new_requests:
                return False
            self._inflight_requests += 1
            self._no_inflight_requests.clear()
            return True

    async def _mark_request_finished(self) -> None:
        async with self._inflight_lock:
            if self._inflight_requests == 0:
                return
            self._inflight_requests -= 1
            if self._inflight_requests == 0:
                self._no_inflight_requests.set()

    async def _wait_for_inflight_requests(self, timeout_s: float) -> None:
        try:
            await asyncio.wait_for(self._no_inflight_requests.wait(), timeout_s)
        except asyncio.TimeoutError as exc:
            async with self._inflight_lock:
                inflight = self._inflight_requests
            raise RuntimeError(
                f"Timed out waiting for {inflight} in-flight request(s) to finish"
            ) from exc

    @staticmethod
    def _controller_needs_resume_recovery(
        controller: TRTLLMEnginePauseController,
    ) -> bool:
        needs_recovery = getattr(controller, "needs_resume_recovery", False)
        return needs_recovery if isinstance(needs_recovery, bool) else False

    async def release_memory_occupation(self, body: dict) -> dict:
        controller = self._pause_controller
        if controller is None:
            return {"status": "error", "message": "engine is not initialized"}

        body = body or {}
        tags = body.get("tags")
        async with self._pause_lock:
            if controller.is_paused:
                return {"status": "ok", "message": "Memory already released"}
            if (
                self._resume_recovery_required
                or self._controller_needs_resume_recovery(controller)
            ):
                return {
                    "status": "error",
                    "message": "resume_memory_occupation required before retrying release",
                }
            try:
                await self._set_reject_new_requests(True)
                timeout_s = float(body.get("timeout_s", 30.0))
                await self._wait_for_inflight_requests(timeout_s)
            except Exception as exc:
                logger.error("release_memory_occupation failed before pause: %s", exc)
                await self._set_reject_new_requests(False)
                return {"status": "error", "message": str(exc)}

            try:
                await controller.pause(tags)
                self._resume_recovery_required = False
                return {"status": "ok", "message": "Memory released"}
            except Exception as exc:
                logger.error("release_memory_occupation pause failed: %s", exc)
                self._resume_recovery_required = True
                return {"status": "error", "message": str(exc)}

    async def resume_memory_occupation(self, body: dict) -> dict:
        controller = self._pause_controller
        if controller is None:
            return {"status": "error", "message": "engine is not initialized"}

        body = body or {}
        tags = body.get("tags")
        async with self._pause_lock:
            needs_recovery = (
                self._resume_recovery_required
                or self._controller_needs_resume_recovery(controller)
            )
            if not controller.is_paused and not needs_recovery:
                return {"status": "ok", "message": "Memory already resumed"}
            try:
                if controller.is_paused or self._controller_needs_resume_recovery(
                    controller
                ):
                    await controller.resume(tags)
                await self._set_reject_new_requests(False)
                controller.mark_resumed()
                self._resume_recovery_required = False
                return {"status": "ok", "message": "Memory resumed"}
            except Exception as exc:
                logger.error("resume_memory_occupation failed: %s", exc)
                self._resume_recovery_required = True
                return {"status": "error", "message": str(exc)}

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if not await self._mark_request_started():
            yield {
                "finish_reason": "error",
                "token_ids": [],
                "index": 0,
                "completion_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            return
        try:
            async for chunk in self._generate_started(request, context):
                yield chunk
        finally:
            await self._mark_request_finished()

    async def _generate_started(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        assert self._engine is not None, "Engine not initialized"

        # Tag the request as structured-output / image so the per-type
        # counters split traffic correctly in Prometheus.
        if self._additional_metrics is not None:
            sampling_options = request.get("sampling_options", {})
            guided = sampling_options.get("guided_decoding")
            if isinstance(guided, dict) and (
                any(
                    guided.get(k) is not None
                    for k in (
                        "json",
                        "regex",
                        "grammar",
                        "json_object",
                        "structural_tag",
                    )
                )
                or bool(guided.get("choice"))
            ):
                self._additional_metrics.record_request_type_structured_output()
            if request.get("multi_modal_data") is not None:
                self._additional_metrics.record_request_type_image()

        token_ids = request.get("token_ids", [])
        sampling_params = self._override_sampling_params(
            self._default_sampling_params, request
        )

        # TRT-LLM's `logprobs=0` disables computation entirely. Floor at 1
        # so a `logprobs=0` request still gets the chosen-token logprob;
        # the raw requested count gates top_logprobs emission below.
        (
            requested_logprobs_count,
            prompt_logprobs_count,
        ) = _shared_logprobs.parse_logprob_options(
            request.get("output_options", {}) or {}
        )
        if requested_logprobs_count is not None and hasattr(
            sampling_params, "logprobs"
        ):
            sampling_params.logprobs = max(1, requested_logprobs_count)
        if prompt_logprobs_count is not None and hasattr(
            sampling_params, "prompt_logprobs"
        ):
            sampling_params.prompt_logprobs = prompt_logprobs_count

        # Prefill: context_only handle → packed into the response.
        # Decode: read prefill peer's handle, flip to generation_only.
        disaggregated_params: LlmDisaggregatedParams | None = None
        is_prefill = self.disaggregation_mode == DisaggregationMode.PREFILL
        is_decode = self.disaggregation_mode == DisaggregationMode.DECODE

        if is_prefill:
            disaggregated_params = LlmDisaggregatedParams(
                request_type="context_only",
                disagg_request_id=get_global_disagg_request_id(self._disagg_machine_id),
            )
        elif is_decode:
            prefill_result = require_prefill_result(
                request, _TRTLLM_TO_COMMON_DISAGG[self.disaggregation_mode]
            )
            disaggregated_params = self._decode_prefill_handoff(prefill_result)

        stop_conditions = request.get("stop_conditions", {})
        if is_prefill:
            # Prefill only needs to populate KV — one token is enough.
            sampling_params.max_tokens = 1
        else:
            max_tokens = stop_conditions.get("max_tokens")
            if max_tokens is not None:
                sampling_params.max_tokens = max_tokens
            elif self.max_seq_len is not None:
                sampling_params.max_tokens = max(1, self.max_seq_len - len(token_ids))

        # TODO: mirror visible/hidden stop-token handling from the disagg
        # path (handler_base.py) into a shared helper. See PR #9778.
        ignore_eos = stop_conditions.get("ignore_eos")
        if ignore_eos:
            sampling_params.ignore_eos = ignore_eos

        # Honour the router's DP rank decision; without it TRT-LLM picks
        # its own rank and KV events land on the wrong publisher.
        rank = validate_global_dp_rank(
            forced_dp_rank(request), 0, self._attention_dp_size, "TRT-LLM"
        )
        scheduling_params = (
            SchedulingParams(attention_dp_rank=rank, attention_dp_relax=False)
            if rank is not None
            else None
        )

        entries = logits_processors_for_request(
            self._logits_processor_spec,
            disaggregation_mode=_TRTLLM_TO_COMMON_DISAGG[self.disaggregation_mode],
        )
        attach_logits_processors(sampling_params, entries)

        # Prefill returns one non-streaming chunk carrying the handoff -
        # matches the legacy disagg wire format.
        streaming = not is_prefill
        cache_salt = request_cache_salt(request)
        generation_result = self._engine.llm.generate_async(
            inputs=token_ids,
            sampling_params=sampling_params,
            streaming=streaming,
            disaggregated_params=disaggregated_params,
            scheduling_params=scheduling_params,
            cache_salt=cache_salt,
            **telemetry.engine_trace_kwargs(context),
        )

        request_id = context.id()
        if request_id is not None:
            self._active_requests[request_id] = generation_result

        try:
            # TRT-LLM reports cumulative token_ids per choice; track the
            # emitted length per index so we can yield deltas (n>1 safe).
            output_tokens_per_choice: dict[int, int] = {}
            async for res in generation_result:
                if not res.outputs and not res.finished:
                    yield {"finish_reason": "error", "token_ids": [], "index": 0}
                    break

                for output in res.outputs:
                    output_idx = getattr(output, "index", 0) or 0
                    tokens_so_far = output_tokens_per_choice.get(output_idx, 0)
                    next_total = len(output.token_ids)
                    out: GenerateChunk = {
                        "token_ids": output.token_ids[tokens_so_far:],
                        "index": output_idx,
                    }

                    # output.logprobs is cumulative in lockstep with
                    # output.token_ids — reuse the same slice offset.
                    (
                        log_probs,
                        top_logprobs,
                    ) = _shared_logprobs.extract_from_completion_output(
                        output,
                        tokens_so_far,
                        fallback_to_first_on_missing=True,
                        include_bytes=False,
                    )
                    if log_probs is not None:
                        out["log_probs"] = log_probs
                    if (
                        top_logprobs is not None
                        and requested_logprobs_count is not None
                        and requested_logprobs_count > 0
                    ):
                        out["top_logprobs"] = top_logprobs

                    if output.finish_reason:
                        out["finish_reason"] = str(output.finish_reason)

                    if out.get("finish_reason") or res.finished:
                        if not out.get("finish_reason"):
                            out["finish_reason"] = "unknown"
                        # TRT-LLM shares vLLM's prompt_logprobs shape.
                        if prompt_logprobs_count is not None:
                            prompt_payload = _shared_logprobs.extract_prompt_logprobs_from_completion_output(
                                res
                            )
                            if prompt_payload is not None:
                                out["engine_data"] = {"prompt_logprobs": prompt_payload}
                        prompt_tokens = len(token_ids)
                        total_completion_tokens = sum(
                            len(o.token_ids) for o in res.outputs
                        )
                        out["completion_usage"] = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                            "total_tokens": prompt_tokens + total_completion_tokens,
                        }
                        # Stamp the handoff payload on the prefill terminal.
                        if is_prefill:
                            params_dict = self._encode_prefill_handoff(
                                output, disaggregated_params
                            )
                            if params_dict is not None:
                                out["disaggregated_params"] = params_dict

                        # On the terminal chunk, record KV-transfer
                        # latency/bytes/speed from `timing_metrics`. Only
                        # meaningful for decode workers — the collector
                        # self-skips on zero-duration timings.
                        if (
                            self._additional_metrics is not None
                            and res.finished
                            and not is_prefill
                        ):
                            try:
                                perf = getattr(res, "request_perf_metrics", None)
                                tm = (
                                    getattr(perf, "timing_metrics", None)
                                    if perf
                                    else None
                                )
                                if (
                                    tm is not None
                                    and self._additional_metrics.record_kv_transfer_perf(
                                        tm
                                    )
                                ):
                                    self._additional_metrics.record_kv_transfer_success()
                            except (AttributeError, TypeError) as e:
                                logger.debug("KV-transfer perf recording failed: %s", e)

                        # Drive trtllm_request_success_total / e2e /
                        # TTFT / ITL / queue_time. Method name resolved
                        # once in `start` to handle the upstream rename.
                        if (
                            res.finished
                            and self._log_request_metrics is not None
                            and hasattr(res, "metrics_dict")
                        ):
                            try:
                                self._log_request_metrics(res.metrics_dict)
                            except (AttributeError, TypeError) as e:
                                logger.debug("TRT-LLM log_metrics_dict failed: %s", e)

                    yield out
                    output_tokens_per_choice[output_idx] = next_total
        finally:
            if request_id is not None:
                self._active_requests.pop(request_id, None)

    @staticmethod
    def _decode_prefill_handoff(
        prefill_result: dict[str, Any],
    ) -> LlmDisaggregatedParams:
        """Decode the prefill peer's handoff payload into a TRT-LLM
        `LlmDisaggregatedParams` ready to drive a generation_only call.
        Mirrors `HandlerBase._decode_disaggregated_params_from_prefill`.
        """
        params_dict = dict(prefill_result.get("disaggregated_params") or {})
        if not params_dict:
            raise ValueError(
                "decode received prefill_result without disaggregated_params"
            )
        # Strip prefill-side routing metadata before codec construction.
        params_dict.pop("worker_id", None)
        DisaggregatedParamsCodec.deserialize_first_gen_log_probs(params_dict)
        params_dict.pop("_epd_metadata", None)
        decoded = DisaggregatedParamsCodec.decode(DisaggregatedParams(**params_dict))
        decoded.request_type = "generation_only"
        # Already baked into the imported KV; clearing avoids a
        # generation_only validation error in TRT-LLM.
        if (
            hasattr(decoded, "multimodal_embedding_handles")
            and decoded.multimodal_embedding_handles
        ):
            decoded.multimodal_embedding_handles = None
        return decoded

    @staticmethod
    def _encode_prefill_handoff(
        output: Any, input_params: LlmDisaggregatedParams | None
    ) -> dict[str, Any] | None:
        """Pack the engine's output `disaggregated_params` for wire
        transport. Falls back to input params when the engine returns
        None (occasionally happens on a successful prefill)."""
        params_to_encode = (
            output.disaggregated_params
            if output.disaggregated_params is not None
            else input_params
        )
        encoded = DisaggregatedParamsCodec.encode(params_to_encode)
        if encoded is None:
            logger.error(
                "PREFILL: encoded disaggregated_params is None; the decode peer will fail"
            )
            return None
        params_dict = asdict(encoded)
        DisaggregatedParamsCodec.serialize_first_gen_log_probs(params_dict)
        return params_dict

    async def abort(self, context: Context) -> None:
        request_id = context.id()
        if request_id is not None:
            result = self._active_requests.get(request_id)
            if result is not None:
                result.abort()
                logger.debug("Aborted request %s", request_id)
                if self._additional_metrics is not None:
                    self._additional_metrics.record_request_abort()

    async def register_prometheus(self, metrics: "EngineMetrics") -> None:
        # Framework owns the dynamo_component_* registry; we just bridge
        # the global REGISTRY (`trtllm_*` family from MetricsCollector +
        # AdditionalMetricsCollector). Always on — the collectors emit
        # vendor metrics independent of KV-event publishing.
        register_global_registry(metrics, engine_prefix="trtllm_")

    async def health_check_payload(self) -> Optional[dict[str, Any]]:
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            logger.warning(
                "DECODE worker: health-check canary disabled — synthesizing a "
                "prefill_result that survives DisaggregatedParamsCodec.decode "
                "is non-trivial. Endpoint readiness will rely on real request traffic."
            )
            return None
        tokenizer = None
        if self._engine is not None and self._engine._llm is not None:
            tokenizer = self._engine.llm.tokenizer
        # priority=1.0 is TRT-LLM's max — keeps the canary off the starvation path.
        return build_health_check_payload(
            bos_token_id=bos_token_id_or(tokenizer),
            extras={"priority": 1.0},
        )

    # TRT-LLM deliberately does not override is_quiescent (inherits None: wait
    # the full drain budget). It has no signal for "pending KV transfers done":
    # iteration stats stop arriving once the worker is idle, so active == 0 is
    # never observed; and _active_requests is popped at handoff, before decode
    # pulls the KV. The budget alone gives decode time to drain.

    async def cleanup(self) -> None:
        # Stop the publisher threads BEFORE engine shutdown so they don't
        # observe a half-torn-down RPC client. Each thread already loops on
        # `_publish_stop`; the join bounds the wait at the poll timeout.
        self._publish_stop.set()
        for thread in (self._kv_events_thread, self._metrics_thread):
            if thread is not None:
                thread.join(timeout=_KV_EVENTS_POLL_TIMEOUT_S * 2 + 0.5)
        self._kv_events_thread = None
        self._metrics_thread = None
        self._kv_publishers.clear()
        self._pause_controller = None
        # Abort any still-tracked requests so llm.shutdown() runs on an idle
        # engine. Mostly a no-op for prefill (popped at handoff); matters for
        # decode/aggregated workers mid-generation.
        for result in list(self._active_requests.values()):
            try:
                result.abort()
            except Exception:
                logger.debug("abort during cleanup failed", exc_info=True)
        self._active_requests.clear()
        if self._engine is not None:
            await self._engine.cleanup()
            logger.info("TensorRT-LLM engine shutdown")

    @staticmethod
    def _override_sampling_params(
        sampling_params: SamplingParams, request: GenerateRequest
    ) -> SamplingParams:
        overrides = {
            key: value
            for key, value in request.get("sampling_options", {}).items()
            if value is not None
        }

        guided_decoding = overrides.pop("guided_decoding", None)
        if guided_decoding is not None and isinstance(guided_decoding, dict):
            regex = guided_decoding.get("regex")
            choice = guided_decoding.get("choice")
            if choice and not regex:
                valid_choices = [c for c in choice if c is not None]
                if valid_choices:
                    regex = "(" + "|".join(re.escape(c) for c in valid_choices) + ")"
            overrides["guided_decoding"] = GuidedDecodingParams(
                json=guided_decoding.get("json"),
                regex=regex,
                grammar=guided_decoding.get("grammar"),
                json_object=guided_decoding.get("json_object", False),
                structural_tag=serialize_structural_tag(
                    guided_decoding.get("structural_tag")
                ),
            )

        n = overrides.get("n")
        if (
            isinstance(n, int)
            and not isinstance(n, bool)
            and n > 1
            and hasattr(sampling_params, "best_of")
        ):
            # Dynamo does not expose best_of here, but TRT-LLM validates that
            # its internal best_of is at least n when cloning SamplingParams.
            # Keep that private field in lockstep so OpenAI n>1 requests do
            # not fail before generation starts.
            best_of = getattr(sampling_params, "best_of", None)
            if best_of is None or best_of < n:
                overrides["best_of"] = n

        return dataclasses.replace(sampling_params, **overrides)
