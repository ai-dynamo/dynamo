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
from vllm.lora.request import LoRARequest
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
from dynamo.common.lora.manager import LoRAInfo, get_lora_manager
from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    lora_name_to_id,
    register_model,
    unregister_model,
)
from dynamo.runtime import Endpoint
from dynamo.vllm.args import Config, configure_rl_logprobs_mode, parse_args
from dynamo.vllm.cache_info import (
    configure_kv_event_block_size,
    get_configured_kv_event_block_size,
)
from dynamo.vllm.capacity import per_rank_kv_blocks
from dynamo.vllm.router_hints import enable_router_hint_support

from .handlers import (
    VllmEnginePauseController,
    _apply_nvext_cache_salt,
    _engine_generate_reasoning_kwargs,
    _request_reasoning_metadata,
    build_sampling_params,
    get_dp_range_for_worker,
)
from .logits_processing import (
    activate_logits_processors,
    register_dynamo_logits_processor,
)
from .multimodal_utils.cache_config import configure_multimodal_embedding_cache
from .multimodal_utils.media_config import create_frontend_media_config
from .multimodal_utils.request_processor import VllmMultimodalRequestProcessor

if TYPE_CHECKING:
    from dynamo._core.backend import EngineMetrics  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class _UnifiedStatLogger(StatLoggerBase):
    """vLLM stat-logger that pushes :class:`ComponentSnapshot` values into
    the Rust-owned :class:`SnapshotPublisher` on every iteration."""

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


# Number of stripe locks serializing per-adapter LoRA load/unload. Fixed so
# lock memory stays bounded no matter how many distinct adapter names are seen;
# distinct names may share a stripe (harmless extra serialization on this
# control-plane path).
_LORA_LOCK_STRIPES = 32


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
        dyn_tool_call_parser: Optional[str] = None,
        dyn_reasoning_parser: Optional[str] = None,
        enable_rl: bool = False,
        enable_multimodal: bool = False,
        frontend_decoding: bool = False,
        multimodal_embedding_cache_capacity_gb: float = 0.0,
        namespace: str = "dynamo",
    ):
        self.engine_args = engine_args
        self.disaggregation_mode = disaggregation_mode
        self._served_model_name = served_model_name
        self._component = component
        self._dyn_tool_call_parser = dyn_tool_call_parser
        self._dyn_reasoning_parser = dyn_reasoning_parser
        self.enable_rl = enable_rl
        self.enable_multimodal = enable_multimodal
        self.frontend_decoding = frontend_decoding
        self.multimodal_embedding_cache_capacity_gb = (
            multimodal_embedding_cache_capacity_gb
        )
        self._namespace = namespace
        self.engine_client: AsyncLLM | None = None
        self._vllm_config: Any = None
        self._default_sampling_params: Any = None
        self._prometheus_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._model_max_len: int | None = None
        self._dp_range: Optional[tuple[int, int]] = None
        # Effective KV-event block size, computed in start(). LoRA MDCs must
        # publish this (not engine_args.block_size) so LoRA block hashes match
        # vLLM's emitted KV events for routing.
        self._kv_event_block_size: int | None = None
        # Per-rank KV block count, computed in start() and published on both the
        # base-model and LoRA MDCs so the router sees the worker's real capacity.
        self._total_kv_blocks: int | None = None
        # Constructed in start() before AsyncLLM init so vLLM's stat-logger
        # factory call sees a valid object. `num_gpu_blocks` is patched
        # after KV profiling finishes.
        self._stat_logger_factory: Optional[_UnifiedStatLoggerFactory] = None
        self._logits_processor_spec: LogitsProcessorSpec | None = None
        self._pause_controller: VllmEnginePauseController | None = None
        self._multimodal_request_processor: VllmMultimodalRequestProcessor | None = None
        self._pause_lock = asyncio.Lock()
        self._scale_ep_lock = asyncio.Lock()
        self._scale_ep_in_progress = False
        # Dynamic LoRA state. `_endpoint` is set by `on_endpoint_ready` before
        # serving begins; LoRA discovery (register_model/unregister_model)
        # publishes against it.
        self._endpoint: Optional[Endpoint] = None
        self.loaded_loras: dict[str, LoRAInfo] = {}
        # Adapters whose discovery ModelDeploymentCard is currently published.
        # Tracked separately from `loaded_loras` because the engine load and the
        # discovery publish can diverge on partial failure: an adapter may be
        # loaded into vLLM yet have no card (publish failed and engine-side
        # rollback also failed), or have a stale card with no engine load
        # (unregister failed and re-add rollback also failed). Keeping the two
        # states apart lets a retried load/unload reconcile the divergence
        # instead of short-circuiting as "already loaded" / "not found".
        self._published_loras: set[str] = set()
        # Striped locks serialize concurrent load/unload of the same adapter.
        # A fixed array keyed by hash bounds lock memory (no per-name growth)
        # and preserves the "same name -> same lock" invariant by construction,
        # so there is no lock-eviction race. Relies only on per-process hash
        # stability, which is all we need within a single worker.
        self._lora_load_locks = [asyncio.Lock() for _ in range(_LORA_LOCK_STRIPES)]

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None, config: Config | None = None
    ) -> tuple[VllmLLMEngine, WorkerConfig]:
        # `config` lets unified_main thread its already-parsed args through so we
        # don't re-parse (idempotent, but avoids a duplicate argparse + doubled
        # vLLM deprecation warnings at startup).
        if config is None:
            config = parse_args(argv, fpm_trace_relay_supported=False)

        if config.disaggregation_mode == DisaggregationMode.ENCODE:
            raise NotImplementedError(
                "ENCODE is not supported by the unified vLLM entry point; "
                "use `python -m dynamo.vllm` for multimodal encode workers"
            )

        # Headless is handled by unified_main before engine construction; a
        # headless config reaching here means run() was driven directly,
        # bypassing the entry point. Fail loud rather than booting a full
        # backend on what should be a vLLM-workers-only secondary node.
        if config.headless:
            raise NotImplementedError(
                "--headless must be launched via `python -m dynamo.vllm.unified_main` "
                "(or the legacy `python -m dynamo.vllm`); it is not supported when "
                "driving the unified Worker directly"
            )

        if config.route_to_encoder:
            raise NotImplementedError(
                "--route-to-encoder is not supported by the unified vLLM entry "
                "point yet; use `python -m dynamo.vllm` until the separate "
                "unified encode worker is available"
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
            dyn_tool_call_parser=config.dyn_tool_call_parser,
            dyn_reasoning_parser=config.dyn_reasoning_parser,
            enable_rl=config.enable_rl,
            enable_multimodal=config.enable_multimodal,
            frontend_decoding=config.frontend_decoding,
            multimodal_embedding_cache_capacity_gb=(
                config.multimodal_embedding_cache_capacity_gb
            ),
            namespace=config.namespace,
        )
        media_decoder, media_fetcher = create_frontend_media_config(
            config.frontend_decoding
        )
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
            media_decoder=media_decoder,
            media_fetcher=media_fetcher,
        )
        return engine, worker_config

    async def on_endpoint_ready(self, endpoint: Endpoint) -> None:
        """Stash the serving endpoint for dynamic-LoRA discovery publishing."""
        self._endpoint = endpoint

    async def start(self, worker_id: int) -> EngineConfig:
        """Start vLLM and return normalized metadata for runtime registration."""
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

        configure_multimodal_embedding_cache(
            self.engine_args,
            route_to_encoder=False,
            capacity_gb=self.multimodal_embedding_cache_capacity_gb,
            namespace=self._namespace,
            component=self._component,
        )

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
        self._multimodal_request_processor = VllmMultimodalRequestProcessor(
            model=self.engine_args.model,
            engine_client=self.engine_client,
            enable_multimodal=self.enable_multimodal,
            enable_frontend_decoding=self.frontend_decoding,
        )
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._multimodal_request_processor.initialize_prefill_handoff()
        # Resolve once the tokenizer is available (see logits_processor_spec()).
        self._logits_processor_spec = await self.logits_processor_spec()
        self._pause_controller = VllmEnginePauseController(self.engine_client)
        num_gpu_blocks = self.engine_client.vllm_config.cache_config.num_gpu_blocks or 0
        per_rank_num_gpu_blocks = per_rank_kv_blocks(num_gpu_blocks, self._dp_range[1])
        if per_rank_num_gpu_blocks is None:
            raise RuntimeError("per-rank KV block count is not set")
        self._stat_logger_factory.num_gpu_blocks = per_rank_num_gpu_blocks
        self._total_kv_blocks = per_rank_num_gpu_blocks
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
        self._kv_event_block_size = block_size

        return EngineConfig(
            model=self.engine_args.model,
            served_model_name=self._served_model_name,
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

        multimodal_processor = self._multimodal_request_processor
        if multimodal_processor is None:
            raise RuntimeError("VllmLLMEngine.start() must complete before generate()")
        prepared_prompt = await multimodal_processor.prepare_prompt(
            dict(request),
            request_id,
            context,
            self.disaggregation_mode,
        )
        prompt = prepared_prompt.prompt
        _apply_nvext_cache_salt(prepared_prompt.request, prompt)

        # Multimodal decode may replace token_ids with the expanded prefill
        # sequence. Sampling limits must use that same effective request.
        sampling_params = build_sampling_params(
            prepared_prompt.request,
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

        # Route to a loaded LoRA adapter when the request names one; the base
        # model resolves to None. With LoRA enabled, an unknown adapter name
        # raises rather than silently falling back to the base model.
        lora_request = self._resolve_lora_request(request.get("model"))
        reasoning_ended, reasoning_parser_kwargs = _request_reasoning_metadata(request)

        gen = self.engine_client.generate(
            prompt,
            sampling_params,
            request_id,
            data_parallel_rank=local_dp_rank,
            lora_request=lora_request,
            **_engine_generate_reasoning_kwargs(
                self.engine_client,
                reasoning_ended,
                reasoning_parser_kwargs,
            ),
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
                        handoff_params: dict[str, Any] = {}
                        if kv_transfer_params is not None:
                            handoff_params["kv_transfer_params"] = kv_transfer_params
                        embedding_params = multimodal_processor.build_prefill_handoff(
                            multi_modal_data=prepared_prompt.multi_modal_data,
                            prompt_token_ids=list(res.prompt_token_ids or []),
                            mm_processor_kwargs=prepared_prompt.mm_processor_kwargs,
                        )
                        if embedding_params is not None:
                            handoff_params["embedding_params"] = embedding_params
                        if handoff_params:
                            out["disaggregated_params"] = handoff_params

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

    def _lora_enabled(self) -> bool:
        """Dynamic-LoRA updates are available only when the engine was built
        with ``--enable-lora`` AND the LoRA manager is initialized
        (``DYN_LORA_ENABLED=true``)."""
        return bool(getattr(self.engine_args, "enable_lora", False)) and (
            get_lora_manager() is not None
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

    def supported_updates(self) -> set[str]:
        # LoRA lifecycle ops mutate engine-managed adapters, so they ride the
        # engine-update surface (/engine/update/<name>) rather than inflating
        # the engine-control surface.
        if self._lora_enabled():
            return {"load_lora", "unload_lora", "list_loras"}
        return set()

    async def engine_update(self, update: str, body: dict) -> dict:
        handlers = {}
        if self._lora_enabled():
            handlers["load_lora"] = self.load_lora
            handlers["unload_lora"] = self.unload_lora
            handlers["list_loras"] = self.list_loras

        handler = handlers.get(update)
        if handler is None:
            return {
                "status": "error",
                "message": f"unsupported engine update: {update}",
            }
        return await handler(body or {})

    def _resolve_lora_request(self, model_name: str | None) -> LoRARequest | None:
        """Return a LoRARequest for a loaded adapter, or None for the base model.

        Raises ValueError when LoRA is enabled and ``model_name`` is a non-base
        name with no loaded adapter, so an unknown or just-unloaded adapter
        fails loudly instead of being silently served by the base model. When
        LoRA is disabled there are no adapters, so a non-base name is left to
        the engine (current behavior) rather than rejected here.
        """
        if not model_name or model_name in (
            self._served_model_name,
            self.engine_args.model,
        ):
            return None
        lora = self.loaded_loras.get(model_name)
        if lora is not None:
            return LoRARequest(
                lora_name=model_name,
                lora_int_id=lora.id,
                lora_path=lora.path,
            )
        if self._lora_enabled():
            raise ValueError(f"unknown model or LoRA adapter: '{model_name}'")
        return None

    def _get_lora_lock(self, lora_name: str) -> asyncio.Lock:
        """Return the stripe lock that serializes load/unload for ``lora_name``.

        A name always maps to the same stripe within the process, so every
        load/unload for a given ``lora_name`` serializes on the *same* lock.
        Because the stripe set is fixed, there is no per-name lock to evict and
        thus no eviction race; the cost is that two distinct names sharing a
        stripe serialize against each other (harmless on this control path).
        """
        return self._lora_load_locks[hash(lora_name) % _LORA_LOCK_STRIPES]

    async def _publish_lora_card(self, lora_name: str, lora_id: int) -> None:
        """Publish a LoRA adapter as a ModelDeploymentCard for discovery.

        Assumes ``self._endpoint`` is set (callers gate on it). Raises on
        failure so callers can roll back or retry; on success the caller is
        responsible for recording ``lora_name`` in ``self._published_loras``.
        """
        assert self._endpoint is not None

        user_data = {
            "lora_adapter": True,
            "lora_id": lora_id,
        }

        # Match the base-model registration topology (see main.py
        # register_vllm_model + worker_factory) so the frontend builds the LoRA
        # pipeline against the right component. Without this, a prefill worker
        # would publish the adapter as a decode-capable chat/completions model
        # and the frontend would route chat traffic straight to prefill, which
        # then waits forever for a KV transfer.
        model_type, worker_type, needs = self._lora_registration_topology()

        runtime_config = ModelRuntimeConfig()
        enable_router_hint_support(runtime_config, self.engine_args)
        # Prefill workers don't run tool/reasoning parsing (mirrors the base
        # model registration in main.py:register_vllm_model).
        if model_type != ModelType.Prefill:
            runtime_config.tool_call_parser = self._dyn_tool_call_parser
            runtime_config.reasoning_parser = self._dyn_reasoning_parser

        # Carry the worker's DP-rank range and capacity metadata (the same
        # effective vLLM values the base-model MDC publishes via EngineConfig in
        # start()), so multi-DP LoRA requests are routed/attributed per rank
        # instead of as if every worker only served rank 0. start() always runs
        # before a load; guard in case a load somehow races ahead of it.
        if self._vllm_config is not None and self._dp_range is not None:
            scheduler_config = self._vllm_config.scheduler_config
            if self._total_kv_blocks is not None:
                runtime_config.total_kv_blocks = self._total_kv_blocks
            runtime_config.max_num_seqs = scheduler_config.max_num_seqs
            runtime_config.max_num_batched_tokens = (
                scheduler_config.max_num_batched_tokens
            )
            runtime_config.data_parallel_start_rank = self._dp_range[0]
            runtime_config.data_parallel_size = self._dp_range[1]

        # Publish the effective KV-event block size (computed in start() and
        # used by the base-model MDC) so LoRA block hashes match vLLM's emitted
        # KV events. start() always runs before a load, but fall back to the
        # engine arg if it somehow hasn't.
        kv_cache_block_size = (
            self._kv_event_block_size
            if self._kv_event_block_size is not None
            else self.engine_args.block_size
        )

        await register_model(
            model_input=ModelInput.Tokens,
            model_type=model_type,
            endpoint=self._endpoint,
            model_path=self.engine_args.model,
            kv_cache_block_size=kv_cache_block_size,
            runtime_config=runtime_config,
            user_data=user_data,
            lora_name=lora_name,
            base_model_path=self.engine_args.model,
            worker_type=worker_type,
            needs=needs,
        )

    def _lora_registration_topology(
        self,
    ) -> tuple[ModelType, WorkerType, list[list[WorkerType]]]:
        """Map the worker's disaggregation role to the LoRA MDC topology.

        Returns ``(model_type, worker_type, needs)`` matching how the base
        model registers (main.py:register_vllm_model + worker_factory).
        """
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            return ModelType.Prefill, WorkerType.Prefill, [[WorkerType.Decode]]
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            return (
                ModelType.Chat | ModelType.Completions,
                WorkerType.Decode,
                [[WorkerType.Prefill]],
            )
        return ModelType.Chat | ModelType.Completions, WorkerType.Aggregated, []

    async def load_lora(self, body: dict) -> dict:
        """Load a LoRA adapter dynamically into vLLM's AsyncLLM engine.

        Request body: ``{"lora_name": str, "source": {"uri": str}}``.

        Idempotent: concurrent loads of the same name are serialized and only
        one load operation happens.
        """
        request = body or {}
        if self.engine_client is None:
            return {"status": "error", "message": "Engine is not running"}
        try:
            lora_name = request.get("lora_name")
            if not lora_name:
                return {
                    "status": "error",
                    "message": "'lora_name' is required in request",
                }

            # Reject names that collide with the base model. A LoRA card shares
            # the frontend model key with its name, so an adapter named after the
            # base model would shadow it and make _resolve_lora_request route
            # plain base-model requests through the adapter.
            if lora_name in (self._served_model_name, self.engine_args.model):
                return {
                    "status": "error",
                    "message": (
                        f"LoRA name '{lora_name}' collides with the base model; "
                        "choose a different adapter name"
                    ),
                }

            logger.debug("load_lora request keys: %s", list(request.keys()))

            source = request.get("source")
            if not source or not isinstance(source, dict):
                return {
                    "status": "error",
                    "message": "'source' object is required in request",
                }

            lora_uri = source.get("uri")
            if not lora_uri:
                return {
                    "status": "error",
                    "message": "'source.uri' is required in request",
                }

            lora_manager = get_lora_manager()
            if lora_manager is None:
                return {
                    "status": "error",
                    "message": "LoRAManager not initialized. Set DYN_LORA_ENABLED=true to enable URI-based LoRA loading.",
                }

            # Serialize load/unload operations per lora_name.
            lock = self._get_lora_lock(lora_name)
            async with lock:
                # Idempotency check after acquiring the lock: a concurrent
                # request may have loaded this LoRA while we waited.
                if lora_name in self.loaded_loras:
                    lora_id = self.loaded_loras[lora_name].id
                    # The adapter is loaded into the engine, but its
                    # discovery card may be missing (a prior publish failed
                    # and the engine-side rollback also failed). Reconcile by
                    # retrying the publish instead of reporting early success.
                    if (
                        self._endpoint is not None
                        and lora_name not in self._published_loras
                    ):
                        logger.info(
                            "LoRA '%s' loaded but unpublished; "
                            "retrying discovery publish",
                            lora_name,
                        )
                        try:
                            await self._publish_lora_card(lora_name, lora_id)
                            self._published_loras.add(lora_name)
                        except Exception as e:
                            logger.exception(
                                "Failed to publish LoRA %s ModelDeploymentCard: %s",
                                lora_name,
                                e,
                            )
                            return {
                                "status": "error",
                                "message": f"LoRA '{lora_name}' is loaded but discovery publish failed: {str(e)}",
                                "lora_name": lora_name,
                            }
                    logger.info(
                        "LoRA adapter already loaded (concurrent request completed): "
                        "%s with ID %s",
                        lora_name,
                        lora_id,
                    )
                    return {
                        "status": "success",
                        "message": f"LoRA adapter '{lora_name}' already loaded",
                        "lora_name": lora_name,
                        "lora_id": lora_id,
                    }

                logger.info("Downloading LoRA adapter: %s from %s", lora_name, lora_uri)
                download_result = await lora_manager.download_lora(lora_uri)

                if download_result["status"] != "success":
                    return {
                        "status": "error",
                        "message": f"Failed to download LoRA: {download_result.get('message', 'Unknown error')}",
                    }

                lora_path = download_result["local_path"]
                logger.debug("LoRA downloaded to: %s", lora_path)

                # Deterministic ID from lora_name before using it.
                lora_id = lora_name_to_id(lora_name)

                await self.engine_client.add_lora(
                    LoRARequest(
                        lora_name=lora_name,
                        lora_int_id=lora_id,
                        lora_path=lora_path,
                    )
                )

                self.loaded_loras[lora_name] = LoRAInfo(id=lora_id, path=lora_path)
                logger.info(
                    "Successfully loaded LoRA adapter: %s with ID %s",
                    lora_name,
                    lora_id,
                )

                # Publish the LoRA as a ModelDeploymentCard so the frontend
                # can discover it and route to this worker instance.
                if self._endpoint is not None:
                    logger.debug(
                        "Publishing LoRA '%s' ModelDeploymentCard to %s",
                        lora_name,
                        self._endpoint,
                    )
                    try:
                        await self._publish_lora_card(lora_name, lora_id)
                        self._published_loras.add(lora_name)
                        logger.info(
                            "Successfully published LoRA '%s' ModelDeploymentCard",
                            lora_name,
                        )
                    except Exception as e:
                        logger.exception(
                            "Failed to publish LoRA %s ModelDeploymentCard: %s",
                            lora_name,
                            e,
                        )

                        # Rollback: remove the LoRA from the engine to keep
                        # engine state and discovery consistent. If the
                        # rollback itself fails, the entry stays in
                        # `loaded_loras` but absent from `_published_loras`,
                        # so a retried load reconciles the publish.
                        try:
                            logger.debug(
                                "Rolling back: removing LoRA '%s' from engine",
                                lora_name,
                            )
                            await self.engine_client.remove_lora(lora_id)
                            self.loaded_loras.pop(lora_name, None)
                            logger.debug(
                                "Successfully rolled back LoRA '%s'", lora_name
                            )
                        except Exception as rollback_error:
                            logger.exception(
                                "Failed to rollback LoRA %s: %s",
                                lora_name,
                                rollback_error,
                            )
                        self._published_loras.discard(lora_name)

                        return {
                            "status": "error",
                            "message": f"Failed to register LoRA '{lora_name}' in discovery registry: {str(e)}",
                            "lora_name": lora_name,
                        }
                else:
                    logger.debug(
                        "Cannot publish LoRA '%s': serving endpoint not ready",
                        lora_name,
                    )

                return {
                    "status": "success",
                    "message": f"LoRA adapter '{lora_name}' loaded successfully",
                    "lora_name": lora_name,
                    "lora_id": lora_id,
                }
        except Exception as e:
            logger.exception("Failed to load LoRA adapter: %s", e)
            return {"status": "error", "message": str(e)}

    async def unload_lora(self, body: dict) -> dict:
        """Unload a LoRA adapter dynamically from vLLM's AsyncLLM engine.

        Request body: ``{"lora_name": str}``.
        """
        request = body or {}
        if self.engine_client is None:
            return {"status": "error", "message": "Engine is not running"}
        try:
            lora_name = request.get("lora_name")
            if not lora_name:
                return {
                    "status": "error",
                    "message": "'lora_name' is required in request",
                }

            # Serialize load/unload operations per lora_name.
            lock = self._get_lora_lock(lora_name)
            async with lock:
                # Check existence *after* waiting for any in-progress load.
                lora = self.loaded_loras.get(lora_name)
                if lora is None:
                    # The adapter is gone from the engine but may still have
                    # a stale discovery card (a prior unload's unregister
                    # failed and the re-add rollback also failed). Reconcile
                    # by retrying the unregister so discovery converges.
                    if (
                        self._endpoint is not None
                        and lora_name in self._published_loras
                    ):
                        logger.info(
                            "LoRA '%s' not loaded but still published; "
                            "retrying discovery unregister",
                            lora_name,
                        )
                        try:
                            await unregister_model(
                                endpoint=self._endpoint,
                                lora_name=lora_name,
                            )
                            self._published_loras.discard(lora_name)
                            return {
                                "status": "success",
                                "message": f"LoRA adapter '{lora_name}' discovery card removed",
                                "lora_name": lora_name,
                            }
                        except Exception as e:
                            logger.exception(
                                "Failed to unregister stale LoRA %s ModelDeploymentCard: %s",
                                lora_name,
                                e,
                            )
                            return {
                                "status": "error",
                                "message": f"Failed to unregister stale LoRA '{lora_name}' from discovery registry: {str(e)}",
                                "lora_name": lora_name,
                            }
                    return {
                        "status": "error",
                        "message": f"LoRA adapter '{lora_name}' not found. Available LoRAs: {list(self.loaded_loras.keys())}",
                    }

                logger.debug("Unloading LoRA adapter: %s", lora_name)
                lora_id = lora.id

                # Stop advertising the adapter *before* removing it from the
                # engine, so the frontend stops routing LoRA traffic here
                # while the adapter still exists. Removing it first would
                # leave a window where requests route to a worker that no
                # longer has the adapter (falling back to base or failing).
                if self._endpoint is not None and lora_name in self._published_loras:
                    logger.debug(
                        "Unregistering LoRA '%s' ModelDeploymentCard",
                        lora_name,
                    )
                    try:
                        await unregister_model(
                            endpoint=self._endpoint,
                            lora_name=lora_name,
                        )
                        self._published_loras.discard(lora_name)
                        logger.info(
                            "Successfully unregistered LoRA '%s' ModelDeploymentCard",
                            lora_name,
                        )
                    except Exception as e:
                        # Nothing mutated yet: the engine still has the
                        # adapter and discovery still advertises it
                        # (consistent and still routable). Surface the error
                        # and leave state intact for a retry.
                        logger.exception(
                            "Failed to unregister LoRA %s ModelDeploymentCard: %s",
                            lora_name,
                            e,
                        )
                        return {
                            "status": "error",
                            "message": f"Failed to unregister LoRA '{lora_name}' from discovery registry: {str(e)}",
                            "lora_name": lora_name,
                        }
                elif self._endpoint is None:
                    logger.debug(
                        "Cannot unregister LoRA '%s': serving endpoint not ready",
                        lora_name,
                    )

                # Discovery no longer routes to this adapter; remove it from
                # the engine.
                try:
                    await self.engine_client.remove_lora(lora_id)
                except Exception as e:
                    # The discovery card is already gone but the engine still
                    # holds the adapter (loaded-but-unpublished). Leave it in
                    # loaded_loras so a retried unload skips the unregister
                    # and retries only the engine removal.
                    logger.exception(
                        "Failed to remove LoRA %s from engine: %s",
                        lora_name,
                        e,
                    )
                    return {
                        "status": "error",
                        "message": f"Failed to remove LoRA '{lora_name}' from engine: {str(e)}",
                        "lora_name": lora_name,
                    }

                del self.loaded_loras[lora_name]

                logger.info(
                    "Successfully unloaded LoRA adapter: %s with ID %s",
                    lora_name,
                    lora_id,
                )
                return {
                    "status": "success",
                    "message": f"LoRA adapter '{lora_name}' unloaded successfully",
                    "lora_name": lora_name,
                    "lora_id": lora_id,
                }
        except Exception as e:
            logger.exception("Failed to unload LoRA adapter: %s", e)
            return {"status": "error", "message": str(e)}

    async def list_loras(self, body: dict) -> dict:
        """List all loaded LoRA adapters as a lora_name -> lora_id mapping."""
        try:
            loras = {name: lora.id for name, lora in self.loaded_loras.items()}
            return {
                "status": "success",
                "loras": loras,
                "count": len(loras),
            }
        except Exception as e:
            logger.error("Failed to list LoRA adapters: %s", e)
            return {"status": "error", "message": str(e)}

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

    # No is_quiescent() override: vLLM's NixlConnector exposes no idle signal,
    # so it inherits the base None and the framework drains prefill workers for
    # the full budget.

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
            self._multimodal_request_processor = None
            # Drop the serving endpoint and dynamic-LoRA bookkeeping so a
            # shut-down engine holds no dangling endpoint reference and no
            # stale adapter state. Discovery cards published for the worker are
            # reclaimed when the endpoint's lease expires on process exit. The
            # stripe locks are fixed process state, not per-adapter, so they
            # are left intact.
            self._endpoint = None
            self.loaded_loras.clear()
            self._published_loras.clear()
            if self._prometheus_temp_dir is not None:
                if (
                    os.environ.get("PROMETHEUS_MULTIPROC_DIR")
                    == self._prometheus_temp_dir.name
                ):
                    os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
                self._prometheus_temp_dir.cleanup()
                self._prometheus_temp_dir = None
            logger.info("vLLM engine shutdown")
