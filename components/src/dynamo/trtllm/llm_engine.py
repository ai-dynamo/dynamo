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
import re
import sys
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import Any

from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi import KvCacheConfig, SchedulerConfig
from tensorrt_llm.llmapi.disagg_utils import get_global_disagg_request_id
from tensorrt_llm.llmapi.llm import SamplingParams
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.sampling_params import GuidedDecodingParams
from torch.cuda import device_count

from dynamo._core import Context
from dynamo.common.backend.disagg import require_prefill_result
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode as CommonDisaggregationMode
from dynamo.llm import ModelInput
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.engine import Backend, TensorRTLLMEngine
from dynamo.trtllm.utils.disagg_utils import (
    DisaggregatedParams,
    DisaggregatedParamsCodec,
)
from dynamo.trtllm.utils.trtllm_utils import deep_update, warn_override_collisions

logger = logging.getLogger(__name__)

# Match legacy `trtllm/main.py` so prefill drain behaves the same.
_DRAIN_TIMEOUT_S = 30.0
_DRAIN_POLL_INTERVAL_S = 0.5

# 1021 is the largest 10-bit prime — spreads machine_ids more evenly
# under modulo than 1024 would. Matches legacy
# `workers/llm_worker.py: connection_id() % 1021`.
_DISAGG_MACHINE_ID_MAX = 1021


# Bridges trtllm's local enum into the common one. ENCODE absent —
# rejected up front in from_args().
_TRTLLM_TO_COMMON_DISAGG = {
    DisaggregationMode.AGGREGATED: CommonDisaggregationMode.AGGREGATED,
    DisaggregationMode.PREFILL: CommonDisaggregationMode.PREFILL,
    DisaggregationMode.DECODE: CommonDisaggregationMode.DECODE,
}


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
        self._engine: TensorRTLLMEngine | None = None
        self._default_sampling_params = SamplingParams(detokenize=False)
        # Per-request GenerationResult handles for `abort()` lookup. TRT-LLM's
        # abort API is `GenerationResult.abort()` (not by request-id), so we
        # need the handle. Other engines (vllm, sglang) abort by id and
        # don't keep this map.
        self._active_requests: dict[str, GenerationResult] = {}
        # Set in start() from worker_id. 10-bit field is a TRT-LLM API
        # constraint; collisions possible at scale (~30 replicas).
        self._disagg_machine_id: int = 0

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
            "backend": Backend.PYTORCH,
            "kv_cache_config": KvCacheConfig(
                free_gpu_memory_fraction=config.free_gpu_memory_fraction,
            ),
            "gpus_per_node": gpus_per_node,
            "max_num_tokens": config.max_num_tokens,
            "max_seq_len": config.max_seq_len,
            "max_beam_width": config.max_beam_width,
            "max_batch_size": config.max_batch_size,
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

        return EngineConfig(
            model=self.model_name,
            served_model_name=self.served_model_name,
            context_length=self.max_seq_len,
            kv_cache_block_size=self.kv_block_size,
            max_num_seqs=self.max_batch_size,
            max_num_batched_tokens=self.max_num_tokens,
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        assert self._engine is not None, "Engine not initialized"

        token_ids = request.get("token_ids", [])
        sampling_params = self._override_sampling_params(
            self._default_sampling_params, request
        )

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

        ignore_eos = stop_conditions.get("ignore_eos")
        if ignore_eos:
            sampling_params.ignore_eos = ignore_eos

        # Prefill returns one non-streaming chunk carrying the handoff —
        # matches the legacy disagg wire format.
        streaming = not is_prefill
        generation_result = self._engine.llm.generate_async(
            inputs=token_ids,
            sampling_params=sampling_params,
            streaming=streaming,
            disaggregated_params=disaggregated_params,
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

                    if output.finish_reason:
                        out["finish_reason"] = str(output.finish_reason)

                    if out.get("finish_reason") or res.finished:
                        if not out.get("finish_reason"):
                            out["finish_reason"] = "unknown"
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

                    yield out
                    output_tokens_per_choice[output_idx] = next_total
        finally:
            if request_id is not None:
                self._active_requests.pop(request_id, None)

    @staticmethod
    def _decode_prefill_handoff(
        prefill_result: dict[str, Any]
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

    async def drain(self) -> None:
        """Prefill-only: poll until in-flight requests finish so a
        decode peer's NIXL pull doesn't see freed GPU memory (#7319).
        Mirrors legacy `_make_drain_callback`."""
        if (
            self._engine is None
            or self.disaggregation_mode != DisaggregationMode.PREFILL
        ):
            return

        deadline = asyncio.get_running_loop().time() + _DRAIN_TIMEOUT_S
        logger.info(
            "Draining in-flight requests on prefill worker (timeout=%.1fs)",
            _DRAIN_TIMEOUT_S,
        )
        # The stats stream can raise asyncio.TimeoutError when the engine
        # has nothing fresh to report, or StopAsyncIteration when the
        # underlying iterator is exhausted — both are benign and just mean
        # "no signal this tick, try again". A wider `Exception` would
        # swallow code bugs introduced later; the test stub raises
        # RuntimeError to exercise the retry path.
        _BENIGN_POLL = (asyncio.TimeoutError, StopAsyncIteration, RuntimeError)
        while asyncio.get_running_loop().time() < deadline:
            try:
                stats_iter = self._engine.llm.get_stats_async(timeout=2)
                stat = await anext(stats_iter)
                active = stat.get("numActiveRequests", 0)
                queued = stat.get("numQueuedRequests", 0)
                if active + queued == 0:
                    logger.info("All in-flight requests drained")
                    return
                logger.info(
                    "Waiting for %d in-flight request(s) (active=%d, queued=%d)",
                    active + queued,
                    active,
                    queued,
                )
            except _BENIGN_POLL as e:
                logger.debug("Stats poll failed during drain: %s", e)
            await asyncio.sleep(_DRAIN_POLL_INTERVAL_S)
        logger.warning(
            "Drain timeout (%.1fs) reached; proceeding with shutdown — "
            "some NIXL transfers may still be in flight",
            _DRAIN_TIMEOUT_S,
        )

    async def cleanup(self) -> None:
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
                structural_tag=guided_decoding.get("structural_tag"),
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
