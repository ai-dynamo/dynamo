# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import dataclasses
import gc
import logging
import os
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any, Optional, Union

import torch
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi.llm import SamplingParams
from tensorrt_llm.sampling_params import GuidedDecodingParams
from tensorrt_llm.scheduling_params import SchedulingParams

from dynamo._core import Context
from dynamo.common.utils.otel_tracing import build_trace_headers
from dynamo.logits_processing.examples import HelloWorldLogitsProcessor
from dynamo.nixl_connect import Connector
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.engine import TensorRTLLMEngine
from dynamo.trtllm.logits_processing.adapter import create_trtllm_adapters
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor
from dynamo.trtllm.publisher import Publisher
from dynamo.trtllm.request_handlers.base_generative_handler import BaseGenerativeHandler
from dynamo.trtllm.utils.disagg_utils import (
    DisaggregatedParams,
    DisaggregatedParamsCodec,
)

configure_dynamo_logging()

DEFAULT_MEMORY_OCCUPATION_TAGS = ["kv_cache", "weights"]


@dataclass
class RequestHandlerConfig:
    """
    Configuration for the request handler
    """

    engine: TensorRTLLMEngine
    default_sampling_params: SamplingParams
    publisher: Publisher
    disaggregation_mode: DisaggregationMode
    encode_client: Optional[object] = None
    multimodal_processor: Optional[
        MultimodalRequestProcessor
    ] = None  # for multimodal support
    generate_endpoint: Optional[Any] = None
    connector: Optional[Connector] = None
    runtime: Optional[
        DistributedRuntime
    ] = None  # DistributedRuntime reference for graceful shutdown
    metrics_collector: Optional[Any] = None  # TensorRT-LLM MetricsCollector
    kv_block_size: int = 32
    shutdown_event: Optional[asyncio.Event] = None
    encoder_cache_capacity_gb: float = 0  # Encoder cache capacity in GB
    disable_request_abort: bool = True


class HandlerBase(BaseGenerativeHandler):
    """
    Base class for LLM request handlers (text generation, multimodal LLM).

    This class is dedicated to LLM-based generation using TensorRT-LLM engine.
    For diffusion-based handlers (video, image), see VideoGenerationHandler
    and ImageGenerationHandler which inherit directly from BaseGenerativeHandler.

    Inherits from BaseGenerativeHandler to ensure a consistent interface
    across all generative handlers (LLM, video, image).
    """

    def __init__(self, config: RequestHandlerConfig):
        self.engine = config.engine
        self.default_sampling_params = config.default_sampling_params
        self.publisher = config.publisher
        self.metrics_collector = config.metrics_collector
        self.disaggregation_mode = config.disaggregation_mode
        self.encode_client = config.encode_client
        self.multimodal_processor = config.multimodal_processor
        self.generate_endpoint = config.generate_endpoint
        self.first_generation = True
        self.connector = config.connector
        # Store runtime reference for graceful shutdown
        self.runtime = config.runtime
        self.kv_block_size: int = config.kv_block_size
        self.shutdown_event = config.shutdown_event
        self.disable_request_abort = config.disable_request_abort
        self._sleep_wake_lock = asyncio.Lock()
        self._inflight_lock = asyncio.Lock()
        self._inflight_requests = 0
        self._no_inflight_requests = asyncio.Event()
        self._no_inflight_requests.set()
        self._memory_released = False
        self._reject_new_requests = False

    def check_error(self, result: dict) -> bool:
        """
        Check if there is an error in the result.
        """
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            return result["finish_reason"] == "error"
        else:
            return (
                result["finish_reason"] == "stop" or result["finish_reason"] == "error"
            )

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
        except TimeoutError as exc:
            async with self._inflight_lock:
                inflight = self._inflight_requests
            raise RuntimeError(
                f"Timed out waiting for {inflight} in-flight request(s) to finish"
            ) from exc

    @staticmethod
    def _normalize_memory_tags(body: dict) -> list[str]:
        tags = body.get("tags")
        if tags is None:
            return list(DEFAULT_MEMORY_OCCUPATION_TAGS)
        if not isinstance(tags, list):
            raise ValueError("'tags' must be a list of strings")
        normalized: list[str] = []
        for tag in tags:
            if not isinstance(tag, str):
                raise ValueError("'tags' must be a list of strings")
            candidate = tag.strip().lower()
            if candidate not in {"kv_cache", "weights"}:
                raise ValueError(f"Unsupported memory tag: {tag!r}")
            if candidate not in normalized:
                normalized.append(candidate)
        if not normalized:
            raise ValueError("'tags' must include at least one value")
        return normalized

    @staticmethod
    def _split_memory_tags(tags: list[str]) -> tuple[list[str], bool]:
        collective_tags = [tag for tag in tags if tag == "kv_cache"]
        handle_weights = "weights" in tags
        return collective_tags, handle_weights

    @staticmethod
    def _is_collective_rpc_unsupported(exc: Exception) -> bool:
        return "does not support collective rpc" in str(exc).lower()

    def _can_use_local_kv_sleep_fallback(self) -> bool:
        """Local tag-based sleep/wake fallback is safe for single-rank executors."""
        try:
            llm_args = getattr(self.engine.llm, "args", None)
            parallel_config = getattr(llm_args, "parallel_config", None)
            world_size = int(getattr(parallel_config, "world_size", 1))
        except Exception:
            return False
        return world_size == 1

    @staticmethod
    def _call_local_virtual_memory_method(method: str, tags: list[str]) -> None:
        from tensorrt_llm._torch.virtual_memory import (
            materialize_with_tag,
            release_with_tag,
            verify_sleep_wakeup_tags,
        )

        normalized_tags = [
            str(tag.value) if hasattr(tag, "value") else str(tag)
            for tag in verify_sleep_wakeup_tags(tags)
        ]
        if not normalized_tags:
            return

        torch.cuda.synchronize()
        if method == "sleep":
            release_with_tag(*normalized_tags)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            return
        if method in {"wakeup", "wake_up"}:
            materialize_with_tag(*normalized_tags)
            torch.cuda.synchronize()
            return
        raise ValueError(f"Unsupported virtual memory method: {method}")

    async def _call_collective_rpc(self, method: str, tags: list[str]) -> None:
        rpc = getattr(self.engine.llm, "_collective_rpc", None)
        if rpc is None:
            raise RuntimeError(
                "TensorRT-LLM runtime does not support collective RPC memory operations"
            )

        kwargs = {
            "args": (tags,),
            "kwargs": {},
            "non_block": False,
        }
        try:
            rpc(method, **kwargs)
        except Exception:
            if method != "wakeup":
                raise
            rpc("wake_up", **kwargs)

    @staticmethod
    def _get_gms_manager():
        try:
            import gpu_memory_service
        except ImportError as exc:
            raise RuntimeError(
                "GPU Memory Service package is not available for TRT-LLM sleep/wake"
            ) from exc

        manager = gpu_memory_service.get_gms_client_memory_manager()
        if manager is None:
            raise RuntimeError("GPU Memory Service manager is not initialized")
        return manager

    async def release_memory_occupation(self, body: dict) -> dict:
        body = body or {}
        try:
            tags = self._normalize_memory_tags(body)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        timeout_raw = body.get("timeout_s", 30.0)
        try:
            timeout_s = float(timeout_raw)
        except (TypeError, ValueError):
            return {"status": "error", "message": "timeout_s must be a number"}
        if timeout_s <= 0:
            return {"status": "error", "message": "timeout_s must be > 0"}

        collective_tags, handle_weights = self._split_memory_tags(tags)

        async with self._sleep_wake_lock:
            if self._memory_released:
                return {"status": "ok", "message": "Memory already released"}

            await self._set_reject_new_requests(True)
            endpoint_unregistered = False
            skipped_tags: list[str] = []
            try:
                if self.generate_endpoint is not None:
                    await self.generate_endpoint.unregister_endpoint_instance()
                    endpoint_unregistered = True

                await self._wait_for_inflight_requests(timeout_s)

                if collective_tags:
                    try:
                        await self._call_collective_rpc("sleep", collective_tags)
                    except Exception as exc:
                        if not self._is_collective_rpc_unsupported(exc):
                            raise
                        if self._can_use_local_kv_sleep_fallback():
                            logging.info(
                                "Collective RPC is unsupported; using local kv_cache sleep fallback"
                            )
                            self._call_local_virtual_memory_method(
                                "sleep", collective_tags
                            )
                        else:
                            logging.warning(
                                "Skipping kv_cache sleep because collective RPC is unsupported: %s",
                                exc,
                            )
                            skipped_tags.append("kv_cache")

                if handle_weights:
                    manager = self._get_gms_manager()
                    manager.unmap_all_vas()
                    manager.disconnect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                self._memory_released = True
                response = {
                    "status": "ok",
                    "message": f"Memory released for tags: {tags}",
                }
                if skipped_tags:
                    response["skipped_tags"] = skipped_tags
                return response
            except Exception as exc:
                logging.error("Failed to release memory occupation: %s", exc)
                if endpoint_unregistered and self.generate_endpoint is not None:
                    try:
                        await self.generate_endpoint.register_endpoint_instance()
                    except Exception:
                        logging.exception(
                            "Failed to restore endpoint registration after release failure"
                        )
                await self._set_reject_new_requests(False)
                return {"status": "error", "message": str(exc)}

    async def resume_memory_occupation(self, body: dict) -> dict:
        body = body or {}
        try:
            tags = self._normalize_memory_tags(body)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        collective_tags, handle_weights = self._split_memory_tags(tags)

        async with self._sleep_wake_lock:
            if not self._memory_released:
                return {"status": "ok", "message": "Memory already resumed"}

            try:
                skipped_tags: list[str] = []
                if handle_weights:
                    manager = self._get_gms_manager()
                    if manager.is_unmapped:
                        from gpu_memory_service.integrations import (
                            trtllm as trtllm_integration,
                        )

                        manager.connect(trtllm_integration.get_gms_lock_mode())
                        manager.remap_all_vas()

                if collective_tags:
                    try:
                        await self._call_collective_rpc("wakeup", collective_tags)
                    except Exception as exc:
                        if not self._is_collective_rpc_unsupported(exc):
                            raise
                        if self._can_use_local_kv_sleep_fallback():
                            logging.info(
                                "Collective RPC is unsupported; using local kv_cache wakeup fallback"
                            )
                            self._call_local_virtual_memory_method(
                                "wakeup", collective_tags
                            )
                        else:
                            logging.warning(
                                "Skipping kv_cache wakeup because collective RPC is unsupported: %s",
                                exc,
                            )
                            skipped_tags.append("kv_cache")

                if self.generate_endpoint is not None:
                    await self.generate_endpoint.register_endpoint_instance()

                await self._set_reject_new_requests(False)
                self._memory_released = False
                response = {
                    "status": "ok",
                    "message": f"Memory resumed for tags: {tags}",
                }
                if skipped_tags:
                    response["skipped_tags"] = skipped_tags
                return response
            except Exception as exc:
                logging.error("Failed to resume memory occupation: %s", exc)
                return {"status": "error", "message": str(exc)}

    @staticmethod
    def _extract_logprobs(
        output, num_output_tokens_so_far: int
    ) -> tuple[list[float] | None, list[list[dict]] | None]:
        """
        Extract logprobs from the TRTLLM output for new tokens.

        Args:
            output: TRTLLM CompletionOutput object
            num_output_tokens_so_far: Number of tokens already processed
        Returns:
            Tuple of (log_probs, top_logprobs) in Dynamo's expected format:
            - log_probs: List of log probabilities for each new token
            - top_logprobs: List of top logprobs dicts for each new token
        """
        if output.logprobs is None:
            return None, None

        # Get logprobs for new tokens only
        new_logprobs = output.logprobs[num_output_tokens_so_far:]
        if not new_logprobs:
            return None, None

        # From TRTLLM CompletionOutput API, logprobs: (TokenLogprobs | List[float], optional)
        # Expect TokenLogprobs output when logprobs is set, check edge case where list[float] is returned instead
        if isinstance(new_logprobs[0], float):
            return [float(lp) for lp in new_logprobs], None

        log_probs = []
        top_logprobs = []

        for token_idx, token_logprobs_dict in enumerate(new_logprobs):
            if token_logprobs_dict is None:
                continue

            # Get the actual token_id that was generated at this position
            actual_token_id = output.token_ids[num_output_tokens_so_far + token_idx]

            # Extract log probability for the selected token
            if actual_token_id in token_logprobs_dict:
                selected_logprob = token_logprobs_dict[actual_token_id]
                log_probs.append(float(selected_logprob.logprob))
            else:
                # Fallback: use the first logprob if selected token not found
                first_logprob = next(iter(token_logprobs_dict.values()), None)
                if first_logprob:
                    log_probs.append(float(first_logprob.logprob))

            # Build top_logprobs list for this token position
            # NOTE: TRTLLM LogProb API doesn't have decoded_token, will default to None
            token_top_logprobs = []
            for tok_id, logprob_info in token_logprobs_dict.items():
                token_top_logprobs.append(
                    {
                        "rank": logprob_info.rank
                        if hasattr(logprob_info, "rank")
                        else 0,
                        "token_id": tok_id,
                        "token": (
                            logprob_info.decoded_token
                            if hasattr(logprob_info, "decoded_token")
                            else None
                        ),
                        "logprob": float(logprob_info.logprob),
                    }
                )
            top_logprobs.append(token_top_logprobs)

        return log_probs if log_probs else None, top_logprobs if top_logprobs else None

    async def _handle_cancellation(
        self, generation_result: GenerationResult, context: Context
    ):
        """
        Background task to trigger cancellation if request is cancelled or shutdown
        event is set.

        Raise GeneratorExit if shutdown event is triggered.
        """
        try:
            cancellation_triggers: list[asyncio.Future[Any]] = [
                context.async_killed_or_stopped(),  # Request cancellation
            ]
            # Shutdown cancellation
            shutdown_task = None
            if self.shutdown_event is not None:
                shutdown_task = asyncio.create_task(self.shutdown_event.wait())
                cancellation_triggers.append(shutdown_task)

            # Wait for cancellation to be triggered
            done, pending = await asyncio.wait(
                cancellation_triggers,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Abort the generation unless disabled
            if self.disable_request_abort:
                logging.debug(
                    f"Request ID {context.id()} cancelled but abort() skipped "
                    "(DYN_TRTLLM_DISABLE_REQUEST_ABORT=true)"
                )
            else:
                generation_result.abort()
                logging.debug(f"Aborted Request ID: {context.id()}")

            # Clean up any remaining background task
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Raise GeneratorExit if cancellation is due to shutdown event triggered
            if shutdown_task in done:
                raise GeneratorExit("Engine was shut down during generation.")

        except asyncio.CancelledError:
            # Task was cancelled, which is expected when generation completes normally
            pass

    @asynccontextmanager
    async def _cancellation_monitor(
        self, generation_result: GenerationResult, context: Context
    ) -> AsyncGenerator[asyncio.Task, None]:
        """
        Monitor for cancellation triggers and cancel by calling
        generation_result.abort().

        Raise GeneratorExit if shutdown event is triggered.

        Yields:
            asyncio.Task: The cancellation monitoring task
        """
        monitor_task = asyncio.create_task(
            self._handle_cancellation(generation_result, context)
        )

        try:
            yield monitor_task
        finally:
            if not monitor_task.done():
                # Cancellation not triggered - clean up the background monitoring task
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            else:
                # Cancellation triggered - propagate any exceptions
                monitor_task.result()

    def _decode_disaggregated_params_from_prefill(
        self, prefill_result: dict
    ) -> tuple[Any, dict]:
        """
        Extract and decode disaggregated params from prefill_result.

        Args:
            prefill_result: Result from prefill worker containing encoded disaggregated params

        Returns:
            Tuple of (disaggregated_params, epd_metadata) where:
            - disaggregated_params: Decoded LlmDisaggregatedParams object
            - epd_metadata: Dictionary containing EPD-specific metadata (_epd_processed_prompt, etc.)
        """
        params_dict = prefill_result["disaggregated_params"]

        # Remove worker_id if present (added by prefill worker, not needed for decode)
        params_dict.pop("worker_id", None)

        # Extract EPD metadata that was packed by prefill worker
        epd_metadata = {}
        if "_epd_metadata" in params_dict:
            epd_metadata = params_dict.pop("_epd_metadata")
            logging.debug(
                f"DECODE: Extracted _epd_metadata with {len(epd_metadata)} fields"
            )

        # Decode the disaggregated params
        disaggregated_params = DisaggregatedParamsCodec.decode(
            DisaggregatedParams(**params_dict)
        )
        # Set to generation_only mode for decode phase
        disaggregated_params.request_type = "generation_only"

        # In generation-only mode, multimodal embeddings are already processed and in KV cache
        # Remove multimodal_embedding_handles to avoid TRT-LLM validation error
        # NOTE: `hasattr` is used because multimodal_embedding_handles may not be present
        # on DisaggregatedParams in all EPD flows (e.g., text-only requests or certain stages).
        if (
            hasattr(disaggregated_params, "multimodal_embedding_handles")
            and disaggregated_params.multimodal_embedding_handles
        ):
            disaggregated_params.multimodal_embedding_handles = None

        logging.debug("DECODE: Set request_type to generation_only")

        return disaggregated_params, epd_metadata

    def _encode_and_pack_disaggregated_params(
        self,
        output: GenerationResult,
        disaggregated_params: Any,
        request: dict,
        res: Any,
        processed_input: Any = None,
    ) -> Optional[dict]:
        """
        Encode and pack disaggregated params for PREFILL mode response.

        Handles:
        - Choosing between output and input disaggregated params
        - Preserving multimodal_embedding_handles in EPD flow
        - Encoding params for transmission
        - Packing prefill metadata for DECODE optimization

        Args:
            output: GenerationResult from the engine
            disaggregated_params: Input disaggregated params
            request: Original request dict
            res: RequestOutput object with prompt and prompt_token_ids attributes
            processed_input: The processed input dict from process_openai_request (contains correct prompt)

        Returns:
            Dictionary with encoded disaggregated params, or None if encoding failed
        """
        # In EPD flow, output.disaggregated_params might be None, use the input params
        params_to_encode = (
            output.disaggregated_params
            if output.disaggregated_params is not None
            else disaggregated_params
        )

        # In EPD flow, manually preserve multimodal_embedding_handles from input
        # because TRT-LLM engine may not propagate them through prefill
        if params_to_encode is not None and disaggregated_params is not None:
            input_handles = getattr(
                disaggregated_params,
                "multimodal_embedding_handles",
                None,
            )
            output_handles = getattr(
                params_to_encode, "multimodal_embedding_handles", None
            )

            if input_handles is not None and output_handles is None:
                params_to_encode.multimodal_embedding_handles = input_handles
                # Also preserve hashes if they exist
                input_hashes = getattr(disaggregated_params, "multimodal_hashes", None)
                if input_hashes is not None:
                    params_to_encode.multimodal_hashes = input_hashes

        encoded_params = DisaggregatedParamsCodec.encode(params_to_encode)

        if encoded_params is None:
            logging.error("PREFILL: encoded_params is None - decode worker will fail!")
            return None

        logging.debug("PREFILL: Successfully encoded disaggregated params")
        params_dict = asdict(encoded_params)

        # Pack prefill metadata for DECODE worker optimization
        # The frontend only forwards disaggregated_params from prefill response
        # Note: max_tokens is already handled by Rust frontend's PrefillRouter
        prefill_metadata = {}

        # ALWAYS pack prompt info for DECODE to skip re-processing
        # Per TRT-LLM team: DECODE never needs to reload images - KV cache has the context
        # Use processed_input['prompt'] (from process_openai_request) which is the actual
        # multimodal prompt used by TRT-LLM, not res.prompt which might be raw
        if (
            processed_input
            and isinstance(processed_input, dict)
            and processed_input.get("prompt")
        ):
            prefill_metadata["_prefill_prompt"] = processed_input["prompt"]
        elif res.prompt:
            prefill_metadata["_prefill_prompt"] = res.prompt
        if res.prompt_token_ids:
            prefill_metadata["_prefill_prompt_token_ids"] = list(res.prompt_token_ids)

        # EPD-specific: use encoder's prompt if available
        if "_epd_processed_prompt" in request and res.prompt:
            prefill_metadata["_epd_processed_prompt"] = res.prompt
        if "_epd_prompt_token_ids" in request and res.prompt_token_ids:
            prefill_metadata["_epd_prompt_token_ids"] = list(res.prompt_token_ids)

        # Add metadata to the disaggregated_params dict
        if prefill_metadata:
            params_dict["_epd_metadata"] = prefill_metadata

        return params_dict

    def _setup_disaggregated_params_for_mode(
        self,
        request: dict,
        ep_disaggregated_params: Optional[Any],
    ) -> tuple[Any, Any, dict]:
        """
        Setup disaggregated_params based on disaggregation mode.

        For PREFILL mode:
        - Uses ep_disaggregated_params from encode worker if available
        - Otherwise creates new LlmDisaggregatedParams with request_type="context_only"

        For DECODE mode:
        - Decodes disaggregated_params from prefill_result
        - Extracts EPD metadata for prompt optimization

        For PREFILL_AND_DECODE (aggregated) mode:
        - Uses ep_disaggregated_params from encode worker if available
          (passes multimodal_embedding_handles to TRT-LLM and sets
          request_type="context_and_generation" for full prefill + decode)

        Args:
            request: Request dictionary (may contain prefill_result)
            ep_disaggregated_params: Optional params from encode worker (EPD flow)

        Returns:
            Tuple of (disaggregated_params, ep_disaggregated_params, epd_metadata)
        """
        disaggregated_params = None
        epd_metadata: dict[str, Any] = {}

        # PREFILL mode: setup context_only params
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if ep_disaggregated_params:
                ep_disaggregated_params.request_type = "context_only"
                disaggregated_params = ep_disaggregated_params
            else:
                disaggregated_params = LlmDisaggregatedParams(
                    request_type="context_only"
                )

        # AGGREGATED (prefill_and_decode) mode with encoder disaggregation:
        # Pass the encode worker's DisaggregatedParams (containing
        # multimodal_embedding_handles) directly so TRT-LLM can import
        # the vision embeddings.  Use "context_and_generation" so the
        # engine runs a full prefill + decode cycle.
        elif (
            self.disaggregation_mode == DisaggregationMode.AGGREGATED
            and ep_disaggregated_params is not None
        ):
            disaggregated_params = DisaggregatedParamsCodec.decode(
                ep_disaggregated_params
            )
            disaggregated_params.request_type = "context_and_generation"

        # DECODE mode: decode params from prefill_result
        prefill_result = request.get("prefill_result")
        if prefill_result and "disaggregated_params" in prefill_result:
            (
                disaggregated_params,
                epd_metadata,
            ) = self._decode_disaggregated_params_from_prefill(prefill_result)
            # For full EPD flow, make decoded params available to multimodal processor
            ep_disaggregated_params = disaggregated_params

        return disaggregated_params, ep_disaggregated_params, epd_metadata

    async def _prepare_input_for_generation(
        self,
        request: dict,
        embeddings: Optional[Union[torch.Tensor, dict]],
        ep_disaggregated_params: Optional[Any],
        epd_metadata: dict,
    ) -> Any:
        """
        Prepare input for TRT-LLM generation (handles multimodal/text flows).

        Three paths:
        1. DECODE with prefill metadata: Use cached prompt, skip image re-processing
        2. Multimodal: Process via multimodal_processor
        3. Text-only: Use token_ids from request

        Args:
            request: Request dictionary
            embeddings: Optional embeddings tensor/dict from encode worker
            ep_disaggregated_params: Optional params from encode worker (EPD flow)
            epd_metadata: Metadata from prefill worker (DECODE optimization)

        Returns:
            Processed input for TRT-LLM (dict with prompt/token_ids, or raw token_ids)
        """
        # DECODE mode: Use prefill metadata to skip re-processing multimodal content
        # Per TRT-LLM team: DECODE never needs to reload images - KV cache has the context
        has_prefill_metadata = epd_metadata and (
            epd_metadata.get("_prefill_prompt")
            or epd_metadata.get("_epd_processed_prompt")
        )

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and has_prefill_metadata
        ):
            # Use prompt/token_ids from PREFILL, skip image re-processing
            prefill_prompt = epd_metadata.get("_prefill_prompt") or epd_metadata.get(
                "_epd_processed_prompt"
            )
            prefill_token_ids = epd_metadata.get(
                "_prefill_prompt_token_ids"
            ) or epd_metadata.get("_epd_prompt_token_ids")

            # Build input without multimodal data (already in KV cache)
            # Use the SAME multimodal key that PREFILL used:
            # - EPD/Embeddings flow: PREFILL used multi_modal_embeddings
            # - Simple P→D (image URL): PREFILL used multi_modal_data
            is_epd_flow = epd_metadata.get("_epd_processed_prompt") is not None

            processed_input = {
                "prompt": prefill_prompt,
                "prompt_token_ids": prefill_token_ids,
            }
            if is_epd_flow:
                processed_input["multi_modal_embeddings"] = None
            else:
                processed_input["multi_modal_data"] = None
            return processed_input

        # PREFILL/ENCODE/AGGREGATED: Process multimodal content if available
        if self.multimodal_processor:
            processed_input = await self.multimodal_processor.process_openai_request(
                request, embeddings, ep_disaggregated_params
            )
            if processed_input:
                return processed_input

            # If multimodal processing returned None but request has multimodal data,
            # this is an error (not a text-only request). Raise instead of falling back.
            if request.get("multi_modal_data"):
                raise RuntimeError(
                    "Failed to process multimodal request. Check server logs for details. "
                    "Common issues: missing allowed_local_media_path configuration, "
                    "file not found, or file outside allowed directory."
                )

        # Fallback: text-only flow (no multimodal processor or no multimodal data)
        return request.get("token_ids")

    def _normalize_request_format(self, request: dict) -> None:
        """
        Convert OpenAI request format to TRT-LLM internal format.

        Moves fields from OpenAI locations to where TRT-LLM expects them:
        - max_tokens: top-level → stop_conditions.max_tokens
        - temperature: top-level → sampling_options.temperature

        Note: The Rust frontend's PrefillRouter handles the *value* of max_tokens
        (sets to 1 for prefill, restores original for decode). This method only
        moves fields to the correct location.

        Args:
            request: Request dictionary to normalize (modified in place)
        """
        # Ensure stop_conditions exists
        if "stop_conditions" not in request:
            request["stop_conditions"] = {}
        if "max_tokens" in request and "max_tokens" not in request["stop_conditions"]:
            request["stop_conditions"]["max_tokens"] = request.pop("max_tokens")

        # Ensure sampling_options exists
        if "sampling_options" not in request:
            request["sampling_options"] = {}
        if (
            "temperature" in request
            and "temperature" not in request["sampling_options"]
        ):
            request["sampling_options"]["temperature"] = request.pop("temperature")

    async def _initiate_shutdown(self, error: Exception):
        """Initiate graceful shutdown after fatal error"""
        logging.warning(f"Initiating graceful shutdown due to: {error}")

        try:
            if self.runtime:
                logging.info("Shutting down Dynamo runtime...")
                self.runtime.shutdown()

            if self.engine:
                logging.info("Shutting down TensorRT-LLM engine...")
                await self.engine.cleanup()
        except Exception as cleanup_error:
            logging.error(f"Error during graceful shutdown: {cleanup_error}")
        finally:
            logging.critical("Forcing process exit for restart")
            os._exit(1)

    async def generate_locally(
        self,
        request: dict,
        context: Context,
        embeddings: Optional[Union[torch.Tensor, dict]] = None,
        ep_disaggregated_params: Optional[DisaggregatedParams] = None,
    ) -> AsyncGenerator[dict, None]:
        started = await self._mark_request_started()
        if not started:
            yield {
                "finish_reason": {
                    "error": "Worker is temporarily rejecting new requests"
                },
                "token_ids": [],
            }
            return

        try:
            async for chunk in self._generate_locally_impl(
                request,
                context,
                embeddings,
                ep_disaggregated_params,
            ):
                yield chunk
        finally:
            await self._mark_request_finished()

    async def _generate_locally_impl(
        self,
        request: dict,
        context: Context,
        embeddings: Optional[Union[torch.Tensor, dict]] = None,
        ep_disaggregated_params: Optional[DisaggregatedParams] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Generate responses based on the disaggregation mode in the request.

        Args:
            request: The request dictionary containing generation parameters
            context: Context object for cancellation handling
            embeddings: Optional tensor or dict containing embeddings for multimodal processing
            ep_disaggregated_params: Optional DisaggregatedParams from encode worker (full EPD flow)
        """
        logging.debug(f"Request: {request}")

        # Normalize OpenAI format to TRT-LLM internal format
        self._normalize_request_format(request)

        # Setup disaggregated params based on PREFILL/DECODE mode
        (
            disaggregated_params,
            ep_disaggregated_params,
            epd_metadata,
        ) = self._setup_disaggregated_params_for_mode(request, ep_disaggregated_params)

        # Prepare input for generation (handles multimodal/text flows)
        processed_input = await self._prepare_input_for_generation(
            request, embeddings, ep_disaggregated_params, epd_metadata
        )

        # Check if there is an error in the publisher error queue
        publishers_error = (
            self.publisher.check_error_queue() if self.publisher else None
        )
        if publishers_error:
            raise publishers_error

        # For PREFILL mode, set max_tokens=1 (we only need to process context)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            request["stop_conditions"]["max_tokens"] = 1
            # disaggregated_params is already set above (lines 460-468)
            # Don't overwrite it here as it may contain multimodal_embedding_handles from encoder

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and disaggregated_params is None
        ):
            logging.error("DECODE: disaggregated_params is None but required!")
            logging.error(f"DECODE: Request keys: {list(request.keys())}")
            raise ValueError("Disaggregated params are required for decode mode")

        num_output_tokens_so_far = 0

        sampling_params = self._override_sampling_params(
            self.default_sampling_params, request
        )

        # Additional sampling params in output options
        output_options = request.get("output_options", {})
        if output_options:
            logprobs_value = output_options.get("logprobs")

            # Handle logprobs
            if logprobs_value is not None:
                if hasattr(sampling_params, "logprobs"):
                    setattr(
                        sampling_params, "logprobs", max(1, int(logprobs_value))
                    )  # If top_logprobs = 0, still want to see chosen token logprob

            # Handle prompt_logprobs
            prompt_logprobs_value = output_options.get("prompt_logprobs")
            if prompt_logprobs_value:
                if hasattr(sampling_params, "prompt_logprobs"):
                    setattr(
                        sampling_params, "prompt_logprobs", int(prompt_logprobs_value)
                    )

        max_tokens = request["stop_conditions"]["max_tokens"]
        if max_tokens:
            sampling_params.max_tokens = max_tokens

        ignore_eos = request["stop_conditions"].get("ignore_eos")
        if ignore_eos:
            sampling_params.ignore_eos = ignore_eos

        min_tokens = request["stop_conditions"].get("min_tokens")
        if min_tokens:
            sampling_params.min_tokens = min_tokens

        stop_token_ids = request["stop_conditions"].get("stop_token_ids_hidden")
        if stop_token_ids:
            existing = sampling_params.stop_token_ids or []
            sampling_params.stop_token_ids = list(set(existing).union(stop_token_ids))

        # TODO: Instead of True, we should use streaming from the request.
        # However, currently dynamo run does not send streaming in the request.
        streaming = (
            False if self.disaggregation_mode == DisaggregationMode.PREFILL else True
        )

        request_id = request.get("id") or request.get("request_id", "unknown-id")

        # Optional test-only logits processing (enable with DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR=1)
        if os.getenv("DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR") == "1":
            processors = [HelloWorldLogitsProcessor(self.engine.llm.tokenizer)]
            adapters = create_trtllm_adapters(processors)
            sampling_params.logits_processor = adapters

        prefill_result = request.get("prefill_result")
        prefill_prompt_tokens_details = (
            prefill_result.get("prompt_tokens_details") if prefill_result else None
        )

        # Build trace headers for distributed tracing
        trace_headers = build_trace_headers(context)

        # Extract dp_rank from request's routing hints for attention DP routing
        routing = request.get("routing", {})
        dp_rank = routing.get("dp_rank") if routing else None
        scheduling_params = None
        if dp_rank is not None:
            scheduling_params = SchedulingParams(
                attention_dp_rank=dp_rank,
                attention_dp_relax=False,  # Strict routing - use the rank dynamo router selected
            )
            logging.debug(
                f"Using dynamo router dp_rank={dp_rank} for TRTLLM attention DP scheduling"
            )

        try:
            # NEW: Updated engine call to include multimodal data
            generation_result = self.engine.llm.generate_async(
                inputs=processed_input,  # Use the correctly extracted inputs
                sampling_params=sampling_params,
                disaggregated_params=disaggregated_params,
                streaming=streaming,
                trace_headers=trace_headers,
                scheduling_params=scheduling_params,
            )

            # Monitor for cancellation triggers and cancel by calling generation_result.abort()
            async with self._cancellation_monitor(generation_result, context):
                async for res in generation_result:
                    # TRTLLM engine needs to start generating tokens first before stats
                    # can be retrieved.
                    if self.first_generation and self.publisher:
                        self.publisher.start()
                        self.first_generation = False

                    # If we are not done generating, but there are no outputs, return an error
                    if not res.outputs and not res.finished:
                        yield {"finish_reason": "error", "token_ids": []}
                        break

                    output = res.outputs[0]
                    # The engine returns all tokens generated so far. We must calculate the new
                    # tokens generated in this iteration to create the "delta".
                    next_total_toks = len(output.token_ids)

                    out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}

                    # Extract logprobs from the output
                    log_probs, top_logprobs = self._extract_logprobs(
                        output, num_output_tokens_so_far
                    )
                    if log_probs:
                        out["log_probs"] = log_probs
                    if top_logprobs:
                        out["top_logprobs"] = top_logprobs

                    if output.finish_reason:
                        out["finish_reason"] = output.finish_reason
                    if output.stop_reason:
                        out["stop_reason"] = output.stop_reason
                    if self.disaggregation_mode == DisaggregationMode.PREFILL:
                        # Return the disaggregated params only when operating in prefill mode.
                        params_dict = self._encode_and_pack_disaggregated_params(
                            output, disaggregated_params, request, res, processed_input
                        )
                        if params_dict is not None:
                            out["disaggregated_params"] = params_dict

                    if out.get("finish_reason"):
                        num_input_tokens = len(request.get("token_ids", []))

                        prompt_tokens_details = None
                        if prefill_prompt_tokens_details:
                            prompt_tokens_details = prefill_prompt_tokens_details
                        else:
                            if output.request_perf_metrics is not None:
                                kv_cache_metrics = (
                                    output.request_perf_metrics.kv_cache_metrics
                                )
                                cached_tokens = min(
                                    num_input_tokens,
                                    kv_cache_metrics.num_reused_blocks
                                    * self.kv_block_size,
                                )
                                if cached_tokens > 0:
                                    prompt_tokens_details = {
                                        "cached_tokens": int(cached_tokens),
                                    }

                        out["completion_usage"] = {
                            "prompt_tokens": int(num_input_tokens),
                            "completion_tokens": int(next_total_toks),
                            "total_tokens": int(num_input_tokens + next_total_toks),
                            "prompt_tokens_details": prompt_tokens_details,
                        }

                    if res.finished and not out.get("finish_reason"):
                        out["finish_reason"] = "unknown"
                        logging.warning(
                            "Request finished with no finish reason set - this indicates a possible bug"
                        )

                    # Log metrics to TensorRT-LLM MetricsCollector when request finishes
                    # NOTE: TRT-LLM 1.3.0rc5 (PR #11243) renamed log_metrics_dict → log_request_metrics_dict
                    if (
                        res.finished
                        and self.metrics_collector
                        and hasattr(res, "metrics_dict")
                    ):
                        try:
                            if hasattr(
                                self.metrics_collector,
                                "log_request_metrics_dict",
                            ):
                                self.metrics_collector.log_request_metrics_dict(
                                    res.metrics_dict
                                )
                            else:
                                self.metrics_collector.log_metrics_dict(
                                    res.metrics_dict
                                )
                        except Exception as e:
                            logging.warning(f"Failed to log TensorRT-LLM metrics: {e}")

                    # Yield the chunk to the client and update the token count for the next iteration.
                    yield out
                    num_output_tokens_so_far = next_total_toks

        # 1. Client cancellation - don't shutdown
        except asyncio.CancelledError:
            logging.debug(f"Request {request_id}: Client cancelled")
            # _cancellation_monitor already called abort_request
            return  # Just stop, no error response

        # 2. Per-request errors - send to client, don't shutdown
        except RequestError as e:
            error_msg = str(e)
            logging.warning(f"Request {request_id} error: {error_msg}")
            yield {
                "finish_reason": {"error": error_msg},
                "token_ids": [],
            }

        # 3. ALL OTHER ERRORS - graceful shutdown
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logging.error(
                f"Fatal {error_type} in request {request_id}: {error_msg}",
                exc_info=True,
            )

            # Try to send error to client before shutdown
            try:
                yield {
                    "finish_reason": {"error": error_msg},
                    "token_ids": [],
                }
            except Exception:
                pass  # Best effort

            # Initiate graceful shutdown
            await self._initiate_shutdown(e)

    @staticmethod
    def _override_sampling_params(sampling_params, request: dict) -> SamplingParams:
        overrides = {
            key: value
            for key, value in request["sampling_options"].items()
            if value is not None
        }

        # Convert guided_decoding dict (from Rust serialization) to GuidedDecodingParams.
        # Explicit field mapping avoids breakage if either side adds fields the other
        # doesn't know about (e.g. Rust's "backend"/"choice" vs TRT-LLM's fields).
        guided_decoding = overrides.pop("guided_decoding", None)
        if guided_decoding is not None and isinstance(guided_decoding, dict):
            # TRT-LLM's GuidedDecodingParams doesn't have a "choice" field.
            # Convert choice list to a regex pattern: (choice1|choice2|...)
            # This matches the approach used by vLLM's outlines backend.
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

        # NOTE: using `dataclasses.replace` has several benefits over a `setattr` based approach:
        # 1. it catches unsupported fields / attributes.
        # 2. it executes the class's `__post_init__`, which may contain helpful validation logic.
        return dataclasses.replace(sampling_params, **overrides)
