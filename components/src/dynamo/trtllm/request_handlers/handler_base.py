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
import logging
import os
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union

import torch
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi.disagg_utils import get_global_disagg_request_id
from tensorrt_llm.llmapi.llm import SamplingParams
from tensorrt_llm.sampling_params import GuidedDecodingParams
from tensorrt_llm.scheduling_params import SchedulingParams

from dynamo._core import Client, Context
from dynamo.common.utils.otel_tracing import build_trace_headers
from dynamo.llm.exceptions import EngineShutdown
from dynamo.logits_processing.examples import HelloWorldLogitsProcessor
from dynamo.nixl_connect import Connector
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.engine import TensorRTLLMEngine
from dynamo.trtllm.logits_processing.adapter import create_trtllm_adapters
from dynamo.trtllm.metrics import AdditionalMetricsCollector
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor
from dynamo.trtllm.publisher import Publisher
from dynamo.trtllm.request_handlers.base_generative_handler import BaseGenerativeHandler
from dynamo.trtllm.utils.disagg_utils import (
    DisaggregatedParams,
    DisaggregatedParamsCodec,
)

if TYPE_CHECKING:
    # tensorrt_llm may use a different version that doesn't have MetricsCollector,
    # so guard this import inside TYPE_CHECKING to avoid runtime import errors.
    from tensorrt_llm.metrics import MetricsCollector

configure_dynamo_logging()

logger = logging.getLogger(__name__)


class TRTLLMEngineQuiesceController:
    """Adapts TRT-LLM sleep/wake to the standard quiesce controller interface.

    Two memory domains: KV cache via TRT-LLM collective_rpc, weights via GMS.
    """

    def __init__(self, engine: TensorRTLLMEngine):
        self._engine = engine
        self._is_quiesced = False

    @property
    def is_quiesced(self) -> bool:
        return self._is_quiesced

    async def quiesce(self, tags: list[str] | None = None) -> bool:
        if self._is_quiesced:
            return False
        tags = tags or ["kv_cache", "weights"]
        if "kv_cache" in tags:
            self._collective_rpc("sleep", ["kv_cache"])
        if "weights" in tags:
            self._release_gms_weights()
        self._is_quiesced = True
        return True

    async def resume(self, tags: list[str] | None = None) -> bool:
        if not self._is_quiesced:
            return False
        tags = tags or ["kv_cache", "weights"]
        if "weights" in tags:
            self._restore_gms_weights()
        if "kv_cache" in tags:
            self._collective_rpc("wakeup", ["kv_cache"])
        return True

    def mark_resumed(self) -> None:
        self._is_quiesced = False

    def _collective_rpc(self, method: str, rpc_tags: list[str]) -> None:
        """Call TRT-LLM collective_rpc for KV cache sleep/wake."""
        rpc = getattr(self._engine.llm, "_collective_rpc", None)
        if rpc is None:
            logger.warning(
                "TRT-LLM does not expose _collective_rpc; skipping %s", method
            )
            return

        rpc(method, args=(rpc_tags,), kwargs={}, non_block=False)

    @staticmethod
    def _release_gms_weights() -> None:
        """Release GMS-managed weight memory."""
        try:
            from gpu_memory_service.client.torch.allocator import (
                get_gms_client_memory_manager,
            )
        except ImportError:
            return
        manager = get_gms_client_memory_manager("weights")
        if manager is None:
            return
        manager.unmap_all_vas()
        manager.abort()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    @staticmethod
    def _restore_gms_weights() -> None:
        """Restore GMS-managed weight memory."""
        try:
            from gpu_memory_service.client.torch.allocator import (
                get_gms_client_memory_manager,
            )
            from gpu_memory_service.integrations.trtllm.model_loader import (
                get_gms_lock_mode,
            )
        except ImportError:
            return
        manager = get_gms_client_memory_manager("weights")
        if manager is None or not manager.is_unmapped:
            return
        manager.connect(get_gms_lock_mode())
        manager.remap_all_vas()


class _Abortable(Protocol):
    """Structural type for objects that support abort(). Satisfied by both
    GenerationResult and _DeferredAbort."""

    def abort(self) -> None:
        ...


class _DeferredAbort:
    """Wraps GenerationResult.abort() to defer until first token in disagg decode.

    When abort() is called before the first generation result, spawns a
    background asyncio.Task that reads from GenerationResult.aqueue (TRT-LLM's
    internal asyncio.Queue, decoupled from Dynamo RPC transport) until the
    first result arrives, then calls the real abort().
    """

    def __init__(self, generation_result: GenerationResult):
        self._generation_result = generation_result
        self._first_token_received = False

    def signal_first_token(self) -> None:
        """Called by generate_locally() when first generation result is yielded."""
        self._first_token_received = True

    def abort(self) -> None:
        """Abort immediately if first token received, otherwise defer."""
        if self._first_token_received:
            self._generation_result.abort()
            logging.debug("Deferred abort: first token already received, aborting now")
        else:
            logging.debug(
                "Deferred abort: first token not received, spawning background task"
            )
            asyncio.create_task(self._wait_and_abort())

    async def _wait_and_abort(self) -> None:
        """Background task: read from GenerationResult until first token, then abort."""
        try:
            async for _ in self._generation_result:
                break  # First result = KV transfer complete
        except Exception:
            pass
        self._generation_result.abort()
        logging.debug("Deferred abort: background task completed, abort fired")


@dataclass
class RequestHandlerConfig:
    """
    Configuration for the request handler
    """

    engine: TensorRTLLMEngine
    default_sampling_params: SamplingParams
    publisher: Optional[Publisher]
    disaggregation_mode: DisaggregationMode
    encode_client: Optional[Client] = None
    multimodal_processor: Optional[
        MultimodalRequestProcessor
    ] = None  # for multimodal support
    connector: Optional[Connector] = None
    runtime: Optional[
        DistributedRuntime
    ] = None  # DistributedRuntime reference for graceful shutdown
    metrics_collector: Optional["MetricsCollector"] = None
    kv_block_size: int = 32
    shutdown_event: Optional[asyncio.Event] = None
    generate_endpoint: Optional[Any] = None
    encoder_cache_capacity_gb: float = 0  # Encoder cache capacity in GB
    additional_metrics: Optional["AdditionalMetricsCollector"] = None
    max_seq_len: Optional[int] = None
    disagg_machine_id: int = 0  # 10-bit machine_id for snowflake disagg_request_id


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
        self.first_generation = True
        self.connector = config.connector
        # Store runtime reference for graceful shutdown
        self.runtime = config.runtime
        self.kv_block_size: int = config.kv_block_size
        self.shutdown_event = config.shutdown_event
        self.generate_endpoint = config.generate_endpoint
        self.additional_metrics = config.additional_metrics
        self.max_seq_len = config.max_seq_len
        self.disagg_machine_id = config.disagg_machine_id
        # Sleep/wake state
        self._quiesce_lock = asyncio.Lock()
        self._inflight_lock = asyncio.Lock()
        self._inflight_requests = 0
        self._no_inflight_requests = asyncio.Event()
        self._no_inflight_requests.set()
        self._quiesce_controller = TRTLLMEngineQuiesceController(config.engine)
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

    # ------------------------------------------------------------------
    # In-flight request tracking (used by sleep/wake)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Sleep / wake public API (delegates to TRTLLMEngineQuiesceController)
    # ------------------------------------------------------------------

    async def release_memory_occupation(self, body: dict) -> dict:
        """Release GPU memory: unregister endpoint, drain requests, quiesce engine."""
        body = body or {}
        tags = body.get("tags")

        async with self._quiesce_lock:
            if self._quiesce_controller.is_quiesced:
                return {"status": "ok", "message": "Memory already released"}

            try:
                await self._set_reject_new_requests(True)

                if self.generate_endpoint is not None:
                    await self.generate_endpoint.unregister_endpoint_instance()

                timeout_s = float(body.get("timeout_s", 30.0))
                await self._wait_for_inflight_requests(timeout_s)
                await self._quiesce_controller.quiesce(tags)

                return {"status": "ok", "message": "Memory released"}
            except Exception as exc:
                logger.error("release_memory_occupation failed: %s", exc)
                # Rollback: TRT-LLM has no pause_generation(), so we
                # manually unregistered the endpoint and set reject flag
                # above. Restore both on failure.
                if self.generate_endpoint is not None:
                    await self.generate_endpoint.register_endpoint_instance()
                await self._set_reject_new_requests(False)
                return {"status": "error", "message": str(exc)}

    async def resume_memory_occupation(self, body: dict) -> dict:
        """Restore GPU memory: resume engine, re-register endpoint."""
        body = body or {}
        tags = body.get("tags")

        async with self._quiesce_lock:
            if not self._quiesce_controller.is_quiesced:
                return {"status": "ok", "message": "Memory already resumed"}

            try:
                await self._quiesce_controller.resume(tags)

                if self.generate_endpoint is not None:
                    await self.generate_endpoint.register_endpoint_instance()

                await self._set_reject_new_requests(False)
                self._quiesce_controller.mark_resumed()
                return {"status": "ok", "message": "Memory resumed"}
            except Exception as exc:
                logger.error("resume_memory_occupation failed: %s", exc)
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
                        "rank": (
                            logprob_info.rank if hasattr(logprob_info, "rank") else 0
                        ),
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
        self,
        generation_result: _Abortable,
        context: Context,
    ):
        """
        Background task to trigger cancellation if request is cancelled or shutdown
        event is set.

        In disaggregated decode mode, generation_result may be a _DeferredAbort
        wrapper that defers abort() until the first token is received (KV
        transfer complete).

        Raise EngineShutdown if shutdown event is triggered.
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

            generation_result.abort()
            logging.debug(f"Aborted Request ID: {context.id()}")

            # Clean up any remaining background task
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Raise EngineShutdown if cancellation is due to shutdown event triggered
            if shutdown_task in done:
                raise EngineShutdown("Engine was shut down during generation.")

        except asyncio.CancelledError:
            # Task was cancelled, which is expected when generation completes normally
            pass

    @asynccontextmanager
    async def _cancellation_monitor(
        self,
        generation_result: _Abortable,
        context: Context,
    ) -> AsyncGenerator[asyncio.Task, None]:
        """
        Monitor for cancellation triggers and cancel by calling
        generation_result.abort().

        In disaggregated decode mode, generation_result may be a _DeferredAbort
        wrapper that defers abort() until the first token.

        Raise EngineShutdown if shutdown event is triggered.

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

        # Deserialize first_gen_log_probs from transport format back to
        # TRT-LLM's internal {token_id: Logprob} dict format.
        DisaggregatedParamsCodec.deserialize_first_gen_log_probs(params_dict)

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

        # Serialize first_gen_log_probs for the Rust transport layer.
        DisaggregatedParamsCodec.serialize_first_gen_log_probs(params_dict)

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
                    request_type="context_only",
                    disagg_request_id=get_global_disagg_request_id(
                        self.disagg_machine_id
                    ),
                )

            # Ensure disagg_request_id is set even when using
            # ep_disaggregated_params, so the PYTHON transceiver can track
            # requests across prefill/decode workers.
            if disaggregated_params.disagg_request_id is None:
                disaggregated_params.disagg_request_id = get_global_disagg_request_id(
                    self.disagg_machine_id
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

        if self.multimodal_processor is None and self._request_has_multimodal(request):
            raise RuntimeError(
                "Multimodal input received but worker started without --modality multimodal. "
                "Restart the worker with --modality multimodal or remove image_url content."
            )

        # PREFILL/ENCODE/AGGREGATED: Process multimodal content if available
        if self.multimodal_processor:
            mm_result = await self.multimodal_processor.process_openai_request(
                request, embeddings, ep_disaggregated_params
            )
            if mm_result:
                return mm_result

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

    def _request_has_multimodal(self, request: dict) -> bool:
        if request.get("multi_modal_data"):
            return True

        extra_args = request.get("extra_args") or {}
        messages = extra_args.get("messages") or request.get("messages") or []
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

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
        """Track in-flight count, reject during sleep, then delegate to implementation."""
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
                request, context, embeddings, ep_disaggregated_params
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

        # Additional metrics: request type detection
        metrics_collector = self.additional_metrics

        if metrics_collector:
            try:
                # Detect request types for metrics
                sampling_options = request.get("sampling_options", {})
                guided = sampling_options.get("guided_decoding")
                if guided and isinstance(guided, dict):
                    has_structured_guidance = any(
                        guided.get(k) is not None
                        for k in (
                            "json",
                            "regex",
                            "grammar",
                            "json_object",
                            "structural_tag",
                        )
                    ) or bool(guided.get("choice"))
                    if has_structured_guidance:
                        metrics_collector.record_request_type_structured_output()
                if (
                    request.get("multi_modal_data")
                    or embeddings is not None
                    or request.get("_epd_processed_prompt") is not None
                ):
                    metrics_collector.record_request_type_image()
            except Exception as e:
                logging.warning("Additional metrics (request type): %s", e)

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
        if max_tokens is not None:
            sampling_params.max_tokens = max_tokens
        elif self.max_seq_len is not None:
            if self.multimodal_processor and processed_input is not None:
                logging.debug(
                    "Skipping dynamic max_tokens default for multimodal request..."
                )
            else:
                token_ids = request.get("token_ids", [])
                input_length = len(token_ids)
                dynamic_default = max(1, self.max_seq_len - input_length)
                sampling_params.max_tokens = dynamic_default

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

            # In disagg decode mode, wrap abort() to defer until first token
            # (KV transfer complete).
            abort_guard = (
                _DeferredAbort(generation_result)
                if self.disaggregation_mode == DisaggregationMode.DECODE
                else None
            )

            # Monitor for cancellation triggers and cancel by calling abort()
            async with self._cancellation_monitor(
                abort_guard or generation_result, context
            ):
                async for res in generation_result:
                    # Signal first token to deferred abort guard
                    if abort_guard is not None:
                        abort_guard.signal_first_token()

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

                    # Record additional metrics on request finish
                    if res.finished and metrics_collector and out.get("finish_reason"):
                        try:
                            # KV transfer metrics from request_perf_metrics
                            if output.request_perf_metrics is not None:
                                # Record KV transfer latency/bytes/speed from timing_metrics
                                tm = output.request_perf_metrics.timing_metrics
                                if tm is not None:
                                    recorded = (
                                        metrics_collector.record_kv_transfer_perf(tm)
                                    )
                                    # Only count success if a transfer actually occurred
                                    if (
                                        recorded
                                        and self.disaggregation_mode
                                        == DisaggregationMode.PREFILL
                                    ):
                                        metrics_collector.record_kv_transfer_success()
                        except Exception as e:
                            logging.warning(
                                "Additional metrics (request finish): %s", e
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
            try:
                if metrics_collector:
                    metrics_collector.record_request_abort()
            except Exception as e:
                logging.debug("Additional metrics (request abort): %s", e)
            return  # Just stop, no error response

        # 2. Per-request errors - send to client, don't shutdown
        except RequestError as e:
            error_msg = str(e)
            logging.warning(f"Request {request_id} error: {error_msg}")
            yield {
                "finish_reason": {"error": error_msg},
                "token_ids": [],
            }

        # 3. EngineShutdown - let it propagate to the Rust bridge
        except EngineShutdown:
            raise

        # 4. ALL OTHER ERRORS - graceful shutdown
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
