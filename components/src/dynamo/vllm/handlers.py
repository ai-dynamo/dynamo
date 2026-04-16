# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import binascii
import io
import logging
import os
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Final, Generic, Optional, TypeVar

import torch
from vllm.config import VllmConfig
from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine.exceptions import EngineDeadError

from dynamo._core import Context
from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal.audio_loader import AudioLoader
from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingReceiver,
    NixlReadEmbeddingReceiver,
    NixlWriteEmbeddingReceiver,
)
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.multimodal.video_loader import VideoLoader
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.common.utils.input_params import InputParamManager
from dynamo.common.utils.otel_tracing import build_trace_headers
from dynamo.common.utils.time_section import time_and_log_code_section
from dynamo.llm import (
    KvEventPublisher,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    lora_name_to_id,
    register_model,
    unregister_model,
)
from dynamo.llm.exceptions import EngineShutdown
from dynamo.runtime import Client
from dynamo.runtime.logging import configure_dynamo_logging

from .args import Config
from .constants import DisaggregationMode, EmbeddingTransferMode
from .engine_monitor import VllmEngineMonitor
from .multimodal_utils.hash_utils import compute_mm_uuids_from_images
from .multimodal_utils.model import construct_qwen_decode_mm_data, is_qwen_vl_model
from .multimodal_utils.models.qwen import (
    build_qwen_embedding_params,
    load_qwen_grid_params,
)
from .multimodal_utils.prefill_worker_utils import MultiModalEmbeddingLoader

# Multimodal data dictionary keys
IMAGE_URL_KEY: Final = "image_url"
VIDEO_URL_KEY: Final = "video_url"
AUDIO_URL_KEY: Final = "audio_url"
URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class VllmEngineQuiesceController:
    def __init__(self, engine_client: Any):
        self._engine_client = engine_client
        self._is_quiesced = False

    @property
    def is_quiesced(self) -> bool:
        return self._is_quiesced

    async def quiesce(self, *args: object) -> bool:
        if self._is_quiesced:
            return False

        level = args[0] if args else None
        await self._engine_client.pause_generation()
        if level is None:
            await self._engine_client.sleep()
        else:
            await self._engine_client.sleep(level)
        self._is_quiesced = True
        return True

    async def resume(self, tags: list[str] | None = None) -> bool:
        if not self._is_quiesced:
            return False

        if tags is None:
            await self._engine_client.wake_up()
        else:
            await self._engine_client.wake_up(tags)
        await self._engine_client.resume_generation()
        return True

    def mark_resumed(self) -> None:
        self._is_quiesced = False


@dataclass(frozen=True)
class LoRAInfo:
    """Metadata for a loaded LoRA adapter."""

    id: int
    path: str


def _compute_mm_uuids(
    multi_modal_data: Dict[str, Any] | None
) -> Dict[str, list[str]] | None:
    """
    Compute multi_modal_uuids from multi_modal_data.

    Each image gets a SHA256 hex digest as its UUID, ensuring consistent
    hashing across the MM Router, vLLM handler, and Rust KV publisher.
    """
    if not multi_modal_data or "image" not in multi_modal_data:
        return None
    images = multi_modal_data["image"]
    # [gluo FIXME] Dict being returned when the mm data has been processed,
    # in this case, we skip computing mm_uuids for now until we better understand
    # what info should be hash on.
    if isinstance(images, dict):
        return None
    if not isinstance(images, list):
        images = [images]
    if not images:
        return None
    uuids = compute_mm_uuids_from_images(images)
    return {"image": uuids}


# LoRAManager singleton - initialized lazily when DYN_LORA_ENABLED is set
# None = not yet initialized, False = disabled/failed, LoRAManager = initialized
_lora_manager = None


def get_lora_manager():
    """Get the LoRAManager singleton, initializing it on first call if enabled."""
    global _lora_manager

    if _lora_manager is not None:
        return _lora_manager

    if os.environ.get("DYN_LORA_ENABLED", "").lower() in ("true", "1", "yes"):
        try:
            from dynamo.common.lora import LoRAManager

            _lora_manager = LoRAManager()
            logger.info("LoRAManager initialized successfully")
            return _lora_manager
        except Exception as e:
            logger.warning(
                f"Failed to initialize LoRAManager: {e}. URI-based LoRA loading will be disabled."
            )

    return None


def build_sampling_params(
    request: Dict[str, Any],
    default_sampling_params: Dict[str, Any],
    model_max_len: int | None = None,
) -> SamplingParams:
    """
    Build SamplingParams from a PreprocessedRequest (internal protocol format).

    Args:
        request: The PreprocessedRequest dict with 'sampling_options', 'stop_conditions',
                 and 'output_options'
        default_sampling_params: Default sampling parameters to initialize with

    Returns:
        SamplingParams configured from the request
    """
    sampling_params = SamplingParams(**default_sampling_params)
    sampling_params.detokenize = False

    # Handle guided_decoding - convert to StructuredOutputsParams
    sampling_options = request.get("sampling_options", {})
    guided_decoding = sampling_options.get("guided_decoding")
    if guided_decoding is not None and isinstance(guided_decoding, dict):
        sampling_params.structured_outputs = StructuredOutputsParams(
            json=guided_decoding.get("json"),
            regex=guided_decoding.get("regex"),
            choice=guided_decoding.get("choice"),
            grammar=guided_decoding.get("grammar"),
            whitespace_pattern=guided_decoding.get("whitespace_pattern"),
        )

    # Apply remaining sampling_options
    for key, value in sampling_options.items():
        # Skip guided_decoding - already handled above
        if key == "guided_decoding":
            continue
        if value is not None and hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    # Apply stop_conditions
    for key, value in request.get("stop_conditions", {}).items():
        if value is not None and hasattr(sampling_params, key):
            # Do not add stop key to sampling params - dynamo handles stop conditions directly
            if key == "stop":
                continue
            setattr(sampling_params, key, value)
        if (
            key == "stop_token_ids_hidden"
            and value is not None
            and hasattr(sampling_params, "stop_token_ids")
        ):
            existing = sampling_params.stop_token_ids or []
            sampling_params.stop_token_ids = list(set(existing).union(value))

    # Apply output_options (logprobs, prompt_logprobs, etc.)
    output_options = request.get("output_options", {})
    if output_options:
        # Handle logprobs - vLLM expects this as an integer or None
        logprobs_value = output_options.get("logprobs")
        if logprobs_value is not None and logprobs_value != "":
            try:
                parsed_logprobs = int(logprobs_value)
                if parsed_logprobs < 0:
                    logger.warning(
                        f"Invalid logprobs value: {logprobs_value} (must be non-negative), ignoring"
                    )
                else:
                    sampling_params.logprobs = parsed_logprobs
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid logprobs value: {logprobs_value} (must be integer), ignoring"
                )

        # Handle prompt_logprobs - vLLM expects this as an integer or None
        prompt_logprobs_value = output_options.get("prompt_logprobs")
        if prompt_logprobs_value is not None and prompt_logprobs_value != "":
            try:
                parsed_prompt_logprobs = int(prompt_logprobs_value)
                if parsed_prompt_logprobs < 0:
                    logger.warning(
                        f"Invalid prompt_logprobs value: {prompt_logprobs_value} (must be non-negative), ignoring"
                    )
                else:
                    sampling_params.prompt_logprobs = parsed_prompt_logprobs
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid prompt_logprobs value: {prompt_logprobs_value} (must be integer), ignoring"
                )

    # If max_tokens wasn't provided (None or missing), compute a dynamic default
    provided_max_tokens = request.get("stop_conditions", {}).get("max_tokens", None)
    token_ids = request.get("token_ids", [])
    input_length = len(token_ids)
    if model_max_len is not None and (provided_max_tokens is None):
        # Ensure at least 1 token generation by default when possible
        dynamic_default = max(1, model_max_len - input_length)
        sampling_params.max_tokens = dynamic_default

    return sampling_params


def build_sampling_params_openai(
    request: Dict[str, Any],
    default_sampling_params: Dict[str, Any],
) -> SamplingParams:
    """
    Build SamplingParams from an OpenAI-compatible request format.

    Args:
        request: The OpenAI-style request dict with parameters like temperature, max_tokens, etc.
        default_sampling_params: Default sampling parameters to initialize with

    Returns:
        SamplingParams configured from the request
    """
    sampling_params = SamplingParams(**default_sampling_params)
    sampling_params.detokenize = True

    # Map common OpenAI parameters to SamplingParams
    openai_mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "seed": "seed",
        "top_k": "top_k",
        "repetition_penalty": "repetition_penalty",
        "min_p": "min_p",
        "length_penalty": "length_penalty",
        "use_beam_search": "use_beam_search",
    }

    for req_key, param_key in openai_mapping.items():
        if req_key in request and request[req_key] is not None:
            if hasattr(sampling_params, param_key):
                setattr(sampling_params, param_key, request[req_key])

    # Handle max_tokens
    if "max_tokens" in request and request["max_tokens"] is not None:
        sampling_params.max_tokens = request["max_tokens"]

    # Handle stop sequences
    if "stop" in request and request["stop"] is not None:
        sampling_params.stop = request["stop"]

    # Handle ignore_eos (custom extension)
    if "ignore_eos" in request and request["ignore_eos"] is not None:
        sampling_params.ignore_eos = request["ignore_eos"]

    # Handle min_tokens (custom extension)
    if "min_tokens" in request and request["min_tokens"] is not None:
        sampling_params.min_tokens = request["min_tokens"]

    return sampling_params


def get_dp_range_for_worker(vllm_config: VllmConfig) -> tuple[int, int]:
    """
    Get the global DP rank range that this worker is responsible for based on vLLM config.
    Note that the 'vllm_config' is normalized so the load balancing flags are set properly.
    The return value is in the format of (start_dp_rank, managed_dp_size)."""
    if vllm_config.parallel_config.data_parallel_external_lb:
        # external load balancing, each worker is responsible for exactly 1 rank
        return (vllm_config.parallel_config.data_parallel_rank, 1)
    elif vllm_config.parallel_config.data_parallel_hybrid_lb:
        # hybrid load balancing, each worker is responsible for a subset of local ranks
        return (
            vllm_config.parallel_config.data_parallel_rank,
            vllm_config.parallel_config.data_parallel_size_local,
        )
    else:
        # internal load balancing, the worker is responsible for all DP ranks
        logger.warning(
            "vLLM selects internal DP load balancing. If you are launching multiple workers for DP deployment,"
            " hybrid or external load balancing is recommended."
        )
        return (
            vllm_config.parallel_config.data_parallel_rank,
            vllm_config.parallel_config.data_parallel_size,
        )


RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class BaseWorkerHandler(ABC, Generic[RequestT, ResponseT]):
    """
    Request handler for the generate and clear_kv_blocks endpoints.
    """

    _benchmark_results: Optional[dict] = None

    def __init__(
        self,
        runtime,
        config: Config,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
        encode_worker_client: Optional[Client] = None,
    ):
        self.runtime = runtime
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params
        self.kv_publishers: list[KvEventPublisher] | None = None
        self.fpm_relays: list | None = None
        self.generate_endpoint = generate_endpoint
        self.config = config
        self.engine_monitor = VllmEngineMonitor(runtime, engine, shutdown_event)
        self.temp_dirs: list[tempfile.TemporaryDirectory] = []
        self.model_max_len = model_max_len
        self.enable_multimodal = enable_multimodal
        # LoRA tracking: name -> LoRAInfo(id, path)
        self.loaded_loras: dict[str, LoRAInfo] = {}
        # Per-LoRA locks to prevent concurrent load operations for the same LoRA
        self._lora_load_locks: dict[str, asyncio.Lock] = {}
        # Guard lock-map access in case handlers are invoked from multiple threads.
        self._lora_load_locks_guard = threading.Lock()

        self.image_loader = ImageLoader(
            enable_frontend_decoding=enable_frontend_decoding
        )
        self.audio_loader = AudioLoader(
            enable_frontend_decoding=enable_frontend_decoding
        )
        self.video_loader = VideoLoader(
            enable_frontend_decoding=enable_frontend_decoding
        )
        self.embedding_loader = self.init_embedding_loader(config, encode_worker_client)

        self.use_vllm_tokenizer = use_vllm_tokenizer

        self.dp_range = get_dp_range_for_worker(self.engine_client.vllm_config)
        self._quiesce_controller = VllmEngineQuiesceController(self.engine_client)
        self._quiesce_lock = asyncio.Lock()

        # Initialize InputParamManager for text-in-text-out mode
        tokenizer = None
        if use_vllm_tokenizer and hasattr(engine, "tokenizer"):
            tokenizer = engine.tokenizer
        self.input_param_manager = InputParamManager(tokenizer)

        # Store shutdown event for graceful shutdown monitoring
        self.shutdown_event = shutdown_event

    def init_embedding_loader(
        self, config: Config, encode_worker_client: Optional[Client] = None
    ) -> Optional[MultiModalEmbeddingLoader]:
        """Initialize the embedding loader with the given encode worker client."""
        # Without encode worker, the embedding will be generated internally by vLLM.
        if encode_worker_client is None:
            return None
        logger.warning(
            "Separate multimodal encode-worker routing only applies to image_url "
            "inputs. video_url inputs are not sent to the encode worker and will "
            "be processed on the prefill/PD worker instead."
        )
        # Embedding loader consist of two main components:
        # 1) An remote encode worker client and matching embedding receiver,
        #    which can request remote encode and handle the transfer of embeddings
        #    from the encode worker to this prefill worker.
        # 2) A local embedding cache manager, which can store previously fetched embeddings
        #    and used to determine whether remote encode is necessary for a given mm data.
        self.encode_worker_client = encode_worker_client
        if config.embedding_transfer_mode == EmbeddingTransferMode.LOCAL:
            self.embedding_receiver = LocalEmbeddingReceiver()  # type: ignore
        elif config.embedding_transfer_mode == EmbeddingTransferMode.NIXL_WRITE:
            self.embedding_receiver = NixlWriteEmbeddingReceiver()  # type: ignore
        elif config.embedding_transfer_mode == EmbeddingTransferMode.NIXL_READ:
            # [gluo FIXME] can't use pre-registered tensor as NIXL requires descriptors
            # to be at matching size, need to overwrite nixl connect library
            self.embedding_receiver = NixlReadEmbeddingReceiver(max_items=0)  # type: ignore
        else:
            raise ValueError(
                f"Invalid embedding transfer mode: {config.embedding_transfer_mode}"
            )
        # [gluo FIXME/NOTE] This embedding cache manager is purely used for caching embedding
        # results from encode worker, but 'config.multimodal_embedding_cache_capacity_gb' is
        # also used to configure the DynamoMultimodalEmbeddingCacheConnector within the vLLM.
        # This results in duplication of memory and ideally we should have single cache manager
        # which can be used by vLLM internal and here. Then we can explore asynchrous embedding
        # transfer as we can process and block until the embedding is actually used within vLLM.
        self.embedding_cache_manager: MultimodalEmbeddingCacheManager | None = None
        if config.multimodal_embedding_cache_capacity_gb > 0:
            capacity_bytes = int(
                config.multimodal_embedding_cache_capacity_gb * 1024**3
            )
            self.embedding_cache_manager = MultimodalEmbeddingCacheManager(
                capacity_bytes
            )
        return MultiModalEmbeddingLoader(
            encode_worker_client=self.encode_worker_client,  # type: ignore
            receiver=self.embedding_receiver,
            embedding_cache_manager=self.embedding_cache_manager,
        )

    async def sleep(self, body: dict) -> dict:
        """Sleep the engine to release GPU memory and unregister from discovery.

        Args:
            body: Dict with optional 'level' key (1=weights only, 2=weights+buffers, 3=everything)

        Order of operations:
        1. Unregister from discovery - stop accepting new requests
        2. Abort and drain in-flight requests
        3. Sleep engine - safe now that GPU is quiesced
        """
        body = body or {}
        level = body.get("level", 1)
        async with self._quiesce_lock:
            if self._quiesce_controller.is_quiesced:
                return {
                    "status": "ok",
                    "message": "Engine already sleeping",
                }

            try:
                # Step 1: Unregister endpoint instance before memory transitions.
                if self.generate_endpoint is not None:
                    await self.generate_endpoint.unregister_endpoint_instance()
                    logger.info(
                        "[Sleep] Unregistered endpoint from discovery - worker removed from routing pool"
                    )

                # Step 2: Abort in-flight requests and wait for them to drain so the
                # GPU is fully quiesced before unmapping memory.
                if not await self._quiesce_controller.quiesce(level):
                    return {
                        "status": "ok",
                        "message": "Engine already sleeping",
                    }

                return {
                    "status": "ok",
                    "message": f"Engine slept (level={level})",
                }
            except Exception as e:
                logger.error(f"Failed to sleep engine: {e}")
                return {"status": "error", "message": str(e)}

    async def scale_elastic_ep(self, body: dict) -> dict:
        """Scale the elastic expert-parallelism data-parallel size live.

        Args:
            body: Dict with required 'new_data_parallel_size' key (int).
                Example::

                    {"new_data_parallel_size": 4}

        The vLLM Ray DP backend will spin up / tear down DP workers on the GPUs
        already reserved by the pod, then hot-swap the expert routing table.
        No pod restart is needed.
        """
        body = body or {}
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

        logger.info(f"[ElasticEP] Scaling to new_data_parallel_size={new_dp_size}")
        try:
            # TODO(upstream-vllm): remove this patch once vLLM fixes
            # add_dp_placement_groups in vllm/v1/engine/utils.py to use ray.nodes()
            # instead of ray.util.state.list_nodes().
            #
            # Patch ray.util.state.list_nodes to use the GCS API instead of the
            # dashboard HTTP API (127.0.0.1:8265/api/v0/nodes). The dynamo image
            # installs ray core only (not ray[default]), so the dashboard HTTP server
            # starts in --minimal mode with the HTTP server disabled. vLLM's
            # add_dp_placement_groups calls list_nodes() which requires that HTTP
            # endpoint, causing scale_elastic_ep to fail with "Failed to connect to
            # API server".
            #
            # ray.nodes() uses the GCS gRPC channel directly (no dashboard process
            # needed) and returns the same information. Imported lazily so ray is not
            # required at module load time (absent in non-elastic-EP deployments).
            #
            # Format mapping:
            #   list_nodes() → objects with .node_ip and .node_id
            #   ray.nodes()  → dicts with "NodeManagerAddress" and "NodeID"
            import ray
            import ray.util.state as _ray_util_state

            class _NodeInfo:
                __slots__ = ("node_id", "node_ip")

                def __init__(self, d: dict) -> None:
                    self.node_ip: str = d["NodeManagerAddress"]
                    self.node_id: str = d["NodeID"]

            _ray_util_state.list_nodes = lambda **kw: [
                _NodeInfo(n) for n in ray.nodes() if n.get("Alive", False)
            ]

            await self.engine_client.scale_elastic_ep(new_dp_size)
            logger.info(f"[ElasticEP] Scaling to dp={new_dp_size} complete")
            return {
                "status": "ok",
                "message": f"Scaled to data_parallel_size={new_dp_size}",
                "new_data_parallel_size": new_dp_size,
            }
        except Exception as e:
            logger.error(f"[ElasticEP] Scaling failed: {e}")
            return {"status": "error", "message": str(e)}

    async def wake_up(self, body: dict) -> dict:
        """Wake the engine to restore GPU memory and re-register to discovery.

        Args:
            body: Optional dict with "tags" to request a partial wake.

        Order of operations:
        1. Wake engine - restore GPU memory
        2. Re-register endpoint instance - allow frontend to route requests here again
        """
        body = body or {}
        tags = body.get("tags")
        async with self._quiesce_lock:
            if not self._quiesce_controller.is_quiesced:
                return {"status": "ok", "message": "Engine already awake"}

            try:
                # Step 1: Wake engine first - must be ready before accepting requests
                await self._quiesce_controller.resume(tags)
                if self.generate_endpoint is not None:
                    await self.generate_endpoint.register_endpoint_instance()
                    logger.info(
                        "[Wake] Re-registered endpoint to discovery - worker added back to routing pool"
                    )
                self._quiesce_controller.mark_resumed()

                return {
                    "status": "ok",
                    "message": "Engine woke",
                }
            except Exception as e:
                logger.error(f"Failed to wake up engine: {e}")
                return {"status": "error", "message": str(e)}

    @abstractmethod
    def generate(self, request: RequestT, context: Context) -> AsyncIterator[ResponseT]:
        raise NotImplementedError

    async def _monitor_abort(self, context, request_id, is_prefill):
        """
        Background task that monitors for context cancellation and shutdown.
        Aborts the request if either occurs. Raises EngineShutdown if shutdown was triggered.
        """
        try:
            # Build list of futures/tasks to wait for
            wait_for = [context.async_killed_or_stopped()]
            shutdown_task = None

            if self.shutdown_event:
                # Create task for shutdown monitoring and add to wait list
                shutdown_task = asyncio.create_task(self.shutdown_event.wait())
                wait_for.append(shutdown_task)

            # Wait for whichever happens first
            done, pending = await asyncio.wait(
                wait_for,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the pending task/future
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Abort the request
            await self.engine_client.abort(request_id)
            logger.debug(
                f"Aborted {'Prefill ' if is_prefill else ''}Request ID: {request_id}"
            )

            # Check which event triggered and raise EngineShutdown if shutdown
            if shutdown_task and shutdown_task in done:
                raise EngineShutdown("Engine was shut down during generation.")

        except asyncio.CancelledError:
            # Task was cancelled, normal cleanup if not aborted
            pass
        except EngineShutdown:
            raise
        except Exception as e:
            logger.error(f"Error in abort monitor for request {request_id}: {e}")

    @asynccontextmanager
    async def _abort_monitor(self, context, request_id, is_prefill=False):
        """
        Context manager that creates and automatically cleans up an abort monitoring task.
        If shutdown event was triggered, raises EngineShutdown on exit.
        """
        task = asyncio.create_task(self._monitor_abort(context, request_id, is_prefill))
        try:
            yield task
        finally:
            # Clean up the abort monitoring task
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:
                # If the task completed, check if it raised EngineShutdown
                task.result()

    async def clear_kv_blocks(self, request=None):
        try:
            await self.engine_client.reset_prefix_cache()
            yield {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    async def get_perf_metrics(self, request=None):
        """Return self-benchmark FPM results, or an error dict if none."""
        result = getattr(self, "_benchmark_results", None)
        if result is None:
            yield {"status": "error", "message": "no benchmark data"}
        else:
            yield result

    def add_temp_dir(self, temp_dir: tempfile.TemporaryDirectory) -> None:
        """Add a temporary directory to be cleaned up later."""
        if temp_dir is not None:
            self.temp_dirs.append(temp_dir)

    def _to_local_dp_rank(self, dp_rank: int | None) -> int | None:
        """Convert global DP rank to local DP rank based on engine config."""
        if dp_rank is None:
            return None
        if dp_rank < self.dp_range[0] or dp_rank >= self.dp_range[0] + self.dp_range[1]:
            logger.warning(
                f"Received DP rank {dp_rank} is out of range [{self.dp_range[0]} - {self.dp_range[0] + self.dp_range[1]}), fallback to vLLM internal DP selection"
            )
            return None
        local_dp_rank = (dp_rank - self.dp_range[0]) % self.dp_range[1]
        logger.debug(
            f"Converted global DP rank {dp_rank} to local DP rank {local_dp_rank}"
        )
        return local_dp_rank

    def _resolve_lora_request(self, model_name: str | None) -> LoRARequest | None:
        """Return a LoRARequest if model_name is a loaded adapter, else None."""
        if model_name and (lora := self.loaded_loras.get(model_name)):
            return LoRARequest(
                lora_name=model_name,
                lora_int_id=lora.id,
                lora_path=lora.path,
            )
        return None

    def _get_lora_lock(self, lora_name: str) -> asyncio.Lock:
        """Get/create the per-LoRA lock without eagerly allocating a new lock each call."""
        with self._lora_load_locks_guard:
            lock = self._lora_load_locks.get(lora_name)
            if lock is None:
                lock = asyncio.Lock()
                self._lora_load_locks[lora_name] = lock
            return lock

    async def load_lora(self, request=None):
        """
        Load a LoRA adapter dynamically into the vLLM's AsyncLLM engine.

        Request format:
        {
            "lora_name": str,
            "source": {
                "uri": str  # e.g., "s3://bucket/path" or "file:///path"
            }
        }

        This method is idempotent - concurrent calls for the same LoRA will be
        serialized and only one load operation will happen.
        """
        try:
            if request is None:
                yield {
                    "status": "error",
                    "message": "Request is required with 'lora_name' and 'source.uri'",
                }
                return

            lora_name = request.get("lora_name")
            if not lora_name:
                yield {
                    "status": "error",
                    "message": "'lora_name' is required in request",
                }
                return

            # Debug: Log the incoming request
            logger.debug(f"load_lora request keys: {list(request.keys())}")
            logger.debug(f"load_lora request: {request}")

            # Check for URI-based API format (source.uri)
            source = request.get("source")
            if not source or not isinstance(source, dict):
                yield {
                    "status": "error",
                    "message": "'source' object is required in request",
                }
                return

            lora_uri = source.get("uri")
            if not lora_uri:
                yield {
                    "status": "error",
                    "message": "'source.uri' is required in request",
                }
                return

            # Use LoRAManager to download from URI
            lora_manager = get_lora_manager()
            if lora_manager is None:
                yield {
                    "status": "error",
                    "message": "LoRAManager not initialized. Set DYN_LORA_ENABLED=true to enable URI-based LoRA loading.",
                }
                return

            # Serialize load/unload operations per lora_name.
            lock = self._get_lora_lock(lora_name)
            async with lock:
                try:
                    # Check if already loaded (idempotency check after acquiring lock).
                    # Another concurrent request may have loaded this LoRA while we waited.
                    if lora_name in self.loaded_loras:
                        lora_id = self.loaded_loras[lora_name].id
                        logger.info(
                            f"LoRA adapter already loaded (concurrent request completed): "
                            f"{lora_name} with ID {lora_id}"
                        )
                        yield {
                            "status": "success",
                            "message": f"LoRA adapter '{lora_name}' already loaded",
                            "lora_name": lora_name,
                            "lora_id": lora_id,
                        }
                        return

                    logger.info(
                        f"Downloading LoRA adapter: {lora_name} from {lora_uri}"
                    )
                    download_result = await lora_manager.download_lora(lora_uri)

                    if download_result["status"] != "success":
                        yield {
                            "status": "error",
                            "message": f"Failed to download LoRA: {download_result.get('message', 'Unknown error')}",
                        }
                        return

                    lora_path = download_result["local_path"]
                    logger.debug(f"LoRA downloaded to: {lora_path}")

                    # Generate deterministic ID from lora_name before using it
                    lora_id = lora_name_to_id(lora_name)

                    # Add the LoRA to the engine
                    await self.engine_client.add_lora(
                        LoRARequest(
                            lora_name=lora_name,
                            lora_int_id=lora_id,
                            lora_path=lora_path,
                        )
                    )

                    # Track the LoRA
                    self.loaded_loras[lora_name] = LoRAInfo(id=lora_id, path=lora_path)
                    logger.info(
                        f"Successfully loaded LoRA adapter: {lora_name} with ID {lora_id}"
                    )

                    # Publish LoRA as a ModelDeploymentCard with format:
                    # v1/mdc/{namespace}/{component}/{endpoint}/{instance_id}/{lora_slug}
                    # This allows the frontend to discover it and route correctly to the worker instance
                    if self.generate_endpoint is not None:
                        logger.debug(
                            f"Publishing LoRA '{lora_name}' ModelDeploymentCard to {self.generate_endpoint}"
                        )
                        try:
                            logger.debug(
                                f"Publishing LoRA '{lora_name}' ModelDeploymentCard"
                            )

                            # Mark this as a LoRA in user_data
                            user_data = {
                                "lora_adapter": True,
                                "lora_id": lora_id,
                            }

                            runtime_config = ModelRuntimeConfig()
                            runtime_config.tool_call_parser = (
                                self.config.dyn_tool_call_parser
                            )
                            runtime_config.reasoning_parser = (
                                self.config.dyn_reasoning_parser
                            )

                            # Publish with format: v1/mdc/dynamo/backend/generate/{instance_id}/{lora_slug}
                            await register_model(
                                model_input=ModelInput.Tokens,
                                model_type=ModelType.Chat | ModelType.Completions,
                                endpoint=self.generate_endpoint,
                                model_path=self.config.model,
                                kv_cache_block_size=self.config.engine_args.block_size,
                                runtime_config=runtime_config,
                                user_data=user_data,
                                lora_name=lora_name,
                                base_model_path=self.config.model,
                            )
                            logger.info(
                                f"Successfully published LoRA '{lora_name}' ModelDeploymentCard"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Failed to publish LoRA {lora_name} ModelDeploymentCard: {e}"
                            )

                            # Rollback: remove the LoRA from the engine to maintain consistency
                            try:
                                logger.debug(
                                    f"Rolling back: removing LoRA '{lora_name}' from engine"
                                )
                                await self.engine_client.remove_lora(lora_id)
                                self.loaded_loras.pop(lora_name, None)
                                logger.debug(
                                    f"Successfully rolled back LoRA '{lora_name}'"
                                )
                            except Exception as rollback_error:
                                logger.exception(
                                    f"Failed to rollback LoRA {lora_name}: {rollback_error}"
                                )

                            # Return error status since registration failed
                            yield {
                                "status": "error",
                                "message": f"Failed to register LoRA '{lora_name}' in discovery registry: {str(e)}",
                                "lora_name": lora_name,
                            }
                            return
                    else:
                        logger.debug(
                            f"Cannot publish LoRA '{lora_name}': generate_endpoint={self.generate_endpoint}, config={self.config}"
                        )

                    yield {
                        "status": "success",
                        "message": f"LoRA adapter '{lora_name}' loaded successfully",
                        "lora_name": lora_name,
                        "lora_id": lora_id,
                    }
                finally:
                    # Avoid lock-map growth on failed loads: if this attempt did not leave the LoRA
                    # loaded, remove the lock entry (best-effort).
                    with self._lora_load_locks_guard:
                        if (
                            lora_name not in self.loaded_loras
                            and self._lora_load_locks.get(lora_name) is lock
                        ):
                            self._lora_load_locks.pop(lora_name, None)
        except Exception as e:
            logger.exception(f"Failed to load LoRA adapter: {e}")
            yield {"status": "error", "message": str(e)}

    async def unload_lora(self, request=None):
        """
        Unload a LoRA adapter dynamically from the vLLM's AsyncLLM engine.
        Expected request format:
        {
            "lora_name": str,
        }
        """
        try:
            if request is None:
                yield {
                    "status": "error",
                    "message": "Request is required with 'lora_name' field",
                }
                return
            lora_name = request.get("lora_name")
            if not lora_name:
                yield {
                    "status": "error",
                    "message": "'lora_name' is required in request",
                }
                return

            # Serialize load/unload operations per lora_name.
            lock = self._get_lora_lock(lora_name)
            async with lock:
                try:
                    # Check if the LoRA exists *after* waiting for any in-progress load.
                    lora = self.loaded_loras.get(lora_name)
                    if lora is None:
                        yield {
                            "status": "error",
                            "message": f"LoRA adapter '{lora_name}' not found. Available LoRAs: {list(self.loaded_loras.keys())}",
                        }
                        return

                    logger.debug(f"Unloading LoRA adapter: {lora_name}")
                    lora_id = lora.id
                    lora_path = lora.path

                    await self.engine_client.remove_lora(lora_id)

                    # Remove from tracking
                    del self.loaded_loras[lora_name]

                    # Unregister the LoRA model from the model registry
                    if self.generate_endpoint is not None:
                        logger.debug(
                            f"Unregistering LoRA '{lora_name}' ModelDeploymentCard"
                        )
                        try:
                            await unregister_model(
                                endpoint=self.generate_endpoint,
                                lora_name=lora_name,
                            )
                            logger.info(
                                f"Successfully unregistered LoRA '{lora_name}' ModelDeploymentCard"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Failed to unregister LoRA {lora_name} ModelDeploymentCard: {e}"
                            )

                            # Rollback: re-add the LoRA to the engine to maintain consistency
                            try:
                                logger.debug(
                                    f"Rolling back: re-adding LoRA '{lora_name}' to engine"
                                )
                                await self.engine_client.add_lora(
                                    LoRARequest(
                                        lora_name=lora_name,
                                        lora_int_id=lora_id,
                                        lora_path=lora_path,
                                    )
                                )
                                # Re-add to tracking
                                self.loaded_loras[lora_name] = LoRAInfo(
                                    id=lora_id, path=lora_path
                                )
                                logger.debug(
                                    f"Successfully rolled back LoRA '{lora_name}'"
                                )
                            except Exception as rollback_error:
                                logger.exception(
                                    f"Failed to rollback LoRA {lora_name}: {rollback_error}"
                                )

                            # Return error status since unregistration failed
                            yield {
                                "status": "error",
                                "message": f"Failed to unregister LoRA '{lora_name}' from discovery registry: {str(e)}",
                                "lora_name": lora_name,
                            }
                            return
                    else:
                        logger.debug(
                            f"Cannot unregister LoRA '{lora_name}': generate_endpoint={self.generate_endpoint}"
                        )

                    logger.info(
                        f"Successfully unloaded LoRA adapter: {lora_name} with ID {lora_id}"
                    )
                    yield {
                        "status": "success",
                        "message": f"LoRA adapter '{lora_name}' unloaded successfully",
                        "lora_name": lora_name,
                        "lora_id": lora_id,
                    }
                finally:
                    # Remove lock entry once the LoRA is not loaded (or never was).
                    with self._lora_load_locks_guard:
                        if (
                            lora_name not in self.loaded_loras
                            and self._lora_load_locks.get(lora_name) is lock
                        ):
                            self._lora_load_locks.pop(lora_name, None)
        except Exception as e:
            logger.exception(f"Failed to unload LoRA adapter: {e}")
            yield {"status": "error", "message": str(e)}

    async def list_loras(self, request=None):
        """
        List all loaded LoRA adapters.
        Returns a dictionary of lora_name -> lora_id mappings.
        """
        try:
            loras = {name: lora.id for name, lora in self.loaded_loras.items()}
            yield {
                "status": "success",
                "loras": loras,
                "count": len(loras),
            }
        except Exception as e:
            logger.error(f"Failed to list LoRA adapters: {e}")
            yield {"status": "error", "message": str(e)}

    def cleanup(self):
        """Clean up resources including temporary directories."""
        for temp_dir in self.temp_dirs:
            try:
                temp_dir.cleanup()
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")

    def _decode_prompt_embeds(self, prompt_embeds_base64: str):
        """
        Decode base64-encoded prompt embeddings in PyTorch format.

        Format: PyTorch tensor serialized with torch.save() and base64-encoded.

        Args:
            prompt_embeds_base64: Base64-encoded PyTorch tensor

        Returns:
            torch.Tensor: Decoded prompt embeddings with preserved shape and dtype

        Raises:
            ValueError: If decoding fails or format is invalid
        """
        try:
            # Step 1: Decode base64 to bytes
            embeds_bytes = base64.b64decode(prompt_embeds_base64)

            # Step 2: Load PyTorch tensor from bytes
            buffer = io.BytesIO(embeds_bytes)
            embeddings_tensor = torch.load(buffer, weights_only=True)

            # Step 3: Validate it's a tensor
            if not isinstance(embeddings_tensor, torch.Tensor):
                raise ValueError(
                    f"prompt_embeds must be a torch.Tensor, got {type(embeddings_tensor)}"
                )

            logger.debug(
                f"Decoded PyTorch format embeddings: shape={embeddings_tensor.shape}, "
                f"dtype={embeddings_tensor.dtype}, size={len(embeds_bytes)} bytes"
            )

            return embeddings_tensor

        except binascii.Error as e:
            logger.error(f"Invalid base64 encoding in prompt_embeds: {e}")
            raise ValueError(f"Invalid base64 encoding in prompt_embeds: {e}")
        except Exception as e:
            logger.error(f"Failed to decode prompt_embeds: {e}")
            raise ValueError(f"Failed to decode prompt_embeds as PyTorch tensor: {e}")

    def _create_prompt_from_embeddings(
        self, prompt_embeds_base64: str
    ) -> tuple[EmbedsPrompt, int, torch.Tensor]:
        """
        Decode prompt embeddings and create EmbedsPrompt for vLLM.

        Args:
            prompt_embeds_base64: Base64-encoded PyTorch tensor

        Returns:
            Tuple of (EmbedsPrompt, sequence_length, tensor) where:
            - EmbedsPrompt: The vLLM prompt input
            - sequence_length: Extracted from tensor shape for usage statistics
            - tensor: The decoded tensor (for logging shape/dtype)

        Raises:
            ValueError: If decoding fails or tensor is invalid
        """
        embeddings_tensor = self._decode_prompt_embeds(prompt_embeds_base64)

        # Extract sequence length from tensor shape for usage reporting
        # Shape is typically (sequence_length, hidden_dim) or (batch, sequence_length, hidden_dim)
        if embeddings_tensor.dim() == 2:
            sequence_length = embeddings_tensor.shape[0]
        elif embeddings_tensor.dim() == 3:
            sequence_length = embeddings_tensor.shape[1]
        else:
            # Fallback for unexpected shapes
            sequence_length = embeddings_tensor.shape[0]

        # EmbedsInputs TypedDict has: {type: 'embeds', prompt_embeds: Tensor, cache_salt?: str}
        prompt = EmbedsPrompt(prompt_embeds=embeddings_tensor)

        return prompt, sequence_length, embeddings_tensor

    @staticmethod
    def _get_mm_processor_kwargs(
        request: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """Extract mm_processor_kwargs from a request dict.

        Checks the top-level key (client router / Rust preprocessor path)
        and falls back to ``extra_args`` (KV router path).
        """
        mm_processor_kwargs = request.get("mm_processor_kwargs")
        if mm_processor_kwargs is None:
            req_extra_args = request.get("extra_args")
            if isinstance(req_extra_args, dict):
                mm_processor_kwargs = req_extra_args.get("mm_processor_kwargs")
        return mm_processor_kwargs

    async def _extract_multimodal_data(
        self,
        request: Dict[str, Any],
        request_id: str,
        context,
        mm_processor_kwargs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        """
        Extract and decode multimodal data from PreprocessedRequest.
        """
        if "multi_modal_data" not in request or request["multi_modal_data"] is None:
            return None

        # Security check: reject multimodal data if not explicitly enabled
        if not self.enable_multimodal:
            raise ValueError(
                "Received multimodal data but multimodal processing is not enabled. "
                "Use --enable-multimodal flag to enable multimodal processing."
            )

        mm_map = request["multi_modal_data"]

        vllm_mm_data = {}

        # [gluo NOTE] If embedding loader is configured, fetch image embeddings first.
        # Still continue below so mixed image+video requests can attach `video`.
        if self.embedding_loader is not None:
            # [gluo FIXME] couldn't simply pass 'mm_map.get(IMAGE_URL_KEY, [])' like below
            # as currently the encode worker is using 'ImageLoader.load_image()' which doesn't
            # support 'Decoded' variant. Need to update the encode worker to unify handling
            image_urls = []
            supported = True
            for item in mm_map.get(IMAGE_URL_KEY, []):
                if isinstance(item, dict) and "Url" in item:
                    image_urls.append(item["Url"])
                elif isinstance(item, dict) and "Decoded" in item:
                    supported = False
            if supported:
                vllm_mm_data = await self.embedding_loader.load_multimodal_embeddings(
                    image_urls, request_id, model=self.config.model, context=context
                )
                logger.debug(
                    f"Fetched multimodal embeddings for {len(vllm_mm_data)} items"
                )

        image_mm_items = mm_map.get(IMAGE_URL_KEY, [])
        if "image" not in vllm_mm_data and image_mm_items:
            images = await self.image_loader.load_image_batch(
                image_mm_items,
            )

            if images:
                # vLLM expects single image or list
                vllm_mm_data["image"] = images[0] if len(images) == 1 else images
                logger.debug(
                    f"Extracted {len(images)} image(s) for multimodal processing"
                )

        video_mm_items = mm_map.get(VIDEO_URL_KEY, [])
        if video_mm_items:
            videos = await self.video_loader.load_video_batch(video_mm_items)

            if videos:
                # vLLM expects single video or list
                vllm_mm_data["video"] = videos[0] if len(videos) == 1 else videos
                logger.debug(
                    f"Extracted {len(videos)} video(s) for multimodal processing"
                )

        # Handle audio_url entries
        audio_mm_items = mm_map.get(AUDIO_URL_KEY, [])
        if audio_mm_items:
            audios = await self.audio_loader.load_audio_batch(audio_mm_items)
            if audios:
                vllm_mm_data["audio"] = audios[0] if len(audios) == 1 else audios
                logger.debug(
                    f"Extracted {len(audios)} audio item(s) for multimodal processing"
                )

        # Extract audio from video URLs when use_audio_in_video is set.
        # Models expect 1:1 audio/video pairing in the same order.
        # We load per-video sequentially to preserve ordering; a video
        # without an audio track raises immediately to avoid corrupting
        # the alignment.
        if (
            video_mm_items
            and mm_processor_kwargs
            and mm_processor_kwargs.get("use_audio_in_video", False)
        ):
            video_audios: list = []
            for item in video_mm_items:
                url = item.get(URL_VARIANT_KEY) if isinstance(item, dict) else None
                if not url:
                    raise ValueError(
                        "use_audio_in_video requires all video items to be "
                        "URL-based. Got a non-URL video item (e.g. frontend-"
                        "decoded). Audio extraction from decoded video data "
                        "is not yet supported."
                    )
                try:
                    audio = await self.audio_loader.load_audio(url)
                    video_audios.append(audio)
                except Exception:
                    logger.error(
                        "Failed to extract audio from video %s. "
                        "use_audio_in_video requires every video to "
                        "contain an audio stream.",
                        url[:80],
                    )
                    raise
            if video_audios:
                existing = vllm_mm_data.get("audio")
                if existing is not None:
                    all_audios = (
                        existing if isinstance(existing, list) else [existing]
                    ) + video_audios
                else:
                    all_audios = video_audios
                vllm_mm_data["audio"] = (
                    all_audios[0] if len(all_audios) == 1 else all_audios
                )
                logger.debug(
                    "Extracted %d audio track(s) from video URL(s) "
                    "(use_audio_in_video=True)",
                    len(video_audios),
                )

        return vllm_mm_data if vllm_mm_data else None

    def _build_prompt_from_request(
        self,
        request: Dict[str, Any],
        request_id: str,
        multi_modal_data: Dict[str, Any] | None,
        log_prefix: str = "",
        mm_processor_kwargs: Dict[str, Any] | None = None,
    ) -> tuple[TokensPrompt | EmbedsPrompt | None, int | None, Dict[str, Any] | None]:
        """
        Build a prompt from request, handling both prompt_embeds and token_ids.

        Args:
            request: The request dict containing either prompt_embeds or token_ids
            request_id: Request ID for logging
            multi_modal_data: Optional multimodal data to attach to TokensPrompt
            log_prefix: Prefix for log messages (e.g., "Prefill " for prefill requests)
            mm_processor_kwargs: Optional multimodal processor kwargs (e.g.
                use_audio_in_video) forwarded to the vLLM engine.

        Returns:
            Tuple of (prompt, embedding_sequence_length, error_dict) where:
            - On success: (prompt, embedding_sequence_length or None, None)
            - On failure: (None, None, error_dict to yield)
        """
        embedding_sequence_length = None

        if "prompt_embeds" in request and request["prompt_embeds"]:
            try:
                (
                    prompt,
                    embedding_sequence_length,
                    tensor,
                ) = self._create_prompt_from_embeddings(request["prompt_embeds"])
                logger.info(
                    f"{log_prefix}Using prompt embeddings: shape={tensor.shape}, "
                    f"dtype={tensor.dtype}, sequence_length={embedding_sequence_length}, "
                    f"request_id={request_id}"
                )
                return prompt, embedding_sequence_length, None
            except Exception as e:
                logger.error(
                    f"Failed to process prompt_embeds for {log_prefix.lower().strip() or 'request'} "
                    f"{request_id}: {e}"
                )
                return (
                    None,
                    None,
                    {
                        "finish_reason": f"error: Invalid prompt_embeds: {e}",
                        "token_ids": [],
                    },
                )
        # Normal path: use token IDs
        mm_uuids = _compute_mm_uuids(multi_modal_data)
        prompt_kwargs = dict[str, Any](
            prompt_token_ids=request["token_ids"],
            multi_modal_data=multi_modal_data,
        )
        if mm_uuids is not None:
            prompt_kwargs["multi_modal_uuids"] = mm_uuids
        if mm_processor_kwargs is not None:
            prompt_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

        prompt = TokensPrompt(**prompt_kwargs)
        return prompt, embedding_sequence_length, None

    @staticmethod
    def _build_completion_usage(
        request_output: RequestOutput,
        embedding_sequence_length: int | None = None,
    ) -> Dict[str, Any]:
        """
        Build completion usage statistics.

        Args:
            request_output: vLLM RequestOutput object
            embedding_sequence_length: If using prompt embeddings, the sequence length
                                     extracted from the embeddings tensor shape

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details
        """
        # Determine prompt token count:
        # - For embeddings: use embedding_sequence_length from tensor shape
        # - For normal text: use len(prompt_token_ids)
        if embedding_sequence_length is not None:
            prompt_tokens = embedding_sequence_length
        elif request_output.prompt_token_ids:
            prompt_tokens = len(request_output.prompt_token_ids)
        else:
            prompt_tokens = None

        completion_tokens = len(request_output.outputs[0].token_ids)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens + completion_tokens if prompt_tokens is not None else None
            ),
            "prompt_tokens_details": (
                {"cached_tokens": num_cached}
                if (num_cached := getattr(request_output, "num_cached_tokens", None))
                else None
            ),
        }

    @staticmethod
    def _extract_logprobs(
        output,
        num_output_tokens_so_far: int,
        tokenizer=None,
        return_tokens_as_token_ids: bool = False,
    ) -> tuple[list[float] | None, list[list[dict]] | None]:
        """
        Extract logprobs from vLLM CompletionOutput for new tokens.

        Args:
            output: vLLM CompletionOutput object
            num_output_tokens_so_far: Number of tokens already processed
            tokenizer: Optional tokenizer for decoding token IDs when
                       decoded_token is not populated by the engine

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

        log_probs = []
        top_logprobs = []

        for token_idx, token_logprobs_dict in enumerate(new_logprobs):
            if token_logprobs_dict is None:
                continue

            # Get the actual token_id that was generated at this position
            actual_token_id = output.token_ids[num_output_tokens_so_far + token_idx]

            # Extract log probability for the selected token
            # vLLM guarantees the selected token is always in the logprobs dict
            selected_logprob = token_logprobs_dict[actual_token_id]
            log_probs.append(float(selected_logprob.logprob))

            # Build top_logprobs list for this token position
            token_top_logprobs = []
            for tok_id, logprob_info in token_logprobs_dict.items():
                if return_tokens_as_token_ids:
                    token_str = f"token_id:{tok_id}"
                else:
                    token_str = getattr(logprob_info, "decoded_token", None)
                    if not token_str and tokenizer:
                        try:
                            token_str = tokenizer.decode([tok_id])
                        except Exception:
                            token_str = None
                token_top_logprobs.append(
                    {
                        "rank": (
                            logprob_info.rank if hasattr(logprob_info, "rank") else 0
                        ),
                        "token_id": tok_id,
                        "token": token_str,
                        "logprob": float(logprob_info.logprob),
                        "bytes": (
                            list(token_str.encode("utf-8")) if token_str else None
                        ),
                    }
                )
            top_logprobs.append(token_top_logprobs)

        return log_probs if log_probs else None, top_logprobs if top_logprobs else None

    @staticmethod
    def _log_with_lora_context(
        message: str,
        request_id: str,
        lora_request=None,
        level: str = "debug",
        **kwargs,
    ) -> None:
        """
        Log a message with optional LoRA context.

        Args:
            message: Base message to log (can include {lora_info} placeholder)
            request_id: Request ID for correlation
            lora_request: Optional LoRA request object
            level: Log level ("debug" or "info")
            **kwargs: Additional format arguments for the message
        """
        if lora_request:
            lora_info = f" with LoRA {lora_request.lora_name}"
        else:
            lora_info = ""

        formatted_message = message.format(
            request_id=request_id,
            lora_info=lora_info,
            **kwargs,
        )

        if level == "info":
            logger.info(formatted_message)
        else:
            logger.debug(formatted_message)

    async def generate_tokens(
        self,
        prompt,
        sampling_params,
        request_id,
        data_parallel_rank=None,
        lora_request=None,
        embedding_sequence_length=None,
        trace_headers=None,
        priority=0,
        return_tokens_as_token_ids=False,
    ):
        try:
            # Log LoRA usage for this generation (debug level to avoid log spam)
            self._log_with_lora_context(
                "Starting token generation for request {request_id}{lora_info}",
                request_id,
                lora_request,
            )
            gen = self.engine_client.generate(
                prompt,
                sampling_params,
                request_id,
                lora_request=lora_request,
                data_parallel_rank=data_parallel_rank,
                trace_headers=trace_headers,
                priority=priority,
            )

            num_output_tokens_so_far = 0
            async for res in gen:
                # res is vllm's RequestOutput

                if not res.outputs:
                    self._log_with_lora_context(
                        "Request {request_id}{lora_info} returned no outputs",
                        request_id,
                        lora_request,
                    )
                    # Use string format "error: message" for consistency with vLLM's string-based finish_reason
                    # Rust will parse this into FinishReason::Error(message)
                    yield {
                        "finish_reason": "error: No outputs from vLLM engine",
                        "token_ids": [],
                    }
                    break

                output = res.outputs[0]
                next_total_toks = len(output.token_ids)
                out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}

                # Extract logprobs for new tokens if available
                tokenizer = getattr(self.engine_client, "tokenizer", None)
                log_probs, top_logprobs = self._extract_logprobs(
                    output,
                    num_output_tokens_so_far,
                    tokenizer=tokenizer,
                    return_tokens_as_token_ids=return_tokens_as_token_ids,
                )
                if log_probs is not None:
                    out["log_probs"] = log_probs
                if top_logprobs is not None:
                    out["top_logprobs"] = top_logprobs

                if output.finish_reason:
                    out["finish_reason"] = normalize_finish_reason(output.finish_reason)
                    out["completion_usage"] = BaseWorkerHandler._build_completion_usage(
                        request_output=res,
                        embedding_sequence_length=embedding_sequence_length,
                    )
                    # Log completion with LoRA info (debug level to avoid log spam)
                    self._log_with_lora_context(
                        "Completed token generation for request {request_id}{lora_info}: "
                        "{output_tokens} output tokens, finish_reason={finish_reason}",
                        request_id,
                        lora_request,
                        output_tokens=next_total_toks,
                        finish_reason=output.finish_reason,
                    )
                if output.stop_reason:
                    out["stop_reason"] = output.stop_reason
                yield out
                num_output_tokens_so_far = next_total_toks

        except EngineDeadError as e:
            logger.error(f"vLLM EngineDeadError: {e}")
            logger.warning("Initiating Dynamo Runtime shutdown.")
            self.runtime.shutdown()
            os._exit(1)


class DecodeWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        runtime,
        config: Config,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
        encode_worker_client: Client | None = None,
    ):
        super().__init__(
            runtime,
            config,
            engine,
            default_sampling_params,
            model_max_len,
            enable_multimodal,
            generate_endpoint,
            use_vllm_tokenizer,
            shutdown_event,
            enable_frontend_decoding,
            encode_worker_client,
        )

    async def generate(self, request, context):
        # Use context ID for request tracking and correlation
        request_id = context.id()
        logger.debug(f"Decode Request ID: {request_id}")
        first_token = True
        with time_and_log_code_section(
            f"[DECODE] request: {request_id} generate"
        ) as decode_timer:
            if self.use_vllm_tokenizer:
                # Text-in-text-out mode: use InputParamManager and OpenAI-compatible format
                generator = self._generate_text_mode(request, context, request_id)
            else:
                # Token-in-token-out mode: internal protocol format
                generator = self._generate_token_mode(request, context, request_id)

            async for chunk in generator:
                if first_token:
                    decode_timer.stop_interval()
                    first_token = False
                yield chunk

    async def _generate_token_mode(self, request, context, request_id):
        """Generate tokens using internal protocol format (token-in-token-out)."""
        # Firstly extract disaggregated params from prefill result if available
        prefill_result = request.get("prefill_result")
        if prefill_result and isinstance(prefill_result, dict):
            kv_params = prefill_result.get("disaggregated_params", {}).get(
                "kv_transfer_params"
            )
            embedding_params = prefill_result.get("disaggregated_params", {}).get(
                "embedding_params"
            )
        else:
            kv_params = None
            embedding_params = None

        is_decode_only = self.config.disaggregation_mode == DisaggregationMode.DECODE
        has_mm_data = (
            "multi_modal_data" in request and request["multi_modal_data"] is not None
        )

        mm_processor_kwargs = self._get_mm_processor_kwargs(request)

        multi_modal_data = None
        if is_decode_only:
            # Decode mode: branch on model, not data.
            if is_qwen_vl_model(self.config.model):
                # Qwen VL needs embedding_params for mRoPE initialization.
                if embedding_params is not None:
                    multi_modal_data = construct_qwen_decode_mm_data(
                        embedding_params["image_grid_thw"],
                        embedding_params["embeddings_shape"],
                        request_id,
                    )
                elif has_mm_data and request["multi_modal_data"].get(IMAGE_URL_KEY):
                    # Guard is on IMAGE_URL_KEY (not just has_mm_data) so
                    # text-only requests pass through and video/audio fall
                    # through to re-download below (TODO: proper support).
                    msg = (
                        "Decode worker received multimodal request without "
                        "prefill result"
                        if prefill_result is None
                        else "Prefill did not produce required multimodal "
                        "embedding metadata (image_grid_thw) for Qwen VL "
                        "decode. Use --route-to-encoder or the P/D launcher "
                        "with grid_thw computation support"
                    )
                    logger.error("Request %s: %s", request_id, msg)
                    yield {"status": "error", "message": msg}
                    return
            # TODO(DIS-1661): video/audio re-downloaded on decode.
            # TODO(DIS-1664): mixed image+video in disagg decode is not
            # supported — synthetic image data would be overwritten.
            if multi_modal_data is None and has_mm_data:
                mm = request["multi_modal_data"]
                if mm.get(VIDEO_URL_KEY) or mm.get(AUDIO_URL_KEY):
                    multi_modal_data = await self._extract_multimodal_data(
                        request,
                        request_id,
                        context,
                        mm_processor_kwargs=mm_processor_kwargs,
                    )
        else:
            # Aggregated mode: load images normally
            multi_modal_data = await self._extract_multimodal_data(
                request,
                request_id,
                context,
                mm_processor_kwargs=mm_processor_kwargs,
            )

        # Build prompt from request (handles both prompt_embeds and token_ids)
        prompt, embedding_sequence_length, error = self._build_prompt_from_request(
            request,
            request_id,
            multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        if error is not None:
            yield error
            return

        # Build sampling params from request
        sampling_params = build_sampling_params(
            request, self.default_sampling_params, self.model_max_len
        )

        if kv_params is not None:
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args["kv_transfer_params"] = kv_params
            logger.debug(
                f"Using disaggregated params from prefill for request {request_id}"
            )
        prefill_prompt_tokens_details = (
            prefill_result.get("prompt_tokens_details") if prefill_result else None
        )

        # Extract LoRA request if present
        model_name = request.get("model")
        lora_request = self._resolve_lora_request(model_name)
        if lora_request:
            logger.info(
                f"Decode request {request_id} will use LoRA adapter: {model_name} (ID: {lora_request.lora_int_id})"
            )
        else:
            logger.debug(
                f"Decode request {request_id} has no LoRA specified (model: {model_name})"
            )
        routing = request.get("routing") or {}
        dp_rank = self._to_local_dp_rank(routing.get("dp_rank"))
        priority = -int(routing.get("priority", 0))

        trace_headers = build_trace_headers(context)

        output_options = request.get("output_options", {})
        return_tokens_as_token_ids = bool(
            output_options.get("return_tokens_as_token_ids")
        )

        async with self._abort_monitor(context, request_id):
            try:
                async for tok in self.generate_tokens(
                    prompt,
                    sampling_params,
                    request_id,
                    data_parallel_rank=dp_rank,
                    lora_request=lora_request,
                    embedding_sequence_length=embedding_sequence_length,
                    trace_headers=trace_headers,
                    priority=priority,
                    return_tokens_as_token_ids=return_tokens_as_token_ids,
                ):
                    if prefill_result is not None and "completion_usage" in tok:
                        tok["completion_usage"][
                            "prompt_tokens_details"
                        ] = prefill_prompt_tokens_details
                    yield tok
            except EngineDeadError as e:
                logger.error(f"vLLM EngineDeadError: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)

    async def _generate_text_mode(self, request, context, request_id):
        """Generate text using OpenAI-compatible format (text-in-text-out)."""
        # Get text input using InputParamManager
        input_data = self.input_param_manager.get_input_param(
            request, use_tokenizer=True
        )

        # Build prompt for vLLM
        if isinstance(input_data, list):
            prompt = TokensPrompt(prompt_token_ids=input_data)
        else:
            prompt = TextPrompt(prompt=input_data)

        # Build sampling params from OpenAI-style request
        sampling_params = build_sampling_params_openai(
            request, self.default_sampling_params
        )

        routing = request.get("routing") or {}
        dp_rank = self._to_local_dp_rank(routing.get("dp_rank"))
        priority = -int(routing.get("priority", 0))
        openai_request_id = request.get("id") or request.get("request_id", request_id)
        previous_text = ""

        trace_headers = build_trace_headers(context)

        async with self._abort_monitor(context, request_id):
            try:
                gen = self.engine_client.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    data_parallel_rank=dp_rank,
                    trace_headers=trace_headers,
                    priority=priority,
                )

                async for res in gen:
                    if not res.outputs:
                        yield {
                            "id": openai_request_id,
                            "created": int(time.time()),
                            "object": "chat.completion.chunk",
                            "model": "unknown",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": "error",
                                }
                            ],
                        }
                        break

                    output = res.outputs[0]
                    # Calculate the delta text (new text since last chunk)
                    delta_text = output.text[len(previous_text) :]
                    previous_text = output.text

                    choice_data = {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": delta_text,
                        },
                        "finish_reason": normalize_finish_reason(output.finish_reason),
                    }

                    chunk = {
                        "id": openai_request_id,
                        "created": int(time.time()),
                        "object": "chat.completion.chunk",
                        "model": "unknown",
                        "choices": [choice_data],
                    }

                    if output.finish_reason:
                        chunk["usage"] = BaseWorkerHandler._build_completion_usage(
                            request_output=res,
                        )

                    yield chunk

            except EngineDeadError as e:
                logger.error(f"vLLM EngineDeadError: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)


class PrefillWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        runtime,
        config: Config,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
        encode_worker_client: Client | None = None,
    ):
        super().__init__(
            runtime,
            config,
            engine,
            default_sampling_params,
            model_max_len,
            enable_multimodal,
            generate_endpoint,
            use_vllm_tokenizer,
            shutdown_event,
            enable_frontend_decoding,
            encode_worker_client,
        )

        # Cache Qwen VL grid parameters for computing image_grid_thw from
        # PIL images in the P/D path (no separate encode worker).
        if is_qwen_vl_model(config.model):
            self._qwen_grid_params = load_qwen_grid_params(config.model)
            if self._qwen_grid_params is None and self.embedding_loader is None:
                logger.error(
                    "Qwen VL grid params failed to load and no encode worker "
                    "is configured. P/D multimodal requests will fail because "
                    "prefill cannot produce embedding_params for decode. "
                    "Use --route-to-encoder or ensure the model is cached."
                )
        else:
            self._qwen_grid_params = None

    async def generate(self, request, context):
        # Use context ID for request tracking and correlation with decode phase
        request_id = context.id()
        logger.debug(f"Prefill Request ID: {request_id}")

        # Token-in-token-out mode: internal protocol format
        with time_and_log_code_section(f"[PREFILL] request: {request_id} generate"):
            async for chunk in self._generate_token_mode(request, context, request_id):
                yield chunk

    async def _generate_token_mode(self, request, context, request_id):
        """Generate prefill using internal protocol format (token-in-token-out)."""
        mm_processor_kwargs = self._get_mm_processor_kwargs(request)

        # Extract and decode multimodal data if present
        multi_modal_data = await self._extract_multimodal_data(
            request,
            request_id,
            context,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        embedding_params = self._build_embedding_params(multi_modal_data or {})

        # Build prompt from request (handles both prompt_embeds and token_ids)
        prompt, embedding_sequence_length, error = self._build_prompt_from_request(
            request,
            request_id,
            multi_modal_data,
            log_prefix="Prefill ",
            mm_processor_kwargs=mm_processor_kwargs,
        )
        if error is not None:
            # Prefill errors need disaggregated_params field
            error["disaggregated_params"] = None
            yield error
            return

        # Build sampling params from request using shared utility
        sampling_params = build_sampling_params(
            request, self.default_sampling_params, self.model_max_len
        )

        # Configure for prefill-only mode with remote decode
        if sampling_params.extra_args is None:
            sampling_params.extra_args = {}
        sampling_params.extra_args["kv_transfer_params"] = {
            "do_remote_decode": True,
        }
        sampling_params_defaults = {
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        # Add only missing keys
        for k, v in sampling_params_defaults.items():
            sampling_params.extra_args["kv_transfer_params"].setdefault(k, v)
        # Override for prefill: only generate 1 token
        sampling_params.max_tokens = 1
        sampling_params.min_tokens = 1

        # Extract LoRA request if present
        model_name = request.get("model")
        lora_request = self._resolve_lora_request(model_name)
        if lora_request:
            logger.info(
                f"Prefill request {request_id} will use LoRA adapter: {model_name} "
                f"(ID: {lora_request.lora_int_id}), path: {lora_request.lora_path}"
            )
        else:
            logger.debug(
                f"Prefill request {request_id} has no LoRA specified (model: {model_name})"
            )

        routing = request.get("routing") or {}
        dp_rank = self._to_local_dp_rank(routing.get("dp_rank"))
        priority = -int(routing.get("priority", 0))

        trace_headers = build_trace_headers(context)

        async with self._abort_monitor(context, request_id, is_prefill=True):
            try:
                gen = self.engine_client.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    data_parallel_rank=dp_rank,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=priority,
                )
            except EngineDeadError as e:
                logger.error(f"vLLM EngineDeadError: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)

            async for res in gen:
                logger.debug(f"kv transfer params: {res.kv_transfer_params}")

                token_ids = res.outputs[0].token_ids if res.outputs else []

                output: Dict[str, Any] = {
                    "token_ids": list(token_ids),
                    "disaggregated_params": self._build_disaggregated_params(
                        res.kv_transfer_params, embedding_params
                    ),
                    "completion_usage": BaseWorkerHandler._build_completion_usage(
                        request_output=res,
                        embedding_sequence_length=embedding_sequence_length,
                    ),
                }

                # Log prefill completion with LoRA info
                self._log_with_lora_context(
                    "Prefill completed for request {request_id}{lora_info}: "
                    "generated {token_count} token(s), has_kv_params={has_kv_params}",
                    request_id,
                    lora_request,
                    level="info" if lora_request else "debug",
                    token_count=len(token_ids),
                    has_kv_params=res.kv_transfer_params is not None,
                )

                yield output

    def _build_disaggregated_params(self, kv_transfer_params, embedding_params=None):
        disaggregated_params = {}
        if kv_transfer_params is not None:
            disaggregated_params["kv_transfer_params"] = kv_transfer_params
        if embedding_params is not None:
            disaggregated_params["embedding_params"] = embedding_params

        return disaggregated_params if disaggregated_params else None

    def _build_embedding_params(
        self, multi_modal_data: dict[str, Any]
    ) -> Dict[str, Any] | None:
        if not is_qwen_vl_model(self.config.model):
            return None
        return build_qwen_embedding_params(multi_modal_data, self._qwen_grid_params)
