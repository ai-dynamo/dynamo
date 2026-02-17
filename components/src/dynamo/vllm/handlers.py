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
from typing import Any, AsyncGenerator, Dict, Final

import torch
from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine.exceptions import EngineDeadError

import dynamo.nixl_connect as nixl_connect
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.common.utils.input_params import InputParamManager
from dynamo.common.utils.media_nixl import read_decoded_media_via_nixl
from dynamo.common.utils.otel_tracing import build_trace_headers
from dynamo.llm import (
    KvEventPublisher,
    ModelInput,
    ModelType,
    lora_name_to_id,
    register_model,
    unregister_model,
)
from dynamo.runtime.logging import configure_dynamo_logging

from .engine_monitor import VllmEngineMonitor
from .multimodal_utils.hash_utils import compute_mm_uuids_from_images
from .multimodal_utils.image_loader import ImageLoader

# Multimodal data dictionary keys
IMAGE_URL_KEY: Final = "image_url"
VIDEO_URL_KEY: Final = "video_url"
URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"

configure_dynamo_logging()
logger = logging.getLogger(__name__)


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
    guided_decoding = request["sampling_options"].get("guided_decoding")
    if guided_decoding is not None and isinstance(guided_decoding, dict):
        sampling_params.structured_outputs = StructuredOutputsParams(
            json=guided_decoding.get("json"),
            regex=guided_decoding.get("regex"),
            choice=guided_decoding.get("choice"),
            grammar=guided_decoding.get("grammar"),
            whitespace_pattern=guided_decoding.get("whitespace_pattern"),
        )

    # Apply remaining sampling_options
    for key, value in request["sampling_options"].items():
        # Skip guided_decoding - already handled above
        if key == "guided_decoding":
            continue
        if value is not None and hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    # Apply stop_conditions
    for key, value in request["stop_conditions"].items():
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


class BaseWorkerHandler(ABC):
    """
    Request handler for the generate and clear_kv_blocks endpoints.
    """

    def __init__(
        self,
        runtime,
        component,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        config=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
    ):
        self.runtime = runtime
        self.component = component
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params
        self.kv_publishers: list[KvEventPublisher] | None = None
        self.generate_endpoint = generate_endpoint
        self.config = config
        self.engine_monitor = VllmEngineMonitor(runtime, engine, shutdown_event)
        self.image_loader = ImageLoader()
        self.temp_dirs: list[tempfile.TemporaryDirectory] = []
        self.model_max_len = model_max_len
        self.enable_multimodal = enable_multimodal
        self.enable_frontend_decoding = enable_frontend_decoding
        # NIXL connector for frontend decoding - lazy initialized
        self._nixl_connector = None
        self._nixl_connector_lock = asyncio.Lock()
        # LoRA tracking
        self.lora_id_for_name: dict[str, int] = {}
        self.lora_name_to_path: dict[str, str] = {}
        # Per-LoRA locks to prevent concurrent load operations for the same LoRA
        self._lora_load_locks: dict[str, asyncio.Lock] = {}
        # Guard lock-map access in case handlers are invoked from multiple threads.
        self._lora_load_locks_guard = threading.Lock()

        self.use_vllm_tokenizer = use_vllm_tokenizer

        # Initialize InputParamManager for text-in-text-out mode
        tokenizer = None
        if use_vllm_tokenizer and hasattr(engine, "tokenizer"):
            tokenizer = engine.tokenizer
        self.input_param_manager = InputParamManager(tokenizer)

        # Store shutdown event for graceful shutdown monitoring
        self.shutdown_event = shutdown_event

    async def sleep(self, body: dict) -> dict:
        """Sleep the engine to release GPU memory and unregister from discovery.

        Args:
            body: Dict with optional 'level' key (1=weights only, 2=weights+buffers, 3=everything)

        Order of operations:
        1. Unregister from discovery - stop accepting new requests
        2. Sleep engine - safe now that no new requests will arrive
        """
        level = body.get("level", 1)
        try:
            # Step 1: Unregister endpoint instance FIRST to stop new requests from arriving
            try:
                await self.generate_endpoint.unregister_endpoint_instance()
                logger.info(
                    "[Sleep] Unregistered endpoint from discovery - worker removed from routing pool"
                )
            except Exception as unreg_err:
                logger.warning(
                    f"[Sleep] Failed to unregister endpoint from discovery: {unreg_err}"
                )

            # Step 2: Now safe to sleep - no new requests will be routed here
            await self.engine_client.sleep(level)

            return {"status": "ok", "message": f"Engine slept (level={level})"}
        except Exception as e:
            logger.error(f"Failed to sleep engine: {e}")
            return {"status": "error", "message": str(e)}

    async def wake_up(self, body: dict) -> dict:
        """Wake the engine to restore GPU memory and re-register to discovery.

        Args:
            body: Dict with optional 'tags' key (e.g., ["weights", "kv_cache"]). None wakes all.

        Order of operations:
        1. Wake engine - restore GPU memory
        2. Re-register endpoint instance - allow frontend to route requests here again
        """
        tags = body.get("tags")
        try:
            # Step 1: Wake engine first - must be ready before accepting requests
            await self.engine_client.wake_up(tags)

            # Step 2: Re-register endpoint instance to discovery so frontend can route to us again
            try:
                await self.generate_endpoint.register_endpoint_instance()
                logger.info(
                    "[Wake] Re-registered endpoint to discovery - worker added back to routing pool"
                )
            except Exception as reg_err:
                logger.warning(
                    f"[Wake] Failed to re-register endpoint to discovery: {reg_err}"
                )

            return {"status": "ok", "message": f"Engine woke (tags={tags})"}
        except Exception as e:
            logger.error(f"Failed to wake up engine: {e}")
            return {"status": "error", "message": str(e)}

    @abstractmethod
    async def generate(self, request, context) -> AsyncGenerator[dict, None]:
        raise NotImplementedError

    async def _monitor_abort(self, context, request_id, is_prefill):
        """
        Background task that monitors for context cancellation and shutdown.
        Aborts the request if either occurs. Raises GeneratorExit if shutdown was triggered.
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

            # Check which event triggered and raise GeneratorExit if shutdown
            if shutdown_task and shutdown_task in done:
                raise GeneratorExit("Engine was shut down during generation.")

        except asyncio.CancelledError:
            # Task was cancelled, normal cleanup if not aborted
            pass
        except Exception as e:
            logger.error(f"Error in abort monitor for request {request_id}: {e}")

    @asynccontextmanager
    async def _abort_monitor(self, context, request_id, is_prefill=False):
        """
        Context manager that creates and automatically cleans up an abort monitoring task.
        If shutdown event was triggered, raises GeneratorExit on exit.
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
                # If the task completed, check if it raised GeneratorExit
                task.result()

    async def clear_kv_blocks(self, request=None):
        try:
            await self.engine_client.reset_prefix_cache()
            yield {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    def add_temp_dir(self, temp_dir: tempfile.TemporaryDirectory) -> None:
        """Add a temporary directory to be cleaned up later."""
        if temp_dir is not None:
            self.temp_dirs.append(temp_dir)

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
                    if lora_name in self.lora_id_for_name:
                        lora_id = self.lora_id_for_name[lora_name]
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
                    self.lora_id_for_name[lora_name] = lora_id
                    self.lora_name_to_path[lora_name] = lora_path
                    logger.info(
                        f"Successfully loaded LoRA adapter: {lora_name} with ID {lora_id}"
                    )

                    # Publish LoRA as a ModelDeploymentCard with format:
                    # v1/mdc/{namespace}/{component}/{endpoint}/{instance_id}/{lora_slug}
                    # This allows the frontend to discover it and route correctly to the worker instance
                    if self.generate_endpoint is not None and self.config is not None:
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

                            # Publish with format: v1/mdc/dynamo/backend/generate/{instance_id}/{lora_slug}
                            await register_model(
                                model_input=ModelInput.Tokens,
                                model_type=ModelType.Chat | ModelType.Completions,
                                endpoint=self.generate_endpoint,
                                model_path=self.config.model,
                                kv_cache_block_size=self.config.engine_args.block_size,
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
                                # Remove from tracking dictionaries
                                if lora_name in self.lora_id_for_name:
                                    del self.lora_id_for_name[lora_name]
                                if lora_name in self.lora_name_to_path:
                                    del self.lora_name_to_path[lora_name]
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
                            lora_name not in self.lora_id_for_name
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
                    if lora_name not in self.lora_id_for_name:
                        yield {
                            "status": "error",
                            "message": f"LoRA adapter '{lora_name}' not found. Available LoRAs: {list(self.lora_id_for_name.keys())}",
                        }
                        return

                    logger.debug(f"Unloading LoRA adapter: {lora_name}")
                    lora_id = self.lora_id_for_name[lora_name]
                    lora_path = self.lora_name_to_path.get(lora_name)

                    await self.engine_client.remove_lora(lora_id)

                    # Remove from tracking dictionaries
                    del self.lora_id_for_name[lora_name]
                    if lora_name in self.lora_name_to_path:
                        del self.lora_name_to_path[lora_name]

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
                            if lora_path is None:
                                logger.error(
                                    f"Cannot rollback LoRA '{lora_name}': lora_path is None (data inconsistency)"
                                )
                            else:
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
                                    # Re-add to tracking dictionaries
                                    self.lora_id_for_name[lora_name] = lora_id
                                    self.lora_name_to_path[lora_name] = lora_path
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
                            lora_name not in self.lora_id_for_name
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
            loras = dict(self.lora_id_for_name)
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

    async def _load_image_batch(
        self, image_mm_items: list[Dict[str, Any]]
    ) -> list[Any]:
        """
        Load a batch of images from multimodal data items.

        Supports two paths:
        1. Url variant: Download and decode image from URL (default)
        2. Decoded variant: Read pre-decoded image via NIXL RDMA (requires --frontend-decoding)

        Args:
            image_mm_items: List of multimodal data items for images
        Returns:
            List of loaded image data
        Raises:
            Exception: If any image fails to load
        """
        image_futures = []

        for item in image_mm_items:
            if isinstance(item, dict) and URL_VARIANT_KEY in item:
                # URL path: download and decode in Python backend
                url = item[URL_VARIANT_KEY]
                image_futures.append(self.image_loader.load_image(url))
                logger.debug(f"Preparing to load image from URL: {url[:80]}...")
            elif isinstance(item, dict) and DECODED_VARIANT_KEY in item:
                if self.enable_frontend_decoding:
                    async with self._nixl_connector_lock:
                        if self._nixl_connector is None:
                            self._nixl_connector = nixl_connect.Connector()
                            await self._nixl_connector.initialize()

                    metadata = item[DECODED_VARIANT_KEY]
                    image_futures.append(
                        read_decoded_media_via_nixl(self._nixl_connector, metadata)
                    )
                else:
                    logger.error(
                        "Received Decoded multimodal data but --frontend-decoding not enabled. "
                        "Use --frontend-decoding flag to enable NIXL RDMA image transfer."
                    )
                    raise ValueError("Could not load decoded media from frontend")

        # Process images in parallel
        results = await asyncio.gather(*image_futures, return_exceptions=True)
        loaded_images = []
        collective_exceptions = ""
        for media_item, result in zip(image_mm_items, results):
            if isinstance(result, Exception):
                source = media_item.get(URL_VARIANT_KEY, "decoded")
                logger.error(f"Failed to load image from {source[:80]}...: {result}")
                collective_exceptions += (
                    f"Failed to load image from {source[:80]}...: {result}\n"
                )
                continue
            loaded_images.append(result)

        if collective_exceptions:
            raise Exception(collective_exceptions)

        return loaded_images

    async def _extract_multimodal_data(
        self, request: Dict[str, Any]
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

        # Process image_url entries
        images = await self._load_image_batch(mm_map.get(IMAGE_URL_KEY, []))

        if images:
            # vLLM expects single image or list
            vllm_mm_data["image"] = images[0] if len(images) == 1 else images
            logger.debug(f"Extracted {len(images)} image(s) for multimodal processing")

        # Handle video_url entries (future expansion)
        if VIDEO_URL_KEY in mm_map:
            logger.warning("Video multimodal data not yet supported in standard worker")

        return vllm_mm_data if vllm_mm_data else None

    def _build_prompt_from_request(
        self,
        request: Dict[str, Any],
        request_id: str,
        multi_modal_data: Dict[str, Any] | None,
        log_prefix: str = "",
    ) -> tuple[TokensPrompt | EmbedsPrompt | None, int | None, Dict[str, Any] | None]:
        """
        Build a prompt from request, handling both prompt_embeds and token_ids.

        Args:
            request: The request dict containing either prompt_embeds or token_ids
            request_id: Request ID for logging
            multi_modal_data: Optional multimodal data to attach to TokensPrompt
            log_prefix: Prefix for log messages (e.g., "Prefill " for prefill requests)

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
                {"cached_tokens": request_output.num_cached_tokens}
                if request_output.num_cached_tokens
                else None
            ),
        }

    @staticmethod
    def _extract_logprobs(
        output, num_output_tokens_so_far: int
    ) -> tuple[list[float] | None, list[list[dict]] | None]:
        """
        Extract logprobs from vLLM CompletionOutput for new tokens.

        Args:
            output: vLLM CompletionOutput object
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
                log_probs, top_logprobs = self._extract_logprobs(
                    output, num_output_tokens_so_far
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
        component,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        config=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
    ):
        super().__init__(
            runtime,
            component,
            engine,
            default_sampling_params,
            model_max_len,
            enable_multimodal,
            generate_endpoint,
            config,
            use_vllm_tokenizer,
            shutdown_event,
            enable_frontend_decoding,
        )

    async def generate(self, request, context):
        # Use context ID for request tracking and correlation
        request_id = context.id()
        logger.debug(f"Decode Request ID: {request_id}")

        if self.use_vllm_tokenizer:
            # Text-in-text-out mode: use InputParamManager and OpenAI-compatible format
            async for chunk in self._generate_text_mode(request, context, request_id):
                yield chunk
        else:
            # Token-in-token-out mode: internal protocol format
            async for chunk in self._generate_token_mode(request, context, request_id):
                yield chunk

    async def _generate_token_mode(self, request, context, request_id):
        """Generate tokens using internal protocol format (token-in-token-out)."""
        # Extract and decode multimodal data if present
        multi_modal_data = await self._extract_multimodal_data(request)

        # Build prompt from request (handles both prompt_embeds and token_ids)
        prompt, embedding_sequence_length, error = self._build_prompt_from_request(
            request, request_id, multi_modal_data
        )
        if error is not None:
            yield error
            return

        # Build sampling params from request
        sampling_params = build_sampling_params(
            request, self.default_sampling_params, self.model_max_len
        )

        prefill_result = request.get("prefill_result")
        if prefill_result and isinstance(prefill_result, dict):
            kv_params = prefill_result.get("disaggregated_params", {}).get(
                "kv_transfer_params"
            )
        else:
            kv_params = None

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
        # Check if model name matches a loaded LoRA adapter
        lora_request = None
        model_name = request.get("model")

        if model_name and model_name in self.lora_id_for_name:
            lora_id = self.lora_id_for_name[model_name]
            lora_request = LoRARequest(
                lora_name=model_name,
                lora_int_id=lora_id,
                lora_path=self.lora_name_to_path[model_name],
            )
            logger.info(
                f"Decode request {request_id} will use LoRA adapter: {model_name} (ID: {lora_id})"
            )
        else:
            logger.debug(
                f"Decode request {request_id} has no LoRA specified (model: {model_name})"
            )
        dp_rank = request.get("routing", {}).get("dp_rank")

        trace_headers = build_trace_headers(context)

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

        dp_rank = request.get("routing", {}).get("dp_rank")
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
        component,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        config=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
    ):
        super().__init__(
            runtime,
            component,
            engine,
            default_sampling_params,
            model_max_len,
            enable_multimodal,
            generate_endpoint,
            config,
            use_vllm_tokenizer,
            shutdown_event,
            enable_frontend_decoding,
        )

    async def generate(self, request, context):
        # Use context ID for request tracking and correlation with decode phase
        request_id = context.id()
        logger.debug(f"Prefill Request ID: {request_id}")

        # Token-in-token-out mode: internal protocol format
        async for chunk in self._generate_token_mode(request, context, request_id):
            yield chunk

    async def _generate_token_mode(self, request, context, request_id):
        """Generate prefill using internal protocol format (token-in-token-out)."""
        # Extract and decode multimodal data if present
        multi_modal_data = await self._extract_multimodal_data(request)

        # Build prompt from request (handles both prompt_embeds and token_ids)
        prompt, embedding_sequence_length, error = self._build_prompt_from_request(
            request, request_id, multi_modal_data, log_prefix="Prefill "
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
        # Check if model name matches a loaded LoRA adapter
        lora_request = None
        model_name = request.get("model")

        if model_name and model_name in self.lora_id_for_name:
            lora_id = self.lora_id_for_name[model_name]
            lora_request = LoRARequest(
                lora_name=model_name,
                lora_int_id=lora_id,
                lora_path=self.lora_name_to_path[model_name],
            )
            logger.info(
                f"Prefill request {request_id} will use LoRA adapter: {model_name} (ID: {lora_id}), "
                f"path: {self.lora_name_to_path[model_name]}"
            )
        else:
            logger.debug(
                f"Prefill request {request_id} has no LoRA specified (model: {model_name})"
            )

        dp_rank = request.get("routing", {}).get("dp_rank")

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
                    "disaggregated_params": (
                        {"kv_transfer_params": res.kv_transfer_params}
                        if res.kv_transfer_params
                        else None
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
