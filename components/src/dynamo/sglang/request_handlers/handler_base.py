# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import json
import logging
import random
import threading
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Generic,
    Optional,
    Tuple,
    TypeVar,
)

import sglang as sgl
from sglang.srt.managers.io_struct import ProfileReq
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

from dynamo._core import Context
from dynamo.common.constants import DisaggregationMode
from dynamo.common.lora.manager import get_lora_manager
from dynamo.common.model_taints import MODEL_TAINT_ROUTE, register_model_taint_route
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.guided_json import reject_nonprogressing_guided_json_ref_cycles
from dynamo.common.utils.input_params import InputParamManager
from dynamo.common.utils.structural_tag import serialize_structural_tag
from dynamo.llm import (
    HttpError,
    KvEventPublisher,
    ModelInput,
    ModelType,
    WorkerMetricsPublisher,
    WorkerType,
    lora_name_to_id,
    register_llm,
    unregister_llm,
)
from dynamo.llm.exceptions import EngineShutdown
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.capacity import kv_event_block_size
from dynamo.sglang.engine_routes import resolve_configured_engine_routes
from dynamo.sglang.pause import SGLangEnginePauseController
from dynamo.sglang.publisher import DynamoSglangPublisher

logger = logging.getLogger(__name__)


RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class BaseGenerativeHandler(ABC, Generic[RequestT, ResponseT]):
    """Minimal base class for all generative handlers (LLM, diffusion, etc.).

    Provides common infrastructure for:
    - Component and configuration management
    - Metrics and KV event publishing
    - Distributed tracing integration
    """

    def __init__(
        self,
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
    ) -> None:
        """Initialize base generative handler.

        Args:
            config: SGLang and Dynamo configuration.
            publisher: Optional metrics publisher for the worker.
        """
        self.config = config
        self.enable_trace = getattr(config.server_args, "enable_trace", False)

        # Set up metrics and KV publishers
        self.metrics_publisher: Optional[WorkerMetricsPublisher] = None
        self.kv_publisher: Optional[KvEventPublisher] = None
        if publisher is not None:
            self.metrics_publisher = publisher.metrics_publisher
            self.kv_publisher = publisher.kv_publisher

    @abstractmethod
    def generate(self, request: RequestT, context: Context) -> AsyncIterator[ResponseT]:
        """Generate response from request.

        Args:
            request: Request with input and parameters.
            context: Context object for cancellation handling.

        Yields:
            Response data (format varies by handler implementation).
        """
        ...

    def cleanup(self) -> None:
        """Cleanup resources. Override in subclasses as needed."""
        pass


class LoraMixin:
    """Mixin providing LoRA adapter load/unload/list management.

    Requires the host class to have ``self.engine``, ``self.config``,
    and ``self.generate_endpoint``.
    """

    engine: sgl.Engine  # provided by BaseWorkerHandler
    config: Config
    generate_endpoint: Any

    def _init_lora_tracking(self) -> None:
        """Initialize LoRA tracking state. Call from host __init__."""
        self.lora_id_for_name: dict[str, int] = {}
        self.lora_name_to_path: dict[str, str] = {}
        # Per-LoRA locks to prevent concurrent load/unload for the same adapter.
        # Matches the vLLM pattern (handlers.py) for cross-backend consistency.
        self._lora_load_locks: dict[str, asyncio.Lock] = {}
        self._lora_load_locks_guard = threading.Lock()

    def _get_lora_lock(self, lora_name: str) -> asyncio.Lock:
        """Get/create the per-LoRA lock without eagerly allocating a new lock each call."""
        with self._lora_load_locks_guard:
            lock = self._lora_load_locks.get(lora_name)
            if lock is None:
                lock = asyncio.Lock()
                self._lora_load_locks[lora_name] = lock
            return lock

    def _resolve_lora(self, request: Dict[str, Any]) -> Optional[str]:
        """Return the LoRA name to pass as ``lora_path`` to SGLang, or *None*.

        SGLang's lora_registry and lora_ref_cache are keyed by lora_name,
        so we pass the name (not the filesystem path) as lora_path.
        """
        model_name = request.get("model")
        if model_name and model_name in self.lora_id_for_name:
            return model_name
        return None

    async def load_lora(self, request: Optional[Dict[str, Any]] = None):
        """
        Load a LoRA adapter dynamically into the SGLang engine.

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
                    # Idempotency check after acquiring lock — another concurrent
                    # request may have loaded this LoRA while we waited.
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

                    # Generate deterministic ID from lora_name
                    lora_id = lora_name_to_id(lora_name)

                    # Add the LoRA to the SGLang engine via tokenizer_manager
                    if (
                        hasattr(self.engine, "tokenizer_manager")
                        and self.engine.tokenizer_manager
                    ):
                        from sglang.srt.managers.io_struct import (
                            LoadLoRAAdapterReqInput,
                        )

                        load_req = LoadLoRAAdapterReqInput(
                            lora_name=lora_name,
                            lora_path=lora_path,
                        )
                        load_result = (
                            await self.engine.tokenizer_manager.load_lora_adapter(
                                load_req
                            )
                        )
                        if not load_result.success:
                            yield {
                                "status": "error",
                                "message": f"SGLang failed to load LoRA adapter '{lora_name}': {load_result.error_message}",
                            }
                            return
                    else:
                        yield {
                            "status": "error",
                            "message": "SGLang engine does not support LoRA loading (tokenizer_manager not available)",
                        }
                        return

                    # Track the LoRA
                    self.lora_id_for_name[lora_name] = lora_id
                    self.lora_name_to_path[lora_name] = lora_path
                    logger.info(
                        f"Successfully loaded LoRA adapter: {lora_name} with ID {lora_id}"
                    )

                    # Publish LoRA as a ModelDeploymentCard
                    if self.generate_endpoint is not None and self.config is not None:
                        try:
                            user_data = {
                                "lora_adapter": True,
                                "lora_id": lora_id,
                            }

                            # Match the base-model registration topology so the
                            # prefill router activates for the LoRA model name
                            # the same way it does for the base model. The prefill
                            # role is carried by `worker_type=Prefill`; we register
                            # the legacy `ModelType.Prefill` marker bit (not a
                            # surface) so an old frontend still detects it during
                            # the cross-version rollout. Non-prefill workers honor
                            # --endpoint-types so the LoRA is exposed on the same
                            # endpoints as the base model.
                            if self.config.serving_mode == DisaggregationMode.PREFILL:
                                lora_model_type = ModelType.Prefill
                                lora_worker_type = WorkerType.Prefill
                                lora_needs: list[list[WorkerType]] = [
                                    [WorkerType.Decode]
                                ]
                            else:
                                lora_model_type = parse_endpoint_types(
                                    self.config.dynamo_args.endpoint_types
                                )
                                if (
                                    self.config.serving_mode
                                    == DisaggregationMode.DECODE
                                ):
                                    lora_worker_type = WorkerType.Decode
                                    lora_needs = [[WorkerType.Prefill]]
                                else:
                                    lora_worker_type = WorkerType.Aggregated
                                    lora_needs = []

                            # Reuse the base-model metadata builder so LoRA
                            # cards advertise the same token-overflow policy,
                            # parser configuration, and routing capabilities.
                            # Lazy import: static test collection lacks parts of SGLang.
                            from dynamo.sglang.register import get_runtime_config

                            runtime_config = await get_runtime_config(
                                self.engine,
                                self.config.server_args,
                                self.config.dynamo_args,
                            )
                            await register_llm(
                                model_input=ModelInput.Tokens,
                                model_type=lora_model_type,
                                endpoint=self.generate_endpoint,
                                model_path=self.config.server_args.model_path,
                                kv_cache_block_size=kv_event_block_size(
                                    self.config.server_args
                                ),
                                user_data=user_data,
                                lora_name=lora_name,
                                base_model_path=self.config.server_args.model_path,
                                worker_type=lora_worker_type,
                                needs=lora_needs,
                                runtime_config=runtime_config,
                                # Publish the worker's per-worker LoRA slot budget so the frontend
                                # allocator sizes placement against real capacity instead of the
                                # hard-coded default.
                                max_gpu_lora_count=getattr(
                                    self.config.server_args, "max_loras_per_batch", None
                                ),
                            )
                            logger.info(
                                f"Successfully published LoRA '{lora_name}' ModelDeploymentCard"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Failed to publish LoRA {lora_name} ModelDeploymentCard"
                            )
                            # Rollback: remove the LoRA from the engine
                            try:
                                from sglang.srt.managers.io_struct import (
                                    UnloadLoRAAdapterReqInput,
                                )

                                rollback_req = UnloadLoRAAdapterReqInput(
                                    lora_name=lora_name
                                )
                                await self.engine.tokenizer_manager.unload_lora_adapter(
                                    rollback_req
                                )
                                self.lora_id_for_name.pop(lora_name, None)
                                self.lora_name_to_path.pop(lora_name, None)
                            except Exception:
                                logger.exception(f"Failed to rollback LoRA {lora_name}")

                            yield {
                                "status": "error",
                                "message": f"Failed to register LoRA '{lora_name}' in discovery registry: {str(e)}",
                                "lora_name": lora_name,
                            }
                            return

                    yield {
                        "status": "success",
                        "message": f"LoRA adapter '{lora_name}' loaded successfully",
                        "lora_name": lora_name,
                        "lora_id": lora_id,
                    }
                finally:
                    # Avoid lock-map growth on failed loads: if this attempt did
                    # not leave the LoRA loaded, remove the lock entry.
                    with self._lora_load_locks_guard:
                        if (
                            lora_name not in self.lora_id_for_name
                            and self._lora_load_locks.get(lora_name) is lock
                        ):
                            self._lora_load_locks.pop(lora_name, None)
        except Exception as e:
            logger.exception("Failed to load LoRA adapter")
            yield {"status": "error", "message": str(e)}

    async def unload_lora(self, request: Optional[Dict[str, Any]] = None):
        """
        Unload a LoRA adapter dynamically from the SGLang engine.

        Request format:
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
                    # Check after acquiring lock — a concurrent unload may have
                    # already removed this LoRA while we waited.
                    if lora_name not in self.lora_id_for_name:
                        yield {
                            "status": "error",
                            "message": f"LoRA adapter '{lora_name}' not found. Available LoRAs: {list(self.lora_id_for_name.keys())}",
                        }
                        return

                    lora_id = self.lora_id_for_name[lora_name]
                    lora_path = self.lora_name_to_path.get(lora_name)

                    # Unload from SGLang engine
                    if (
                        hasattr(self.engine, "tokenizer_manager")
                        and self.engine.tokenizer_manager
                    ):
                        from sglang.srt.managers.io_struct import (
                            UnloadLoRAAdapterReqInput,
                        )

                        unload_req = UnloadLoRAAdapterReqInput(lora_name=lora_name)
                        unload_result = (
                            await self.engine.tokenizer_manager.unload_lora_adapter(
                                unload_req
                            )
                        )
                        if not unload_result.success:
                            yield {
                                "status": "error",
                                "message": f"SGLang failed to unload LoRA adapter '{lora_name}': {unload_result.error_message}",
                            }
                            return
                    else:
                        yield {
                            "status": "error",
                            "message": "SGLang engine does not support LoRA unloading (tokenizer_manager not available)",
                        }
                        return

                    # Remove from tracking
                    del self.lora_id_for_name[lora_name]
                    self.lora_name_to_path.pop(lora_name, None)

                    # Unregister from discovery
                    if self.generate_endpoint is not None:
                        try:
                            await unregister_llm(
                                endpoint=self.generate_endpoint,
                                lora_name=lora_name,
                            )
                            logger.info(
                                f"Successfully unregistered LoRA '{lora_name}' ModelDeploymentCard"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Failed to unregister LoRA {lora_name} ModelDeploymentCard"
                            )
                            # Rollback: re-add the LoRA to engine
                            try:
                                from sglang.srt.managers.io_struct import (
                                    LoadLoRAAdapterReqInput,
                                )

                                rollback_req = LoadLoRAAdapterReqInput(
                                    lora_name=lora_name,
                                    lora_path=lora_path,
                                )
                                await self.engine.tokenizer_manager.load_lora_adapter(
                                    rollback_req
                                )
                                self.lora_id_for_name[lora_name] = lora_id
                                if lora_path:
                                    self.lora_name_to_path[lora_name] = lora_path
                            except Exception:
                                logger.exception(f"Failed to rollback LoRA {lora_name}")

                            yield {
                                "status": "error",
                                "message": f"Failed to unregister LoRA '{lora_name}' from discovery registry: {str(e)}",
                                "lora_name": lora_name,
                            }
                            return

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
            logger.exception("Failed to unload LoRA adapter")
            yield {"status": "error", "message": str(e)}

    async def list_loras(self, _request: Optional[Dict[str, Any]] = None):
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
            logger.exception("Failed to list LoRA adapters")
            yield {"status": "error", "message": str(e)}


class BaseWorkerHandler(LoraMixin, BaseGenerativeHandler[RequestT, ResponseT]):
    """Abstract base class for SGLang LLM worker handlers.

    Extends BaseGenerativeHandler with LLM-specific functionality:
    - SGLang Engine integration
    - Tokenization and input parameter management
    - Disaggregated serving support
    """

    def __init__(
        self,
        engine: sgl.Engine,
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
        generate_endpoint=None,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Initialize base worker handler.

        Args:
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: Optional metrics publisher for the worker.
            generate_endpoint: The endpoint handle for discovery registration.
            shutdown_event: Optional event to signal shutdown.
        """
        # Call parent constructor
        super().__init__(config, publisher)

        # LLM-specific initialization
        self.engine = engine
        self.config = config
        self.generate_endpoint = generate_endpoint
        self.publisher = publisher
        self.shutdown_event = shutdown_event
        if publisher is not None:
            self.metrics_publisher = publisher.metrics_publisher
            self.kv_publisher = publisher.kv_publisher
        self.serving_mode = config.serving_mode
        self.use_sglang_tokenizer = config.dynamo_args.use_sglang_tokenizer
        self.enable_trace = getattr(config.server_args, "enable_trace", False)
        self._max_input_token_id: Optional[int] = None

        if engine is not None:
            self._max_input_token_id = self._resolve_max_input_token_id(engine)
            self.input_param_manager = InputParamManager(
                self.engine.tokenizer_manager.tokenizer
                if self.use_sglang_tokenizer
                else None
            )
            self._engine_supports_priority = (
                "priority" in inspect.signature(engine.async_generate).parameters
            )
        else:
            # Encode-only workers (e.g. MultimodalEncodeWorkerHandler) don't
            # have an sgl.Engine.
            self.input_param_manager = InputParamManager(None)
            self._engine_supports_priority = False
        self._pause_controller = (
            SGLangEnginePauseController(engine) if engine is not None else None
        )
        self._pause_lock = asyncio.Lock()

        # LoRA tracking (via LoraMixin)
        self._init_lora_tracking()

    def _priority_kwargs(self, priority: Any) -> Dict[str, Any]:
        if priority is not None and self._engine_supports_priority:
            normalized = int(priority)
            if getattr(
                self.config.server_args, "schedule_low_priority_values_first", False
            ):
                normalized = -normalized
            return {"priority": normalized}
        return {}

    async def release_memory_occupation(self, body: dict) -> dict:
        """Release GPU memory occupation and unregister from discovery.

        Args:
            body: Optional dict with "tags" to target specific memory regions.

        Order of operations:
        1. Unregister from discovery - stop accepting new requests
        2. Pause generation - drain in-flight requests
        3. Release memory - safe now that no requests are active
        """
        if self._pause_controller is None:
            return {
                "status": "error",
                "message": "memory control not supported on this worker",
            }

        body = body or {}
        tags = body.get("tags")
        async with self._pause_lock:
            if self._pause_controller.is_paused:
                return {
                    "status": "ok",
                    "message": "Memory already released",
                }
            if self._pause_controller.needs_resume_recovery:
                return {
                    "status": "error",
                    "message": "resume_memory_occupation required before retrying release",
                }

            unregistered = False
            try:
                # Stop new requests and drain in-flight work before releasing memory.
                if self.generate_endpoint is not None:
                    await self.generate_endpoint.unregister_endpoint_instance()
                    unregistered = True

                await self._pause_controller.pause(tags)

                return {
                    "status": "ok",
                    "message": (
                        f"Memory released for tags: {tags}"
                        if tags is not None
                        else "Memory released"
                    ),
                }
            except Exception as e:
                logging.error(f"Failed to release memory occupation: {e}")
                # If pause rolled back cleanly the engine is serving-safe again,
                # but discovery still shows us unregistered and resume will
                # early-return. Re-register so the worker rejoins the routing pool.
                if (
                    unregistered
                    and not self._pause_controller.is_paused
                    and not self._pause_controller.needs_resume_recovery
                    and self.generate_endpoint is not None
                ):
                    try:
                        await self.generate_endpoint.register_endpoint_instance()
                        logging.info(
                            "Re-registered endpoint after failed memory release rollback"
                        )
                    except Exception as reg_err:
                        logging.error(
                            f"Failed to re-register endpoint after release failure: {reg_err}"
                        )
                return {"status": "error", "message": str(e)}

    async def resume_memory_occupation(self, body: dict) -> dict:
        """Resume GPU memory occupation and re-register to discovery.

        Args:
            body: Optional dict with "tags" to target specific memory regions.

        Order of operations:
        1. Resume memory - restore GPU allocations
        2. Continue generation - ready to serve requests
        3. Re-register to discovery - allow frontend to route here
        """
        if self._pause_controller is None:
            return {
                "status": "error",
                "message": "memory control not supported on this worker",
            }

        body = body or {}
        tags = body.get("tags")
        async with self._pause_lock:
            needs_recovery = self._pause_controller.needs_resume_recovery
            if not self._pause_controller.is_paused and not needs_recovery:
                return {
                    "status": "ok",
                    "message": "Memory already resumed",
                }

            try:
                await self._pause_controller.resume(tags)

                if self.generate_endpoint is not None:
                    await self.generate_endpoint.register_endpoint_instance()
                self._pause_controller.mark_resumed()

                return {
                    "status": "ok",
                    "message": (
                        f"Memory resumed for tags: {tags}"
                        if tags is not None
                        else "Memory resumed"
                    ),
                }
            except Exception as e:
                logging.error(f"Failed to resume memory occupation: {e}")
                return {"status": "error", "message": str(e)}

    async def clear_kv_blocks(self, request: Optional[Dict[str, Any]] = None):
        """Flush SGLang's local cache when no requests are active."""
        tokenizer_manager = (
            getattr(self.engine, "tokenizer_manager", None)
            if self.engine is not None
            else None
        )
        if tokenizer_manager is None:
            yield {
                "status": "error",
                "message": "KV cache clear not supported on this worker",
            }
            return

        try:
            async with self._pause_lock:
                if getattr(tokenizer_manager, "rid_to_state", None):
                    yield {
                        "status": "error",
                        "message": "Cannot clear KV cache while requests are active",
                    }
                    return

                if hasattr(tokenizer_manager, "auto_create_handle_loop"):
                    tokenizer_manager.auto_create_handle_loop()
                result = await tokenizer_manager.flush_cache()

                if not result.success:
                    yield {
                        "status": "error",
                        "message": getattr(result, "message", None)
                        or "KV cache clear failed",
                    }
                    return

                backend = tokenizer_manager.server_args.hicache_storage_backend
                if backend and backend != "none":
                    result = await tokenizer_manager.clear_hicache_storage()
                    if not result.success:
                        yield {
                            "status": "error",
                            "message": getattr(result, "message", None)
                            or "External KV cache clear failed",
                        }
                        return

                yield {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            logging.error("Failed to clear KV cache: %s", e)
            yield {"status": "error", "message": str(e)}

    async def start_profile(self, body: dict) -> dict:
        """Start profiling on the engine.

        Args:
            body: Dict with profiling parameters passed to start_profile.
        """
        await self.engine.tokenizer_manager.start_profile(ProfileReq(**body))
        return {"status": "ok", "message": "Profiling started"}

    async def stop_profile(self, body: dict) -> dict:
        """Stop profiling on the engine.

        Args:
            body: Unused, but required for handler signature.
        """
        await self.engine.tokenizer_manager.stop_profile()
        return {"status": "ok", "message": "Profiling stopped"}

    async def update_weights_from_disk(self, body: dict) -> dict:
        """Update model weights from disk without restarting the server."""
        from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput

        req = UpdateWeightFromDiskReqInput(**body)
        (
            success,
            message,
            num_paused_requests,
        ) = await self.engine.tokenizer_manager.update_weights_from_disk(req, None)
        return {
            "success": success,
            "message": message,
            "num_paused_requests": num_paused_requests,
        }

    async def update_weights_from_tensor(self, body: dict) -> dict:
        """Update model weights from tensors without restarting the server."""
        from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

        req = UpdateWeightsFromTensorReqInput(**body)
        (
            success,
            message,
        ) = await self.engine.tokenizer_manager.update_weights_from_tensor(req, None)
        return {"success": success, "message": message}

    async def update_weights_from_distributed(self, body: dict) -> dict:
        """Update model weights using distributed online synchronization."""
        from sglang.srt.managers.io_struct import UpdateWeightsFromDistributedReqInput

        req = UpdateWeightsFromDistributedReqInput(**body)
        (
            success,
            message,
        ) = await self.engine.tokenizer_manager.update_weights_from_distributed(
            req, None
        )
        return {"success": success, "message": message}

    async def update_weights_from_ipc(self, body: dict) -> dict:
        """Update model weights from IPC for checkpoint-engine integration."""
        from sglang.srt.managers.io_struct import UpdateWeightsFromIPCReqInput

        req = UpdateWeightsFromIPCReqInput(**body)
        success, message = await self.engine.tokenizer_manager.update_weights_from_ipc(
            req, None
        )
        if success and not self.engine.tokenizer_manager.initial_weights_loaded:
            self.engine.tokenizer_manager.initial_weights_loaded = True
        return {"success": success, "message": message}

    async def update_weight_version(self, body: dict) -> dict:
        """Update the active weight version without changing model weights."""
        from sglang.srt.managers.io_struct import UpdateWeightVersionReqInput

        req = UpdateWeightVersionReqInput(**body)
        if req.abort_all_requests:
            self.engine.tokenizer_manager.abort_request(abort_all=True)

        self.engine.tokenizer_manager.server_args.weight_version = req.new_version
        return {
            "success": True,
            "message": f"Weight version updated to {req.new_version}",
            "new_version": req.new_version,
        }

    def register_engine_routes(self, runtime: DistributedRuntime) -> None:
        """Register all engine routes for this handler.

        Args:
            runtime: The DistributedRuntime instance to register routes on.
        """
        configured_routes = resolve_configured_engine_routes(
            self.engine,
            self.config.dynamo_args.engine_routes,
        )
        built_in_routes = {
            "control/start_profile": self.start_profile,
            "control/stop_profile": self.stop_profile,
            "control/release_memory_occupation": self.release_memory_occupation,
            "control/resume_memory_occupation": self.resume_memory_occupation,
            "control/update_weights_from_disk": self.update_weights_from_disk,
            "control/update_weights_from_tensor": self.update_weights_from_tensor,
            "control/update_weights_from_distributed": (
                self.update_weights_from_distributed
            ),
            "control/update_weights_from_ipc": self.update_weights_from_ipc,
            "control/update_weight_version": self.update_weight_version,
        }
        reserved_routes = {*built_in_routes, MODEL_TAINT_ROUTE}
        for path, _ in configured_routes:
            if path in reserved_routes:
                raise ValueError(
                    f"Configured SGLang engine route /engine/{path} collides "
                    "with a built-in route"
                )

        register_model_taint_route(runtime, self.generate_endpoint)
        for path, handler in built_in_routes.items():
            runtime.register_engine_route(path, handler)
        for path, configured_handler in configured_routes:
            runtime.register_engine_route(path, configured_handler)

    @abstractmethod
    def generate(self, request: RequestT, context: Context) -> AsyncIterator[ResponseT]:
        """Generate response from request.

        Args:
            request: Request with input and parameters.
            context: Context object for cancellation handling.

        Yields:
            Response data (format varies by handler implementation).
        """
        ...

    def cleanup(self) -> None:
        """Cleanup resources. Override in subclasses as needed."""
        if self.publisher is not None:
            self.publisher.cleanup()

    def _get_input_param(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request_input = self.input_param_manager.get_input_param(
            request, use_tokenizer=self.use_sglang_tokenizer
        )
        self._validate_nvext_token_data(request, request_input)

        return {
            "prompt" if isinstance(request_input, str) else "input_ids": request_input
        }

    @staticmethod
    def _resolve_max_input_token_id(engine: sgl.Engine) -> Optional[int]:
        """Resolve the largest token ID accepted by the model embedding table."""
        tokenizer_manager = getattr(engine, "tokenizer_manager", None)
        model_config = getattr(tokenizer_manager, "model_config", None)
        return BaseWorkerHandler._resolve_max_input_token_id_from_model_config(
            model_config
        )

    @staticmethod
    def _resolve_max_input_token_id_from_model_config(
        model_config: Any,
    ) -> Optional[int]:
        model_vocab_size: object = getattr(model_config, "vocab_size", None)

        # Compatibility fallback for SGLang model configs that expose the
        # Hugging Face text config but not the derived vocab_size attribute.
        if model_vocab_size is None:
            hf_text_config = getattr(model_config, "hf_text_config", None)
            model_vocab_size = getattr(hf_text_config, "vocab_size", None)

        if (
            isinstance(model_vocab_size, bool)
            or not isinstance(model_vocab_size, int)
            or model_vocab_size <= 0
        ):
            return None
        return model_vocab_size - 1

    def _resolve_request_multimodal_token_ids(
        self, request: Dict[str, Any]
    ) -> frozenset[int]:
        mm_data = request.get("multi_modal_data")
        if not isinstance(mm_data, dict):
            return frozenset()

        tokenizer_manager = getattr(self.engine, "tokenizer_manager", None)
        mm_processor = getattr(tokenizer_manager, "mm_processor", None)
        mm_tokens = getattr(mm_processor, "mm_tokens", None)
        token_ids = set()

        if mm_tokens is not None:
            for modality in ("image", "video", "audio"):
                if not mm_data.get(f"{modality}_url"):
                    continue
                token_id = getattr(mm_tokens, f"{modality}_token_id", None)
                if isinstance(token_id, int) and not isinstance(token_id, bool):
                    token_ids.add(token_id)

        # Some processors, including LLaVA's wrapper, expose only the image
        # token on ModelConfig. LLaVA also represents video frames as images.
        if mm_data.get("image_url") or mm_data.get("video_url"):
            model_config = getattr(tokenizer_manager, "model_config", None)
            image_token_id = getattr(model_config, "image_token_id", None)
            if isinstance(image_token_id, int) and not isinstance(image_token_id, bool):
                token_ids.add(image_token_id)

        return frozenset(token_ids)

    def _validate_token_ids(
        self,
        token_ids: Any,
        allowed_oov_ids: frozenset[int] = frozenset(),
    ) -> None:
        if not isinstance(token_ids, list):
            raise HttpError(400, "nvext.token_data must resolve to a token ID list")

        max_input_token_id = self._max_input_token_id
        for index, token_id in enumerate(token_ids):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise HttpError(
                    400,
                    f"nvext.token_data[{index}] must be an integer token ID",
                )
            # Dynamo's Rust frontend uses u32 token IDs, so negatives are not expected.
            if (
                max_input_token_id is not None and token_id > max_input_token_id
            ) and token_id not in allowed_oov_ids:
                raise HttpError(400, f"Token id {token_id} is out of vocabulary")

    def _validate_nvext_token_data(
        self,
        request: Dict[str, Any],
        token_ids: Any,
    ) -> None:
        """Reject out-of-vocabulary IDs supplied through ``nvext.token_data``."""
        extra_args = request.get("extra_args")
        if not isinstance(extra_args, dict):
            return
        nvext = extra_args.get("nvext")
        if not isinstance(nvext, dict) or nvext.get("token_in") is not True:
            return

        self._validate_token_ids(
            token_ids,
            self._resolve_request_multimodal_token_ids(request),
        )

    @staticmethod
    def _get_guided_decoding_params(
        guided_decoding: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract guided decoding params (e.g. json_schema) for SGLang sampling_params."""
        if isinstance(guided_decoding, dict):
            json_schema = guided_decoding.get("json")
            if json_schema is not None:
                reject_nonprogressing_guided_json_ref_cycles(json_schema)
                return {"json_schema": json.dumps(json_schema)}
            structural_tag = guided_decoding.get("structural_tag")
            if structural_tag is not None:
                return {"structural_tag": serialize_structural_tag(structural_tag)}
        return {}

    @staticmethod
    def _generate_bootstrap_room() -> int:
        """Generate a unique bootstrap room ID for disaggregated serving.

        Returns:
            Random 63-bit integer.
        """
        return random.randint(0, 2**63 - 1)

    @staticmethod
    def _get_bootstrap_info(engine: sgl.Engine) -> Tuple[str, int]:
        """Extract bootstrap host and port from SGLang engine.

        Args:
            engine: The SGLang engine instance.

        Returns:
            Tuple of (bootstrap_host, bootstrap_port).
        """
        inner_tm = engine.tokenizer_manager
        bootstrap_port = inner_tm.server_args.disaggregation_bootstrap_port

        if inner_tm.server_args.dist_init_addr:
            dist_init = NetworkAddress.parse(inner_tm.server_args.dist_init_addr)
            bootstrap_host = (
                NetworkAddress(dist_init.resolved().host, bootstrap_port)
                .to_host_port_str()
                .rsplit(":", 1)[0]
            )
        else:
            bootstrap_host = (
                NetworkAddress(get_local_ip_auto(), bootstrap_port)
                .to_host_port_str()
                .rsplit(":", 1)[0]
            )

        return bootstrap_host, bootstrap_port

    async def _handle_cancellation(
        self, request_id_future: asyncio.Future, context: Context
    ):
        """Background task to handle cancellation and shutdown by monitoring both signals.

        Args:
            request_id_future: Future that will be set with the SGLang request ID
                              when the first response arrives.
            context: Context object for cancellation handling.

        Raises:
            EngineShutdown: If shutdown event was triggered.
        """
        cancellation_future: asyncio.Future[Any] | None = None
        shutdown_task: asyncio.Task[Any] | None = None
        try:
            logging.debug(f"Cancellation monitor started for Context: {context.id()}")

            # Always wait for the request ID to ensure we can abort the request
            sglang_request_id = await request_id_future
            logging.debug(
                f"Cancellation monitor received SGLang Request ID {sglang_request_id} for Context: {context.id()}"
            )
            logging.debug(f"Request ID future cancelled for Context: {context.id()}")

            # Get the cancellation future
            cancellation_future = context.async_killed_or_stopped()

            # Build list of futures/tasks to wait for
            wait_for: list[asyncio.Future[Any]] = [cancellation_future]

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

            logging.info(
                f"Cancellation or shutdown signal received for SGLang Request ID {sglang_request_id}, Context: {context.id()}"
            )

            # Call abort_request on the tokenizer_manager through the engine
            if (
                hasattr(self.engine, "tokenizer_manager")
                and self.engine.tokenizer_manager
            ):
                logging.info(
                    f"Calling SGLang abort_request for Request ID {sglang_request_id}"
                )
                self.engine.tokenizer_manager.abort_request(
                    rid=sglang_request_id, abort_all=False
                )
                logging.info(f"Aborted Request ID: {context.id()}")
            else:
                logging.error(
                    f"SGLang tokenizer_manager not found for abort request: {context.id()}"
                )

            # Check which event triggered and raise EngineShutdown if shutdown
            if shutdown_task and shutdown_task in done:
                raise EngineShutdown("Engine was shut down during token generation")

        except asyncio.CancelledError:
            # Task was cancelled, which is expected when generation completes
            request_id = "unknown"
            if request_id_future.done() and not request_id_future.cancelled():
                try:
                    request_id = request_id_future.result()
                except Exception:
                    pass
            logging.debug(
                f"Cancellation monitor task cancelled for SGLang Request ID {request_id}, Context: {context.id()}"
            )
            raise
        finally:
            for awaitable in (cancellation_future, shutdown_task):
                if awaitable is None or awaitable.done():
                    continue
                awaitable.cancel()
                try:
                    await awaitable
                except (asyncio.CancelledError, Exception):
                    pass

    @asynccontextmanager
    async def _cancellation_monitor(
        self, request_id_future: asyncio.Future, context: Context
    ) -> AsyncGenerator[asyncio.Task, None]:
        """
        Context manager for monitoring request cancellation and shutdown.
        Automatically creates a background task to monitor for cancellation and
        shutdown events, cleaning it up when the context exits.

        If shutdown event was triggered, raises EngineShutdown on exit.

        Args:
            request_id_future: Future that will be set with the SGLang request ID
                              when the first response arrives.
            context: Context object for cancellation handling

        Yields:
            asyncio.Task: The cancellation monitoring task being managed
        """
        logging.debug(f"Creating cancellation monitor task for Context: {context.id()}")

        # Start the cancellation monitoring task
        cancellation_task = asyncio.create_task(
            self._handle_cancellation(request_id_future, context)
        )

        try:
            yield cancellation_task
        finally:
            # Clean up the background cancellation task
            request_id = "unknown"
            if request_id_future.done() and not request_id_future.cancelled():
                try:
                    request_id = request_id_future.result()
                except Exception:
                    pass

            if not cancellation_task.done():
                logging.debug(
                    f"Cancelling cancellation monitor task for SGLang Request ID {request_id}, Context: {context.id()}"
                )
                cancellation_task.cancel()
                try:
                    await cancellation_task
                except asyncio.CancelledError:
                    pass
            else:
                cancellation_task.result()
