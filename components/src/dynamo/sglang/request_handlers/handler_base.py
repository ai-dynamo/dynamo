# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import importlib
import inspect
import json
import logging
import os
import random
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

from dynamo._core import Context
from dynamo.common.utils.input_params import InputParamManager
from dynamo.llm import (
    KvEventPublisher,
    ModelInput,
    ModelType,
    WorkerMetricsPublisher,
    lora_name_to_id,
    register_llm,
    unregister_llm,
)
from dynamo.llm.exceptions import EngineShutdown
from dynamo.runtime import DistributedRuntime
from dynamo.sglang._compat import NetworkAddress, get_local_ip_auto
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher

logger = logging.getLogger(__name__)

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


class SGLangEngineQuiesceController:
    def __init__(self, engine: sgl.Engine):
        self._engine = engine
        self._is_quiesced = False

    @property
    def is_quiesced(self) -> bool:
        return self._is_quiesced

    async def quiesce(self, tags: Optional[list[str]] = None) -> bool:
        if self._is_quiesced:
            return False

        from sglang.srt.managers.io_struct import (
            PauseGenerationReqInput,
            ReleaseMemoryOccupationReqInput,
        )

        await self._engine.tokenizer_manager.pause_generation(PauseGenerationReqInput())
        await self._engine.tokenizer_manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=tags),
            None,
        )
        self._is_quiesced = True
        return True

    async def resume(self, tags: Optional[list[str]] = None) -> bool:
        if not self._is_quiesced:
            return False

        from sglang.srt.managers.io_struct import (
            ContinueGenerationReqInput,
            ResumeMemoryOccupationReqInput,
        )

        await self._engine.tokenizer_manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=tags),
            None,
        )
        await self._engine.tokenizer_manager.continue_generation(
            ContinueGenerationReqInput()
        )
        return True

    def mark_resumed(self) -> None:
        self._is_quiesced = False


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
        self.enable_trace = config.server_args.enable_trace

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


class RLMixin:
    """Mixin providing generic tokenizer_manager passthrough for RL training.

    Requires the host class to have ``self.engine`` with a
    ``tokenizer_manager`` attribute.
    """

    engine: sgl.Engine  # provided by BaseWorkerHandler

    def _resolve_arg(self, arg: Any) -> Any:
        """Resolve a single argument from the generic call body.

        If ``arg`` is a dict with exactly one key starting with ``"io_struct."``,
        treat it as a typed constructor: import the class from
        ``sglang.srt.managers.io_struct`` and construct it with the nested kwargs.
        Otherwise return the value as-is.
        """
        if isinstance(arg, dict) and len(arg) == 1:
            key = next(iter(arg))
            if isinstance(key, str) and key.startswith("io_struct."):
                class_name = key[len("io_struct.") :]
                module = importlib.import_module("sglang.srt.managers.io_struct")
                cls = getattr(module, class_name)
                return cls(**arg[key])
        return arg

    def _normalize_result(self, result: Any) -> dict:
        """Convert a tokenizer_manager method return value to a JSON-safe dict."""
        if result is None:
            return {"status": "ok"}
        if isinstance(result, tuple):
            if len(result) == 2:
                return {"success": result[0], "message": result[1]}
            if len(result) == 3:
                return {
                    "success": result[0],
                    "message": result[1],
                    "num_paused_requests": result[2],
                }
        if isinstance(result, list):
            return {
                "result": [
                    dataclasses.asdict(item)
                    if dataclasses.is_dataclass(item) and not isinstance(item, type)
                    else item
                    for item in result
                ]
            }
        if dataclasses.is_dataclass(result) and not isinstance(result, type):
            return dataclasses.asdict(result)
        if isinstance(result, dict):
            return result
        if isinstance(result, (str, int, float, bool)):
            return {"result": result}
        return {"result": str(result)}

    async def call_tokenizer_manager(self, body: dict) -> dict:
        """Generic passthrough to any tokenizer_manager method.

        Body format::

            {
                "method": "method_name",
                "args": [arg1, arg2, ...],
                "kwargs": {"key": value, ...}
            }

        Each element in args/kwargs is either a plain value or a typed
        constructor ``{"io_struct.ClassName": {kwargs}}``.
        """
        method_name = body["method"]
        raw_args = body.get("args", [])
        raw_kwargs = body.get("kwargs", {})

        args = [self._resolve_arg(a) for a in raw_args]
        kwargs = {k: self._resolve_arg(v) for k, v in raw_kwargs.items()}

        tm = self.engine.tokenizer_manager
        # Ensure the handle_loop task is running so communicator responses
        # are received.  Several tokenizer_manager methods call this
        # internally, but not all of them (e.g. flush_cache does not).
        if hasattr(tm, "auto_create_handle_loop"):
            tm.auto_create_handle_loop()

        method = getattr(tm, method_name)
        result = await method(*args, **kwargs)
        return self._normalize_result(result)

    def register_rl_engine_routes(self, runtime) -> None:
        """Register RL-specific engine routes.

        Args:
            runtime: The DistributedRuntime instance to register routes on.
        """
        runtime.register_engine_route(
            "call_tokenizer_manager", self.call_tokenizer_manager
        )


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

            logger.info(f"Downloading LoRA adapter: {lora_name} from {lora_uri}")
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
                from sglang.srt.managers.io_struct import LoadLoRAAdapterReqInput

                load_req = LoadLoRAAdapterReqInput(
                    lora_name=lora_name,
                    lora_path=lora_path,
                )
                load_result = await self.engine.tokenizer_manager.load_lora_adapter(
                    load_req
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

                    await register_llm(
                        model_input=ModelInput.Tokens,
                        model_type=ModelType.Chat | ModelType.Completions,
                        endpoint=self.generate_endpoint,
                        model_path=self.config.server_args.model_path,
                        kv_cache_block_size=self.config.server_args.page_size,
                        user_data=user_data,
                        lora_name=lora_name,
                        base_model_path=self.config.server_args.model_path,
                    )
                    logger.info(
                        f"Successfully published LoRA '{lora_name}' ModelDeploymentCard"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to publish LoRA {lora_name} ModelDeploymentCard: {e}"
                    )
                    # Rollback: remove the LoRA from the engine
                    try:
                        from sglang.srt.managers.io_struct import (
                            UnloadLoRAAdapterReqInput,
                        )

                        rollback_req = UnloadLoRAAdapterReqInput(lora_name=lora_name)
                        await self.engine.tokenizer_manager.unload_lora_adapter(
                            rollback_req
                        )
                        self.lora_id_for_name.pop(lora_name, None)
                        self.lora_name_to_path.pop(lora_name, None)
                    except Exception as rollback_error:
                        logger.error(
                            f"Failed to rollback LoRA {lora_name}: {rollback_error}"
                        )

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
                from sglang.srt.managers.io_struct import UnloadLoRAAdapterReqInput

                unload_req = UnloadLoRAAdapterReqInput(lora_name=lora_name)
                unload_result = await self.engine.tokenizer_manager.unload_lora_adapter(
                    unload_req
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
                    logger.error(
                        f"Failed to unregister LoRA {lora_name} ModelDeploymentCard: {e}"
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
                    except Exception as rollback_error:
                        logger.error(
                            f"Failed to rollback LoRA {lora_name}: {rollback_error}"
                        )

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


class BaseWorkerHandler(LoraMixin, RLMixin, BaseGenerativeHandler[RequestT, ResponseT]):
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
        self.enable_trace = config.server_args.enable_trace

        if engine is not None:
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
        self._quiesce_controller = (
            SGLangEngineQuiesceController(engine) if engine is not None else None
        )
        self._quiesce_lock = asyncio.Lock()

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
        if self._quiesce_controller is None:
            return {
                "status": "error",
                "message": "memory control not supported on this worker",
            }

        body = body or {}
        tags = body.get("tags")
        async with self._quiesce_lock:
            if self._quiesce_controller.is_quiesced:
                return {
                    "status": "ok",
                    "message": "Memory already released",
                }

            try:
                # Stop new requests and drain in-flight work before releasing memory.
                if self.generate_endpoint is not None:
                    await self.generate_endpoint.unregister_endpoint_instance()

                await self._quiesce_controller.quiesce(tags)

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
        if self._quiesce_controller is None:
            return {
                "status": "error",
                "message": "memory control not supported on this worker",
            }

        body = body or {}
        tags = body.get("tags")
        async with self._quiesce_lock:
            if not self._quiesce_controller.is_quiesced:
                return {
                    "status": "ok",
                    "message": "Memory already resumed",
                }

            try:
                await self._quiesce_controller.resume(tags)

                if self.generate_endpoint is not None:
                    await self.generate_endpoint.register_endpoint_instance()
                self._quiesce_controller.mark_resumed()

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

    async def start_profile(self, body: dict) -> dict:
        """Start profiling on the engine.

        Args:
            body: Dict with profiling parameters passed to start_profile.
        """
        await self.engine.tokenizer_manager.start_profile(**body)
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

    async def open_session(self, body: dict) -> dict:
        """Open a streaming session for subagent KV isolation.

        Args:
            body: Dict with "session_id", optional "timeout" (default 120),
                  and optional "capacity_of_str_len" (default 65536).
        """
        from sglang.srt.managers.io_struct import OpenSessionReqInput

        session_id = body.get("session_id")
        if not session_id:
            return {"status": "error", "message": "session_id required"}
        timeout = body.get("timeout", 120)
        capacity = body.get("capacity_of_str_len", 65536)
        try:
            obj = OpenSessionReqInput(
                capacity_of_str_len=capacity,
                session_id=session_id,
                streaming=True,
                timeout=float(timeout),
            )
            result = await self.engine.tokenizer_manager.open_session(obj, None)
            if result is None:
                return {
                    "status": "ok",
                    "session_id": session_id,
                    "message": "Session already exists",
                }
            return {"status": "ok", "session_id": result}
        except Exception as e:
            logging.error(f"Failed to open session {session_id}: {e}")
            return {"status": "error", "message": str(e)}

    async def close_session(self, body: dict) -> dict:
        """Close a streaming session and release its KV resources.

        Args:
            body: Dict with "session_id".
        """
        from sglang.srt.managers.io_struct import CloseSessionReqInput

        session_id = body.get("session_id")
        if not session_id:
            return {"status": "error", "message": "session_id required"}
        try:
            obj = CloseSessionReqInput(session_id=session_id)
            await self.engine.tokenizer_manager.close_session(obj, None)
            return {"status": "ok", "session_id": session_id}
        except Exception as e:
            logging.error(f"Failed to close session {session_id}: {e}")
            return {"status": "error", "message": str(e)}

    async def session_control(self, request, context=None):
        """Service mesh endpoint for session lifecycle operations.

        Args:
            request: Dict with "action" key ("open_session" or "close_session")
                     and action-specific parameters.
            context: Optional Dynamo context (unused but required by protocol).

        Yields:
            Single dict with operation result.
        """
        action = request.get("action")
        if action == "open_session":
            result = await self.open_session(request)
        elif action == "close_session":
            result = await self.close_session(request)
        else:
            result = {"status": "error", "message": f"Unknown action: {action}"}
        yield result

    def register_engine_routes(self, runtime: DistributedRuntime) -> None:
        """Register all engine routes for this handler.

        Args:
            runtime: The DistributedRuntime instance to register routes on.
        """
        runtime.register_engine_route("start_profile", self.start_profile)
        runtime.register_engine_route("stop_profile", self.stop_profile)
        runtime.register_engine_route(
            "release_memory_occupation", self.release_memory_occupation
        )
        runtime.register_engine_route(
            "resume_memory_occupation", self.resume_memory_occupation
        )
        runtime.register_engine_route(
            "update_weights_from_disk", self.update_weights_from_disk
        )
        runtime.register_engine_route(
            "update_weights_from_tensor", self.update_weights_from_tensor
        )
        runtime.register_engine_route(
            "update_weights_from_distributed", self.update_weights_from_distributed
        )
        runtime.register_engine_route(
            "update_weights_from_ipc", self.update_weights_from_ipc
        )
        runtime.register_engine_route(
            "update_weight_version", self.update_weight_version
        )
        if getattr(self.config, "dynamo_args", None) and getattr(
            self.config.dynamo_args, "enable_rl", False
        ):
            self.register_rl_engine_routes(runtime)

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

        return {
            "prompt" if isinstance(request_input, str) else "input_ids": request_input
        }

    def _session_kwargs(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not getattr(self.config.server_args, "enable_streaming_session", False):
            return {}
        routing = request.get("routing") or {}
        session_control = routing.get("session_control") or {}
        session_id = session_control.get("session_id")
        if not session_id:
            return {}

        # Streaming sessions only need the session identifier on each turn.
        return {"session_params": {"id": session_id}}

    @staticmethod
    def _get_guided_decoding_params(
        guided_decoding: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract guided decoding params (e.g. json_schema) for SGLang sampling_params."""
        if isinstance(guided_decoding, dict):
            json_schema = guided_decoding.get("json")
            if json_schema is not None:
                return {"json_schema": json.dumps(json_schema)}
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
