# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import logging
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
from dynamo.llm import KvEventPublisher, WorkerMetricsPublisher
from dynamo.runtime import DistributedRuntime
from dynamo.sglang._compat import NetworkAddress, get_local_ip_auto
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher


class SGLangEngineQuiesceController:
    def __init__(self, engine: sgl.Engine):
        self._engine = engine
        self._quiesced_tags: Optional[list[str]] = None
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
        self._quiesced_tags = None if tags is None else list(tags)
        self._is_quiesced = True
        return True

    async def resume(self, tags: Optional[list[str]] = None) -> bool:
        if not self._is_quiesced:
            return False

        from sglang.srt.managers.io_struct import (
            ContinueGenerationReqInput,
            ResumeMemoryOccupationReqInput,
        )

        request_tags = self._quiesced_tags if tags is None else list(tags)
        await self._engine.tokenizer_manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=request_tags),
            None,
        )
        await self._engine.tokenizer_manager.continue_generation(
            ContinueGenerationReqInput()
        )
        return True

    def mark_resumed(self) -> None:
        self._quiesced_tags = None
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

    def _get_trace_header(self, context: Context) -> Optional[Dict[str, str]]:
        """Get trace header dict for passing to generation functions.

        Args:
            context: Dynamo Context object containing trace information.

        Returns:
            Dict with traceparent header if trace context available, None otherwise.
        """
        trace_id = context.trace_id
        span_id = context.span_id
        if not trace_id or not span_id:
            return None
        return {"traceparent": f"00-{trace_id}-{span_id}-01"}


class BaseWorkerHandler(BaseGenerativeHandler[RequestT, ResponseT]):
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
        self.skip_tokenizer_init = config.server_args.skip_tokenizer_init
        self.enable_trace = config.server_args.enable_trace

        if engine is not None:
            self.input_param_manager = InputParamManager(
                self.engine.tokenizer_manager.tokenizer
                if not self.skip_tokenizer_init
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

    async def pin_prefix(self, body: dict) -> dict:
        """Pin a prefix by token_ids to resist eviction.

        Args:
            body: Dict with "token_ids" list of token IDs and optional
                  "ttl_seconds" (default 300).
        """
        token_ids = body.get("token_ids", [])
        ttl_seconds = body.get("ttl_seconds", 300)
        if not token_ids:
            return {"status": "error", "message": "token_ids required"}
        try:
            result = await self.engine.tokenizer_manager.pin_prefix(
                token_ids, ttl_seconds
            )
            return {
                "status": "ok" if result.success else "error",
                "nodes_pinned": result.nodes_pinned,
                "message": result.message,
            }
        except Exception as e:
            logging.error(f"Failed to pin prefix: {e}")
            return {"status": "error", "message": str(e)}

    async def cache_control(self, request, context=None):
        """Service mesh endpoint for cache control operations.

        Args:
            request: Dict with "action" key and action-specific parameters.
            context: Optional Dynamo context (unused but required by protocol).

        Yields:
            Single dict with operation result.
        """
        action = request.get("action")
        if action == "pin_prefix":
            result = await self.pin_prefix(request)
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
        runtime.register_engine_route("pin_prefix", self.pin_prefix)
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
            request, use_tokenizer=not self.skip_tokenizer_init
        )

        return {
            "prompt" if isinstance(request_input, str) else "input_ids": request_input
        }

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
            GeneratorExit: If shutdown event was triggered.
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

            # Check which event triggered and raise GeneratorExit if shutdown
            if shutdown_task and shutdown_task in done:
                raise GeneratorExit("Engine was shut down during token generation")

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

        If shutdown event was triggered, raises GeneratorExit on exit.

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
