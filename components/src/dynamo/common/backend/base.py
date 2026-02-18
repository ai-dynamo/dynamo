# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes for Dynamo backend workers.

This module provides the foundational classes for implementing LLM backend workers:
- BackendConfig: Common configuration fields shared across all backends
- BaseBackend: Abstract base class with common worker lifecycle management
"""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from prometheus_client import CollectorRegistry

from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.prometheus import LLMBackendMetrics
from dynamo.common.utils.runtime import create_runtime
from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, register_llm
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)

# Shared Prometheus registry for dynamo_component metrics across all worker backends.
DYNAMO_COMPONENT_REGISTRY = CollectorRegistry()


class BackendConfig(ConfigBase):
    """Base configuration class with common fields shared across all backends.

    This class provides the common configuration fields needed by all LLM backend
    workers. Framework-specific backends should extend this class and add their
    own fields.

    Attributes:
        namespace: Dynamo namespace for the worker.
        component: Dynamo component name.
        endpoint: Dynamo endpoint name.
        model: Path or identifier for the model.
        served_model_name: Name to serve the model under.
        store_kv: Key-value backend type (etcd, file, mem).
        request_plane: Request distribution mechanism (tcp, nats, http).
        event_plane: Event publishing mechanism (nats, zmq).
        use_kv_events: Whether NATS KV events are enabled.
        enable_local_indexer: Whether local indexer is enabled for KV events.
        custom_jinja_template: Path to custom Jinja template for chat.
        endpoint_types: Comma-separated list of endpoint types to enable.
        dump_config_to: Path to dump configuration for debugging.
    """

    # Dynamo hierarchy
    namespace: str
    component: str
    endpoint: str

    # Model identification
    model: str
    served_model_name: Optional[str] = None

    # Runtime configuration
    store_kv: str
    request_plane: str
    event_plane: str
    use_kv_events: bool = False
    enable_local_indexer: bool = True

    # Template and endpoint configuration
    custom_jinja_template: Optional[str] = None
    endpoint_types: str = "chat,completions"

    # Debugging
    dump_config_to: Optional[str] = None

    def get_model_name(self) -> str:
        """Get the effective model name for display and metrics.

        Returns:
            served_model_name if set, otherwise model path/identifier.
        """
        return self.served_model_name or self.model

    def validate(self) -> None:
        """Validate the configuration.

        Subclasses should call super().validate() and add their own validation.
        """
        # Validate custom Jinja template path if provided
        if self.custom_jinja_template:
            expanded_path = os.path.expandvars(
                os.path.expanduser(self.custom_jinja_template)
            )
            if not os.path.isfile(expanded_path):
                raise FileNotFoundError(
                    f"Custom Jinja template file not found: {expanded_path}"
                )
            self.custom_jinja_template = expanded_path

    def get_metrics_labels(self) -> List[Tuple[str, str]]:
        """Get common metrics labels for Prometheus.

        Returns:
            List of (label_name, label_value) tuples.
        """
        from dynamo import prometheus_names

        model_name = self.get_model_name()
        return [
            (prometheus_names.labels.MODEL, model_name),
            (prometheus_names.labels.MODEL_NAME, model_name),
        ]


class BaseBackend(ABC):
    """Abstract base class for LLM backend workers.

    This class provides the common lifecycle management for backend workers
    using the Template Method pattern. The run() method orchestrates the full
    lifecycle, calling hook methods that subclasses can override.

    Lifecycle (run()):
      1. pre_runtime_setup()          - Hook: e.g. pre-runtime configuration
      2. setup_runtime()              - Create DistributedRuntime (skip if pre-provided)
      3. setup_component()            - Create namespace/component/endpoint
      4. engine_context():            - Context manager for engine lifecycle
         5. create_engine()           - ABSTRACT: framework-specific engine creation
         6. setup_metrics()           - Hook: framework-specific metrics setup
         7. is_non_leader_node()      - Hook: check + early exit for non-leader nodes
         8. create_handler()          - ABSTRACT: framework-specific handler creation
         9. setup_kv_publishers()     - Hook: KV event publisher setup
        10. register_engine_routes()  - Hook: engine-specific routes
        11. register_and_serve()      - Register model then serve; overridable
        12. cleanup                   - Cancel metrics_task, handler cleanup, post_serve_cleanup()

    Subclasses must implement abstract methods and can override hooks.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        runtime: Optional[DistributedRuntime] = None,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Initialize the backend.

        Args:
            config: Backend-specific configuration object.
            runtime: Optional pre-created DistributedRuntime. If provided,
                    setup_runtime() will skip runtime creation.
            shutdown_event: Optional pre-created shutdown event. Used when
                          runtime is pre-provided.
        """
        self.config = config
        self.runtime: Optional[DistributedRuntime] = runtime
        self.component = None
        self.endpoint = None
        self.engine = None
        self.handler = None
        self.shutdown_event: Optional[asyncio.Event] = shutdown_event
        self._metrics_task: Optional[asyncio.Task] = None

    @property
    def backend_name(self) -> str:
        """Class name used as prefix in lifecycle log messages."""
        return self.__class__.__name__

    # ── Abstract methods (must be implemented) ──────────────────────

    @abstractmethod
    async def create_engine(self) -> Any:
        """Create and initialize the LLM engine.

        Returns:
            The initialized engine instance.
        """
        pass

    @abstractmethod
    def create_handler(self, engine: Any, component: Any, endpoint: Any) -> Any:
        """Create the request handler for the backend.

        Args:
            engine: The initialized LLM engine.
            component: The Dynamo component.
            endpoint: The Dynamo endpoint.

        Returns:
            Handler instance that implements the generate() method.
        """
        pass

    @abstractmethod
    def get_health_check_payload(self, engine: Any) -> Dict[str, Any]:
        """Get the health check payload for the endpoint.

        Args:
            engine: The initialized LLM engine.

        Returns:
            Dictionary containing health check configuration.
        """
        pass

    # ── Hook methods (override as needed) ───────────────────────────

    def pre_runtime_setup(self) -> None:
        """Hook called before runtime creation.

        Override for pre-runtime setup.
        """
        pass

    def setup_runtime(self) -> None:
        """Create and configure the Dynamo runtime.

        Skips creation if a runtime was provided in the constructor.
        """
        logger.info(f"[{self.backend_name}] Setting up runtime...")
        if self.runtime is not None:
            logger.info(
                f"[{self.backend_name}] Runtime ready (using pre-provided runtime)"
            )
            return

        if self.shutdown_event is None:
            self.shutdown_event = asyncio.Event()

        runtime, loop = create_runtime(
            store_kv=self.config.store_kv,
            request_plane=self.config.request_plane,
            event_plane=self.config.event_plane,
            use_kv_events=self.config.use_kv_events,
            shutdown_event=self.shutdown_event,
        )
        self.runtime = runtime
        logger.info(f"[{self.backend_name}] Runtime ready")

    def setup_component(self) -> Tuple[Any, Any]:
        """Set up the Dynamo component and endpoint.

        Returns:
            Tuple of (component, endpoint).
        """
        logger.info(
            f"[{self.backend_name}] Setting up component "
            f"(namespace={self.config.namespace}, component={self.config.component})..."
        )
        if self.runtime is None:
            raise RuntimeError("Runtime not initialized. Call setup_runtime() first.")

        self.component = self.runtime.namespace(self.config.namespace).component(
            self.config.component
        )
        self.endpoint = self.component.endpoint(self.config.endpoint)
        logger.info(f"[{self.backend_name}] Component and endpoint ready")
        return self.component, self.endpoint

    @asynccontextmanager
    async def engine_context(self):
        """Async context manager for engine lifecycle.

        Default: creates engine via _run_create_engine() and yields it.
        Override for backends that need a context manager (e.g., engines
        requiring async context management).
        """
        engine = await self._run_create_engine()
        try:
            yield engine
        finally:
            pass

    def setup_metrics(self, endpoint: Any) -> Optional[asyncio.Task]:
        """Set up metrics collection for the endpoint.

        Override to add framework-specific metrics setup.

        Args:
            endpoint: The Dynamo endpoint to register metrics for.

        Returns:
            Optional asyncio.Task for background metrics collection.
        """
        return None

    def is_non_leader_node(self) -> bool:
        """Check if this is a non-leader node in multi-node deployments.

        Non-leader nodes typically run engines but don't serve Dynamo endpoints.

        Returns:
            True if this is a non-leader node.
        """
        return False

    async def handle_non_leader_node(self) -> None:
        """Handle non-leader node behavior.

        Called when is_non_leader_node() returns True.
        Default: wait indefinitely (process terminated via signal handlers).
        """
        logger.info("Non-leader node detected. Waiting indefinitely.")
        await asyncio.Event().wait()

    def setup_kv_publishers(self) -> None:
        """Hook for setting up KV event publishers.

        Called after handler creation. Override to set up KV event
        publishing (e.g., prefix caching publishers).
        """
        pass

    def register_engine_routes(self, runtime: Any, handler: Any) -> None:
        """Hook for registering engine-specific routes.

        Override to register engine-specific routes.

        Args:
            runtime: The DistributedRuntime.
            handler: The request handler.
        """
        pass

    def get_metrics_labels(self) -> List[Tuple[str, str]]:
        """Get metrics labels for the endpoint.

        Returns:
            List of (label_name, label_value) tuples.
        """
        from dynamo import prometheus_names

        model_name = self._get_model_name()
        return [
            (prometheus_names.labels.MODEL, model_name),
            (prometheus_names.labels.MODEL_NAME, model_name),
        ]

    def _get_model_name(self) -> str:
        """Get effective model name from config.

        Handles different config structures across backends.
        """
        # Try BackendConfig-style
        if hasattr(self.config, "get_model_name"):
            return self.config.get_model_name()
        if hasattr(self.config, "served_model_name") and self.config.served_model_name:
            return self.config.served_model_name
        if hasattr(self.config, "model"):
            return self.config.model
        if hasattr(self.config, "model_path"):
            return self.config.served_model_name or self.config.model_path
        return "unknown"

    def _get_endpoint_types(self) -> str:
        """Get endpoint types from config."""
        return getattr(self.config, "endpoint_types", "chat,completions")

    def _get_custom_jinja_template(self) -> Optional[str]:
        """Get custom Jinja template path from config."""
        return getattr(self.config, "custom_jinja_template", None)

    # ── Shared model-registration hooks ──────────────────────────

    def _is_prefill(self) -> bool:
        """Whether this worker is a prefill-only node. Default False."""
        return False

    def _is_decode(self) -> bool:
        """Whether this worker is a decode-only node. Default False."""
        return False

    def _get_model_path(self) -> str:
        """Get the model path from config."""
        if hasattr(self.config, "model"):
            return self.config.model
        if hasattr(self.config, "model_path"):
            return self.config.model_path
        return "unknown"

    def _get_served_model_name(self) -> Optional[str]:
        """Get the served model name from config."""
        return getattr(self.config, "served_model_name", None)

    def _get_kv_block_size(self, engine: Any) -> Optional[int]:
        """Get the KV cache block size. Override for engine-specific retrieval."""
        return getattr(self.config, "kv_block_size", None)

    def _get_input_type(self) -> "ModelInput":
        """Get the model input type. Default: Tokens. Override for Text input."""
        return ModelInput.Tokens

    def extract_runtime_config(self, engine: Any) -> Optional[ModelRuntimeConfig]:
        """Extract runtime configuration from the engine.

        Override in subclasses to provide engine-specific runtime config.
        Return None to skip model registration in the default register_model().
        """
        return None

    def create_component_gauges(self) -> LLMBackendMetrics:
        """Create LLMBackendMetrics for this component."""
        return LLMBackendMetrics(
            registry=DYNAMO_COMPONENT_REGISTRY,
            model_name=self._get_model_name(),
            component_name=getattr(self.config, "component", ""),
        )

    def get_additional_endpoints(self) -> Dict[str, Callable]:
        """Get additional endpoints to serve alongside the main generate endpoint.

        Returns:
            Dict mapping endpoint_name -> handler_callable.
            Example: {"clear_kv_blocks": handler.clear_kv_blocks}
        """
        return {}

    async def register_and_serve(
        self, handler: Any, endpoint: Any, engine: Any
    ) -> None:
        """Register the model and serve the endpoint.

        Default: register model, then serve endpoint with additional endpoints.
        Override for concurrent patterns (e.g., concurrent registration and serving).

        Args:
            handler: The request handler.
            endpoint: The Dynamo endpoint.
            engine: The LLM engine.
        """
        await self.register_model(endpoint, engine)

        health_check_payload = self.get_health_check_payload(engine)
        metrics_labels = self.get_metrics_labels()

        # Build list of serve coroutines
        serve_coros = [
            endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
                health_check_payload=health_check_payload,
            )
        ]

        # Add additional endpoints
        additional = self.get_additional_endpoints()
        for ep_name, ep_handler in additional.items():
            additional_endpoint = self.component.endpoint(ep_name)
            serve_coros.append(
                additional_endpoint.serve_endpoint(
                    ep_handler,
                    metrics_labels=metrics_labels,
                )
            )

        await asyncio.gather(*serve_coros)

    async def register_model(self, endpoint: Any, engine: Any) -> None:
        """Register the model with the Dynamo frontend.

        Default implementation uses extract_runtime_config() to build
        ModelRuntimeConfig and register via register_llm(). If
        extract_runtime_config() returns None, this is a no-op
        (backward compatible for subclasses that don't override it).

        Args:
            endpoint: The Dynamo endpoint.
            engine: The LLM engine.
        """
        runtime_config = self.extract_runtime_config(engine)
        if runtime_config is None:
            return

        # Post-process: disable local indexer for decode workers
        runtime_config.enable_local_indexer = (
            getattr(self.config, "enable_local_indexer", True)
            and not self._is_decode()
        )

        # Determine input and model type
        input_type = self._get_input_type()

        if self._is_prefill():
            model_type = ModelType.Prefill
            # Prefill workers don't parse tool calls or reasoning
            runtime_config.tool_call_parser = None
            runtime_config.reasoning_parser = None
        else:
            model_type = parse_endpoint_types(self._get_endpoint_types())

        # When input is text (e.g. using framework tokenizer),
        # restrict to Chat for non-embedding models
        if input_type == ModelInput.Text and model_type != ModelType.Embedding:
            model_type = ModelType.Chat

        # Log runtime config
        logger.info(
            f"Set runtime config max_num_seqs: {runtime_config.max_num_seqs}"
        )
        logger.info(
            f"Set runtime config max_num_batched_tokens: "
            f"{runtime_config.max_num_batched_tokens}"
        )
        logger.info(
            f"Set runtime config data_parallel_size: "
            f"{runtime_config.data_parallel_size}"
        )

        await register_llm(
            input_type,
            model_type,
            endpoint,
            self._get_model_path(),
            self._get_served_model_name(),
            kv_cache_block_size=self._get_kv_block_size(engine),
            runtime_config=runtime_config,
            custom_template_path=self._get_custom_jinja_template(),
        )

    def post_serve_cleanup(self) -> None:
        """Hook for sync post-serve cleanup.

        Called by async_post_serve_cleanup() by default.
        Override for backend-specific sync cleanup.
        """
        pass

    async def async_post_serve_cleanup(self) -> None:
        """Async hook for post-serve cleanup.

        Default calls sync post_serve_cleanup() for backward compatibility.
        Override for async cleanup (e.g., deferred handler cleanup).
        """
        self.post_serve_cleanup()

    # ── Logged wrapper methods ────────────────────────────────────

    async def _run_create_engine(self) -> Any:
        """Logged wrapper for create_engine() with timing."""
        logger.info(f"[{self.backend_name}] Creating engine...")
        start = time.time()
        engine = await self.create_engine()
        elapsed = time.time() - start
        logger.info(f"[{self.backend_name}] Engine created ({elapsed:.2f}s)")
        return engine

    def _run_create_handler(self, engine: Any, component: Any, endpoint: Any) -> Any:
        """Logged wrapper for create_handler()."""
        logger.info(f"[{self.backend_name}] Creating request handler...")
        handler = self.create_handler(engine, component, endpoint)
        logger.info(f"[{self.backend_name}] Request handler created")
        return handler

    async def _run_setup_metrics(self, endpoint: Any) -> None:
        """Logged wrapper for metrics setup."""
        logger.info(f"[{self.backend_name}] Setting up metrics...")
        await self._do_setup_metrics(endpoint)
        logger.info(f"[{self.backend_name}] Metrics setup complete")

    async def _do_setup_metrics(self, endpoint: Any) -> None:
        """Hook for metrics setup. Override for async metrics setup.
        Should set self._metrics_task if applicable."""
        self._metrics_task = self.setup_metrics(endpoint)

    async def _run_handle_non_leader(self) -> bool:
        """Check and handle non-leader node. Returns True if non-leader."""
        if self.is_non_leader_node():
            logger.info(f"[{self.backend_name}] Non-leader node detected")
            await self.handle_non_leader_node()
            return True
        return False

    def _run_setup_kv_publishers(self) -> None:
        """Logged wrapper for setup_kv_publishers()."""
        logger.info(f"[{self.backend_name}] Setting up KV publishers...")
        self.setup_kv_publishers()
        logger.info(f"[{self.backend_name}] KV publishers ready")

    def _run_register_engine_routes(self, runtime: Any, handler: Any) -> None:
        """Logged wrapper for register_engine_routes()."""
        logger.info(f"[{self.backend_name}] Registering engine routes...")
        self.register_engine_routes(runtime, handler)
        logger.info(f"[{self.backend_name}] Engine routes registered")

    async def _run_serve(self, handler: Any, endpoint: Any, engine: Any) -> None:
        """Logged wrapper for register_and_serve() with endpoint type info."""
        endpoint_types = self._get_endpoint_types()
        logger.info(
            f"[{self.backend_name}] Registering model with endpoint types: "
            f"{endpoint_types}"
        )

        custom_template = self._get_custom_jinja_template()
        if custom_template and "chat" not in endpoint_types:
            logger.warning(
                "Custom Jinja template provided (--custom-jinja-template) but "
                "'chat' not in --dyn-endpoint-types. The chat template will be "
                "loaded but the /v1/chat/completions endpoint will not be available."
            )

        logger.info(f"[{self.backend_name}] Starting endpoint serving...")
        try:
            await self.register_and_serve(handler, endpoint, engine)
        except Exception as e:
            logger.error(f"[{self.backend_name}] Endpoint serving failed: {e}")
            raise

    async def _run_cleanup(self) -> None:
        """Logged cleanup for metrics task, handler, and post-serve."""
        logger.info(f"[{self.backend_name}] Cleaning up...")
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        if self.handler and hasattr(self.handler, "cleanup"):
            self.handler.cleanup()
        await self.async_post_serve_cleanup()
        logger.info(f"[{self.backend_name}] Cleanup complete")

    async def run(self) -> None:
        """Main entry point for running the backend worker.

        This method orchestrates the complete lifecycle using the
        Template Method pattern. Each step can be overridden by subclasses.
        """
        # Step 1: Pre-runtime setup
        self.pre_runtime_setup()

        # Step 2: Set up runtime (skip if pre-provided)
        self.setup_runtime()

        # Step 3: Set up component and endpoint
        component, endpoint = self.setup_component()

        # Step 4: Engine context (default: create engine; subclasses may override)
        async with self.engine_context() as engine:
            self.engine = engine

            # Step 6: Set up metrics
            await self._run_setup_metrics(endpoint)

            # Step 7: Check non-leader node
            if await self._run_handle_non_leader():
                await self._run_cleanup()
                return

            # Step 8: Create handler
            self.handler = self._run_create_handler(engine, component, endpoint)

            # Step 9: Set up KV publishers
            self._run_setup_kv_publishers()

            # Step 10: Register engine routes
            self._run_register_engine_routes(self.runtime, self.handler)

            # Step 11: Register and serve
            try:
                await self._run_serve(self.handler, endpoint, engine)
            finally:
                # Step 12: Cleanup
                await self._run_cleanup()

    def shutdown(self) -> None:
        """Trigger graceful shutdown of the backend."""
        if self.shutdown_event:
            self.shutdown_event.set()
        if self.runtime:
            self.runtime.shutdown()
