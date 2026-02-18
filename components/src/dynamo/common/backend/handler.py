# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base handler interface for Dynamo backend workers.

This module provides the common interface for request handlers across all
backend implementations.

The handler interface is designed to be minimal and flexible, allowing each
backend to implement its specific logic while maintaining a consistent API
for the Dynamo endpoint serving infrastructure.

Example usage for implementing a new backend handler:

    from dynamo.common.backend import BaseHandler

    class MyBackendHandler(BaseHandler):
        def __init__(self, component, engine, config):
            super().__init__(component)
            self.engine = engine
            self.config = config

        async def generate(self, request, context):
            # Framework-specific generation logic
            async for chunk in self.engine.generate(...):
                yield {"output_ids": chunk.tokens, ...}

        def cleanup(self):
            super().cleanup()
            # Framework-specific cleanup
            self.engine.shutdown()
"""

import asyncio
import logging
import tempfile
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from dynamo._core import Context
from dynamo.common.utils.logprobs import extract_logprobs

logger = logging.getLogger(__name__)


class BaseHandler(ABC):
    """Abstract base class for all Dynamo LLM request handlers.

    This class defines the minimal interface that all backend-specific handlers
    must implement. It provides a consistent API for the Dynamo endpoint serving
    infrastructure while allowing flexibility in implementation.

    The core contract is the generate() method - an async generator that receives
    requests and yields response chunks. All handlers must implement this method.

    Attributes:
        shutdown_event: Optional event to signal shutdown for graceful termination.
        kv_publishers: Optional list of KV event publishers.

    Note:
        Subclasses typically add engine, component, and config attributes
        specific to their framework.
    """

    def __init__(
        self,
        component: Optional[Any] = None,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Initialize the base handler.

        Args:
            component: Optional Dynamo runtime component.
            shutdown_event: Optional event to signal shutdown for graceful termination.
        """
        self.component = component
        self.shutdown_event = shutdown_event

        # KV event publishers (set by subclasses if needed)
        self.kv_publishers: Optional[List[Any]] = None

        # Temporary directories to clean up
        self._temp_dirs: List[tempfile.TemporaryDirectory] = []

    @abstractmethod
    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate response from the request.

        This is the main entry point for handling inference requests.
        It is called by Dynamo's endpoint.serve_endpoint() method.

        Subclasses must implement this method with framework-specific logic.
        The method should be an async generator that yields response dictionaries.

        Args:
            request: Request dictionary containing:
                - input_ids: Token IDs for the input (if using Dynamo tokenizer)
                - prompt: Text prompt (if using framework tokenizer)
                - sampling_params: Generation parameters
                - Additional framework-specific fields

            context: Dynamo context providing:
                - Request ID and trace information
                - Cancellation handling (context.async_killed_or_stopped())
                - Trace header extraction (context.trace_id, context.span_id)

        Yields:
            Response dictionaries. The exact format depends on the backend
            implementation, but typically includes:
            - output_ids: Generated token IDs (list or incremental)
            - finish_reason: Why generation stopped ("stop", "length", etc.)
            - Additional framework-specific fields

        Example:
            async def generate(self, request, context):
                input_ids = request.get("input_ids", [])
                params = request.get("sampling_params", {})

                async for chunk in self.engine.generate(input_ids, **params):
                    if context.is_cancelled():
                        break
                    yield {
                        "output_ids": chunk.tokens,
                        "finish_reason": chunk.finish_reason,
                    }
        """
        # This yield is needed to make this an async generator
        # Subclasses should replace this with actual implementation
        raise NotImplementedError("Subclasses must implement generate()")
        yield {}  # pragma: no cover

    @staticmethod
    def process_generation_output(
        output, num_output_tokens_so_far: int
    ) -> tuple[dict, int]:
        """Process a cumulative generation output into a response dict.

        Extracts token deltas, logprobs, finish_reason, and stop_reason from
        an engine output object.

        Args:
            output: Engine output with .token_ids, .logprobs, .finish_reason,
                    .stop_reason attributes.
            num_output_tokens_so_far: Tokens already yielded (for delta extraction).

        Returns:
            Tuple of (response_dict, new_num_output_tokens):
            - response_dict: Dict with token_ids, and optional log_probs,
              top_logprobs, finish_reason, stop_reason.
            - new_num_output_tokens: Updated cumulative token count.
        """
        next_total_toks = len(output.token_ids)
        out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}

        log_probs, top_logprobs = extract_logprobs(
            output.logprobs, output.token_ids, num_output_tokens_so_far
        )
        if log_probs is not None:
            out["log_probs"] = log_probs
        if top_logprobs is not None:
            out["top_logprobs"] = top_logprobs

        if output.finish_reason:
            out["finish_reason"] = output.finish_reason
        if output.stop_reason:
            out["stop_reason"] = output.stop_reason

        return out, next_total_toks

    def cleanup(self) -> None:
        """Clean up resources held by the handler.

        Called when the handler is being shut down. Subclasses should override
        this to clean up framework-specific resources and call super().cleanup()
        to clean up common resources.

        The default implementation:
        - Shuts down KV publishers
        - Cleans up temporary directories

        Cleanup contract for subclasses:
        1. BaseHandler.cleanup() handles: kv_publishers, temp_dirs
        2. Subclasses MUST call super().cleanup() and handle their own engine shutdown
        3. Recommended cleanup order:
           a. Cancel pending tasks (e.g., consume tasks, background monitors)
           b. Call super().cleanup() for publisher and temp dir cleanup
           c. Shut down the engine (e.g., engine.shutdown(), engine.cleanup())
        """
        # Clean up KV publishers
        if self.kv_publishers:
            for publisher in self.kv_publishers:
                if hasattr(publisher, "shutdown"):
                    try:
                        publisher.shutdown()
                    except Exception as e:
                        logger.warning(f"Error shutting down KV publisher: {e}")

        # Clean up temporary directories
        for temp_dir in self._temp_dirs:
            try:
                temp_dir.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up temp directory: {e}")
        self._temp_dirs.clear()

    def add_temp_dir(self, temp_dir: Optional[tempfile.TemporaryDirectory]) -> None:
        """Add a temporary directory to be cleaned up on shutdown.

        Args:
            temp_dir: TemporaryDirectory instance to track, or None (ignored).
        """
        if temp_dir is not None:
            self._temp_dirs.append(temp_dir)

    def get_trace_header(self, context: Context) -> Optional[Dict[str, str]]:
        """Get trace header dictionary for distributed tracing.

        This creates a traceparent header from the Dynamo context that can be
        passed to LLM frameworks that support distributed tracing.

        Args:
            context: Dynamo Context object containing trace information.

        Returns:
            Dictionary with traceparent header if trace context is available,
            None otherwise.

        Example:
            headers = self.get_trace_header(context)
            if headers:
                # Pass to framework that supports W3C trace context
                engine.generate(..., trace_headers=headers)
        """
        trace_id = context.trace_id
        span_id = context.span_id

        if not trace_id or not span_id:
            return None

        return {"traceparent": f"00-{trace_id}-{span_id}-01"}

    async def _handle_cancellation(
        self, context: Context, abort_callback: Optional[Callable] = None
    ) -> None:
        """Background task that monitors for context cancellation and shutdown.

        Aborts the request if either occurs. Raises GeneratorExit if shutdown
        was triggered.

        Args:
            context: Dynamo context for the request.
            abort_callback: Optional async callable invoked when cancellation detected.
                           Backends pass their engine-specific abort here.
        """
        try:
            wait_for = [context.async_killed_or_stopped()]
            shutdown_task = None

            if self.shutdown_event:
                shutdown_task = asyncio.create_task(self.shutdown_event.wait())
                wait_for.append(shutdown_task)

            done, pending = await asyncio.wait(
                wait_for,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            if abort_callback is not None:
                await abort_callback()

            if shutdown_task and shutdown_task in done:
                raise GeneratorExit("Engine was shut down during generation.")

        except asyncio.CancelledError:
            pass

    @asynccontextmanager
    async def _cancellation_monitor(
        self,
        context: Context,
        abort_callback: Optional[Callable] = None,
    ):
        """Monitor for request cancellation or server shutdown.

        Context manager that creates and automatically cleans up a cancellation
        monitoring task. If shutdown event was triggered, raises GeneratorExit
        on exit.

        Args:
            context: Dynamo context for the request.
            abort_callback: Optional async callable invoked when cancellation detected.
                           Backends pass their engine-specific abort here.

        Yields:
            The cancellation monitoring task.
        """
        task = asyncio.create_task(
            self._handle_cancellation(context, abort_callback)
        )
        try:
            yield task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:
                task.result()

    async def _unregister_for_maintenance(self, endpoint) -> None:
        """Unregister endpoint from discovery before maintenance.

        Args:
            endpoint: The endpoint handle to unregister.
        """
        try:
            await endpoint.unregister_endpoint_instance()
            logger.info("Unregistered endpoint from discovery for maintenance")
        except Exception as e:
            logger.warning(f"Failed to unregister endpoint: {e}")

    async def _register_after_maintenance(self, endpoint) -> None:
        """Re-register endpoint to discovery after maintenance.

        Args:
            endpoint: The endpoint handle to re-register.
        """
        try:
            await endpoint.register_endpoint_instance()
            logger.info("Re-registered endpoint to discovery after maintenance")
        except Exception as e:
            logger.warning(f"Failed to re-register endpoint: {e}")
