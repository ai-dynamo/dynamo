# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import logging
import random
import socket
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import sglang as sgl
from sglang.srt.tracing import trace as sglang_trace
from sglang.srt.utils import get_local_ip_auto

from dynamo._core import Component, Context
from dynamo.common.utils.input_params import InputParamManager
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher


class BaseWorkerHandler(ABC):
    """Abstract base class for SGLang worker handlers."""

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
    ) -> None:
        """Initialize base worker handler.

        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: Optional metrics publisher for the worker.
        """
        self.component = component
        self.engine = engine
        self.config = config
        if publisher is not None:
            self.metrics_publisher = publisher.metrics_publisher
            self.kv_publisher = publisher.kv_publisher
        else:
            self.metrics_publisher = None
            self.kv_publisher = None
        self.serving_mode = config.serving_mode
        self.skip_tokenizer_init = config.server_args.skip_tokenizer_init
        self.enable_trace = config.server_args.enable_trace

        self.input_param_manager = InputParamManager(
            self.engine.tokenizer_manager.tokenizer
            if not self.skip_tokenizer_init
            else None
        )

    @abstractmethod
    async def generate(self, request: Dict[str, Any], context: Context):
        """Generate response from request.

        Args:
            request: Request dict with input and parameters.
            context: Context object for cancellation handling.

        Yields:
            Response data (format varies by handler implementation).
        """
        pass

    def cleanup(self) -> None:
        """Cleanup resources. Override in subclasses as needed."""
        pass

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
            # IPv6-ready host extraction and resolution:
            # 1) Extract raw host from "host:port" or "[IPv6]:port"/"[IPv6]".
            # 2) Resolve via AF_UNSPEC to accept A/AAAA and literals.
            # 3) Bracket-wrap IPv6 for safe "{host}:{port}" URL formatting.
            addr = inner_tm.server_args.dist_init_addr.strip()
            if addr.startswith("["):
                end = addr.find("]")
                host_core = addr[1:end] if end != -1 else addr.strip("[]")
            else:
                # Only treat single ':' with numeric suffix as host:port; otherwise it's an IPv6/FQDN host.
                if addr.count(":") == 1:
                    host_candidate, maybe_port = addr.rsplit(":", 1)
                    host_core = host_candidate if maybe_port.isdigit() else addr
                else:
                    host_core = addr
            try:
                infos = socket.getaddrinfo(
                    host_core,
                    None,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                )
                resolved = infos[0][4][0]  # let OS policy pick v4/v6
                bootstrap_host = resolved
            except socket.gaierror:
                # Fallback: keep literal/FQDN as-is (still wrap IPv6 below)
                bootstrap_host = host_core
        else:
            bootstrap_host = get_local_ip_auto()

        # Wrap IPv6 literal with brackets so f"{host}:{port}" stays valid.
        if ":" in bootstrap_host and not bootstrap_host.startswith("["):
            bootstrap_host = f"[{bootstrap_host}]"

        return bootstrap_host, bootstrap_port

    def _propagate_trace_context_to_sglang(
        self, context: Context, bootstrap_room: int = 0
    ):
        """Propagate Dynamo's trace context to SGLang for distributed tracing. SGLang expects a certain
        format derived by loooking at https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/tracing/trace.py
        in the to_dict() method.

        Args:
            context: Dynamo Context object containing trace information.
            bootstrap_room: Bootstrap room ID (0 for aggregated, actual room for disaggregated).
        """
        trace_id = context.trace_id
        span_id = context.span_id
        if not trace_id or not span_id:
            return

        # Build trace context for SGLang
        trace_context = {
            str(bootstrap_room): {
                "root_span": {"traceparent": f"00-{trace_id}-{span_id}-01"},
                "prev_span": {
                    "span_id": int(span_id, 16),
                    "trace_id": int(trace_id, 16),
                },
            }
        }

        # Encode and propagate
        base64_context = base64.b64encode(
            json.dumps(trace_context, ensure_ascii=False).encode("utf-8")
        ).decode("utf-8")
        sglang_trace.trace_set_remote_propagate_context(base64_context)

    async def _handle_cancellation(
        self, request_id_future: asyncio.Future, context: Context
    ):
        """Background task to handle cancellation by monitoring context state.

        Args:
            request_id_future: Future that will be set with the SGLang request ID
                              when the first response arrives.
            context: Context object for cancellation handling.
        """
        try:
            logging.debug(f"Cancellation monitor started for Context: {context.id()}")

            # Always wait for the request ID to ensure we can abort the request
            sglang_request_id = await request_id_future
            logging.debug(
                f"Cancellation monitor received SGLang Request ID {sglang_request_id} for Context: {context.id()}"
            )
            logging.debug(f"Request ID future cancelled for Context: {context.id()}")

            await context.async_killed_or_stopped()

            logging.info(
                f"Cancellation signal received for SGLang Request ID {sglang_request_id}, Context: {context.id()}"
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
        Context manager for monitoring request cancellation.
        Automatically creates a background task to monitor for cancellation and
        cleans it up when the context exits.

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
                logging.debug(
                    f"Cancellation monitor task already completed for SGLang Request ID {request_id}, Context: {context.id()}"
                )
