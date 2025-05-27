# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Universal Request Tracing Support for Dynamo

This module provides automatic X-Request-Id header support across all Dynamo components
without requiring manual implementation in each frontend/processor/router/worker.
"""

import functools
import inspect
import logging
import threading
import uuid
from typing import Callable, Optional, Union

from fastapi import Request, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# Thread-local storage for request context
_local = threading.local()


def extract_or_generate_request_id(
    request: Optional[Request] = None,
    headers: Optional[dict] = None,
    existing_id: Optional[str] = None,
) -> str:
    """
    Universal function to extract request ID from various sources or generate a new UUID.

    Args:
        request: FastAPI Request object (optional)
        headers: Dictionary of headers (optional)
        existing_id: Already extracted request ID (optional)

    Returns:
        str: Request ID from header or newly generated UUID
    """
    if existing_id:
        return existing_id

    request_id = None

    if request is not None:
        request_id = request.headers.get("x-request-id")

    if request_id is None and headers is not None:
        request_id = headers.get("x-request-id") or headers.get("X-Request-Id")

    if request_id is None:
        request_id = str(uuid.uuid4())

    return request_id


def set_request_context(request_id: str) -> None:
    """Set the current request ID in thread-local context."""
    _local.request_id = request_id


def get_current_request_id() -> Optional[str]:
    """Get the current request ID from thread-local context."""
    return getattr(_local, "request_id", None)


def clear_request_context() -> None:
    """Clear the current request context."""
    if hasattr(_local, "request_id"):
        delattr(_local, "request_id")


def add_request_id_to_response(
    response: Union[Response, StreamingResponse], request_id: str
) -> None:
    """Add X-Request-Id header to any FastAPI response."""
    if hasattr(response, "headers"):
        response.headers["X-Request-Id"] = request_id


def with_request_tracing(func: Callable) -> Callable:
    """
    Decorator to automatically add X-Request-Id support to any dynamo_endpoint.

    This decorator:
    1. Extracts X-Request-Id from incoming requests
    2. Sets it in thread-local context
    3. Passes it to the wrapped function if it accepts request_id parameter
    4. Adds X-Request-Id header to responses
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        request = None
        request_id = None

        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break

        if "request" in kwargs and isinstance(kwargs["request"], Request):
            request = kwargs["request"]

        if request:
            request_id = extract_or_generate_request_id(request)
            set_request_context(request_id)
            logger.debug(
                f"Request tracing: extracted/generated request_id={request_id}"
            )

        sig = inspect.signature(func)
        if "request_id" in sig.parameters and request_id:
            kwargs["request_id"] = request_id

        try:
            result = await func(*args, **kwargs)

            if request_id and isinstance(result, (Response, StreamingResponse)):
                add_request_id_to_response(result, request_id)

            return result

        finally:
            clear_request_context()

    return wrapper


def auto_trace_endpoints(cls):
    """
    Class decorator to automatically add request tracing to all dynamo_endpoints.

    Usage:
        @auto_trace_endpoints
        @service(...)
        class MyFrontend:
            @dynamo_endpoint(is_api=True)
            async def my_endpoint(self, request: Request, data: MyData):
                # X-Request-Id is automatically handled
                pass
    """

    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)

        if (
            hasattr(attr, "__call__")
            and hasattr(attr, "_dynamo_endpoint")
            and not attr_name.startswith("_")
        ):
            wrapped_method = with_request_tracing(attr)
            setattr(cls, attr_name, wrapped_method)

            logger.debug(f"Added request tracing to {cls.__name__}.{attr_name}")

    return cls


class RequestTracingMixin:
    """
    Mixin class that provides request tracing utilities to any component.

    Usage:
        class MyProcessor(RequestTracingMixin):
            async def process(self, data, request_id: Optional[str] = None):
                request_id = self.ensure_request_id(request_id)
                # Use request_id for logging, etc.
    """

    def ensure_request_id(self, request_id: Optional[str] = None) -> str:
        """Ensure we have a request ID, either from parameter or context."""
        if request_id:
            return request_id

        context_id = get_current_request_id()
        if context_id:
            return context_id

        new_id = str(uuid.uuid4())
        set_request_context(new_id)
        return new_id

    def log_with_request_id(
        self, level: str, message: str, request_id: Optional[str] = None
    ):
        """Log message with request ID."""
        req_id = self.ensure_request_id(request_id)
        log_message = f"[request_id={req_id}] {message}"
        getattr(logger, level.lower())(log_message)


def trace_frontend_endpoint(func: Callable) -> Callable:
    """Specialized decorator for frontend endpoints."""
    return with_request_tracing(func)


def trace_processor_method(func: Callable) -> Callable:
    """Specialized decorator for processor methods."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        if "request_id" not in kwargs or not kwargs["request_id"]:
            kwargs["request_id"] = get_current_request_id() or str(uuid.uuid4())

        return await func(*args, **kwargs)

    return wrapper
