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

import contextvars
import functools
import inspect
import logging
import uuid
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union, cast

from fastapi import Request, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


# Async-safe storage for request context
_request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def extract_or_generate_request_id(
    headers: Optional[dict] = None,
    existing_id: Optional[str] = None,
) -> str:
    """
    Universal function to extract request ID from various sources or generate a new UUID.

    Args:
        headers: Dictionary of headers (optional)
        existing_id: Already extracted request ID (optional)

    Returns:
        str: Request ID from header or newly generated UUID
    """
    if existing_id:
        return existing_id

    request_id = None
    if headers is not None:
        request_id = headers.get("x-request-id") or headers.get("X-Request-Id")
    if request_id is None:
        request_id = str(uuid.uuid4())

    return request_id


def set_request_context(request_id: str) -> None:
    """Set the current request ID in thread-local context."""
    _request_id_ctx.set(request_id)


def get_current_request_id() -> Optional[str]:
    """Get the current request ID from thread-local context."""
    return _request_id_ctx.get()


def clear_request_context() -> None:
    """Clear the current request context."""
    _request_id_ctx.set(None)


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
            headers_opt = request.headers if hasattr(request, "headers") else None
            request_id = extract_or_generate_request_id(headers_opt)
            # preserve previous value (if any) for nested invocations
            _token = _request_id_ctx.set(request_id)
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
            # restore previous request-id (or None)
            if request_id is not None:
                _request_id_ctx.reset(_token)

    return wrapper


def auto_trace_endpoints(cls):
    """
    Class decorator to automatically add request tracing to all endpoints.

    Usage:
        @auto_trace_endpoints
        @service(...)
        class MyFrontend:
            @endpoint(is_api=True)
            async def my_endpoint(self, request: Request, data: MyData):
                # X-Request-Id is automatically handled
                pass
    """

    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)

        if (
            hasattr(attr, "__call__")
            and hasattr(attr, "_endpoint")
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
            @with_request_id()
            async def process(self, data, req_id_opt: Optional[str] = None):
                # After decoration, req_id_opt is guaranteed to be non-None
                # And thread-local storage contains the request ID
                self.log("info", "Processing data")  # No need to pass request_id
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

    def log(self, level: str, message: str):
        """
        Log message with request ID automatically retrieved from context.

        When used with @with_request_id decorator, the request ID is automatically
        available in thread-local storage, so there's no need to pass it explicitly.

        Args:
            level: Log level (info, debug, warning, error, etc.)
            message: The log message
        """
        # Get request ID from thread-local storage
        req_id = get_current_request_id()

        # If not available, generate a new one
        if req_id is None:
            log_message = f"[no_request_id] {message}"
        else:
            log_message = f"[request_id={req_id}] {message}"

        getattr(logger, level.lower())(log_message)

    # For backward compatibility
    def log_with_request_id(
        self, level: str, message: str, request_id: Optional[str] = None
    ):
        """
        Legacy method for backward compatibility.
        Prefer using the simpler log() method instead.
        """
        # Get request ID from parameter or thread-local storage
        req_id = request_id or get_current_request_id()

        # If still None, generate a new one
        if req_id is None:
            req_id = str(uuid.uuid4())
            set_request_context(req_id)

        log_message = f"[request_id={req_id}] {message}"
        _method = getattr(logger, level.lower(), None)
        if _method is None:
            logger.warning(f"Unknown log level '{level}', defaulting to INFO")
            _method = logger.info
        _method(log_message)


# Type variables for generic functions
F = TypeVar("F", bound=Callable[..., Any])


def with_request_id(param_name: str = "request_id"):
    """
    Decorator that ensures a non-None request ID is available in the function.

    This decorator:
    1. Accepts a request ID parameter that may be None in the function signature
    2. Ensures the parameter is a non-None string when the function executes
    3. Sets the request ID in thread-local storage for logging

    Usage:
        @with_request_id()
        async def process(self, data, request_id: str = None):
            # Inside the function, request_id is guaranteed to be a non-None str
            self.log("info", f"Processing data for request {request_id}")

    Args:
        param_name: Name of the request ID parameter in the function signature (default: 'request_id')
    """

    def decorator(func: F) -> F:
        sig = inspect.signature(func)

        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Check if the function has the request ID parameter
            if param_name in sig.parameters:
                # Extract the request ID parameter from kwargs or args
                request_id = None
                if param_name in kwargs:
                    request_id = kwargs[param_name]
                else:
                    # Locate positional parameter index
                    param_idx = next(
                        (
                            i
                            for i, p in enumerate(sig.parameters.values())
                            if p.name == param_name
                        ),
                        None,
                    )
                    if param_idx is not None:
                        adj_idx = param_idx - 1  # skip `self`
                        if 0 <= adj_idx < len(args):
                            request_id = args[adj_idx]

                            # Replace value in the *args* tuple instead of duplicating it in **kwargs**
                            args = (
                                *args[:adj_idx],
                                None,
                                *args[adj_idx + 1 :],
                            )

                # … rest of wrapper …

                # Ensure we have a non-None request ID
                if hasattr(self, "ensure_request_id") and callable(
                    self.ensure_request_id
                ):
                    request_id = self.ensure_request_id(request_id)
                else:
                    request_id = request_id or str(uuid.uuid4())

                # Update the parameter value
                kwargs[param_name] = request_id

                # Also set it in thread-local storage for logging
                set_request_context(request_id)

            # Call the original function
            return await func(self, *args, **kwargs)

        return cast(F, wrapper)

    return decorator


def trace_frontend_endpoint(func: Callable) -> Callable:
    """Specialized decorator for frontend endpoints."""
    return with_request_tracing(func)


def trace_processor_method(func: Callable) -> Callable:
    """Specialized decorator for processor methods."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        if "request_id" not in kwargs or not kwargs["request_id"]:
            new_id = get_current_request_id() or str(uuid.uuid4())
            kwargs["request_id"] = new_id
            set_request_context(new_id)

        return await func(*args, **kwargs)

    return wrapper
