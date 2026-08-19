# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import signal
import warnings
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Optional, Type, Union

from pydantic import BaseModel, ValidationError

# List all the classes in the _core module for re-export
# import * causes "unable to detect undefined names"
from dynamo._core import Client as Client
from dynamo._core import Context as Context
from dynamo._core import DistributedRuntime as DistributedRuntime
from dynamo._core import Endpoint as Endpoint
from dynamo._core import PyAsyncRequestStream as PyAsyncRequestStream


def dynamo_worker(enable_nats: Optional[bool] = None, register_shutdown: bool = False):
    """
    Decorator that creates a DistributedRuntime and passes it to the worker function.

    Args:
        enable_nats: Deprecated. NATS enablement is now determined automatically
            from the event-plane configuration. This parameter is accepted for
            backwards compatibility but will be removed in a future release.
        register_shutdown: Boolean flag to gracefully shutdown a dynamo worker post SIGTERM, SIGINT.
    """
    if enable_nats is not None:
        warnings.warn(
            "The 'enable_nats' parameter is deprecated and will be removed in a "
            "future release. NATS enablement is now determined automatically from "
            "the event-plane configuration.",
            DeprecationWarning,
            stacklevel=2,
        )

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_running_loop()
            request_plane = os.environ.get("DYN_REQUEST_PLANE", "tcp")
            discovery_backend = os.environ.get("DYN_DISCOVERY_BACKEND", "etcd")
            runtime = DistributedRuntime(loop, discovery_backend, request_plane)
            if register_shutdown:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, runtime.shutdown)
            await func(runtime, *args, **kwargs)

        return wrapper

    return decorator


def dynamo_endpoint(
    request_model: Union[Type[BaseModel], Type[Any]], response_model: Type[BaseModel]
) -> Callable:
    """Decorator that can parse a request payload into a Pydantic model before the endpoint runs.

    Parsing applies only when ``request_model`` is a ``BaseModel`` subclass
    *and* the wrapper receives one or two positional arguments -- ``(request)``
    or ``(self, request)``. With three or more positional arguments, or when the
    payload arrives by keyword, it is forwarded untouched. A ``str`` payload is
    parsed with ``parse_raw`` and a ``dict`` with ``parse_obj``; any other type,
    including an already-constructed ``request_model`` instance, is rejected.
    ``response_model`` is reserved for future validation; yielded items pass
    through unchanged today.

    Args:
        request_model: Request class used to parse ``str`` or ``dict`` payloads.
            Pass a non-``BaseModel`` type, as ``examples/custom_backend``
            does with ``str``, to skip parsing entirely.
        response_model: Expected response class. Currently accepted but not enforced.

    Raises:
        ValueError: On the first ``__anext__()`` of the returned generator, not
            when the decorated function is called, because the wrapper is
            itself an async generator. Raised when the payload fails validation
            or is neither ``str`` nor ``dict``.

    Examples:
        >>> from pydantic import BaseModel
        >>> from dynamo.runtime import dynamo_endpoint
        >>>
        >>> class Request(BaseModel):
        ...     data: str
        >>> class Response(BaseModel):
        ...     char: str
        >>>
        >>> @dynamo_endpoint(Request, Response)
        ... async def generate(request):
        ...     for char in request.data:
        ...         yield char
    """

    def decorator(
        func: Callable[..., AsyncGenerator[Any, None]],
    ) -> Callable[..., AsyncGenerator[Any, None]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> AsyncGenerator[Any, None]:
            # Validate the request
            try:
                args_list = list(args)
                if len(args) in [1, 2] and issubclass(request_model, BaseModel):
                    if isinstance(args[-1], str):
                        args_list[-1] = request_model.parse_raw(args[-1])
                    elif isinstance(args[-1], dict):
                        args_list[-1] = request_model.parse_obj(args[-1])
                    else:
                        raise ValueError(f"Invalid request: {args[-1]}")
            except ValidationError as e:
                raise ValueError(f"Invalid request: {e}")

            # Wrap the async generator
            async for item in func(*args_list, **kwargs):
                # Validate the response
                # TODO: Validate the response
                try:
                    yield item
                except ValidationError as e:
                    raise ValueError(f"Invalid response: {e}")

        return wrapper

    return decorator
