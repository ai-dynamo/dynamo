# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Type, Union

from pydantic import BaseModel, ValidationError

# List all the classes in the _core module for re-export
# import * causes "unable to detect undefined names"
from dynamo._core import Client as Client
from dynamo._core import Context as Context
from dynamo._core import DistributedRuntime as DistributedRuntime
from dynamo._core import Endpoint as Endpoint


def dynamo_worker(enable_nats: bool = True):
    """
    Decorator that creates a DistributedRuntime and passes it to the worker function.

    Args:
        enable_nats: Whether to enable NATS for KV events. Defaults to True.
                    If request_plane is "nats", NATS is always enabled.
                    Pass False (via --no-kv-events flag) to disable NATS initialization.

    Examples:
        >>> from dynamo.runtime import DistributedRuntime, dynamo_worker
        >>>
        >>> @dynamo_worker()
        ... async def worker(runtime: DistributedRuntime):
        ...     endpoint = runtime.endpoint("dynamo.backend.generate")
        ...     await endpoint.serve_endpoint(handler.generate)
        >>>
        >>> asyncio.run(worker())
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_running_loop()
            request_plane = os.environ.get("DYN_REQUEST_PLANE", "tcp")
            discovery_backend = os.environ.get("DYN_DISCOVERY_BACKEND", "etcd")
            runtime = DistributedRuntime(
                loop, discovery_backend, request_plane, enable_nats
            )

            await func(runtime, *args, **kwargs)

            # # wait for one of
            # # 1. the task to complete
            # # 2. the task to be cancelled

            # done, pending = await asyncio.wait({task, cancelled}, return_when=asyncio.FIRST_COMPLETED)

            # # i want to catch a SIGINT or SIGTERM or a cancellation event here

            # try:
            #     # Call the actual function
            #     return await func(runtime, *args, **kwargs)
            # finally:
            #     print("Decorator: Cleaning up runtime resources")
            #     # Perform cleanup actions here

        return wrapper

    return decorator


def dynamo_endpoint(
    request_model: Union[Type[BaseModel], Type[Any]], response_model: Type[BaseModel]
) -> Callable:
    """
    Decorator that validates request/response Pydantic models on an async generator endpoint.

    Args:
        request_model: Pydantic model class (or Any) for incoming requests.
        response_model: Pydantic model class for streamed response items.

    Examples:
        >>> from pydantic import BaseModel
        >>> from dynamo.runtime import dynamo_endpoint
        >>>
        >>> class Request(BaseModel):
        ...     data: str
        >>> class Response(BaseModel):
        ...     char: str
        >>>
        >>> class RequestHandler:
        ...     @dynamo_endpoint(Request, Response)
        ...     async def generate(self, request):
        ...         for char in request.data:
        ...             yield char
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
