#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

import asyncio
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, get_type_hints
import inspect

from dynamo.runtime import DistributedRuntime
from dynamo.sdk.core.protocol.interface import (
    DynamoEndpointInterface,
    DynamoTransport,
    ServiceInterface,
)

T = TypeVar("T")


class DynamoEndpoint(DynamoEndpointInterface):
    """
    Base class for dynamo endpoints
    Dynamo endpoints are methods decorated with @endpoint.
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        transports: Optional[List[DynamoTransport]] = None,
        **kwargs,
    ):
        self.func = func
        self._name = name or func.__name__
        self._transports = transports or [DynamoTransport.DEFAULT]
        # Extract request type from hints
        hints = get_type_hints(func)
        args = list(hints.items())

        # Skip self/cls argument if present
        if args and args[0][0] in ("self", "cls"):
            args = args[1:]

        # Get request type from first arg if available
        self.request_type = args[0][1] if args else None
        wraps(func)(self)

        # Store additional metadata
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def name(self) -> str:
        return self._name

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self.func(*args, **kwargs)

    @property
    def transports(self) -> List[DynamoTransport]:
        return self._transports


def endpoint(
    name: Optional[str] = None,
    transports: Optional[List[DynamoTransport]] = None,
    **kwargs,
) -> Callable[[Callable], DynamoEndpoint]:
    """Decorator for dynamo endpoints."""

    def decorator(func: Callable) -> DynamoEndpoint:
        return DynamoEndpoint(func, name, transports, **kwargs)

    return decorator


def api(
    name: Optional[str] = None,
    **kwargs,
) -> Callable[[Callable], DynamoEndpoint]:
    """Decorator for dynamo endpoints."""

    def decorator(func: Callable) -> DynamoEndpoint:
        return DynamoEndpoint(func, name, transports=[DynamoTransport.HTTP], **kwargs)

    return decorator


def liveness(
    name: str = "/health/live",
    **kwargs,
) -> Callable[[Callable], DynamoEndpoint]:
    """Decorator for liveness probes.
    
    Args:
        name: Endpoint path, defaults to "/health/live"
        
    Example:
        @liveness
        def is_alive(self) -> tuple[int, dict]:
            return 200, {"status": "alive"}  # (status_code, response_dict)
    """
    
    def decorator(func: Callable) -> DynamoEndpoint:
        setattr(func, "__is_liveness_probe__", True)
        
        # Create a wrapper that converts the return value to a proper response
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from fastapi.responses import JSONResponse
            
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
                
            # Handle tuple return (status_code, content)
            if isinstance(result, tuple) and len(result) == 2:
                status_code, content = result
                return JSONResponse(content=content, status_code=status_code)
            # Already a response object
            elif hasattr(result, "status_code") and hasattr(result, "body"):
                return result
            # Just return the value as JSON with 200
            else:
                return JSONResponse(content=result, status_code=200)
                
        return DynamoEndpoint(wrapper, name, transports=[DynamoTransport.HTTP], **kwargs)
    
    return decorator


def readiness(
    name: str = "/health/ready",
    **kwargs,
) -> Callable[[Callable], DynamoEndpoint]:
    """Decorator for readiness probes.
    
    Args:
        name: Endpoint path, defaults to "/health/ready"
        
    Example:
        @readiness
        def is_ready(self) -> tuple[int, dict]:
            if self.model_loaded:
                return 200, {"status": "ready"}
            return 503, {"status": "initializing"}
    """
    
    def decorator(func: Callable) -> DynamoEndpoint:
        setattr(func, "__is_readiness_probe__", True)
        
        # Create a wrapper that converts the return value to a proper response
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from fastapi.responses import JSONResponse
            
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
                
            # Handle tuple return (status_code, content)
            if isinstance(result, tuple) and len(result) == 2:
                status_code, content = result
                return JSONResponse(content=content, status_code=status_code)
            # Already a response object
            elif hasattr(result, "status_code") and hasattr(result, "body"):
                return result
            # Just return the value as JSON with 200
            else:
                return JSONResponse(content=result, status_code=200)
                
        return DynamoEndpoint(wrapper, name, transports=[DynamoTransport.HTTP], **kwargs)
    
    return decorator


class DynamoClient:
    """Client for calling Dynamo endpoints with streaming support"""

    def __init__(self, service: ServiceInterface[Any]):
        self._service = service
        self._endpoints = service.get_dynamo_endpoints()
        self._dynamo_clients: Dict[str, Any] = {}
        self._runtime = None

    def __getattr__(self, name: str) -> Any:
        if name not in self._endpoints:
            raise AttributeError(
                f"No Dynamo endpoint '{name}' found on service '{self._service.name}'. "
                f"Available endpoints: {list(self._endpoints.keys())}"
            )

        # For streaming endpoints, create/cache the stream function
        if name not in self._dynamo_clients:
            namespace, component_name = self._service.dynamo_address()

            # Create async generator function that directly yields from the stream
            async def get_stream(*args, **kwargs):
                if self._runtime is not None:
                    runtime = self._runtime
                else:
                    loop = asyncio.get_running_loop()
                    runtime = DistributedRuntime(loop, False)
                    self._runtime = runtime
                    # Use existing runtime if available
                try:
                    # TODO: bis - dont recreate the client every time
                    client = (
                        await runtime.namespace(namespace)
                        .component(component_name)
                        .endpoint(name)
                        .client()
                    )
                    # Directly yield items from the stream
                    stream = await client.generate(*args, **kwargs)
                    async for item in stream:
                        yield item.data()
                except Exception as e:
                    raise e

            self._dynamo_clients[name] = get_stream
        return self._dynamo_clients[name]
