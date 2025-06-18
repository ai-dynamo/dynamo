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
import functools
import logging
import os
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from fastapi import FastAPI

from dynamo.sdk.core.protocol.interface import (
    AbstractService,
    DependencyInterface,
    DeploymentTarget,
    DynamoConfig,
    ServiceConfig,
    ServiceInterface,
    validate_dynamo_interfaces,
)

G = TypeVar("G", bound=Callable[..., Any])

#  Note: global service provider.
# this should be set to a concrete implementation of the DeploymentTarget interface
_target: DeploymentTarget

# Add global cache for abstract services
_abstract_service_cache: Dict[Type[AbstractService], ServiceInterface[Any]] = {}


logger = logging.getLogger(__name__)

DYNAMO_IMAGE = os.getenv("DYNAMO_IMAGE", "dynamo:latest-vllm")


def set_target(target: DeploymentTarget) -> None:
    """Set the global service provider implementation"""
    global _target
    _target = target


def get_target() -> DeploymentTarget:
    """Get the current service provider implementation"""
    global _target
    return _target


# Helper function to get or create service instance for AbstractService
def _get_or_create_abstract_service_instance(
    abstract_service_cls: Type[AbstractService],
) -> ServiceInterface[Any]:
    """
    Retrieves a service instance from cache or creates a new one
    for the given AbstractService class.
    """
    global _abstract_service_cache

    if abstract_service_cls in _abstract_service_cache:
        return _abstract_service_cache[abstract_service_cls]

    # This placeholder service will be a singleton, and will be used for all dependencies that depend on this abstract service.
    # The name for DynamoConfig will be the class name of the abstract service.
    dynamo_config_for_abstract = DynamoConfig(enabled=True)

    # Call the main service() decorator/function to create the service instance
    # validate_dynamo_interfaces is False because validating an interface has implemented dynamo endpoints will obviously fail
    service_instance = service(
        abstract_service_cls,
        dynamo=dynamo_config_for_abstract,
        should_validate_dynamo_interfaces=False,
    )
    _abstract_service_cache[abstract_service_cls] = service_instance
    return service_instance


def service(
    inner: Optional[Type[G]] = None,
    /,
    *,
    app: Optional[FastAPI] = None,
    should_validate_dynamo_interfaces: bool = True,
    system_app: Optional[FastAPI] = None,
    **kwargs: Any,
) -> Any:
    """Service decorator that's adapter-agnostic"""

    config = ServiceConfig(**kwargs)

    def decorator(inner: Type[G]) -> ServiceInterface[G]:
        # Ensures that all declared dynamo endpoints on the parent interfaces are implemented
        if should_validate_dynamo_interfaces:
            validate_dynamo_interfaces(inner)
        provider = get_target()
        if inner is not None:
            config.dynamo.name = inner.__name__
        service_instance = provider.create_service(
            service_cls=inner,
            config=config,
            app=app,
            system_app=system_app,
            **kwargs,
        )
        return service_instance

    ret = decorator(inner) if inner is not None else decorator
    return ret


def depends(
    on: Optional[Union[ServiceInterface[G], Type[AbstractService]]] = None,
    **kwargs: Any,
) -> DependencyInterface[G]:
    """Create a dependency using the current service provider.

    If 'on' is an AbstractService type, a placeholder service will be
    created and used for the dependency.
    """
    provider = get_target()
    actual_on_service: Optional[ServiceInterface[Any]] = None

    if isinstance(on, type) and issubclass(on, AbstractService):
        actual_on_service = _get_or_create_abstract_service_instance(on)
        # The type of actual_on_service here would be ServiceInterface[NameOfAbstractClass]
        # So, T would be NameOfAbstractClass.
        return provider.create_dependency(on=actual_on_service, **kwargs)
    elif isinstance(on, ServiceInterface):
        # This handles both 'on=None' and 'on=SomeServiceInterfaceInstance'
        # If 'on' is ServiceInterface[K], T could be K. If 'on' is None, T remains unbound here.
        actual_on_service = on
        return provider.create_dependency(on=actual_on_service, **kwargs)
    else:
        raise TypeError(
            "depends() expects 'on' to be a ServiceInterface, an AbstractService type"
        )


def liveness(func: G) -> G:
    """Decorator for liveness probe."""
    if not callable(func):
        raise TypeError("@liveness can only decorate callable methods")

    func.__is_liveness_probe__ = True  # type: ignore
    return func


def get_liveness_handler(obj):
    for attr in dir(obj):
        fn = getattr(obj, attr)
        if callable(fn) and getattr(fn, "__is_liveness_probe__", False):
            return fn
    return None


def readiness(func: G) -> G:
    """Decorator for readiness probe."""
    if not callable(func):
        raise TypeError("@readiness can only decorate callable methods")

    func.__is_readiness_probe__ = True  # type: ignore
    return func


def get_readiness_handler(obj):
    for attr in dir(obj):
        fn = getattr(obj, attr)
        if callable(fn) and getattr(fn, "__is_readiness_probe__", False):
            return fn
    return None


def health_check(func):
    """
    Decorator for custom health check handler.
    The function must return a tuple of (status, details) where status is either 'healthy' or 'unhealthy'
    and details is a string describing the health status.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            try:
                loop = asyncio.get_running_loop()
                result = loop.run_until_complete(result)  # Already in event loop
            except RuntimeError:
                result = asyncio.run(result)  # No event loop
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                "Health check function must return (status, details) tuple"
            )
        status, details = result
        if status not in ["healthy", "unhealthy"]:
            raise ValueError("status must be 'healthy' or 'unhealthy'")
        return (status, details)

    wrapper.__is_health_check__ = True  # Add a marker to the wrapper function
    return wrapper


def get_health_check_handlers(obj):
    """Get all health check handlers from an object."""
    handlers = []
    for attr in dir(obj):
        fn = getattr(obj, attr)
        if callable(fn) and getattr(fn, "__is_health_check__", False):
            handlers.append(fn)
    return handlers
