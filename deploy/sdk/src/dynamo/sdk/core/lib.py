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

import os
from typing import Any, Dict, Optional, Type, TypeVar, Union, Set, List, Tuple

from fastapi import FastAPI

from dynamo.sdk.core.protocol.interface import (
    DependencyInterface,
    DeploymentTarget,
    DynamoConfig,
    ServiceConfig,
    ServiceInterface,
    AbstractDynamoService,
    validate_dynamo_interfaces,
)
from dynamo.sdk.core.decorators.endpoint import DynamoEndpoint

T = TypeVar("T", bound=object)

#  Note: global service provider.
# this should be set to a concrete implementation of the DeploymentTarget interface
_target: DeploymentTarget

# Add global cache for abstract services
_abstract_service_cache: Dict[Type[AbstractDynamoService], ServiceInterface[Any]] = {}



DYNAMO_IMAGE = os.getenv("DYNAMO_IMAGE", "dynamo:latest-vllm")


def set_target(target: DeploymentTarget) -> None:
    """Set the global service provider implementation"""
    global _target
    _target = target


def get_target() -> DeploymentTarget:
    """Get the current service provider implementation"""
    global _target
    return _target


# Helper function to get or create service instance for AbstractDynamoService
def _get_or_create_abstract_service_instance(
    abstract_service_cls: Type[AbstractDynamoService], provider: DeploymentTarget
) -> ServiceInterface[Any]:
    """
    Retrieves a service instance from cache or creates a new one
    for the given AbstractDynamoService class.
    """
    global _abstract_service_cache
    if abstract_service_cls in _abstract_service_cache:
        return _abstract_service_cache[abstract_service_cls]
    else:
        # This placeholder service will be a singleton, and will be used for all dependencies that depend on this abstract service.
        # The name for DynamoConfig will be the class name of the abstract service.
        dynamo_config_for_abstract = DynamoConfig(enabled=True)

        # Call the main service() decorator/function to create the service instance
        service_instance = service(
            inner=abstract_service_cls,
            dynamo=dynamo_config_for_abstract
            # app will be None by default
            # **kwargs for ServiceConfig will be empty, so ServiceConfig({}) is created inside service()
        )
        _abstract_service_cache[abstract_service_cls] = service_instance
        return service_instance


# TODO: dynamo_component
def service(
    inner: Optional[Type[T]] = None,
    /,
    *,
    dynamo: Optional[Union[Dict[str, Any], DynamoConfig]] = None,
    app: Optional[FastAPI] = None,
    **kwargs: Any,
) -> Any:
    """Service decorator that's adapter-agnostic"""
    config = ServiceConfig(kwargs)
    # Parse dict into DynamoConfig object
    dynamo_config: Optional[DynamoConfig] = None
    if dynamo is not None:
        if isinstance(dynamo, dict):
            dynamo_config = DynamoConfig(**dynamo)
        else:
            dynamo_config = dynamo

    assert isinstance(dynamo_config, DynamoConfig)

    def decorator(inner: Type[T]) -> ServiceInterface[T]:
        # Validate Dynamo interfaces before creating the service
        _validate_dynamo_interfaces(inner)

        provider = get_target()
        if inner is not None:
            dynamo_config.name = inner.__name__
        return provider.create_service(
            service_cls=inner,
            config=config,
            dynamo_config=dynamo_config,
            app=app,
            **kwargs,
        )

    ret = decorator(inner) if inner is not None else decorator
    return ret


def depends(
    on: Optional[Union[ServiceInterface[T], Type[AbstractDynamoService]]] = None,
    **kwargs: Any
) -> DependencyInterface[T]:
    """Create a dependency using the current service provider.

    If 'on' is an AbstractDynamoService type, a placeholder service will be
    created and used for the dependency.
    """
    provider = get_target()
    actual_on_service: Optional[ServiceInterface[Any]] = None

    if isinstance(on, type) and issubclass(on, AbstractDynamoService):
        actual_on_service = _get_or_create_abstract_service_instance(on, provider)
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
            "depends() expects 'on' to be a ServiceInterface, an AbstractDynamoService type"
        )
