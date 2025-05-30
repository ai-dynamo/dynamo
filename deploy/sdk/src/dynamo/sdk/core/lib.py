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

import logging
import os
from typing import Any, Callable, Optional, Type, TypeVar

from fastapi import FastAPI

from dynamo.sdk.core.protocol.interface import (
    DependencyInterface,
    DeploymentTarget,
    ServiceConfig,
    ServiceInterface,
)

G = TypeVar("G", bound=Callable[..., Any])

#  Note: global service provider.
# this should be set to a concrete implementation of the DeploymentTarget interface
_target: DeploymentTarget

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


def service(
    inner: Optional[Type[G]] = None,
    /,
    *,
    app: Optional[FastAPI] = None,
    system_app: Optional[FastAPI] = None,
    **kwargs: Any,
) -> Any:
    """Service decorator that's adapter-agnostic"""
    config = ServiceConfig(**kwargs)

    def decorator(inner: Type[G]) -> ServiceInterface[G]:
        provider = get_target()
        if inner is not None:
            config.dynamo.name = inner.__name__
        return provider.create_service(
            service_cls=inner,
            config=config,
            app=app,
            system_app=system_app,
            **kwargs,
        )

    ret = decorator(inner) if inner is not None else decorator
    return ret


def depends(
    on: Optional[ServiceInterface[G]] = None, **kwargs: Any
) -> DependencyInterface[G]:
    """Create a dependency using the current service provider"""
    provider = get_target()
    return provider.create_dependency(on=on, **kwargs)


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
