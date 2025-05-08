import os
from typing import Any, Dict, Optional, Type, TypeVar, Union

from fastapi import FastAPI

from dynamo.sdk.core.protocol.interface import (
    DependencyInterface,
    DeploymentTarget,
    DynamoConfig,
    ServiceConfig,
    ServiceInterface,
)

T = TypeVar("T", bound=object)

#  Note: global service provider.
# this should be set to a concrete implementation of the DeploymentTarget interface
_target: DeploymentTarget


DYNAMO_IMAGE = os.getenv("DYNAMO_IMAGE", "dynamo:latest-vllm")


def set_target(target: DeploymentTarget) -> None:
    """Set the global service provider implementation"""
    global _target
    _target = target


def get_target() -> DeploymentTarget:
    """Get the current service provider implementation"""
    global _target
    return _target


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

    def decorator(inner: Type[T]) -> ServiceInterface[T]:
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
    on: Optional[ServiceInterface[T]] = None, **kwargs: Any
) -> DependencyInterface[T]:
    """Create a dependency using the current service provider"""
    provider = get_target()
    return provider.create_dependency(on=on, **kwargs)
