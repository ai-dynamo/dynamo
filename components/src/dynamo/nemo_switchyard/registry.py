#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Router registry and factory.

Concrete router modules call :func:`register_router` at import time to
make themselves discoverable.  The CLI uses :func:`create_router` to
instantiate the right implementation by name.
"""

from typing import Dict, List, Type

from .base import BaseModelRouter, RouterConfig

_ROUTER_REGISTRY: Dict[str, Type[BaseModelRouter]] = {}


def register_router(name: str, cls: Type[BaseModelRouter]) -> None:
    """Register a concrete router class under *name*."""
    if name in _ROUTER_REGISTRY:
        raise ValueError(
            f"Router '{name}' is already registered (existing: {_ROUTER_REGISTRY[name]}, "
            f"new: {cls})"
        )
    _ROUTER_REGISTRY[name] = cls


def get_router_class(name: str) -> Type[BaseModelRouter]:
    """Return the router class registered under *name*."""
    try:
        return _ROUTER_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_ROUTER_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown router type: {name!r}. Available routers: {available}"
        ) from None


def create_router(name: str, config: RouterConfig) -> BaseModelRouter:
    """Instantiate the router registered under *name* with the given *config*."""
    cls = get_router_class(name)
    return cls(config)


def list_routers() -> List[str]:
    """Return sorted list of registered router names."""
    return sorted(_ROUTER_REGISTRY)
