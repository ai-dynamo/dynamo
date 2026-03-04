#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Built-in router implementations.

RouteLLM is optional — if ``routellm`` / ``transformers`` are not
installed, the RouteLLM router simply won't be available.
"""

_ROUTERS = {}

try:
    from .routellm_router import RouteLLMRouter

    _ROUTERS["routellm"] = RouteLLMRouter
except ImportError:
    pass


def get_router_class(name: str):
    """Return the router class for *name*, or raise KeyError."""
    try:
        return _ROUTERS[name]
    except KeyError:
        available = ", ".join(sorted(_ROUTERS)) or "(none)"
        raise KeyError(
            f"Unknown router type: {name!r}. Available routers: {available}"
        ) from None


def list_routers():
    """Return sorted list of available router names."""
    return sorted(_ROUTERS)
