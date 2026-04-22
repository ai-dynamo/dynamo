# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Semantic KV cache integration for the standalone router.

Provides helper functions to create and wire a SemanticKvCacheProvider
into the StandaloneRouterHandler's request flow.

Usage in router __main__.py::

    from dynamo.router.semantic import create_semantic_provider

    provider = create_semantic_provider(config.semantic_kv_provider)
"""

from __future__ import annotations

import logging
from typing import Optional

from dynamo.llm.semantic_kv import SemanticKvCacheProvider

logger = logging.getLogger(__name__)

# Registry of built-in provider names → factory functions.
_BUILTIN_PROVIDERS = {}


def _register_builtin(name: str):
    """Decorator to register a built-in provider factory."""

    def decorator(fn):
        _BUILTIN_PROVIDERS[name] = fn
        return fn

    return decorator


@_register_builtin("simple")
def _create_simple_provider(
    min_similarity: float = 0.6,
    max_donors: int = 10_000,
) -> SemanticKvCacheProvider:
    """Create the SimpleSemanticProvider reference implementation."""
    from dynamo.llm.semantic_kv_simple import SimpleSemanticProvider

    return SimpleSemanticProvider(
        min_similarity=min_similarity,
        max_donors=max_donors,
    )


def create_semantic_provider(
    provider_name: Optional[str],
) -> Optional[SemanticKvCacheProvider]:
    """Create a semantic KV cache provider by name.

    Args:
        provider_name: Name of the provider to create. Built-in options:
            - ``"simple"`` — Jaccard similarity reference implementation.
            - ``None`` — Disable semantic KV cache (default).

    Returns:
        A provider instance, or ``None`` if semantic KV is disabled.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    if not provider_name:
        return None

    factory = _BUILTIN_PROVIDERS.get(provider_name)
    if factory is not None:
        provider = factory()
        logger.info("Semantic KV cache enabled: provider=%s", provider_name)
        return provider

    raise ValueError(
        f"Unknown semantic KV provider: {provider_name!r}. "
        f"Available: {', '.join(sorted(_BUILTIN_PROVIDERS))} "
        f"or implement SemanticKvCacheProvider protocol."
    )
