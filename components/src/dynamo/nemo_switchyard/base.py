#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Base classes for the pluggable router abstraction layer.

Provides :class:`RouterConfig` (configuration dataclass) and
:class:`BaseModelRouter` (abstract base class) that every concrete
router must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RouterConfig:
    """Configuration shared by all router implementations.

    Attributes:
        pool_names: Ordered list of pool names (e.g. ["strong", "weak"]).
        fallback_pool: Pool to route to on error.  Defaults to ``pool_names[0]``.
        extra: Router-specific passthrough options (e.g. threshold, model_path).
    """

    pool_names: List[str]
    fallback_pool: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.pool_names:
            raise ValueError("pool_names must contain at least one pool")
        if self.fallback_pool is None:
            self.fallback_pool = self.pool_names[0]
        if self.fallback_pool not in self.pool_names:
            raise ValueError(
                f"fallback_pool '{self.fallback_pool}' is not in pool_names {self.pool_names}"
            )


class BaseModelRouter(ABC):
    """Abstract base class for model routers.

    Subclasses must implement :meth:`route`.  Optionally override
    :meth:`initialize` for async setup, :meth:`cleanup` for teardown,
    and :meth:`get_stats` for observability.
    """

    def __init__(self, config: RouterConfig):
        self.config = config

    async def initialize(self) -> None:
        """Optional async setup hook (load models, connect to services, etc.)."""

    async def cleanup(self) -> None:
        """Optional teardown hook — release models, GPU memory, connections, etc."""

    @abstractmethod
    async def route(self, request: dict) -> str:
        """Determine which pool should handle *request*.

        Args:
            request: The full ``PreprocessedRequest`` dict (with token_ids, etc.).

        Returns:
            A pool name from ``self.config.pool_names``.
        """
        ...

    def get_stats(self) -> dict:
        """Optional observability hook — return routing statistics."""
        return {}
