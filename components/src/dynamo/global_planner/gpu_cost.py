# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared GPU cost per pool, for pools whose DGD does not request GPUs.

The GPU budget is computed from each pool's ``gpu_per_replica``, read from the
DGD's ``resources.limits["nvidia.com/gpu"]``. A pool that requests no GPU has no
cost to read, and :meth:`Service.get_gpu_count` raises -- so with a budget
enabled the whole snapshot fails and every scale request errors, and with the
budget disabled there is no arbitration at all.

That rules out the entire class of GPU-free deployments, most importantly
**mocker-based testing**: mockers exist precisely so a topology can be exercised
without hardware, but they cannot participate in the budget they are meant to
help test.

This table closes that gap by letting an operator declare what a pool *costs*
without its DGD requesting real hardware:

.. code-block:: yaml

    gpu_cost:
      pools:
        - selector: "sachalm/dsv4-**"
          gpu_per_replica: 8
        - selector: "sachalm/gpt-oss-**"
          gpu_per_replica: 4

**A declared cost is a fallback, not an override.** It applies only where the
DGD is silent. A pool that really does request GPUs is always charged what the
cluster says, so a stale or wrong config can never cause the planner to
under-count real hardware.

The local planner already works this way -- ``_initialize_gpu_counts`` falls
back to configured GPU counts when the DGD read fails, noting it is "useful for
mockers that don't specify GPU resources". This is the same idea on the global
side.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamo.global_planner.pool_selectors import (
    PoolSelector,
    first_match,
    order_by_specificity,
    reject_duplicate_selectors,
)

logger = logging.getLogger(__name__)


class PoolGpuCost(PoolSelector):
    """GPUs charged per replica for the pools this selector covers."""

    gpu_per_replica: int = Field(
        ge=0,
        description=(
            "GPUs to charge per replica for matching pools whose DGD does not "
            "request GPUs. 0 excludes them from budget totals entirely."
        ),
    )


class GpuCostConfig(BaseModel):
    """Declared GPU costs, most specific selector wins."""

    model_config = ConfigDict(extra="forbid")

    pools: list[PoolGpuCost] = Field(
        default_factory=list,
        description="Selector-scoped GPU costs for pools the DGD prices at nothing.",
    )

    @model_validator(mode="after")
    def _reject_duplicate_selectors(self) -> "GpuCostConfig":
        reject_duplicate_selectors(self.pools)
        return self


class GpuCostResolver:
    """Resolves a pool's declared GPU cost from a :class:`GpuCostConfig`."""

    def __init__(self, config: Optional[GpuCostConfig] = None):
        self.config = config or GpuCostConfig()
        self._ordered = order_by_specificity(self.config.pools)

    def resolve(self, participant_id: str, sub_type: str) -> Optional[int]:
        """Declared GPUs per replica for one pool, or ``None`` if undeclared.

        ``None`` means "no opinion" and leaves the caller to fail the way it
        would have without this table at all, rather than silently pricing an
        unknown pool at zero and quietly shrinking the budget.
        """
        entry = first_match(self._ordered, participant_id, sub_type)
        return entry.gpu_per_replica if entry is not None else None

    def declared(self) -> bool:
        """Whether any cost is declared at all."""
        return bool(self._ordered)
