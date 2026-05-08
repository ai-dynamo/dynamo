# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin output merge algorithms (DEP-XXXX PR 4).

Two algorithms (both **pure functions** — no I/O, no Clock dependency,
deterministic):

- ``type_aware_merge``: PROPOSE / RECONCILE / CONSTRAIN. Collects
  per-plugin ``OverrideResult``, groups by
  ``(sub_component_type, component_name)``, computes floor (max AT_LEAST) /
  ceiling (min AT_MOST) / recommendation (priority-smallest SET), clamps.
  v11 § G-2: REJECT > final priority.
- ``chain_augment``: PREDICT. Sequential layered prediction with
  partial-merge on ``optional float`` fields; ``final=True`` stops the chain.
  v11 § P1-2: runtime detection of "final at non-lowest priority" misuse.

``type_aware_merge`` is sync; ``chain_augment`` is async only because it
awaits plugin RPCs — the algorithmic logic itself is synchronous.

PR 4 landing order: 4-1 (this file + ``types``) → 4-2 (``type_aware``) →
4-5 (``chain_augment``). Re-exports below are added incrementally as each
sub-task lands; see ``DEP-XXXX_PR4_Detailed_zh.md``.
"""

from dynamo.planner.plugins.merge.chain_augment import chain_augment
from dynamo.planner.plugins.merge.type_aware import type_aware_merge
from dynamo.planner.plugins.merge.types import (
    ChainAugmentOutcome,
    ComponentKey,
    MergeOutcome,
    PluginResult,
    PredictPluginCallable,
)

__all__ = [
    "PluginResult",
    "ComponentKey",
    "MergeOutcome",
    "ChainAugmentOutcome",
    "PredictPluginCallable",
    "type_aware_merge",
    "chain_augment",
]
