# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallelization config shared between the profiler and the planner.

``PickedParallelConfig`` stores the full ``(tp, pp, dp, moe_tp, moe_ep)`` tuple
that AIConfigurator's picker emits. Both the profiler (which picks) and the
planner (which consumes the pick to bootstrap perf models) need this type, so
it lives under ``dynamo.planner.config`` rather than the profiler tree.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PickedParallelConfig:
    """Lightweight representation of a picked parallelization config.

    Uses the same (tp, pp, dp, moe_tp, moe_ep) tuple that AIC's enumeration
    and picking pipelines produce.
    """

    tp: int = 1
    pp: int = 1
    dp: int = 1
    moe_tp: int = 1
    moe_ep: int = 1

    @property
    def num_gpus(self) -> int:
        return self.tp * self.pp * self.dp

    @property
    def tp_size(self) -> int:
        """Effective TP for KV-head splitting (TP or TEP; 1 for DEP).

        .. warning::
            This property has KV-head-split semantics — it is **NOT** the same
            quantity as AIConfigurator's ``ModelConfig.tp_size`` (which means
            attention TP per rank). Do not pass this value into AIC kwargs.
            Use the dedicated helper that derives AIC's tp_size from the full
            (tp, dp, moe_tp, moe_ep) tuple instead.
        """
        if self.moe_ep > 1:
            return 1
        if self.moe_tp > 1:
            return self.moe_tp
        return self.tp

    def label(self) -> str:
        if self.moe_ep > 1:
            return f"dep{self.moe_ep}"
        elif self.moe_tp > 1:
            return f"tep{self.moe_tp}"
        return f"tp{self.tp}"
