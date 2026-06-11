# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallelization config shared between the profiler and the planner.

``PickedParallelConfig`` stores the full ``(tp, pp, dp, moe_tp, moe_ep)`` tuple
that AIConfigurator's picker emits. Both the profiler (which picks) and the
planner (which consumes the pick to bootstrap perf models) need this type, so
it lives under ``dynamo.planner.config`` rather than the profiler tree.

It is a pydantic ``BaseModel`` so it serialises cleanly into the planner
ConfigMap as part of ``AICInterpolationSpec``.
"""

from pydantic import BaseModel, ConfigDict


class PickedParallelConfig(BaseModel):
    """Lightweight representation of a picked parallelization config.

    Uses the same ``(tp, pp, dp, moe_tp, moe_ep)`` tuple that AIC's enumeration
    and picking pipelines produce. Frozen so instances are hashable.
    """

    model_config = ConfigDict(frozen=True)

    tp: int = 1
    pp: int = 1
    dp: int = 1
    moe_tp: int = 1
    moe_ep: int = 1

    @property
    def num_gpus(self) -> int:
        """Physical GPU count per engine across all pipeline stages.

        ``dp`` models attention-DP for AIC and interpolation purposes. When
        attention-DP is layered on top of tensor parallelism it reuses the
        existing TP ranks rather than multiplying the engine width again, so we
        treat the widest parallel axis per pipeline stage as the physical width.

        For dense picks this reduces to ``max(tp, dp)``; for MoE picks the
        expert width ``moe_tp * moe_ep`` captures the same physical engine
        width and keeps pure DEP / hybrid MoE picks at the expected GPU count.
        """
        stage_width = max(self.tp, self.dp, self.moe_tp * self.moe_ep)
        return self.pp * stage_width

    @property
    def tp_size(self) -> int:
        """Effective TP for KV-head splitting (TP or TEP; 1 for DEP).

        .. warning::
            KV-head-split semantics ONLY. This is **NOT** the same quantity as
            AIConfigurator's ``ModelConfig.tp_size`` (which is attention TP
            per rank). Never pass this value into AIC kwargs — use
            :func:`picked_to_aic_model_config_kwargs` instead.
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


def picked_to_aic_model_config_kwargs(p: PickedParallelConfig) -> dict[str, int]:
    """Map a ``PickedParallelConfig`` to AIConfigurator ``ModelConfig`` kwargs.

    Returned keys: ``tp_size``, ``pp_size``, ``moe_tp_size``, ``moe_ep_size``,
    ``attention_dp_size``.

    For MoE picks AIC's picker always emits
    ``tp × dp == moe_tp × moe_ep`` (the attention-layer GPU width matches the
    MoE-layer GPU width per replica), so the mapping is simply:

    * ``tp_size = p.tp`` (AIC's attention TP per rank)
    * ``attention_dp_size = p.dp``
    * ``moe_tp_size = p.moe_tp``
    * ``moe_ep_size = p.moe_ep``
    * ``pp_size = p.pp``

    This satisfies AIC's MoE-only assertion
    ``tp_size × attention_dp_size == moe_tp_size × moe_ep_size`` by
    construction. For dense picks (``moe_tp = moe_ep = 1``) the assertion
    does not apply — AIC's ``BaseModel`` ignores the MoE fields.

    Do **not** derive ``tp_size`` from :attr:`PickedParallelConfig.tp_size`
    — that property has KV-head-split semantics that conflict with AIC's
    definition (it returns 1 for DEP, which breaks the identity).
    """
    return {
        "tp_size": p.tp,
        "pp_size": p.pp,
        "moe_tp_size": p.moe_tp,
        "moe_ep_size": p.moe_ep,
        "attention_dp_size": p.dp,
    }
