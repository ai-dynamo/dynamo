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
        """Build a label that is injective across all five parallel dimensions.

        Encodes ``(tp, pp, dp, moe_tp, moe_ep)`` so distinct topologies never
        collide. Default-1 dimensions are omitted for readability.

        Examples (MoE picks)::

            (tp=2, dp=1, moe_tp=2, moe_ep=1) -> "tp2-moetp2"
            (tp=2, dp=1, moe_tp=1, moe_ep=2) -> "tp2-moeep2"
            (tp=1, dp=2, moe_tp=1, moe_ep=2) -> "tp1-dp2-moeep2"
            (tp=4, dp=1, moe_tp=4, moe_ep=1) -> "tp4-moetp4"

        The prior 3-bucket form (``dep{moe_ep}`` / ``tep{moe_tp}`` / ``tp{tp}``)
        collapsed ``(tp=2, moe_ep=2)`` and ``(tp=1, dp=2, moe_ep=2)`` to the
        same ``"dep2"`` string, which corrupted ``thorough.py`` work_dir
        naming and ``aiconfigurator.sdk.picking`` ``groupby("parallel")``
        dedup.
        """
        parts = [f"tp{self.tp}"]
        if self.pp > 1:
            parts.append(f"pp{self.pp}")
        if self.dp > 1:
            parts.append(f"dp{self.dp}")
        if self.moe_tp > 1:
            parts.append(f"moetp{self.moe_tp}")
        if self.moe_ep > 1:
            parts.append(f"moeep{self.moe_ep}")
        return "-".join(parts)


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
