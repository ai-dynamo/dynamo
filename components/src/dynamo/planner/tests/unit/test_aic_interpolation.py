# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AIC interpolation handoff types and helpers.

The ``run_aic_interpolation`` sweep itself is tested in a separate follow-up
file once that module lands; this file covers the pure-Python helpers that
don't require ``aiconfigurator`` to be installed.
"""

import pytest

from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
from dynamo.planner.config.parallelization import (
    PickedParallelConfig,
    picked_to_aic_model_config_kwargs,
)


class TestPickedToAicKwargs:
    """Verify the pick → AIC ModelConfig kwargs helper for each strategy.

    The invariant AIC enforces for MoE models (aiconfigurator sdk/models.py,
    ~8 assertion sites) is::

        tp_size * attention_dp_size == moe_tp_size * moe_ep_size

    Each test case asserts both the expected kwargs and the identity.
    """

    @staticmethod
    def _assert_identity(kw: dict) -> None:
        assert (
            kw["tp_size"] * kw["attention_dp_size"]
            == kw["moe_tp_size"] * kw["moe_ep_size"]
        ), f"AIC identity violated for {kw}"

    def test_tp_only_dense(self):
        # Dense 8-GPU TP pick; moe_tp/moe_ep are 1. The MoE identity does NOT
        # apply to dense models — AIC's BaseModel doesn't assert it. We only
        # check that tp_size carries p.tp and the MoE slots default to 1.
        p = PickedParallelConfig(tp=8, pp=1, dp=1, moe_tp=1, moe_ep=1)
        kw = picked_to_aic_model_config_kwargs(p)
        assert kw == {
            "tp_size": 8,
            "pp_size": 1,
            "moe_tp_size": 1,
            "moe_ep_size": 1,
            "attention_dp_size": 1,
        }

    def test_tp_only_moe(self):
        # MoE TP-only on a MOE_ADDITIONAL_TP_ARCHITECTURES model, 8 GPUs.
        p = PickedParallelConfig(tp=8, pp=1, dp=1, moe_tp=8, moe_ep=1)
        kw = picked_to_aic_model_config_kwargs(p)
        assert kw["tp_size"] == 8
        assert kw["moe_tp_size"] == 8
        assert kw["moe_ep_size"] == 1
        assert kw["attention_dp_size"] == 1
        self._assert_identity(kw)

    def test_tep(self):
        # TEP-8: attention and experts both sharded across 8 ranks.
        p = PickedParallelConfig(tp=8, pp=1, dp=1, moe_tp=8, moe_ep=1)
        kw = picked_to_aic_model_config_kwargs(p)
        self._assert_identity(kw)

    def test_dep(self):
        # DEP-8: attention replicated across 8 DP ranks; experts split by EP=8.
        p = PickedParallelConfig(tp=1, pp=1, dp=8, moe_tp=1, moe_ep=8)
        kw = picked_to_aic_model_config_kwargs(p)
        assert kw == {
            "tp_size": 1,
            "pp_size": 1,
            "moe_tp_size": 1,
            "moe_ep_size": 8,
            "attention_dp_size": 8,
        }
        self._assert_identity(kw)

    def test_hybrid_tep_plus_dp(self):
        # Hybrid: attention TP=2 × DP=4, MoE TP=2 × EP=4, 16 total GPUs.
        p = PickedParallelConfig(tp=2, pp=1, dp=4, moe_tp=2, moe_ep=4)
        kw = picked_to_aic_model_config_kwargs(p)
        assert kw["tp_size"] == 2
        assert kw["attention_dp_size"] == 4
        assert kw["moe_tp_size"] == 2
        assert kw["moe_ep_size"] == 4
        self._assert_identity(kw)

    def test_never_uses_tp_size_property(self):
        # Regression guard: the KV-head-split .tp_size returns 1 for DEP
        # which would silently break AIC's identity. Confirm the helper
        # does NOT derive from that property.
        p = PickedParallelConfig(tp=1, dp=8, moe_ep=8)
        assert p.tp_size == 1  # KV-head-split semantics
        kw = picked_to_aic_model_config_kwargs(p)
        # tp_size in AIC terms equals p.tp (1 here), not derived from p.tp_size
        assert kw["tp_size"] == p.tp == 1
        self._assert_identity(kw)


class TestAICInterpolationSpec:
    def test_json_roundtrip(self):
        spec = AICInterpolationSpec(
            hf_id="Qwen/Qwen3-235B-A22B-FP8",
            system="h200_sxm",
            backend="trtllm",
            isl=3000,
            osl=300,
            sweep_max_context_length=8192,
            prefill_interpolation_granularity=16,
            decode_interpolation_granularity=6,
            prefill_pick=PickedParallelConfig(tp=4, pp=1, dp=4, moe_tp=1, moe_ep=4),
            decode_pick=PickedParallelConfig(tp=1, pp=1, dp=8, moe_tp=2, moe_ep=4),
        )
        roundtrip = AICInterpolationSpec.model_validate_json(spec.model_dump_json())
        assert roundtrip == spec

    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError):
            AICInterpolationSpec(
                hf_id="x",
                system="h200_sxm",
                backend="bogus",  # type: ignore[arg-type]
                isl=1,
                osl=1,
                sweep_max_context_length=1,
                prefill_interpolation_granularity=1,
                decode_interpolation_granularity=1,
                prefill_pick=PickedParallelConfig(),
                decode_pick=PickedParallelConfig(),
            )

    def test_positive_int_constraints(self):
        # isl, osl, sweep_max_context_length, granularities must all be > 0.
        base_kwargs = dict(
            hf_id="x",
            system="h200_sxm",
            backend="trtllm",
            sweep_max_context_length=8192,
            prefill_interpolation_granularity=16,
            decode_interpolation_granularity=6,
            prefill_pick=PickedParallelConfig(),
            decode_pick=PickedParallelConfig(),
        )
        with pytest.raises(ValueError):
            AICInterpolationSpec(isl=0, osl=100, **base_kwargs)
        with pytest.raises(ValueError):
            AICInterpolationSpec(isl=100, osl=0, **base_kwargs)
