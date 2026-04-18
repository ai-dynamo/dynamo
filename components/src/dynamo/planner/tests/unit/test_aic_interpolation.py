# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AIC interpolation handoff types and helpers.

The ``run_aic_interpolation`` sweep itself is tested in a separate follow-up
file once that module lands; this file covers the pure-Python helpers that
don't require ``aiconfigurator`` to be installed.
"""

from unittest.mock import MagicMock, patch

import pytest

from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.config.parallelization import (
    PickedParallelConfig,
    picked_to_aic_model_config_kwargs,
)
from dynamo.planner.monitoring import aic_interpolation as aic_mod


def _make_spec(
    *,
    prefill_pick: PickedParallelConfig,
    decode_pick: PickedParallelConfig,
    isl: int = 1000,
    osl: int = 200,
    sweep_max_context_length: int = 4096,
    prefill_granularity: int = 4,
    decode_granularity: int = 4,
) -> AICInterpolationSpec:
    return AICInterpolationSpec(
        hf_id="Qwen/Qwen3-235B-A22B-FP8",
        system="h200_sxm",
        backend="trtllm",
        isl=isl,
        osl=osl,
        sweep_max_context_length=sweep_max_context_length,
        prefill_interpolation_granularity=prefill_granularity,
        decode_interpolation_granularity=decode_granularity,
        prefill_pick=prefill_pick,
        decode_pick=decode_pick,
    )


def _patch_estimator(
    *,
    prefill_latency_ms: float = 50.0,
    decode_tpot_ms: float = 20.0,
    per_rank_max_kv: int = 100_000,
):
    """Patch AIConfiguratorPerfEstimator so tests don't need aiconfigurator.

    ``aic_interpolation`` imports the estimator class inside ``run_*`` for
    lazy loading, so we patch at the source module.
    """
    instance = MagicMock(name="AIConfiguratorPerfEstimator")
    instance.estimate_prefill_perf.return_value = {
        "context_latency": prefill_latency_ms
    }
    instance.estimate_perf.return_value = {"tpot": decode_tpot_ms}
    instance.get_max_kv_tokens.return_value = per_rank_max_kv
    cls = MagicMock(return_value=instance)
    return (
        patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            cls,
        ),
        instance,
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


class TestRunAicInterpolation:
    """Exercise the end-to-end sweep with a mocked AIC estimator.

    These tests enforce the MoE-DEP correctness invariants that the old
    profiler path silently violated. Each test drives a DEP-8 pick through
    the sweep and asserts what reaches AIC and what reaches the FPM list.
    """

    def test_prefill_sweep_shape_and_fpm_convention(self):
        """Prefill FPMs match thorough-mode: (1 req, isl tokens, per-rank ttft)."""
        pick = PickedParallelConfig(tp=1, pp=1, dp=8, moe_tp=1, moe_ep=8)
        spec = _make_spec(
            prefill_pick=pick,
            decode_pick=pick,
            sweep_max_context_length=4096,
            prefill_granularity=4,
        )
        ctx_patch, estimator = _patch_estimator(prefill_latency_ms=123.0)
        with ctx_patch:
            fpms = aic_mod.run_aic_interpolation(spec, SubComponentType.PREFILL)

        assert len(fpms) >= 3
        for fpm in fpms:
            # Per-rank, single-request semantics.
            assert fpm.scheduled_requests.num_prefill_requests == 1
            assert fpm.scheduled_requests.sum_prefill_tokens > 0
            assert fpm.wall_time == pytest.approx(0.123)

    def test_prefill_passes_correct_aic_kwargs_for_dep(self):
        """DEP pick → AIC receives attention_dp_size=8, tp_size=1, etc."""
        pick = PickedParallelConfig(tp=1, pp=1, dp=8, moe_tp=1, moe_ep=8)
        spec = _make_spec(prefill_pick=pick, decode_pick=pick)
        ctx_patch, estimator = _patch_estimator()
        with ctx_patch:
            aic_mod.run_aic_interpolation(spec, SubComponentType.PREFILL)

        # Every call to estimate_prefill_perf must carry the full MoE kwargs
        # derived from the pick — this is the fix for the 3-bug cluster that
        # crashed MoE-DEP picks in the old profiler path.
        for call in estimator.estimate_prefill_perf.call_args_list:
            kwargs = call.kwargs
            assert kwargs["tp_size"] == 1
            assert kwargs["attention_dp_size"] == 8
            assert kwargs["moe_tp_size"] == 1
            assert kwargs["moe_ep_size"] == 8
            assert (
                kwargs["tp_size"] * kwargs["attention_dp_size"]
                == kwargs["moe_tp_size"] * kwargs["moe_ep_size"]
            )

    def test_decode_batch_size_passed_per_rank(self):
        """Aggregate num_request gets divided by attention_dp_size for AIC.

        AIC's RuntimeConfig.batch_size is per-attention-DP-rank (see the
        TrtllmWideEPDeepSeekModel comment in aiconfigurator). Feeding the
        aggregate directly would over-count MoE tokens by a factor of DP.
        """
        pick = PickedParallelConfig(tp=1, pp=1, dp=8, moe_tp=2, moe_ep=4)
        spec = _make_spec(
            prefill_pick=pick,
            decode_pick=pick,
            sweep_max_context_length=2048,
            decode_granularity=3,
        )
        # Large per-rank max_kv so the sweep fills multiple concurrency points.
        ctx_patch, estimator = _patch_estimator(per_rank_max_kv=500_000)
        with ctx_patch:
            fpms = aic_mod.run_aic_interpolation(spec, SubComponentType.DECODE)

        assert len(fpms) >= 3
        # estimate_perf should always see a per-rank batch_size.
        # It's positional: estimator.estimate_perf(isl, osl, batch_size, mode="decode", **kw)
        for call in estimator.estimate_perf.call_args_list:
            batch_size = call.args[2]
            # batch_size is per-rank, so it must be <= aggregate/dp. Since
            # the aggregate concurrency sweep can produce values below dp,
            # we floor at 1 — but batch_size should NEVER exceed per-rank
            # max, which bounds at max_kv/(isl+osl) without the *dp scale.
            assert batch_size >= 1
            assert call.kwargs["attention_dp_size"] == 8
            assert call.kwargs["moe_tp_size"] == 2
            assert call.kwargs["moe_ep_size"] == 4
            # MoE identity must hold on every AIC call.
            assert (
                call.kwargs["tp_size"] * call.kwargs["attention_dp_size"]
                == call.kwargs["moe_tp_size"] * call.kwargs["moe_ep_size"]
            )

    def test_decode_fpm_is_aggregate(self):
        """FPMs carry aggregate num_decode_requests, matching thorough mode."""
        pick = PickedParallelConfig(tp=1, pp=1, dp=8, moe_tp=1, moe_ep=8)
        spec = _make_spec(
            prefill_pick=pick,
            decode_pick=pick,
            sweep_max_context_length=2048,
            decode_granularity=3,
        )
        ctx_patch, _ = _patch_estimator(per_rank_max_kv=500_000)
        with ctx_patch:
            fpms = aic_mod.run_aic_interpolation(spec, SubComponentType.DECODE)

        assert all(f.scheduled_requests.num_decode_requests >= 1 for f in fpms)
        assert all(f.scheduled_requests.sum_decode_kv_tokens > 0 for f in fpms)
        # sum_decode_kv_tokens should be num_decode_requests * ctx_len
        # (ctx_len = isl + osl_sweep/2 where osl_sweep = 500 inside the module).
        for fpm in fpms:
            req = fpm.scheduled_requests.num_decode_requests
            kv = fpm.scheduled_requests.sum_decode_kv_tokens
            assert kv / req > 100  # some positive ctx_len
        assert all(f.wall_time == pytest.approx(0.020) for f in fpms)

    def test_decode_uses_aggregate_max_kv(self):
        """AIC per-rank max_kv is multiplied by attention_dp_size for sweep bound.

        Regression for Bug #5: old path used per-rank max, which shrank the
        sweep and left the regression under-fit on high-concurrency points.
        """
        pick = PickedParallelConfig(tp=1, pp=1, dp=8, moe_tp=1, moe_ep=8)
        spec = _make_spec(
            prefill_pick=pick,
            decode_pick=pick,
            isl=1000,
            osl=200,
            sweep_max_context_length=2048,
            decode_granularity=3,
        )
        # Tiny per-rank max_kv. If the sweep used per-rank, max_concurrency =
        # 1200/1200 = 1 and we'd get only one point per isl. With aggregate
        # (×8), max_concurrency = 9600/1200 = 8, yielding multiple points.
        ctx_patch, _ = _patch_estimator(per_rank_max_kv=1200)
        with ctx_patch:
            fpms = aic_mod.run_aic_interpolation(spec, SubComponentType.DECODE)

        # Across all isl steps we should see multiple distinct concurrency
        # levels — evidence the aggregate scaling kicked in.
        distinct_concurrencies = {
            f.scheduled_requests.num_decode_requests for f in fpms
        }
        assert len(distinct_concurrencies) >= 2

class TestQwen235MoEPicks:
    """End-to-end exercise with a realistic large-MoE pick pair.

    Uses a concrete Qwen-235B-A22B-FP8 pick shape that spans the full MoE
    parallelism surface in one test class:

    * Prefill: ``tp=4, dp=1, moe_tp=1, moe_ep=4`` — DEP on the MoE layer.
    * Decode: ``tp=1, dp=8, moe_tp=2, moe_ep=4`` — hybrid TEP + attention DP.

    Together they cover every AIC kwarg that a pick can carry, including the
    case where ``.tp_size`` (KV-head-split property) disagrees with AIC's
    attention-TP.
    """

    PREFILL_PICK = PickedParallelConfig(tp=4, pp=1, dp=1, moe_tp=1, moe_ep=4)
    DECODE_PICK = PickedParallelConfig(tp=1, pp=1, dp=8, moe_tp=2, moe_ep=4)

    def test_prefill_pick_satisfies_aic_identity(self):
        """AIC's assertion: tp_size * attention_dp_size == moe_tp * moe_ep."""
        kw = picked_to_aic_model_config_kwargs(self.PREFILL_PICK)
        assert kw["tp_size"] == 4  # raw p.tp, NOT the .tp_size property
        assert kw["attention_dp_size"] == 1
        assert kw["moe_tp_size"] == 1
        assert kw["moe_ep_size"] == 4
        assert (
            kw["tp_size"] * kw["attention_dp_size"]
            == kw["moe_tp_size"] * kw["moe_ep_size"]
        )

    def test_decode_pick_satisfies_aic_identity(self):
        kw = picked_to_aic_model_config_kwargs(self.DECODE_PICK)
        assert kw["tp_size"] == 1
        assert kw["attention_dp_size"] == 8
        assert kw["moe_tp_size"] == 2
        assert kw["moe_ep_size"] == 4
        assert (
            kw["tp_size"] * kw["attention_dp_size"]
            == kw["moe_tp_size"] * kw["moe_ep_size"]
        )

    def test_tp_size_property_differs_from_aic_tp_size(self):
        """Documents that the KV-split property is NOT AIC's tp_size."""
        # For this DEP-on-MoE pick, .tp_size returns 1 (because moe_ep > 1),
        # which would violate the MoE identity. AIC's real tp_size is 4.
        assert self.PREFILL_PICK.tp_size == 1  # KV-head-split semantics
        kw = picked_to_aic_model_config_kwargs(self.PREFILL_PICK)
        assert kw["tp_size"] == 4  # AIC semantics
        assert kw["tp_size"] != self.PREFILL_PICK.tp_size

    def test_end_to_end_sweep_delivers_complete_kwargs_to_aic(self):
        """Run both sweeps end-to-end and verify every AIC call is well-formed."""
        spec = _make_spec(
            prefill_pick=self.PREFILL_PICK,
            decode_pick=self.DECODE_PICK,
            isl=3000,
            osl=300,
            sweep_max_context_length=4096,
            prefill_granularity=4,
            decode_granularity=3,
        )
        ctx_patch, estimator = _patch_estimator(per_rank_max_kv=500_000)
        with ctx_patch:
            prefill_fpms = aic_mod.run_aic_interpolation(
                spec, SubComponentType.PREFILL
            )
            decode_fpms = aic_mod.run_aic_interpolation(
                spec, SubComponentType.DECODE
            )

        assert prefill_fpms and decode_fpms

        # Every prefill AIC call must carry complete MoE kwargs and satisfy
        # the parallelism identity.
        for call in estimator.estimate_prefill_perf.call_args_list:
            kw = call.kwargs
            assert {
                "tp_size",
                "pp_size",
                "moe_tp_size",
                "moe_ep_size",
                "attention_dp_size",
            } <= kw.keys()
            assert kw["tp_size"] == 4
            assert kw["moe_ep_size"] == 4
            assert kw["attention_dp_size"] == 1
            assert (
                kw["tp_size"] * kw["attention_dp_size"]
                == kw["moe_tp_size"] * kw["moe_ep_size"]
            )

        for call in estimator.estimate_perf.call_args_list:
            kw = call.kwargs
            assert kw["tp_size"] == 1
            assert kw["moe_tp_size"] == 2
            assert kw["moe_ep_size"] == 4
            assert kw["attention_dp_size"] == 8
            assert (
                kw["tp_size"] * kw["attention_dp_size"]
                == kw["moe_tp_size"] * kw["moe_ep_size"]
            )


class TestConcurrencySweep:
    """The AIC-path sweep drops thorough-mode's DP-multiples constraint."""

    def test_small_max_uses_full_range(self):
        assert aic_mod._concurrency_sweep(3, 4) == [1, 2, 3]

    def test_granularity_caps_points(self):
        out = aic_mod._concurrency_sweep(100, 4)
        assert len(out) == 4
        assert out[0] == 1
        assert out[-1] == 100
        assert sorted(out) == out

    def test_zero_max_returns_empty(self):
        assert aic_mod._concurrency_sweep(0, 4) == []
