# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AIC-spec integration in profiler DGD generation."""

import pytest

try:
    from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
    from dynamo.planner.config.parallelization import PickedParallelConfig
    from dynamo.planner.config.planner_config import (
        PlannerConfig,
        PlannerPreDeploymentSweepMode,
    )
    from dynamo.profiler.utils.dgd_generation import (
        _build_planner_config,
        build_aic_interpolation_spec,
    )
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        FeaturesSpec,
    )
except ImportError as e:
    pytest.skip(f"Missing dependency: {e}", allow_module_level=True)


def _dgdr(
    planner: PlannerConfig | None = None,
    model: str = "Qwen/Qwen3-32B",
) -> DynamoGraphDeploymentRequestSpec:
    features = FeaturesSpec(planner=planner) if planner else None
    return DynamoGraphDeploymentRequestSpec(model=model, features=features)


class TestBuildAICInterpolationSpec:
    def _rapid_planner(self) -> PlannerConfig:
        return PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        )

    def test_rapid_planner_produces_spec(self):
        dgdr = _dgdr(planner=self._rapid_planner())
        pick = PickedParallelConfig(tp=1, dp=8, moe_tp=1, moe_ep=8)
        spec = build_aic_interpolation_spec(
            dgdr,
            best_prefill_pick=pick,
            best_decode_pick=pick,
            isl=3000,
            osl=300,
            sweep_max_context_length=8192,
            resolved_backend="trtllm",
            system="h200_sxm",
            prefill_interpolation_granularity=16,
            decode_interpolation_granularity=6,
        )
        assert isinstance(spec, AICInterpolationSpec)
        assert spec.hf_id == "Qwen/Qwen3-32B"
        assert spec.backend == "trtllm"
        assert spec.system == "h200_sxm"
        assert spec.prefill_pick == pick
        assert spec.decode_pick == pick

    def test_thorough_planner_returns_none(self):
        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Thorough,
        )
        dgdr = _dgdr(planner=planner)
        pick = PickedParallelConfig(tp=1, dp=8, moe_ep=8)
        got = build_aic_interpolation_spec(
            dgdr,
            best_prefill_pick=pick,
            best_decode_pick=pick,
            isl=3000,
            osl=300,
            sweep_max_context_length=8192,
            resolved_backend="trtllm",
            system="h200_sxm",
            prefill_interpolation_granularity=16,
            decode_interpolation_granularity=6,
        )
        assert got is None

    def test_throughput_disabled_returns_none(self):
        planner = PlannerConfig(
            enable_throughput_scaling=False,
            enable_load_scaling=True,
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        )
        dgdr = _dgdr(planner=planner)
        pick = PickedParallelConfig(tp=1)
        got = build_aic_interpolation_spec(
            dgdr,
            best_prefill_pick=pick,
            best_decode_pick=pick,
            isl=1000,
            osl=100,
            sweep_max_context_length=4096,
            resolved_backend="trtllm",
            system="h200_sxm",
            prefill_interpolation_granularity=8,
            decode_interpolation_granularity=4,
        )
        assert got is None

    def test_missing_picks_returns_none(self):
        dgdr = _dgdr(planner=self._rapid_planner())
        got = build_aic_interpolation_spec(
            dgdr,
            best_prefill_pick=None,
            best_decode_pick=None,
            isl=1000,
            osl=100,
            sweep_max_context_length=4096,
            resolved_backend="trtllm",
            system="h200_sxm",
            prefill_interpolation_granularity=8,
            decode_interpolation_granularity=4,
        )
        assert got is None

    def test_unsupported_backend_returns_none(self):
        dgdr = _dgdr(planner=self._rapid_planner())
        pick = PickedParallelConfig(tp=1)
        got = build_aic_interpolation_spec(
            dgdr,
            best_prefill_pick=pick,
            best_decode_pick=pick,
            isl=1000,
            osl=100,
            sweep_max_context_length=4096,
            resolved_backend="mocker",
            system="h200_sxm",
            prefill_interpolation_granularity=8,
            decode_interpolation_granularity=4,
        )
        assert got is None

    def test_no_planner_returns_none(self):
        dgdr = _dgdr(planner=None)
        pick = PickedParallelConfig(tp=1)
        got = build_aic_interpolation_spec(
            dgdr,
            best_prefill_pick=pick,
            best_decode_pick=pick,
            isl=1000,
            osl=100,
            sweep_max_context_length=4096,
            resolved_backend="trtllm",
            system="h200_sxm",
            prefill_interpolation_granularity=8,
            decode_interpolation_granularity=4,
        )
        assert got is None


class TestBuildPlannerConfigEmbedsAicSpec:
    def test_spec_threads_into_planner_config(self):
        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        )
        dgdr = _dgdr(planner=planner)
        pick = PickedParallelConfig(tp=1, dp=8, moe_ep=8)
        spec = AICInterpolationSpec(
            hf_id="x",
            system="h200_sxm",
            backend="trtllm",
            isl=1000,
            osl=100,
            sweep_max_context_length=4096,
            prefill_interpolation_granularity=4,
            decode_interpolation_granularity=4,
            prefill_pick=pick,
            decode_pick=pick,
        )
        cfg = _build_planner_config(dgdr, pick, pick, aic_spec=spec)
        assert cfg.aic_interpolation == spec
        # Regression: num-gpu injection still works.
        assert cfg.prefill_engine_num_gpu == pick.num_gpus
        assert cfg.decode_engine_num_gpu == pick.num_gpus

    def test_no_spec_leaves_aic_interpolation_none(self):
        planner = PlannerConfig(
            enable_throughput_scaling=False,
            enable_load_scaling=True,
        )
        dgdr = _dgdr(planner=planner)
        pick = PickedParallelConfig(tp=8)
        cfg = _build_planner_config(dgdr, pick, pick, aic_spec=None)
        assert cfg.aic_interpolation is None


class TestNeedsProfileDataRapid:
    def test_rapid_planner_only_returns_false(self):
        """Planner-only rapid: no files needed; planner will use aic_spec."""
        from dynamo.profiler.utils.profile_common import needs_profile_data

        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        )
        dgdr = _dgdr(planner=planner)
        assert needs_profile_data(dgdr) is False

    def test_thorough_planner_returns_true(self):
        """Thorough still needs files."""
        from dynamo.profiler.utils.profile_common import needs_profile_data

        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Thorough,
        )
        dgdr = _dgdr(planner=planner)
        assert needs_profile_data(dgdr) is True
