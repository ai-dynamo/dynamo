# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SLA target propagation from spec.sla into PlannerConfig.

Covers the behaviour added by _build_planner_config() that propagates
spec.sla.ttft / spec.sla.itl onto PlannerConfig.ttft_ms / itl_ms while
preserving backwards compatibility when the user did not explicitly set
those SLA fields.
"""

import pytest

try:
    from dynamo.planner.config.defaults import SLAPlannerDefaults
    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.profiler.utils.dgd_generation import _build_planner_config
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        FeaturesSpec,
        GPUSKUType,
        HardwareSpec,
        SLASpec,
        WorkloadSpec,
    )
    from dynamo.profiler.utils.dgdr_validate import valid_dgdr_spec
except ImportError as e:
    pytest.skip(f"Missing dependency: {e}", allow_module_level=True)


pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _sla_dgdr(
    sla: SLASpec | None = None,
    planner: PlannerConfig | None = None,
) -> DynamoGraphDeploymentRequestSpec:
    """Build a minimally-valid DGDR that passes valid_dgdr_spec()."""
    return DynamoGraphDeploymentRequestSpec(
        model="Qwen/Qwen3-0.6B",
        image="dummy:latest",
        hardware=HardwareSpec(gpuSku=GPUSKUType.H200SXM, numGpusPerNode=8),
        workload=WorkloadSpec(),
        sla=sla,
        features=FeaturesSpec(planner=planner) if planner is not None else None,
    )


def _build(dgdr: DynamoGraphDeploymentRequestSpec) -> PlannerConfig:
    return _build_planner_config(
        dgdr, best_prefill_mapping=None, best_decode_mapping=None
    )


class TestPlannerSLAPropagation:
    """Propagation of spec.sla onto PlannerConfig.{ttft_ms,itl_ms}."""

    def test_omitted_sla_preserves_planner_defaults(self):
        """spec.sla omitted -> validator injects SLASpec() defaults."""
        dgdr = _sla_dgdr(sla=None)
        valid_dgdr_spec(dgdr)

        cfg = _build(dgdr)

        assert cfg.ttft_ms == SLAPlannerDefaults.ttft_ms
        assert cfg.itl_ms == SLAPlannerDefaults.itl_ms

    def test_empty_explicit_sla_preserves_planner_defaults(self):
        """user passed SLASpec() explicitly with empty fields.model_fields_set, so propagation must be skipped and SLAPlannerDefaults must be preserved."""
        dgdr = _sla_dgdr(sla=SLASpec())

        cfg = _build(dgdr)

        assert cfg.ttft_ms == SLAPlannerDefaults.ttft_ms
        assert cfg.itl_ms == SLAPlannerDefaults.itl_ms

    def test_sla_none_without_validator_is_safe(self):
        """Defensive: _build_planner_config tolerates dgdr.sla is None and keeps SLAPlannerDefaults."""
        dgdr = _sla_dgdr(sla=None)
        # intentionally NOT calling valid_dgdr_spec()

        cfg = _build(dgdr)

        assert cfg.ttft_ms == SLAPlannerDefaults.ttft_ms
        assert cfg.itl_ms == SLAPlannerDefaults.itl_ms

    def test_explicit_sla_targets_propagate(self):
        """user explicitly sets sla.ttft and sla.itl.
        Both must propagate to PlannerConfig."""
        dgdr = _sla_dgdr(sla=SLASpec(ttft=300.0, itl=15.0))

        cfg = _build(dgdr)

        assert cfg.ttft_ms == 300.0
        assert cfg.itl_ms == 15.0

    def test_planner_overrides_take_precedence_over_sla(self):
        """when features.planner.{ttft_ms,itl_ms} are explicitly set, they win over spec.sla values."""
        dgdr = _sla_dgdr(
            sla=SLASpec(ttft=300.0, itl=15.0),
            planner=PlannerConfig(
                ttft_ms=1000.0, itl_ms=20.0, optimization_target="sla"
            ),
        )

        cfg = _build(dgdr)

        assert cfg.ttft_ms == 1000.0
        assert cfg.itl_ms == 20.0

    def test_partial_planner_override_mixes_with_sla(self):
        """only ttft_ms is set explicitly on planner, itl_ms must still be filled from sla.itl."""
        dgdr = _sla_dgdr(
            sla=SLASpec(ttft=300.0, itl=15.0),
            planner=PlannerConfig(ttft_ms=1000.0, optimization_target="sla"),
        )

        cfg = _build(dgdr)

        assert cfg.ttft_ms == 1000.0  # explicit planner override wins
        assert cfg.itl_ms == 15.0  # falls through to sla.itl

    def test_planner_override_with_defaulted_sla(self):
        """features.planner has explicit values, spec.sla omitted."""
        dgdr = _sla_dgdr(
            sla=None,
            planner=PlannerConfig(
                ttft_ms=1000.0, itl_ms=20.0, optimization_target="sla"
            ),
        )
        valid_dgdr_spec(dgdr)

        cfg = _build(dgdr)

        assert cfg.ttft_ms == 1000.0
        assert cfg.itl_ms == 20.0

    def test_e2e_latency_mode_skips_ttft_itl_propagation(self):
        """when SLA uses e2eLatency mode, ttft/itl propagation must be skipped (planner keeps SLAPlannerDefaults)."""
        dgdr = _sla_dgdr(sla=SLASpec(e2eLatency=5000.0))

        cfg = _build(dgdr)

        assert cfg.ttft_ms == SLAPlannerDefaults.ttft_ms
        assert cfg.itl_ms == SLAPlannerDefaults.itl_ms

    def test_partial_sla_ttft_only_propagates_effective_values(self):
        """only ttft → effective (ttft, sla-default itl) both propagated"""

        dgdr = _sla_dgdr(sla=SLASpec(ttft=300.0))

        cfg = _build(dgdr)

        assert cfg.ttft_ms == 300.0
        assert cfg.itl_ms == 30.0  # SLASpec default

    def test_partial_sla_itl_only_propagates_effective_values(self):
        """only itl → effective (sla-default ttft, itl) both propagated"""

        dgdr = _sla_dgdr(sla=SLASpec(itl=15.0))

        cfg = _build(dgdr)

        assert cfg.ttft_ms == 2000.0  # SLASpec default
        assert cfg.itl_ms == 15.0
