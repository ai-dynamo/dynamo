# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profile_sla.py private helper functions.

These tests exercise each helper in isolation, without running the full
profiling pipeline.  External I/O (DGD generation, deployment) is mocked
where needed.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from dynamo.planner.utils.planner_config import (
        PlannerConfig,
        PlannerPreDeploymentSweepMode,
    )
    from dynamo.profiler.profile_sla import (
        _assemble_final_config,
        _extract_profiler_params,
        _write_final_output,
    )
    from dynamo.profiler.utils.config_modifiers.parallelization_mapping import (
        PickedParallelConfig,
    )
    from dynamo.profiler.utils.defaults import SearchStrategy
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        FeaturesSpec,
        HardwareSpec,
        MockerSpec,
        SLASpec,
        WorkloadSpec,
    )
    from dynamo.profiler.utils.dgdr_validate import run_gate_checks
    from dynamo.profiler.utils.profile_common import ProfilerOperationalConfig
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_dgdr(**overrides) -> DynamoGraphDeploymentRequestSpec:
    """Build a minimal dgdr with all required fields set."""
    base = dict(
        model="Qwen/Qwen3-32B",
        backend="trtllm",
        image="nvcr.io/nvidia/ai-dynamo/dynamo-frontend:latest",
        hardware=HardwareSpec(gpuSku="h200_sxm", totalGpus=8, numGpusPerNode=8),
        workload=WorkloadSpec(isl=4000, osl=1000),
        sla=SLASpec(ttft=2000.0, itl=50.0),
    )
    base.update(overrides)
    return DynamoGraphDeploymentRequestSpec(**base)


def _make_planner(**overrides) -> PlannerConfig:
    base = dict(
        enable_throughput_scaling=True,
        enable_load_scaling=False,
        pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        mode="disagg",
        backend="trtllm",
    )
    base.update(overrides)
    return PlannerConfig(**base)


def _make_ops(tmp_path, **kwargs) -> ProfilerOperationalConfig:
    return ProfilerOperationalConfig(
        output_dir=str(tmp_path / "out"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# _extract_profiler_params
# ---------------------------------------------------------------------------


class TestExtractProfilerParams:
    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_basic_ttft_itl(self):
        """Returns correct values when ttft/itl SLA is used."""
        dgdr = _make_dgdr()
        (
            model,
            backend,
            system,
            total_gpus,
            isl,
            osl,
            req_lat,
            ttft,
            tpot,
            strategy,
            picking,
        ) = _extract_profiler_params(dgdr)

        assert model == "Qwen/Qwen3-32B"
        assert backend == "trtllm"
        assert system == "h200_sxm"
        assert total_gpus == 8
        assert isl == 4000
        assert osl == 1000
        assert req_lat is None
        assert ttft == 2000.0
        assert tpot == 50.0
        assert strategy == SearchStrategy.RAPID
        assert picking == "default"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_e2e_latency_sets_both_targets(self):
        """Both ttft and tpot equal e2eLatency when it is set."""
        dgdr = _make_dgdr(sla=SLASpec(ttft=None, itl=None, e2eLatency=35000.0))
        _, _, _, _, _, _, req_lat, ttft, tpot, _, _ = _extract_profiler_params(dgdr)
        assert req_lat == 35000.0
        assert ttft == 35000.0
        assert tpot == 35000.0

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_request_rate_yields_load_match_picking(self):
        """requestRate present in workload → picking_mode == 'load_match'."""
        dgdr = _make_dgdr(workload=WorkloadSpec(isl=4000, osl=1000, requestRate=5.0))
        _, _, _, _, _, _, _, _, _, _, picking = _extract_profiler_params(dgdr)
        assert picking == "load_match"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_backend_lowercased(self):
        """backend value is always lower-cased."""
        dgdr = _make_dgdr(backend="trtllm")
        _, backend, _, _, _, _, _, _, _, _, _ = _extract_profiler_params(dgdr)
        assert backend == "trtllm"
        assert backend == backend.lower()

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_thorough_strategy_extracted(self):
        """searchStrategy: thorough is correctly reflected in the returned tuple."""
        dgdr = _make_dgdr(searchStrategy="thorough")
        _, _, _, _, _, _, _, _, _, strategy, _ = _extract_profiler_params(dgdr)
        assert strategy == SearchStrategy.THOROUGH


# ---------------------------------------------------------------------------
# run_gate_checks
# ---------------------------------------------------------------------------


class TestRunGateChecks:
    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_thorough_auto_backend_raises(self):
        """THOROUGH + 'auto' backend is rejected."""
        dgdr = _make_dgdr()
        with pytest.raises(ValueError, match="does not support 'auto' backend"):
            run_gate_checks(
                dgdr,
                aic_supported=True,
                search_strategy=SearchStrategy.THOROUGH,
                backend="auto",
            )

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_thorough_concrete_backend_passes(self):
        """THOROUGH + concrete backend is fine."""
        dgdr = _make_dgdr()
        run_gate_checks(
            dgdr,
            aic_supported=True,
            search_strategy=SearchStrategy.THOROUGH,
            backend="trtllm",
        )

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_rapid_auto_backend_passes(self):
        """RAPID allows 'auto' backend."""
        dgdr = _make_dgdr()
        run_gate_checks(
            dgdr,
            aic_supported=False,
            search_strategy=SearchStrategy.RAPID,
            backend="auto",
        )

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_no_planner_aic_unsupported_passes(self):
        """No planner, AIC unsupported — no error."""
        dgdr = _make_dgdr()
        run_gate_checks(
            dgdr,
            aic_supported=False,
            search_strategy=SearchStrategy.RAPID,
            backend="vllm",
        )

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_planner_throughput_scaling_aic_unsupported_raises(self):
        """Throughput-based planner scaling requires AIC support."""
        dgdr = _make_dgdr(
            features=FeaturesSpec(
                planner=_make_planner(
                    enable_throughput_scaling=True,
                    backend="vllm",
                )
            )
        )
        with pytest.raises(
            ValueError, match="Throughput-based planner scaling requires AIC support"
        ):
            run_gate_checks(
                dgdr,
                aic_supported=False,
                search_strategy=SearchStrategy.RAPID,
                backend="vllm",
            )

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_planner_rapid_sweep_aic_unsupported_mutates_to_none(self):
        """Rapid pre-deployment sweep falls back to None when AIC is unsupported."""
        planner = _make_planner(
            enable_throughput_scaling=False,
            enable_load_scaling=True,
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
            backend="vllm",
        )
        dgdr = _make_dgdr(features=FeaturesSpec(planner=planner))
        run_gate_checks(
            dgdr,
            aic_supported=False,
            search_strategy=SearchStrategy.RAPID,
            backend="vllm",
        )
        assert (
            dgdr.features.planner.pre_deployment_sweeping_mode
            == PlannerPreDeploymentSweepMode.None_
        )

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_planner_aic_supported_no_mutation(self):
        """When AIC is supported, planner config is left unchanged."""
        planner = _make_planner(
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        )
        dgdr = _make_dgdr(features=FeaturesSpec(planner=planner))
        run_gate_checks(
            dgdr,
            aic_supported=True,
            search_strategy=SearchStrategy.RAPID,
            backend="trtllm",
        )
        assert (
            dgdr.features.planner.pre_deployment_sweeping_mode
            == PlannerPreDeploymentSweepMode.Rapid
        )


# ---------------------------------------------------------------------------
# _write_final_output
# ---------------------------------------------------------------------------


class TestWriteFinalOutput:
    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_normal_config_writes_file_and_returns_true(self, tmp_path):
        ops = _make_ops(tmp_path)
        os.makedirs(ops.output_dir, exist_ok=True)
        final_config = {"apiVersion": "v1", "kind": "Deployment"}

        result = _write_final_output(ops, final_config)

        assert result is True
        out = Path(ops.output_dir) / "final_config.yaml"
        assert out.exists()
        assert yaml.safe_load(out.read_text()) == final_config

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_list_config_writes_multi_doc_yaml(self, tmp_path):
        ops = _make_ops(tmp_path)
        os.makedirs(ops.output_dir, exist_ok=True)
        final_config = [{"kind": "A"}, {"kind": "B"}]

        result = _write_final_output(ops, final_config)

        assert result is True
        out = Path(ops.output_dir) / "final_config.yaml"
        docs = list(yaml.safe_load_all(out.read_text()))
        assert len(docs) == 2

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_none_config_not_dry_run_returns_false(self, tmp_path):
        ops = _make_ops(tmp_path, dry_run=False)
        os.makedirs(ops.output_dir, exist_ok=True)

        result = _write_final_output(ops, None)

        assert result is False

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_none_config_dry_run_writes_empty_yaml_and_returns_true(self, tmp_path):
        ops = _make_ops(tmp_path, dry_run=True)
        os.makedirs(ops.output_dir, exist_ok=True)

        result = _write_final_output(ops, None)

        assert result is True
        out = Path(ops.output_dir) / "final_config.yaml"
        assert out.exists()
        assert yaml.safe_load(out.read_text()) is None  # empty YAML == None


# ---------------------------------------------------------------------------
# _assemble_final_config
# ---------------------------------------------------------------------------


class TestAssembleFinalConfig:
    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_no_planner_no_mocker_returns_dgd_config_unchanged(self, tmp_path):
        dgdr = _make_dgdr()
        ops = _make_ops(tmp_path)
        dgd_config = {"kind": "DynamoGraphDeployment"}

        result = _assemble_final_config(
            dgdr,
            ops,
            dgd_config,
            PickedParallelConfig(tp=1),
            PickedParallelConfig(tp=1),
        )

        assert result is dgd_config

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_none_dgd_config_passes_through_as_none(self, tmp_path):
        dgdr = _make_dgdr()
        ops = _make_ops(tmp_path)

        result = _assemble_final_config(
            dgdr,
            ops,
            None,
            PickedParallelConfig(tp=1),
            PickedParallelConfig(tp=1),
        )

        assert result is None

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_planner_no_mocker_returns_real_config(self, tmp_path):
        dgdr = _make_dgdr(features=FeaturesSpec(planner=_make_planner()))
        ops = _make_ops(tmp_path)
        os.makedirs(ops.output_dir, exist_ok=True)
        dgd_config = {"kind": "DGD"}
        real_cfg = {"kind": "real"}
        mocker_cfg = {"kind": "mocker"}

        with patch(
            "dynamo.profiler.profile_sla.generate_dgd_config_with_planner",
            return_value=(real_cfg, mocker_cfg),
        ):
            result = _assemble_final_config(
                dgdr,
                ops,
                dgd_config,
                PickedParallelConfig(tp=1),
                PickedParallelConfig(tp=1),
            )

        assert result is real_cfg

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_mocker_enabled_returns_mocker_config(self, tmp_path):
        dgdr = _make_dgdr(
            features=FeaturesSpec(
                planner=_make_planner(),
                mocker=MockerSpec(enabled=True),
            )
        )
        ops = _make_ops(tmp_path)
        os.makedirs(ops.output_dir, exist_ok=True)
        dgd_config = {"kind": "DGD"}
        real_cfg = {"kind": "real"}
        mocker_cfg = {"kind": "mocker"}

        with patch(
            "dynamo.profiler.profile_sla.generate_dgd_config_with_planner",
            return_value=(real_cfg, mocker_cfg),
        ):
            result = _assemble_final_config(
                dgdr,
                ops,
                dgd_config,
                PickedParallelConfig(tp=1),
                PickedParallelConfig(tp=1),
            )

        assert result is mocker_cfg
