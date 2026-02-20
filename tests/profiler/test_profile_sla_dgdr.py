# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for profile_sla with DynamoGraphDeploymentRequestSpec input.

Tests the new DGDR-based profiler entry point across different configurations:
rapid/thorough, supported/unsupported, planner/no-planner, load-match, PVC, mocker.

All tests are no-GPU (gpu_0) and pre_merge.
"""

import asyncio
import sys
from pathlib import Path

import pytest
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from dynamo.profiler.profile_sla import run_profile
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        BackendType,
        DynamoGraphDeploymentRequestSpec,
        SearchStrategy,
    )
    from dynamo.profiler.utils.profile_common import ProfilerOperationalConfig
except ImportError as _e:
    pytest.skip(f"Skip testing (refactor in progress): {_e}", allow_module_level=True)


@pytest.fixture(autouse=True)
def logger(request):
    """Override the logger fixture to prevent test directory creation."""
    yield


def _load_dgdr(yaml_path) -> DynamoGraphDeploymentRequestSpec:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return DynamoGraphDeploymentRequestSpec.model_validate(data)


def _make_ops(tmp_path, **overrides) -> ProfilerOperationalConfig:
    defaults = {
        "output_dir": str(tmp_path / "profiling_results"),
        "dry_run": False,
    }
    defaults.update(overrides)
    return ProfilerOperationalConfig(**defaults)


CONFIGS_DIR = Path(__file__).parent / "configs"


class TestRapidSupported:
    """Rapid strategy with AIC-supported model (Qwen3-32B on h200_sxm/trtllm)."""

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_no_planner_no_load(self, tmp_path):
        """Case 1: default picking mode, no planner, no target load."""
        dgdr = _load_dgdr(CONFIGS_DIR / "1_rapid_supported_no_planner_no_load.yaml")
        ops = _make_ops(tmp_path)
        asyncio.run(run_profile(dgdr, ops))

        output = tmp_path / "profiling_results" / "final_config.yaml"
        assert output.exists()
        config = yaml.safe_load(output.read_text())
        assert config, "final_config.yaml should not be empty"
        assert "spec" in config

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_no_planner_with_load(self, tmp_path):
        """Case 2: load-match picking mode with requestRate."""
        dgdr = _load_dgdr(CONFIGS_DIR / "2_rapid_supported_no_planner_with_load.yaml")
        ops = _make_ops(tmp_path)
        asyncio.run(run_profile(dgdr, ops))

        output = tmp_path / "profiling_results" / "final_config.yaml"
        assert output.exists()
        config = yaml.safe_load(output.read_text())
        assert config

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_pvc_no_planner_with_load(self, tmp_path):
        """Case 2b: load-match with PVC model cache."""
        dgdr = _load_dgdr(
            CONFIGS_DIR / "2b_rapid_supported_pvc_no_planner_with_load.yaml"
        )
        ops = _make_ops(tmp_path)
        asyncio.run(run_profile(dgdr, ops))

        output = tmp_path / "profiling_results" / "final_config.yaml"
        assert output.exists()
        config = yaml.safe_load(output.read_text())
        assert config
        spec = config.get("spec", {})
        pvcs = spec.get("pvcs", [])
        assert any(
            p.get("name") == "model-cache" for p in pvcs
        ), "PVC should be mounted"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_e2e_latency_sla(self, tmp_path):
        """Case 2c: e2eLatency SLA instead of ttft/itl."""
        dgdr = _load_dgdr(CONFIGS_DIR / "2c_rapid_supported_e2e_latency.yaml")
        ops = _make_ops(tmp_path)
        asyncio.run(run_profile(dgdr, ops))

        output = tmp_path / "profiling_results" / "final_config.yaml"
        assert output.exists()
        config = yaml.safe_load(output.read_text())
        assert config
        # Verify ttft/itl were cleared by the validator
        assert dgdr.sla.ttft is None
        assert dgdr.sla.itl is None
        assert dgdr.sla.e2eLatency == 35000.0

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_both_concurrency_and_rate_rejected(self):
        """Case 2d: both concurrency and requestRate should fail validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="concurrency.*requestRate"):
            _load_dgdr(CONFIGS_DIR / "2d_rapid_both_concurrency_and_rate_error.yaml")

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_planner_rapid_sweep(self, tmp_path):
        """Case 3: autoscale picking with planner + rapid pre-deployment sweep."""
        dgdr = _load_dgdr(CONFIGS_DIR / "3_rapid_supported_planner_rapid_sweep.yaml")
        ops = _make_ops(tmp_path)
        asyncio.run(run_profile(dgdr, ops))

        output = tmp_path / "profiling_results" / "final_config.yaml"
        assert output.exists()
        raw = output.read_text()
        docs = list(yaml.safe_load_all(raw))
        assert len(docs) >= 2, "Planner config should produce multi-doc YAML"
        dgd = docs[-1]
        assert "Planner" in dgd.get("spec", {}).get(
            "services", {}
        ), "Planner service should be added"


class TestRapidUnsupported:
    """Rapid strategy with AIC-unsupported model (Qwen3-32B on l40s/vllm)."""

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_no_planner_naive_fallback(self, tmp_path):
        """Case 4: falls back to naive config generation."""
        dgdr = _load_dgdr(CONFIGS_DIR / "4_rapid_unsupported_no_planner.yaml")
        ops = _make_ops(tmp_path)
        asyncio.run(run_profile(dgdr, ops))

        output = tmp_path / "profiling_results" / "final_config.yaml"
        assert output.exists()
        config = yaml.safe_load(output.read_text())
        assert config, "Naive fallback should produce a non-empty config"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_planner_load_scaling_rapid_sweep_fallback(self, tmp_path):
        """Case 5: planner with load scaling, rapid sweep falls back to none."""
        dgdr = _load_dgdr(CONFIGS_DIR / "5_rapid_unsupported_planner.yaml")
        ops = _make_ops(tmp_path)
        asyncio.run(run_profile(dgdr, ops))

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_planner_throughput_scaling_raises(self, tmp_path):
        """Case 5b: planner with throughput scaling on unsupported combo should fail."""
        dgdr = _load_dgdr(
            CONFIGS_DIR / "5b_rapid_unsupported_planner_throughput_error.yaml"
        )
        ops = _make_ops(tmp_path)
        with pytest.raises(
            ValueError, match="Throughput-based planner scaling requires AIC support"
        ):
            asyncio.run(run_profile(dgdr, ops))


class TestThoroughDryRun:
    """Thorough strategy tested with --dry-run (no real deployments)."""

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_no_planner_with_load(self, tmp_path):
        """Case 6: thorough + load-match, dry-run."""
        dgdr = _load_dgdr(CONFIGS_DIR / "6_thorough_no_planner_with_load.yaml")
        ops = _make_ops(tmp_path, dry_run=True)
        asyncio.run(run_profile(dgdr, ops))

        output = tmp_path / "profiling_results" / "final_config.yaml"
        assert output.exists()

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_planner_rapid_sweep(self, tmp_path):
        """Case 7: thorough + planner + rapid pre-deployment sweep, dry-run."""
        dgdr = _load_dgdr(CONFIGS_DIR / "7_thorough_planner_rapid_sweep.yaml")
        ops = _make_ops(tmp_path, dry_run=True)
        asyncio.run(run_profile(dgdr, ops))

        output = tmp_path / "profiling_results" / "final_config.yaml"
        assert output.exists()


class TestMockerEnabled:
    """Mocker feature flag selects mocker DGD over real worker DGD."""

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_mocker_config_selected(self, tmp_path):
        """Case 3b: planner + mocker enabled, should produce mocker DGD."""
        config_path = CONFIGS_DIR / "3b_rapid_supported_planner_rapid_sweep_mocker.yaml"
        if not config_path.exists():
            pytest.skip("3b mocker config not found")
        dgdr = _load_dgdr(config_path)
        ops = _make_ops(tmp_path)
        asyncio.run(run_profile(dgdr, ops))

        output = tmp_path / "profiling_results" / "final_config.yaml"
        assert output.exists()


class TestGateChecks:
    """Validate gate checks at profiler startup."""

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_thorough_auto_backend_rejected(self, tmp_path):
        """Thorough + auto backend should raise ValueError."""
        dgdr = _load_dgdr(CONFIGS_DIR / "1_rapid_supported_no_planner_no_load.yaml")
        dgdr.searchStrategy = SearchStrategy.Thorough
        dgdr.backend = BackendType.Auto
        ops = _make_ops(tmp_path)
        with pytest.raises(ValueError, match="does not support 'auto' backend"):
            asyncio.run(run_profile(dgdr, ops))


class TestThoroughEdgeCases:
    """Edge cases for thorough mode."""

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_empty_candidates_due_to_small_gpu(self, tmp_path):
        """Case 8: DeepSeek-R1 on 1 L40S GPU — model too large, no candidates."""
        dgdr = _load_dgdr(CONFIGS_DIR / "8_thorough_empty_candidates.yaml")
        ops = _make_ops(tmp_path, dry_run=True)
        asyncio.run(run_profile(dgdr, ops))

        # Dry-run with thorough should complete but produce empty config
        output = tmp_path / "profiling_results" / "final_config.yaml"
        assert output.exists()
        status_file = tmp_path / "profiling_results" / "profiler_status.yaml"
        if status_file.exists():
            status = yaml.safe_load(status_file.read_text())
            assert status.get("status") in ("success", "failed")
