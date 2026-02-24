# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for profile_sla with DynamoGraphDeploymentRequestSpec input.

Tests the new DGDR-based profiler entry point across different configurations:
rapid/thorough, supported/unsupported, planner/no-planner, load-match, PVC, mocker.

All tests are no-GPU (gpu_0) and pre_merge.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
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
    def test_both_concurrency_and_rate_rejected(self, tmp_path):
        """Case 2d: both concurrency and requestRate should fail profiler validation."""
        dgdr = _load_dgdr(CONFIGS_DIR / "2d_rapid_both_concurrency_and_rate_error.yaml")
        ops = _make_ops(tmp_path)
        with pytest.raises(ValueError, match="concurrency.*requestRate"):
            asyncio.run(run_profile(dgdr, ops))

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

        output = tmp_path / "profiling_results" / "final_config.yaml"
        assert output.exists()
        status_file = tmp_path / "profiling_results" / "profiler_status.yaml"
        if status_file.exists():
            status = yaml.safe_load(status_file.read_text())
            assert status.get("status") in ("success", "failed")


# ---------------------------------------------------------------------------
# Helpers for mocking K8s deployment + benchmark functions
# ---------------------------------------------------------------------------


def _mock_deployment_client():
    """Create a mock DynamoDeploymentClient that returns immediately."""
    client = MagicMock()
    client.create_deployment = AsyncMock()
    client.wait_for_deployment_ready = AsyncMock()
    client.get_deployment_logs = AsyncMock()
    client.delete_deployment = AsyncMock()
    client.get_service_url = MagicMock(return_value="http://mock:8000")
    client.deployment_name = "mock-deployment"
    client.base_log_dir = "/tmp"
    return client


def _save_dummy_npz(output_dir: str):
    """Save dummy prefill + decode NPZ files matching the interpolation format."""
    prefill_dir = os.path.join(output_dir, "selected_prefill_interpolation")
    os.makedirs(prefill_dir, exist_ok=True)
    np.savez(
        os.path.join(prefill_dir, "raw_data.npz"),
        prefill_isl=np.array([500, 1000, 2000, 4000]),
        prefill_ttft=np.array([10.0, 20.0, 40.0, 80.0]),
        prefill_thpt_per_gpu=np.array([50000.0, 50000.0, 50000.0, 50000.0]),
    )

    decode_dir = os.path.join(output_dir, "selected_decode_interpolation")
    os.makedirs(decode_dir, exist_ok=True)
    np.savez(
        os.path.join(decode_dir, "raw_data.npz"),
        x_kv_usage=np.array([0.1, 0.3, 0.5, 0.8]),
        y_context_length=np.array([500, 1000, 2000, 4000]),
        z_itl=np.array([5.0, 6.0, 7.0, 8.0]),
        z_thpt_per_gpu=np.array([200.0, 180.0, 160.0, 140.0]),
        max_kv_tokens=np.array([100000]),
    )


_DECODE_SVC_NAMES = {
    "sglang": "decode",
    "vllm": "VllmDecodeWorker",
    "trtllm": "TRTLLMDecodeWorker",
}


def _make_thorough_patches(backend: str = "trtllm"):
    """Build mock-patches for thorough mode, parameterised by backend."""
    svc_name = _DECODE_SVC_NAMES.get(backend, "TRTLLMDecodeWorker")
    return [
        patch(
            "dynamo.profiler.thorough.DynamoDeploymentClient",
            side_effect=lambda **kw: _mock_deployment_client(),
        ),
        patch("dynamo.profiler.thorough.get_prefill_ttft", return_value=50.0),
        patch(
            "dynamo.profiler.thorough.get_decode_itl_and_thpt_per_gpu",
            return_value=(8.0, 125.0),
        ),
        patch("dynamo.profiler.thorough.get_num_request_range", return_value=[1, 4, 8]),
        patch(
            "dynamo.profiler.thorough.get_service_name_by_type",
            return_value=svc_name,
        ),
    ]


# Backward compat: existing tests use the trtllm-flavored list
_THOROUGH_PATCHES = _make_thorough_patches("trtllm")


def _patch_kv_cache_log(backend: str = "trtllm"):
    """Patch get_kv_cache_size_from_dynamo_log on the real config modifier."""
    from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS

    real_modifier = CONFIG_MODIFIERS[backend]
    return patch.object(
        real_modifier, "get_kv_cache_size_from_dynamo_log", return_value=100000
    )


class TestThoroughMocked:
    """Thorough mode with mocked K8s deployments and benchmark functions.

    Only K8s client, AIPerf benchmarks, and log-file reads are mocked.
    Enumeration, picking, and DGD generation run for real via AIC.
    """

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_thorough_no_planner_with_load(self, tmp_path):
        """Case 6 (mocked): thorough + load-match, full pipeline without real GPUs."""
        dgdr = _load_dgdr(CONFIGS_DIR / "6_thorough_no_planner_with_load.yaml")
        ops = _make_ops(tmp_path)

        with _patch_kv_cache_log("thorough"):
            for p in _THOROUGH_PATCHES:
                p.start()
            try:
                asyncio.run(run_profile(dgdr, ops))
            finally:
                for p in _THOROUGH_PATCHES:
                    p.stop()

        output = tmp_path / "profiling_results" / "final_config.yaml"
        assert output.exists()
        config = yaml.safe_load(output.read_text())
        assert config, "Mocked thorough should produce a non-empty config"
        assert "spec" in config

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_thorough_planner_thorough_sweep(self, tmp_path):
        """Case 7b: thorough search + thorough interpolation, fully mocked."""
        dgdr = _load_dgdr(CONFIGS_DIR / "7b_thorough_planner_thorough_sweep.yaml")
        ops = _make_ops(tmp_path)

        def mock_profile_prefill(work_dir, *args, **kwargs):
            _save_dummy_npz(ops.output_dir)

        def mock_profile_decode(work_dir, *args, **kwargs):
            _save_dummy_npz(ops.output_dir)

        interp_patches = [
            patch(
                "dynamo.profiler.interpolation.DynamoDeploymentClient",
                side_effect=lambda **kw: _mock_deployment_client(),
            ),
            patch(
                "dynamo.profiler.interpolation.profile_prefill",
                side_effect=mock_profile_prefill,
            ),
            patch(
                "dynamo.profiler.interpolation.profile_decode",
                side_effect=mock_profile_decode,
            ),
            patch(
                "dynamo.profiler.interpolation.get_service_name_by_type",
                return_value="TRTLLMWorker",
            ),
        ]

        with _patch_kv_cache_log("thorough"):
            all_patches = _THOROUGH_PATCHES + interp_patches
            for p in all_patches:
                p.start()
            try:
                asyncio.run(run_profile(dgdr, ops))
            finally:
                for p in all_patches:
                    p.stop()

        output = tmp_path / "profiling_results" / "final_config.yaml"
        assert output.exists()
        raw = output.read_text()
        docs = list(yaml.safe_load_all(raw))
        assert len(docs) >= 2, "Planner + profiling data should produce multi-doc YAML"

        prefill_npz = (
            tmp_path
            / "profiling_results"
            / "selected_prefill_interpolation"
            / "raw_data.npz"
        )
        decode_npz = (
            tmp_path
            / "profiling_results"
            / "selected_decode_interpolation"
            / "raw_data.npz"
        )
        assert prefill_npz.exists(), "Prefill interpolation data should be saved"
        assert decode_npz.exists(), "Decode interpolation data should be saved"


# ---------------------------------------------------------------------------
# Shared helper for mocked-thorough + override tests
# ---------------------------------------------------------------------------


def _run_mocked_thorough(dgdr, ops, backend: str):
    """Run the full mocked-thorough pipeline for an arbitrary backend."""
    thorough_patches = _make_thorough_patches(backend)
    kv_patch = _patch_kv_cache_log(backend)

    with kv_patch:
        for p in thorough_patches:
            p.start()
        try:
            asyncio.run(run_profile(dgdr, ops))
        finally:
            for p in thorough_patches:
                p.stop()


def _assert_overrides_applied(final_config_path: Path, dgdr):
    """Assert the final DGD exists and that overrides are reflected."""
    assert final_config_path.exists(), "final_config.yaml should exist"
    raw = final_config_path.read_text()
    docs = list(yaml.safe_load_all(raw))
    dgd = docs[-1] if docs else {}
    assert dgd and "spec" in dgd, "DGD should have a spec"

    override_spec = dgdr.overrides.dgd.get("spec", {})

    for ovr_key in ("envs", "backendFramework"):
        if ovr_key in override_spec:
            assert ovr_key in dgd["spec"], f"Override field spec.{ovr_key} should exist"

    svc_overrides = override_spec.get("services", {})
    dgd_services = dgd.get("spec", {}).get("services", {})
    for svc_name, svc_ovr in svc_overrides.items():
        if svc_name in dgd_services:
            dgd_svc = dgd_services[svc_name]
            if "sharedMemory" in svc_ovr:
                assert (
                    "sharedMemory" in dgd_svc
                ), f"Override sharedMemory on {svc_name} should be applied"
            mc = svc_ovr.get("extraPodSpec", {}).get("mainContainer", {})
            if "args" in mc:
                dgd_args = (
                    dgd_svc.get("extraPodSpec", {})
                    .get("mainContainer", {})
                    .get("args", [])
                )
                for arg in mc["args"]:
                    assert (
                        arg in dgd_args
                    ), f"Override arg '{arg}' should be in {svc_name} args"


class TestThoroughMockedOverrides:
    """Cases 9a/9b/9c: thorough + DGD overrides for DeepSeek-R1 per backend."""

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_9a_sglang_overrides(self, tmp_path):
        """Case 9a: SGLang thorough sweep with DSR1 overrides."""
        dgdr = _load_dgdr(CONFIGS_DIR / "9a_thorough_dsr1_sglang_overrides.yaml")
        ops = _make_ops(tmp_path)
        _run_mocked_thorough(dgdr, ops, "sglang")
        _assert_overrides_applied(
            tmp_path / "profiling_results" / "final_config.yaml",
            dgdr,
        )
