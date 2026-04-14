# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profile_sla.py private helper functions.

These tests exercise each helper in isolation, without running the full
profiling pipeline.  External I/O (DGD generation, deployment) is mocked
where needed.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]

try:
    from dynamo.planner.config.planner_config import (
        PlannerConfig,
        PlannerPreDeploymentSweepMode,
    )
    from dynamo.profiler.profile_sla import (
        _extract_profiler_params,
        _write_final_output,
    )
    from dynamo.profiler.utils.config_modifiers.parallelization_mapping import (
        PickedParallelConfig,
    )
    from dynamo.profiler.utils.defaults import SearchStrategy
    from dynamo.profiler.utils.dgd_generation import (
        add_profile_data_to_config,
        assemble_final_config,
    )
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        FeaturesSpec,
        HardwareSpec,
        MockerSpec,
        SLASpec,
        WorkloadSpec,
    )
    from dynamo.profiler.utils.dgdr_validate import (
        valid_dgdr_spec,
        validate_dgdr_dynamo_features,
    )
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
        optimization_target="sla",
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
# valid_dgdr_spec
# ---------------------------------------------------------------------------


class TestValidDgdrSpec:
    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_thorough_auto_backend_raises(self):
        """THOROUGH + 'auto' backend is rejected."""
        dgdr = _make_dgdr(searchStrategy="thorough", backend="auto")
        with pytest.raises(ValueError, match="does not support 'auto' backend"):
            valid_dgdr_spec(dgdr)

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_thorough_concrete_backend_passes(self):
        """THOROUGH + concrete backend is fine."""
        dgdr = _make_dgdr(searchStrategy="thorough", backend="trtllm")
        valid_dgdr_spec(dgdr)

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_rapid_auto_backend_passes(self):
        """RAPID allows 'auto' backend."""
        dgdr = _make_dgdr(backend="auto")
        valid_dgdr_spec(dgdr)

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_missing_image_raises(self):
        """image is required."""
        dgdr = _make_dgdr(image="")
        with pytest.raises(ValueError, match="image.*required"):
            valid_dgdr_spec(dgdr)

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_missing_hardware_raises(self):
        """hardware is required."""
        dgdr = _make_dgdr(hardware=None)
        with pytest.raises(ValueError, match="hardware.*required"):
            valid_dgdr_spec(dgdr)

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_missing_gpu_sku_raises(self):
        """hardware.gpuSku is required."""
        dgdr = _make_dgdr(hardware=HardwareSpec(gpuSku=None, numGpusPerNode=8))
        with pytest.raises(ValueError, match="gpuSku.*required"):
            valid_dgdr_spec(dgdr)

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_zero_gpus_per_node_raises(self):
        """hardware.numGpusPerNode must be positive."""
        dgdr = _make_dgdr(hardware=HardwareSpec(gpuSku="h200_sxm", numGpusPerNode=0))
        with pytest.raises(ValueError, match="numGpusPerNode.*positive"):
            valid_dgdr_spec(dgdr)

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_none_workload_gets_default(self):
        """None workload is populated with a default WorkloadSpec."""
        dgdr = _make_dgdr(workload=None)
        valid_dgdr_spec(dgdr)
        assert dgdr.workload is not None

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_none_sla_gets_default(self):
        """None sla is populated with a default SLASpec."""
        dgdr = _make_dgdr(sla=None)
        valid_dgdr_spec(dgdr)
        assert dgdr.sla is not None

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_both_concurrency_and_rate_raises(self):
        """concurrency and requestRate are mutually exclusive."""
        dgdr = _make_dgdr(
            workload=WorkloadSpec(isl=4000, osl=1000, concurrency=10, requestRate=5.0)
        )
        with pytest.raises(ValueError, match="concurrency.*requestRate"):
            valid_dgdr_spec(dgdr)

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_negative_sla_ttft_raises(self):
        """Negative SLA ttft must be rejected."""
        dgdr = _make_dgdr(sla=SLASpec(ttft=-1.0, itl=30.0))
        with pytest.raises(ValueError, match="ttft.*positive"):
            valid_dgdr_spec(dgdr)

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_e2e_latency_clears_ttft_itl(self):
        """e2eLatency takes precedence and nulls out ttft/itl."""
        dgdr = _make_dgdr(sla=SLASpec(ttft=None, itl=None, e2eLatency=35000.0))
        valid_dgdr_spec(dgdr)
        assert dgdr.sla.ttft is None
        assert dgdr.sla.itl is None
        assert dgdr.sla.e2eLatency == 35000.0

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_missing_ttft_and_itl_and_e2e_raises(self):
        """At least ttft+itl or e2eLatency must be provided."""
        dgdr = _make_dgdr(sla=SLASpec(ttft=None, itl=None, e2eLatency=None))
        with pytest.raises(ValueError, match="ttft.*itl.*e2eLatency"):
            valid_dgdr_spec(dgdr)


# ---------------------------------------------------------------------------
# validate_dgdr_dynamo_features
# ---------------------------------------------------------------------------


class TestValidateDgdrDynamoFeatures:
    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_no_features_passes(self):
        """No features → no error."""
        dgdr = _make_dgdr()
        validate_dgdr_dynamo_features(dgdr, aic_supported=False)

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_planner_throughput_scaling_aic_unsupported_rapid_sweep_raises(self):
        """Throughput scaling + rapid sweep + AIC unsupported is rejected."""
        dgdr = _make_dgdr(
            features=FeaturesSpec(
                planner=_make_planner(
                    enable_throughput_scaling=True,
                    pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
                    backend="vllm",
                )
            )
        )
        with pytest.raises(ValueError, match="AIC does not support"):
            validate_dgdr_dynamo_features(dgdr, aic_supported=False)

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_planner_throughput_scaling_aic_supported_passes(self):
        """Throughput scaling + rapid sweep + AIC supported is fine."""
        planner = _make_planner(
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        )
        dgdr = _make_dgdr(features=FeaturesSpec(planner=planner))
        validate_dgdr_dynamo_features(dgdr, aic_supported=True)
        assert (
            dgdr.features.planner.pre_deployment_sweeping_mode
            == PlannerPreDeploymentSweepMode.Rapid
        )

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_planner_load_scaling_only_aic_unsupported_passes(self):
        """Load scaling only (no throughput scaling) + AIC unsupported passes."""
        planner = _make_planner(
            enable_throughput_scaling=False,
            enable_load_scaling=True,
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
            backend="vllm",
        )
        dgdr = _make_dgdr(features=FeaturesSpec(planner=planner))
        validate_dgdr_dynamo_features(dgdr, aic_supported=False)
        assert (
            dgdr.features.planner.pre_deployment_sweeping_mode
            == PlannerPreDeploymentSweepMode.Rapid
        )

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_mocker_enabled_sweep_none_raises(self):
        """Mocker enabled + sweep mode None_ is rejected."""
        dgdr = _make_dgdr(
            features=FeaturesSpec(
                planner=_make_planner(
                    enable_throughput_scaling=False,
                    enable_load_scaling=True,
                    pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.None_,
                ),
                mocker=MockerSpec(enabled=True),
            )
        )
        with pytest.raises(ValueError, match="cannot be 'none'.*mocker"):
            validate_dgdr_dynamo_features(dgdr, aic_supported=True)

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_mocker_enabled_sweep_rapid_passes(self):
        """Mocker enabled + sweep mode Rapid is fine."""
        dgdr = _make_dgdr(
            features=FeaturesSpec(
                planner=_make_planner(
                    enable_throughput_scaling=False,
                    enable_load_scaling=True,
                    pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
                ),
                mocker=MockerSpec(enabled=True),
            )
        )
        validate_dgdr_dynamo_features(dgdr, aic_supported=True)


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
# assemble_final_config
# ---------------------------------------------------------------------------

_DGD_GEN = "dynamo.profiler.utils.dgd_generation"


class TestAssembleFinalConfig:
    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_no_planner_no_mocker_returns_dgd_config_unchanged(self, tmp_path):
        dgdr = _make_dgdr()
        ops = _make_ops(tmp_path)
        dgd_config = {"kind": "DynamoGraphDeployment"}

        result = assemble_final_config(
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

        result = assemble_final_config(
            dgdr,
            ops,
            None,
            PickedParallelConfig(tp=1),
            PickedParallelConfig(tp=1),
        )

        assert result is None

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_planner_no_mocker_returns_config_with_planner_cm(self, tmp_path):
        """Planner enabled, no mocker: result is [planner_cm, profile_cm, dgd_config]."""
        dgdr = _make_dgdr(features=FeaturesSpec(planner=_make_planner()))
        ops = _make_ops(tmp_path)
        os.makedirs(ops.output_dir, exist_ok=True)
        dgd_config = {"kind": "DGD", "spec": {"services": {}}}
        planner_cm = {"kind": "ConfigMap", "metadata": {"name": "planner-cm"}}
        profile_cm = {"kind": "ConfigMap", "metadata": {"name": "profile-cm"}}

        with (
            patch(
                f"{_DGD_GEN}.add_planner_to_config",
                return_value=planner_cm,
            ) as mock_planner,
            patch(
                f"{_DGD_GEN}.add_profile_data_to_config",
                return_value=profile_cm,
            ) as mock_profile,
        ):
            result = assemble_final_config(
                dgdr,
                ops,
                dgd_config,
                PickedParallelConfig(tp=1),
                PickedParallelConfig(tp=1),
            )

        mock_planner.assert_called_once()
        mock_profile.assert_called_once()
        assert result == [planner_cm, profile_cm, dgd_config]

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_mocker_plus_planner_uses_mocker_base(self, tmp_path):
        """Mocker + planner: mocker base is created first, then planner layered."""
        dgdr = _make_dgdr(
            features=FeaturesSpec(
                planner=_make_planner(),
                mocker=MockerSpec(enabled=True),
            )
        )
        ops = _make_ops(tmp_path)
        os.makedirs(ops.output_dir, exist_ok=True)
        dgd_config = {"kind": "DGD"}
        mocker_base = {"kind": "MockerDGD", "spec": {"services": {}}}
        planner_cm = {"kind": "ConfigMap", "metadata": {"name": "planner-cm"}}
        profile_cm = {"kind": "ConfigMap", "metadata": {"name": "profile-cm"}}

        with (
            patch(
                f"{_DGD_GEN}.generate_mocker_config",
                return_value=mocker_base,
            ) as mock_mocker,
            patch(
                f"{_DGD_GEN}.add_planner_to_config",
                return_value=planner_cm,
            ) as mock_planner,
            patch(
                f"{_DGD_GEN}.add_profile_data_to_config",
                return_value=profile_cm,
            ),
        ):
            result = assemble_final_config(
                dgdr,
                ops,
                dgd_config,
                PickedParallelConfig(tp=1),
                PickedParallelConfig(tp=1),
            )

        mock_mocker.assert_called_once()
        mock_planner.assert_called_once()
        assert mock_planner.call_args.args[1] is mocker_base
        assert result == [planner_cm, profile_cm, mocker_base]

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_mocker_only_no_planner_returns_mocker_config(self, tmp_path):
        """Mocker-only (no planner): generate_mocker_config is called,
        add_planner_to_config is not, profile data is still attached."""
        dgdr = _make_dgdr(features=FeaturesSpec(mocker=MockerSpec(enabled=True)))
        ops = _make_ops(tmp_path)
        os.makedirs(ops.output_dir, exist_ok=True)
        dgd_config = {"kind": "DGD"}
        mocker_base = {"kind": "MockerDGD", "spec": {"services": {}}}
        profile_cm = {"kind": "ConfigMap", "metadata": {"name": "profile-cm"}}

        with (
            patch(
                f"{_DGD_GEN}.generate_mocker_config",
                return_value=mocker_base,
            ) as mock_mocker,
            patch(
                f"{_DGD_GEN}.add_planner_to_config",
            ) as mock_planner,
            patch(
                f"{_DGD_GEN}.add_profile_data_to_config",
                return_value=profile_cm,
            ) as mock_profile,
        ):
            result = assemble_final_config(
                dgdr,
                ops,
                dgd_config,
                PickedParallelConfig(tp=1),
                PickedParallelConfig(tp=1),
            )

        mock_mocker.assert_called_once()
        mock_planner.assert_not_called()
        mock_profile.assert_called_once()
        assert result == [profile_cm, mocker_base]


# ---------------------------------------------------------------------------
# add_profile_data_to_config — mocker_enabled guard (DYN-2409)
# ---------------------------------------------------------------------------


class TestAddProfileDataMockerGuard:
    """Verify --planner-profile-data is only injected for mocker workers."""

    @staticmethod
    def _sglang_dgd():
        """Minimal DGD with sglang-style 'prefill' and 'decode' workers."""
        return {
            "spec": {
                "services": {
                    "Planner": {
                        "extraPodSpec": {
                            "mainContainer": {"args": ["--config", "{}"]},
                        }
                    },
                    "prefill": {
                        "extraPodSpec": {
                            "mainContainer": {
                                "args": [
                                    "-m",
                                    "dynamo.sglang",
                                    "--model-path",
                                    "Qwen/Qwen3-32B",
                                    "--disaggregation-mode",
                                    "prefill",
                                ]
                            }
                        }
                    },
                    "decode": {
                        "extraPodSpec": {
                            "mainContainer": {
                                "args": [
                                    "-m",
                                    "dynamo.sglang",
                                    "--model-path",
                                    "Qwen/Qwen3-32B",
                                    "--disaggregation-mode",
                                    "decode",
                                ]
                            }
                        }
                    },
                }
            }
        }

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_mocker_disabled_no_planner_profile_data_in_workers(self, tmp_path):
        """When mocker is disabled, workers must NOT receive --planner-profile-data."""
        dgd = self._sglang_dgd()
        with patch(f"{_DGD_GEN}._load_profiling_data", return_value={"prefill": {}}):
            add_profile_data_to_config(dgd, str(tmp_path), mocker_enabled=False)

        for name in ("prefill", "decode"):
            args = dgd["spec"]["services"][name]["extraPodSpec"]["mainContainer"][
                "args"
            ]
            assert (
                "--planner-profile-data" not in args
            ), f"sglang worker '{name}' should not have --planner-profile-data"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_mocker_enabled_injects_planner_profile_data(self, tmp_path):
        """When mocker is enabled, mocker workers MUST receive --planner-profile-data."""
        dgd = self._sglang_dgd()
        with patch(f"{_DGD_GEN}._load_profiling_data", return_value={"prefill": {}}):
            add_profile_data_to_config(dgd, str(tmp_path), mocker_enabled=True)

        for name in ("prefill", "decode"):
            args = dgd["spec"]["services"][name]["extraPodSpec"]["mainContainer"][
                "args"
            ]
            assert (
                "--planner-profile-data" in args
            ), f"mocker worker '{name}' should have --planner-profile-data"


# ---------------------------------------------------------------------------
# Regression tests: naive fallback resolved_backend propagation (bug fix)
# ---------------------------------------------------------------------------


class TestNaiveFallbackResolvedBackend:
    """Regression tests for the 'auto' backend KeyError bug.

    When backend='auto', _run_naive_fallback resolves it to a concrete
    backend (e.g. 'vllm') and must expose that in the result dict so
    run_profile() can pass the concrete name to run_interpolation().
    """

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    @pytest.mark.parallel
    def test_naive_fallback_resolved_backend_auto(self):
        """_run_naive_fallback sets 'resolved_backend' to the concrete backend
        when the input backend is 'auto'."""
        try:
            from unittest.mock import patch

            from dynamo.profiler.rapid import _run_naive_fallback
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")

        dgdr = _make_dgdr(backend="auto")

        with (
            patch(
                "dynamo.profiler.rapid.build_naive_generator_params",
                return_value={},
            ),
            patch(
                "dynamo.profiler.rapid.generate_backend_artifacts",
                return_value={},
            ),
        ):
            result = _run_naive_fallback(
                dgdr,
                model="Qwen/Qwen3-32B",
                total_gpus=8,
                system="h200_sxm",
                backend="auto",
            )

        # The resolved backend must be a concrete name, not 'auto'
        assert (
            "resolved_backend" in result
        ), "result dict must contain 'resolved_backend' key"
        resolved = result["resolved_backend"]
        assert (
            resolved != "auto"
        ), f"resolved_backend must not be 'auto', got {resolved!r}"
        assert resolved in (
            "vllm",
            "sglang",
            "trtllm",
        ), f"resolved_backend must be a concrete backend, got {resolved!r}"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    @pytest.mark.parallel
    def test_naive_fallback_resolved_backend_concrete(self):
        """_run_naive_fallback preserves the concrete backend in 'resolved_backend'
        when a concrete backend (e.g. 'vllm') is passed directly."""
        try:
            from unittest.mock import patch

            from dynamo.profiler.rapid import _run_naive_fallback
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")

        dgdr = _make_dgdr(backend="vllm")

        with (
            patch(
                "dynamo.profiler.rapid.build_naive_generator_params",
                return_value={},
            ),
            patch(
                "dynamo.profiler.rapid.generate_backend_artifacts",
                return_value={},
            ),
        ):
            result = _run_naive_fallback(
                dgdr,
                model="Qwen/Qwen3-32B",
                total_gpus=8,
                system="h200_sxm",
                backend="vllm",
            )

        assert result.get("resolved_backend") == "vllm"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    @pytest.mark.parallel
    def test_naive_fallback_chosen_exp_is_agg(self):
        """_run_naive_fallback always returns chosen_exp='agg' (aggregated config)."""
        try:
            from unittest.mock import patch

            from dynamo.profiler.rapid import _run_naive_fallback
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")

        dgdr = _make_dgdr(backend="vllm")
        with (
            patch(
                "dynamo.profiler.rapid.build_naive_generator_params",
                return_value={},
            ),
            patch(
                "dynamo.profiler.rapid.generate_backend_artifacts",
                return_value={},
            ),
        ):
            result = _run_naive_fallback(
                dgdr,
                model="Qwen/Qwen3-32B",
                total_gpus=8,
                system="h200_sxm",
                backend="vllm",
            )

        assert result.get("chosen_exp") == "agg"


# ---------------------------------------------------------------------------
# Regression tests: run_profile skips interpolation for aggregated configs
# ---------------------------------------------------------------------------


class TestRunProfileSkipsInterpolationForAggConfig:
    """Regression tests for the aggregated-config crash in run_interpolation.

    When the picked DGD config is aggregated (chosen_exp='agg'), run_profile()
    must not call run_interpolation(), which only works with disaggregated
    configs that have separate prefill and decode services.
    """

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    @pytest.mark.parallel
    def test_run_profile_skips_interpolation_when_agg(self, tmp_path):
        """run_profile skips run_interpolation when chosen_exp='agg'."""
        try:
            import asyncio
            from unittest.mock import AsyncMock, patch

            from dynamo.planner.config.planner_config import (
                PlannerPreDeploymentSweepMode,
            )
            from dynamo.profiler.profile_sla import run_profile
            from dynamo.profiler.utils.dgdr_v1beta1_types import FeaturesSpec
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")

        dgdr = _make_dgdr(
            backend="auto",
            features=FeaturesSpec(
                planner=_make_planner(
                    enable_throughput_scaling=True,
                    pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
                )
            ),
        )
        ops = _make_ops(tmp_path)
        os.makedirs(ops.output_dir, exist_ok=True)

        # Simulate naive fallback result: agg config, resolved backend
        agg_dgd = {
            "metadata": {"name": "vllm-agg"},
            "spec": {"services": {"Frontend": {}, "VllmWorker": {}}},
        }
        pick_result = {
            "best_config_df": None,
            "best_latencies": {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0},
            "dgd_config": agg_dgd,
            "chosen_exp": "agg",
            "resolved_backend": "vllm",
        }

        _PROFILE_SLA = "dynamo.profiler.profile_sla"
        with (
            patch(
                f"{_PROFILE_SLA}._extract_profiler_params",
                return_value=(
                    "Qwen/Qwen3-32B",
                    "auto",
                    "h200_sxm",
                    8,
                    4000,
                    1000,
                    None,
                    2000.0,
                    50.0,
                    __import__(
                        "dynamo.profiler.utils.defaults", fromlist=["SearchStrategy"]
                    ).SearchStrategy.RAPID,
                    "autoscale",
                ),
            ),
            patch(f"{_PROFILE_SLA}.check_model_hardware_support", return_value=False),
            patch(f"{_PROFILE_SLA}._check_auto_backend_support", return_value=False),
            patch(f"{_PROFILE_SLA}.validate_dgdr_dynamo_features"),
            patch(
                f"{_PROFILE_SLA}._execute_strategy",
                new_callable=AsyncMock,
                return_value=(
                    pick_result,
                    PickedParallelConfig(tp=1),
                    PickedParallelConfig(tp=1),
                    2000.0,
                    50.0,
                ),
            ),
            patch(
                f"{_PROFILE_SLA}.run_interpolation", new_callable=AsyncMock
            ) as mock_interp,
            patch(f"{_PROFILE_SLA}.assemble_final_config", return_value=agg_dgd),
            patch(f"{_PROFILE_SLA}.needs_profile_data", return_value=True),
            patch(
                f"{_PROFILE_SLA}.get_model_config_from_model_path",
                side_effect=Exception("no model"),
            ),
            patch(
                f"{_PROFILE_SLA}.cleanup_remaining_deployments", new_callable=AsyncMock
            ),
            patch(f"{_PROFILE_SLA}.valid_dgdr_spec"),
        ):
            asyncio.run(run_profile(dgdr, ops))

        # run_interpolation must NOT have been called for agg configs
        mock_interp.assert_not_called()

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    @pytest.mark.parallel
    def test_run_profile_calls_interpolation_with_resolved_backend_for_disagg(
        self, tmp_path
    ):
        """run_profile passes the concrete resolved backend (not 'auto') to
        run_interpolation when the picked config is disaggregated."""
        try:
            import asyncio
            from unittest.mock import AsyncMock, patch

            from dynamo.planner.config.planner_config import (
                PlannerPreDeploymentSweepMode,
            )
            from dynamo.profiler.profile_sla import run_profile
            from dynamo.profiler.utils.dgdr_v1beta1_types import FeaturesSpec
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")

        dgdr = _make_dgdr(
            backend="auto",
            features=FeaturesSpec(
                planner=_make_planner(
                    enable_throughput_scaling=True,
                    pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
                )
            ),
        )
        ops = _make_ops(tmp_path)
        os.makedirs(ops.output_dir, exist_ok=True)

        # Simulate AIC disagg result with 'auto' resolved to 'vllm'
        disagg_dgd = {
            "metadata": {"name": "vllm-disagg"},
            "spec": {
                "services": {
                    "Frontend": {},
                    "VllmPrefillWorker": {},
                    "VllmDecodeWorker": {},
                }
            },
        }
        pick_result = {
            "best_config_df": None,
            "best_latencies": {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0},
            "dgd_config": disagg_dgd,
            "chosen_exp": "disagg",
            "resolved_backend": "vllm",
        }

        _PROFILE_SLA = "dynamo.profiler.profile_sla"
        with (
            patch(
                f"{_PROFILE_SLA}._extract_profiler_params",
                return_value=(
                    "Qwen/Qwen3-32B",
                    "auto",
                    "h200_sxm",
                    8,
                    4000,
                    1000,
                    None,
                    2000.0,
                    50.0,
                    __import__(
                        "dynamo.profiler.utils.defaults", fromlist=["SearchStrategy"]
                    ).SearchStrategy.RAPID,
                    "autoscale",
                ),
            ),
            patch(f"{_PROFILE_SLA}.check_model_hardware_support", return_value=False),
            patch(f"{_PROFILE_SLA}._check_auto_backend_support", return_value=False),
            patch(f"{_PROFILE_SLA}.validate_dgdr_dynamo_features"),
            patch(
                f"{_PROFILE_SLA}._execute_strategy",
                new_callable=AsyncMock,
                return_value=(
                    pick_result,
                    PickedParallelConfig(tp=1),
                    PickedParallelConfig(tp=1),
                    2000.0,
                    50.0,
                ),
            ),
            patch(
                f"{_PROFILE_SLA}.run_interpolation", new_callable=AsyncMock
            ) as mock_interp,
            patch(f"{_PROFILE_SLA}.assemble_final_config", return_value=disagg_dgd),
            patch(f"{_PROFILE_SLA}.needs_profile_data", return_value=True),
            patch(
                f"{_PROFILE_SLA}.get_model_config_from_model_path",
                side_effect=Exception("no model"),
            ),
            patch(
                f"{_PROFILE_SLA}.cleanup_remaining_deployments", new_callable=AsyncMock
            ),
            patch(f"{_PROFILE_SLA}.valid_dgdr_spec"),
        ):
            asyncio.run(run_profile(dgdr, ops))

        # run_interpolation must be called, and with the resolved 'vllm' backend, not 'auto'
        mock_interp.assert_called_once()
        call_kwargs = mock_interp.call_args
        # backend is the 8th positional argument (index 7)
        called_backend = (
            call_kwargs.args[7]
            if call_kwargs.args
            else call_kwargs.kwargs.get("backend")
        )
        assert (
            called_backend == "vllm"
        ), f"run_interpolation must be called with resolved backend 'vllm', got {called_backend!r}"
