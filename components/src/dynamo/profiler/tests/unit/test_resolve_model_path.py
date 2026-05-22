# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for resolve_model_path() and the rapid.py / thorough.py call
sites that feed its result into aiconfigurator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]

try:
    from dynamo.profiler.rapid import _run_autoscale_sim, _run_default_sim
    from dynamo.profiler.thorough import run_thorough
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        HardwareSpec,
        ModelCacheSpec,
        SLASpec,
        WorkloadSpec,
    )
    from dynamo.profiler.utils.profile_common import (
        ProfilerOperationalConfig,
        resolve_model_path,
    )
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


_HF_ID = "Qwen/Qwen3-32B"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_dgdr(**overrides) -> DynamoGraphDeploymentRequestSpec:
    base = dict(
        model=_HF_ID,
        backend="trtllm",
        image="nvcr.io/nvidia/ai-dynamo/dynamo-frontend:latest",
        hardware=HardwareSpec(gpuSku="h200_sxm", totalGpus=8, numGpusPerNode=8),
        workload=WorkloadSpec(isl=4000, osl=1000),
        sla=SLASpec(ttft=2000.0, itl=50.0),
    )
    base.update(overrides)
    return DynamoGraphDeploymentRequestSpec(**base)


def _pvc_model_cache(mount_path: str, model_path: str) -> ModelCacheSpec:
    """A fully-populated modelCache spec."""
    return ModelCacheSpec(
        pvcName="model-cache",
        pvcMountPath=mount_path,
        pvcModelPath=model_path,
    )


# ---------------------------------------------------------------------------
# resolve_model_path() helper
# ---------------------------------------------------------------------------


class TestResolveModelPath:
    """Tests for the resolve_model_path() helper."""

    def test_returns_local_path_when_pvc_dir_exists(self, tmp_path):
        """Full PVC spec + an existing directory -> the local path."""
        local_dir = tmp_path / "hub" / "models--Qwen--Qwen3-32B"
        local_dir.mkdir(parents=True)
        dgdr = _make_dgdr(
            modelCache=_pvc_model_cache(str(tmp_path), "hub/models--Qwen--Qwen3-32B")
        )
        assert resolve_model_path(dgdr) == str(local_dir)

    def test_returns_hf_id_when_pvc_dir_missing(self, tmp_path):
        """Full PVC spec + no directory -> the HF id."""
        dgdr = _make_dgdr(
            modelCache=_pvc_model_cache(str(tmp_path), "hub/does-not-exist")
        )
        assert resolve_model_path(dgdr) == _HF_ID

    def test_returns_hf_id_when_no_model_cache(self):
        """No modelCache at all -> the HF id."""
        dgdr = _make_dgdr()
        assert dgdr.modelCache is None
        assert resolve_model_path(dgdr) == _HF_ID

    def test_returns_hf_id_when_pvc_name_missing(self, tmp_path):
        """modelCache present but pvcName unset -> the HF id, even if the directory exists."""
        local_dir = tmp_path / "model"
        local_dir.mkdir()
        dgdr = _make_dgdr(
            modelCache=ModelCacheSpec(pvcMountPath=str(tmp_path), pvcModelPath="model")
        )
        assert resolve_model_path(dgdr) == _HF_ID

    def test_returns_hf_id_when_pvc_model_path_missing(self, tmp_path):
        """modelCache present but pvcModelPath unset -> the HF id."""
        dgdr = _make_dgdr(
            modelCache=ModelCacheSpec(pvcName="model-cache", pvcMountPath=str(tmp_path))
        )
        assert resolve_model_path(dgdr) == _HF_ID

    def test_returns_hf_id_when_pvc_mount_path_missing(self):
        """modelCache present but pvcMountPath empty -> the HF id."""
        dgdr = _make_dgdr(
            modelCache=ModelCacheSpec(
                pvcName="model-cache", pvcMountPath="", pvcModelPath="model"
            )
        )
        assert resolve_model_path(dgdr) == _HF_ID

    def test_normalizes_surrounding_slashes(self, tmp_path):
        """slash on the model path -> resolve to the correct local directory."""
        local_dir = tmp_path / "model"
        local_dir.mkdir()
        dgdr = _make_dgdr(modelCache=_pvc_model_cache(f"{tmp_path}/", "/model"))
        assert resolve_model_path(dgdr) == str(local_dir)

    def test_returns_hf_id_when_local_path_is_a_file(self, tmp_path):
        """The resolved path exists but is a file, not a directory -> the HF id."""
        (tmp_path / "model").write_text("not a directory")
        dgdr = _make_dgdr(modelCache=_pvc_model_cache(str(tmp_path), "model"))
        assert resolve_model_path(dgdr) == _HF_ID


# ---------------------------------------------------------------------------
# rapid.py call sites
# ---------------------------------------------------------------------------


class TestRapidResolvesModelPath:
    """rapid.py call sites pass resolve_model_path()'s result to aiconfigurator."""

    @staticmethod
    def _execute_return(chosen="disagg"):
        """A fake _execute_task_configs return value (chosen, configs, _, _, latencies)."""
        best_df = pd.DataFrame([{"tp(p)": 1}])
        latencies = {"ttft": 100.0, "tpot": 10.0, "request_latency": 0.0}
        return chosen, {chosen: best_df}, None, None, {chosen: latencies}

    def test_default_sim_uses_local_path_when_pvc_mounted(self, tmp_path):
        """_run_default_sim -> build_default_task_configs gets the local PVC path."""
        local_dir = tmp_path / "model"
        local_dir.mkdir()
        dgdr = _make_dgdr(modelCache=_pvc_model_cache(str(tmp_path), "model"))

        with (
            patch(
                "dynamo.profiler.rapid.build_default_task_configs", return_value={}
            ) as mock_build,
            patch(
                "dynamo.profiler.rapid._execute_task_configs",
                return_value=self._execute_return(),
            ),
            patch("dynamo.profiler.rapid._generate_dgd_from_pick", return_value=None),
        ):
            _run_default_sim(
                dgdr,
                _HF_ID,
                "h200_sxm",
                "trtllm",
                8,
                4000,
                1000,
                2000.0,
                50.0,
                None,
                "default",
            )

        assert mock_build.call_args.kwargs["model_path"] == str(local_dir)

    def test_default_sim_uses_hf_id_when_no_pvc(self):
        """_run_default_sim -> build_default_task_configs gets the HF id when no PVC."""
        dgdr = _make_dgdr()

        with (
            patch(
                "dynamo.profiler.rapid.build_default_task_configs", return_value={}
            ) as mock_build,
            patch(
                "dynamo.profiler.rapid._execute_task_configs",
                return_value=self._execute_return(),
            ),
            patch("dynamo.profiler.rapid._generate_dgd_from_pick", return_value=None),
        ):
            _run_default_sim(
                dgdr,
                _HF_ID,
                "h200_sxm",
                "trtllm",
                8,
                4000,
                1000,
                2000.0,
                50.0,
                None,
                "default",
            )

        assert mock_build.call_args.kwargs["model_path"] == _HF_ID

    def test_autoscale_sim_uses_local_path_when_pvc_mounted(self, tmp_path):
        """_run_autoscale_sim -> TaskConfig gets the local PVC path."""
        local_dir = tmp_path / "model"
        local_dir.mkdir()
        dgdr = _make_dgdr(modelCache=_pvc_model_cache(str(tmp_path), "model"))

        runner = MagicMock()
        runner.run.return_value = {"pareto_df": pd.DataFrame()}
        with (
            patch("dynamo.profiler.rapid.TaskConfig") as mock_task_config,
            patch("dynamo.profiler.rapid.TaskRunner", return_value=runner),
            patch("dynamo.profiler.rapid._generate_dgd_from_pick", return_value=None),
        ):
            _run_autoscale_sim(
                dgdr, _HF_ID, "h200_sxm", "trtllm", 8, 4000, 1000, 2000.0, 50.0, None
            )

        assert mock_task_config.call_args.kwargs["model_path"] == str(local_dir)

    def test_autoscale_sim_uses_hf_id_when_no_pvc(self):
        """_run_autoscale_sim -> TaskConfig gets the HF id when no PVC."""
        dgdr = _make_dgdr()

        runner = MagicMock()
        runner.run.return_value = {"pareto_df": pd.DataFrame()}
        with (
            patch("dynamo.profiler.rapid.TaskConfig") as mock_task_config,
            patch("dynamo.profiler.rapid.TaskRunner", return_value=runner),
            patch("dynamo.profiler.rapid._generate_dgd_from_pick", return_value=None),
        ):
            _run_autoscale_sim(
                dgdr, _HF_ID, "h200_sxm", "trtllm", 8, 4000, 1000, 2000.0, 50.0, None
            )

        assert mock_task_config.call_args.kwargs["model_path"] == _HF_ID


# ---------------------------------------------------------------------------
# thorough.py call sites
# ---------------------------------------------------------------------------


class TestThoroughResolvesModelPath:
    """run_thorough() passes resolve_model_path()'s result into enumerate_profiling_configs."""

    async def _capture_enumerate(self, dgdr, output_dir) -> MagicMock:
        ops = ProfilerOperationalConfig(output_dir=str(output_dir))
        with (
            patch(
                "dynamo.profiler.thorough.enumerate_profiling_configs",
                return_value=([], []),
            ) as mock_enumerate,
            patch(
                "dynamo.profiler.thorough._benchmark_prefill_candidates",
                new=AsyncMock(return_value=pd.DataFrame()),
            ),
            patch(
                "dynamo.profiler.thorough._benchmark_decode_candidates",
                new=AsyncMock(return_value=pd.DataFrame()),
            ),
        ):
            await run_thorough(
                dgdr,
                ops,
                "default",
                _HF_ID,
                "h200_sxm",
                "trtllm",
                8,
                4000,
                1000,
                2000.0,
                50.0,
                None,
                [],
            )
        return mock_enumerate

    async def test_enumerate_uses_local_path_when_pvc_mounted(self, tmp_path):
        """run_thorough -> enumerate_profiling_configs gets the local PVC path."""
        pvc_root = tmp_path / "pvc"
        local_dir = pvc_root / "model"
        local_dir.mkdir(parents=True)
        dgdr = _make_dgdr(modelCache=_pvc_model_cache(str(pvc_root), "model"))

        mock_enumerate = await self._capture_enumerate(dgdr, tmp_path)

        assert mock_enumerate.call_args.kwargs["model_path"] == str(local_dir)

    async def test_enumerate_uses_hf_id_when_no_pvc(self, tmp_path):
        """run_thorough -> enumerate_profiling_configs gets the HF id when no PVC."""
        dgdr = _make_dgdr()

        mock_enumerate = await self._capture_enumerate(dgdr, tmp_path)

        assert mock_enumerate.call_args.kwargs["model_path"] == _HF_ID

    async def _capture_task_config(self, dgdr, output_dir) -> MagicMock:
        """The benchmark stages return non-empty DataFrames so run_thorough gets
        past the 'no results' early return and reaches Stage 4 DGD generation.
        """
        ops = ProfilerOperationalConfig(output_dir=str(output_dir))
        nonempty_df = pd.DataFrame([{"tp": 1}])
        with (
            patch(
                "dynamo.profiler.thorough.enumerate_profiling_configs",
                return_value=([], []),
            ),
            patch(
                "dynamo.profiler.thorough._benchmark_prefill_candidates",
                new=AsyncMock(return_value=nonempty_df),
            ),
            patch(
                "dynamo.profiler.thorough._benchmark_decode_candidates",
                new=AsyncMock(return_value=nonempty_df),
            ),
            patch(
                "dynamo.profiler.thorough._pick_thorough_best_config",
                return_value={"best_config_df": pd.DataFrame()},
            ),
            patch("dynamo.profiler.thorough.TaskConfig") as mock_task_config,
            patch(
                "dynamo.profiler.thorough._generate_dgd_from_pick",
                return_value=None,
            ),
        ):
            await run_thorough(
                dgdr,
                ops,
                "default",
                _HF_ID,
                "h200_sxm",
                "trtllm",
                8,
                4000,
                1000,
                2000.0,
                50.0,
                None,
                [],
            )
        return mock_task_config

    async def test_taskconfig_uses_local_path_when_pvc_mounted(self, tmp_path):
        """run_thorough -> TaskConfig gets the local PVC path."""
        pvc_root = tmp_path / "pvc"
        local_dir = pvc_root / "model"
        local_dir.mkdir(parents=True)
        dgdr = _make_dgdr(modelCache=_pvc_model_cache(str(pvc_root), "model"))

        mock_task_config = await self._capture_task_config(dgdr, tmp_path)

        assert mock_task_config.call_args.kwargs["model_path"] == str(local_dir)

    async def test_taskconfig_uses_hf_id_when_no_pvc(self, tmp_path):
        """run_thorough -> TaskConfig gets the HF id when no PVC."""
        dgdr = _make_dgdr()

        mock_task_config = await self._capture_task_config(dgdr, tmp_path)

        assert mock_task_config.call_args.kwargs["model_path"] == _HF_ID
