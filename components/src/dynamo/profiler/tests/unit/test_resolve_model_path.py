# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for resolve_model_path() and the rapid.py / thorough.py call
sites that feed its result into aiconfigurator."""

import copy
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
    from dynamo.profiler.rapid import (
        _generate_dgd_from_pick,
        _run_autoscale_sim,
        _run_default_sim,
    )
    from dynamo.profiler.thorough import (
        _normalize_candidate_model_identity,
        run_thorough,
    )
    from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS
    from dynamo.profiler.utils.dgd_materialization import DGDMaterializationPurpose
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        HardwareSpec,
        ModelCacheSpec,
        OverridesSpec,
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


def _make_model_dir(path) -> str:
    """Create an AIC-loadable model directory (one containing config.json)
    and return its path as a string."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    return str(path)


# ---------------------------------------------------------------------------
# resolve_model_path() helper
# ---------------------------------------------------------------------------


class TestResolveModelPath:
    """Tests for the resolve_model_path() helper."""

    def test_returns_local_path_when_pvc_dir_has_config(self, tmp_path):
        """Full PVC spec + a directory containing config.json -> the local path."""
        local_dir = tmp_path / "hub" / "models--Qwen--Qwen3-32B"
        _make_model_dir(local_dir)
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
        _make_model_dir(local_dir)
        dgdr = _make_dgdr(modelCache=_pvc_model_cache(f"{tmp_path}/", "/model"))
        assert resolve_model_path(dgdr) == str(local_dir)

    def test_returns_hf_id_when_local_path_is_a_file(self, tmp_path):
        """The resolved path exists but is a file, not a directory -> the HF id."""
        (tmp_path / "model").write_text("not a directory")
        dgdr = _make_dgdr(modelCache=_pvc_model_cache(str(tmp_path), "model"))
        assert resolve_model_path(dgdr) == _HF_ID

    def test_returns_hf_id_when_dir_has_no_config_json(self, tmp_path):
        """The PVC dir exists but has no config.json -> the HF id. AIC treats
        any directory as a local model and would otherwise hard-fail with no
        HuggingFace fallback."""
        local_dir = tmp_path / "model"
        local_dir.mkdir()  # directory only, no config.json
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
        _make_model_dir(local_dir)
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
        _make_model_dir(local_dir)
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
# DGD generation must keep the served model name as the HF id
# ---------------------------------------------------------------------------


class TestGenerateDgdKeepsServedModelName:
    """_generate_dgd_from_pick() must not leak a resolved local PVC path into
    the generated DGD's served model name.
    """

    @staticmethod
    def _generator_cfg(model_path: str) -> dict:
        return {
            "ServiceConfig": {
                "model_path": model_path,
                "served_model_path": model_path,
                "include_frontend": True,
            },
            "K8sConfig": {},
        }

    @staticmethod
    def _task_config() -> MagicMock:
        tc = MagicMock()
        tc.total_gpus = 8
        tc.backend_name = "trtllm"
        tc.backend_version = None
        return tc

    def _capture_generate(self, dgdr, cfg) -> MagicMock:
        """Run _generate_dgd_from_pick with the generator stubbed out and
        return the generate_backend_artifacts mock for assertions."""
        best_df = pd.DataFrame([{"tp": 1}])
        task_configs = {"disagg": self._task_config()}
        with (
            patch(
                "dynamo.profiler.rapid.task_config_to_generator_config",
                return_value=cfg,
            ),
            patch(
                "dynamo.profiler.rapid.generate_backend_artifacts",
                return_value={"k8s_deploy.yaml": ""},
            ) as mock_generate,
        ):
            _generate_dgd_from_pick(dgdr, best_df, "disagg", task_configs)
        return mock_generate

    def test_local_pvc_path_does_not_leak_into_served_model_name(self, tmp_path):
        """the ServiceConfig handed to the generator carries dgdr.model,
        not the resolved local path."""
        local_dir = tmp_path / "model"
        local_dir.mkdir()
        dgdr = _make_dgdr(modelCache=_pvc_model_cache(str(tmp_path), "model"))

        mock_generate = self._capture_generate(
            dgdr, self._generator_cfg(str(local_dir))
        )

        service_cfg = mock_generate.call_args.kwargs["params"]["ServiceConfig"]
        assert service_cfg["model_path"] == _HF_ID
        assert service_cfg["served_model_path"] == _HF_ID

    def test_no_pvc_keeps_hf_id(self):
        """No PVC: ServiceConfig.model_path is already the HF id and stays so."""
        dgdr = _make_dgdr()

        mock_generate = self._capture_generate(dgdr, self._generator_cfg(_HF_ID))

        service_cfg = mock_generate.call_args.kwargs["params"]["ServiceConfig"]
        assert service_cfg["model_path"] == _HF_ID
        assert service_cfg["served_model_path"] == _HF_ID


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
        _make_model_dir(local_dir)
        dgdr = _make_dgdr(modelCache=_pvc_model_cache(str(pvc_root), "model"))

        mock_enumerate = await self._capture_enumerate(dgdr, tmp_path)

        assert mock_enumerate.call_args.kwargs["model_path"] == str(local_dir)

    async def test_enumerate_uses_hf_id_when_no_pvc(self, tmp_path):
        """run_thorough -> enumerate_profiling_configs gets the HF id when no PVC."""
        dgdr = _make_dgdr()

        mock_enumerate = await self._capture_enumerate(dgdr, tmp_path)

        assert mock_enumerate.call_args.kwargs["model_path"] == _HF_ID

    async def test_materializes_each_candidate_once_with_resolved_model_path(
        self, tmp_path
    ):
        dgdr = _make_dgdr()
        prefill = MagicMock(dgd_config={"candidate": "prefill"})
        decode = MagicMock(dgd_config={"candidate": "decode"})

        def _materialize(config, **_kwargs):
            return {"materialized": config["candidate"]}

        with (
            patch(
                "dynamo.profiler.thorough.enumerate_profiling_configs",
                return_value=([prefill], [decode]),
            ),
            patch(
                "dynamo.profiler.thorough.materialize_dgd",
                side_effect=_materialize,
            ) as materialize,
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
                ProfilerOperationalConfig(output_dir=str(tmp_path)),
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

        assert materialize.call_count == 2
        assert [call.args[0] for call in materialize.call_args_list] == [
            {"candidate": "prefill"},
            {"candidate": "decode"},
        ]
        assert all(
            call.kwargs
            == {
                "purpose": DGDMaterializationPurpose.BENCHMARK_CANDIDATE,
                "override": None,
                "tolerations": [],
                "runtime_backend": "trtllm",
                "model_name_or_path": _HF_ID,
            }
            for call in materialize.call_args_list
        )
        assert prefill.dgd_config == {"materialized": "prefill"}
        assert decode.dgd_config == {"materialized": "decode"}

    def test_cache_only_pvc_does_not_rewrite_candidate_model(self):
        """A PVC without pvcModelPath remains an HF_HOME-style cache mount."""
        dgdr = _make_dgdr(
            modelCache=ModelCacheSpec(
                pvcName="model-cache",
                pvcMountPath="/opt/model-cache",
            )
        )
        candidate_config = {"sentinel": "unchanged"}
        candidate = MagicMock(dgd_config=candidate_config)
        modifier = MagicMock()

        _normalize_candidate_model_identity(
            [candidate], dgdr, dgdr.modelCache, modifier
        )

        modifier.update_model_from_pvc.assert_not_called()
        assert candidate.dgd_config is candidate_config

    async def test_candidates_keep_hf_name_and_pvc_runtime_path(self, tmp_path):
        """Every sweep candidate separates API identity from PVC load path."""
        pvc_root = tmp_path / "pvc"
        local_dir = pvc_root / "model"
        _make_model_dir(local_dir)
        modifier = CONFIG_MODIFIERS["vllm"]
        candidate_config = modifier.build_dgd_config(
            mode="agg",
            model_name=str(local_dir),
            image="example/vllm:base",
            agg_cli_args=[],
            agg_replicas=1,
            agg_gpus=1,
            pvc_name="model-cache",
            pvc_mount_path=str(pvc_root),
            model_path=str(local_dir),
        )
        worker_name = next(
            name
            for name in candidate_config["spec"]["services"]
            if name not in {"Frontend", "Planner"}
        )
        dgdr = _make_dgdr(
            backend="vllm",
            modelCache=_pvc_model_cache(str(pvc_root), "model"),
            overrides=OverridesSpec(
                dgd={
                    "spec": {
                        "services": {
                            worker_name: {
                                "extraPodSpec": {
                                    "mainContainer": {
                                        "image": "example/vllm:override",
                                        "args": [
                                            "--model=/stale/path",
                                            "--served-model-name=stale/model",
                                        ],
                                    }
                                }
                            }
                        }
                    }
                }
            ),
        )
        prefill = MagicMock(dgd_config=copy.deepcopy(candidate_config))
        decode = MagicMock(dgd_config=copy.deepcopy(candidate_config))
        ops = ProfilerOperationalConfig(output_dir=str(tmp_path))

        def _apply_override(config, _override):
            result = copy.deepcopy(config)
            main_container = result["spec"]["services"][worker_name]["extraPodSpec"][
                "mainContainer"
            ]
            main_container["image"] = "example/vllm:override"
            main_container["args"] = [
                "--model=/stale/path",
                "--served-model-name=stale/model",
            ]
            return result

        with (
            patch(
                "dynamo.profiler.thorough.enumerate_profiling_configs",
                return_value=([prefill], [decode], True, 1),
            ),
            patch(
                "dynamo.profiler.utils.dgd_materialization.apply_dgd_overrides",
                side_effect=_apply_override,
            ) as apply_override,
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
                "vllm",
                8,
                4000,
                1000,
                2000.0,
                50.0,
                None,
                [],
            )

        assert apply_override.call_count == 2
        assert all(
            call.args[1] == dgdr.overrides.dgd for call in apply_override.call_args_list
        )
        for candidate in (prefill, decode):
            assert modifier.get_model_name(candidate.dgd_config) == (
                _HF_ID,
                str(local_dir),
            )
            services = candidate.dgd_config["spec"]["services"]
            worker = services[worker_name]
            args = worker["extraPodSpec"]["mainContainer"]["args"]
            assert worker["extraPodSpec"]["mainContainer"]["image"] == (
                "example/vllm:override"
            )
            assert [
                arg for arg in args if arg == "--model" or arg.startswith("--model=")
            ] == ["--model"]
            assert [
                arg
                for arg in args
                if arg == "--served-model-name"
                or arg.startswith("--served-model-name=")
            ] == ["--served-model-name"]
            frontend_args = services["Frontend"]["extraPodSpec"]["mainContainer"][
                "args"
            ]
            assert frontend_args[frontend_args.index("--model-name") + 1] == _HF_ID
            assert frontend_args[frontend_args.index("--model-path") + 1] == str(
                local_dir
            )
            assert all(
                any(vm.get("name") == "model-cache" for vm in service["volumeMounts"])
                for service in services.values()
            )

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
        _make_model_dir(local_dir)
        dgdr = _make_dgdr(modelCache=_pvc_model_cache(str(pvc_root), "model"))

        mock_task_config = await self._capture_task_config(dgdr, tmp_path)

        assert mock_task_config.call_args.kwargs["model_path"] == str(local_dir)

    async def test_taskconfig_uses_hf_id_when_no_pvc(self, tmp_path):
        """run_thorough -> TaskConfig gets the HF id when no PVC."""
        dgdr = _make_dgdr()

        mock_task_config = await self._capture_task_config(dgdr, tmp_path)

        assert mock_task_config.call_args.kwargs["model_path"] == _HF_ID
