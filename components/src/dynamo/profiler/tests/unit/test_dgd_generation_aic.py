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
        _inject_mocker_aic_args,
        _mocker_worker_names,
        build_aic_interpolation_spec,
        build_aic_perf_model_spec,
        enable_vllm_benchmark_mode,
        generate_mocker_config,
    )
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        FeaturesSpec,
        MockerSpec,
        ModelCacheSpec,
    )
except ImportError as e:
    pytest.skip(f"Missing dependency: {e}", allow_module_level=True)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def _dgdr(
    planner: PlannerConfig | None = None,
    model: str = "Qwen/Qwen3-32B",
    mocker_enabled: bool = False,
) -> DynamoGraphDeploymentRequestSpec:
    features = None
    if planner is not None or mocker_enabled:
        features = FeaturesSpec(
            planner=planner,
            mocker=MockerSpec(enabled=True) if mocker_enabled else None,
        )
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

    def test_mocker_rapid_without_throughput_scaling_produces_spec(self):
        """Mocker-only consumer still gets an AIC spec so --aic-* flags can be
        injected on its worker args."""
        planner = PlannerConfig(
            enable_throughput_scaling=False,
            enable_load_scaling=True,
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        )
        dgdr = _dgdr(planner=planner, mocker_enabled=True)
        pick = PickedParallelConfig(tp=1, dp=8, moe_ep=8)
        spec = build_aic_interpolation_spec(
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
        assert isinstance(spec, AICInterpolationSpec)
        assert spec.backend == "trtllm"


class TestInjectMockerAicArgs:
    def _spec(self, backend: str = "trtllm") -> AICInterpolationSpec:
        pick = PickedParallelConfig(tp=1, dp=8, moe_tp=1, moe_ep=8)
        return AICInterpolationSpec(
            hf_id="Qwen/Qwen3-235B",
            system="h200_sxm",
            backend=backend,
            isl=1000,
            osl=100,
            sweep_max_context_length=4096,
            prefill_interpolation_granularity=8,
            decode_interpolation_granularity=4,
            prefill_pick=pick,
            decode_pick=pick,
        )

    def test_injects_all_required_flags(self):
        spec = self._spec("trtllm")
        args = ["--model-path", "Qwen/Qwen3-235B", "--disaggregation-mode", "prefill"]
        out = _inject_mocker_aic_args(args, spec, spec.prefill_pick)
        assert "--aic-perf-model" in out
        assert out[out.index("--aic-backend") + 1] == "trtllm"
        assert out[out.index("--aic-system") + 1] == "h200_sxm"
        assert out[out.index("--aic-tp-size") + 1] == "1"
        assert out[out.index("--aic-moe-tp-size") + 1] == "1"
        assert out[out.index("--aic-moe-ep-size") + 1] == "8"
        assert out[out.index("--aic-attention-dp-size") + 1] == "8"
        # trtllm is not a mocker engine_type; leave --engine-type alone.
        assert "--engine-type" not in out

    def test_matches_engine_type_for_vllm(self):
        spec = self._spec("vllm")
        out = _inject_mocker_aic_args([], spec, spec.prefill_pick)
        assert out[out.index("--engine-type") + 1] == "vllm"
        assert out[out.index("--aic-backend") + 1] == "vllm"

    def test_matches_engine_type_for_sglang(self):
        spec = self._spec("sglang")
        out = _inject_mocker_aic_args([], spec, spec.decode_pick)
        assert out[out.index("--engine-type") + 1] == "sglang"
        assert out[out.index("--aic-backend") + 1] == "sglang"


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

    def test_num_gpu_injection_ignores_attention_dp_overcount(self):
        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        )
        dgdr = _dgdr(planner=planner)
        pick = PickedParallelConfig(tp=8, pp=1, dp=8, moe_tp=1, moe_ep=1)
        cfg = _build_planner_config(dgdr, pick, pick, aic_spec=None)
        assert cfg.prefill_engine_num_gpu == 8
        assert cfg.decode_engine_num_gpu == 8

    def test_aic_perf_model_threads_into_planner_config(self):
        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        )
        dgdr = _dgdr(planner=planner)
        prefill_pick = PickedParallelConfig(tp=1, dp=1)
        decode_pick = PickedParallelConfig(tp=2, dp=1)
        spec = build_aic_perf_model_spec(
            dgdr,
            best_prefill_pick=prefill_pick,
            best_decode_pick=decode_pick,
            resolved_backend="vllm",
            system="h200_sxm",
        )

        cfg = _build_planner_config(
            dgdr,
            prefill_pick,
            decode_pick,
            aic_perf_model=spec,
        )

        assert cfg.aic_perf_model is not None
        assert cfg.aic_perf_model.hf_id == dgdr.model
        assert cfg.aic_perf_model.system == "h200_sxm"
        assert cfg.aic_perf_model.backend == "vllm"
        assert cfg.aic_perf_model.prefill_pick == prefill_pick
        assert cfg.aic_perf_model.decode_pick == decode_pick

    @pytest.mark.parametrize(
        ("mode", "prefill_pick", "decode_pick"),
        [
            ("prefill", None, PickedParallelConfig(tp=1)),
            ("decode", PickedParallelConfig(tp=1), None),
            ("agg", PickedParallelConfig(tp=1), None),
            ("disagg", None, PickedParallelConfig(tp=1)),
            ("disagg", PickedParallelConfig(tp=1), None),
        ],
    )
    def test_aic_perf_model_skips_mode_missing_required_pick(
        self, mode, prefill_pick, decode_pick
    ):
        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            mode=mode,
            optimization_target="sla",
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        )
        dgdr = _dgdr(planner=planner)

        spec = build_aic_perf_model_spec(
            dgdr,
            best_prefill_pick=prefill_pick,
            best_decode_pick=decode_pick,
            resolved_backend="vllm",
            system="h200_sxm",
        )

        assert spec is None

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

    def test_none_planner_only_returns_false(self):
        """Planner-only none mode can warm from native AIC or live FPMs."""
        from dynamo.profiler.utils.profile_common import needs_profile_data

        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.None_,
        )
        dgdr = _dgdr(planner=planner)
        assert needs_profile_data(dgdr) is False

    def test_mocker_rapid_returns_false(self):
        """Mocker + rapid: mocker pulls AIC perf data at runtime; no NPZ files."""
        from dynamo.profiler.utils.profile_common import needs_profile_data

        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        )
        dgdr = _dgdr(planner=planner, mocker_enabled=True)
        assert needs_profile_data(dgdr) is False

    def test_mocker_thorough_returns_true(self):
        """Mocker + thorough: mocker consumes real-GPU NPZ."""
        from dynamo.profiler.utils.profile_common import needs_profile_data

        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Thorough,
        )
        dgdr = _dgdr(planner=planner, mocker_enabled=True)
        assert needs_profile_data(dgdr) is True


def _benchmark_mode(svc: dict) -> str | None:
    env = svc.get("extraPodSpec", {}).get("mainContainer", {}).get("env", [])
    for e in env:
        if isinstance(e, dict) and e.get("name") == "DYN_BENCHMARK_MODE":
            return e.get("value")
    return None


class TestEnableVllmBenchmarkMode:
    def test_disagg_sets_prefill_and_decode(self):
        cfg = {
            "spec": {
                "services": {
                    "Frontend": {},
                    "VllmPrefillWorker": {},
                    "VllmDecodeWorker": {},
                }
            }
        }
        enable_vllm_benchmark_mode(cfg)
        services = cfg["spec"]["services"]
        assert _benchmark_mode(services["VllmPrefillWorker"]) == "prefill"
        assert _benchmark_mode(services["VllmDecodeWorker"]) == "decode"
        # Frontend service is untouched — no env list injected.
        assert "env" not in services["Frontend"].get("extraPodSpec", {}).get(
            "mainContainer", {}
        )

    def test_agg_sets_single_worker(self):
        cfg = {"spec": {"services": {"Frontend": {}, "VllmWorker": {}}}}
        enable_vllm_benchmark_mode(cfg)
        assert _benchmark_mode(cfg["spec"]["services"]["VllmWorker"]) == "agg"

    def test_idempotent_replaces_existing_value(self):
        # Simulates a user override that sets DYN_BENCHMARK_MODE to an
        # incorrect role; the helper must overwrite with the canonical value.
        cfg = {
            "spec": {
                "services": {
                    "VllmDecodeWorker": {
                        "extraPodSpec": {
                            "mainContainer": {
                                "env": [
                                    {"name": "SOMETHING_ELSE", "value": "keep"},
                                    {"name": "DYN_BENCHMARK_MODE", "value": "wrong"},
                                ]
                            }
                        }
                    }
                }
            }
        }
        enable_vllm_benchmark_mode(cfg)
        env = cfg["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
            "mainContainer"
        ]["env"]
        names = [e["name"] for e in env]
        assert names.count("DYN_BENCHMARK_MODE") == 1
        assert _benchmark_mode(cfg["spec"]["services"]["VllmDecodeWorker"]) == "decode"
        # Unrelated env vars are preserved.
        assert {"name": "SOMETHING_ELSE", "value": "keep"} in env

    def test_non_vllm_services_unchanged(self):
        cfg = {
            "spec": {
                "services": {
                    "prefill": {},
                    "decode": {},
                    "Frontend": {},
                }
            }
        }
        enable_vllm_benchmark_mode(cfg)
        for svc in cfg["spec"]["services"].values():
            assert _benchmark_mode(svc) is None

    def test_preserves_unrelated_service_keys(self):
        cfg = {
            "spec": {
                "services": {
                    "VllmPrefillWorker": {
                        "extraPodSpec": {
                            "mainContainer": {
                                "image": "nvcr.io/foo:1.0",
                                "args": ["--model-path", "x"],
                            }
                        }
                    }
                }
            }
        }
        enable_vllm_benchmark_mode(cfg)
        mc = cfg["spec"]["services"]["VllmPrefillWorker"]["extraPodSpec"][
            "mainContainer"
        ]
        assert mc["image"] == "nvcr.io/foo:1.0"
        assert mc["args"] == ["--model-path", "x"]
        assert (
            _benchmark_mode(cfg["spec"]["services"]["VllmPrefillWorker"]) == "prefill"
        )


class TestMockerModelCachePvc:
    """DGH-1124: mocker workers must load from the local PVC path (not the HF
    id) and get the model-cache PVC mounted, so offline deployments work."""

    def _mocker_worker_args(self, cfg, worker):
        svc = cfg["spec"]["services"][worker]
        return svc, svc["extraPodSpec"]["mainContainer"]["args"]

    def test_mocker_offline_pvc_uses_local_path_and_mounts_pvc(self, tmp_path):
        # Stage a local model dir with config.json on the "PVC" so
        # resolve_model_path prefers it over the HF id.
        mount = tmp_path / "models"
        model_dir = mount / "org" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")

        dgdr = DynamoGraphDeploymentRequestSpec(
            model="org/model",
            features=FeaturesSpec(mocker=MockerSpec(enabled=True)),
            modelCache=ModelCacheSpec(
                pvcName="model-cache",
                pvcMountPath=str(mount),
                pvcModelPath="org/model",
            ),
        )

        cfg = generate_mocker_config(dgdr, aic_spec=None)

        for worker in _mocker_worker_names():
            svc, args = self._mocker_worker_args(cfg, worker)
            # served name stays the model id so requests match
            assert args[args.index("--model-name") + 1] == "org/model"
            # load path is the local PVC path, not the HF id
            assert args[args.index("--model-path") + 1] == str(model_dir)
            # the model-cache PVC is mounted onto the worker
            mounts = svc["extraPodSpec"]["mainContainer"]["volumeMounts"]
            assert any(m["mountPath"] == str(mount) for m in mounts)
            volumes = svc["extraPodSpec"]["volumes"]
            assert any(
                v.get("persistentVolumeClaim", {}).get("claimName") == "model-cache"
                for v in volumes
            )

    def test_mocker_without_pvc_falls_back_to_model_id(self):
        dgdr = DynamoGraphDeploymentRequestSpec(
            model="org/model",
            features=FeaturesSpec(mocker=MockerSpec(enabled=True)),
        )
        cfg = generate_mocker_config(dgdr, aic_spec=None)
        for worker in _mocker_worker_names():
            svc, args = self._mocker_worker_args(cfg, worker)
            assert args[args.index("--model-path") + 1] == "org/model"
            # no PVC configured -> no model-cache mount added
            mounts = svc["extraPodSpec"]["mainContainer"].get("volumeMounts", [])
            assert not any(m.get("name") == "model-cache" for m in mounts)
