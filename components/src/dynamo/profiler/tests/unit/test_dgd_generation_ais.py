# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AISimulate and AIConfigurator interpolation in profiler DGD generation."""

from pathlib import Path

import pytest
import yaml

try:
    from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
    from dynamo.planner.config.parallelization import PickedParallelConfig
    from dynamo.planner.config.planner_config import (
        PlannerConfig,
        PlannerPreDeploymentSweepMode,
    )
    from dynamo.profiler.utils.dgd_generation import (
        _build_planner_config,
        _inject_mocker_ais_args,
        _load_latest_database_version,
        build_aic_interpolation_spec,
        build_ais_perf_model_spec,
        enable_vllm_benchmark_mode,
    )
    from dynamo.profiler.utils.dgd_template import load_dgd_template
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        FeaturesSpec,
        MockerSpec,
    )
    from dynamo.profiler.utils.profile_common import needs_profile_data
except ImportError as e:
    pytest.skip(f"Missing dependency: {e}", allow_module_level=True)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_aic_import_treats_missing_root_package_as_optional(monkeypatch):
    def raise_missing_root(_):
        raise ModuleNotFoundError(name="aiconfigurator_core")

    monkeypatch.setattr(
        "dynamo.profiler.utils.dgd_generation.importlib.import_module",
        raise_missing_root,
    )

    assert _load_latest_database_version() is None


@pytest.mark.parametrize(
    "missing_module",
    ["aiconfigurator_core.sdk.operations.attention", "unrelated_dependency"],
)
def test_aic_import_propagates_internal_or_unrelated_missing_module(
    monkeypatch, missing_module
):
    def raise_missing_dependency(_):
        raise ModuleNotFoundError(name=missing_module)

    monkeypatch.setattr(
        "dynamo.profiler.utils.dgd_generation.importlib.import_module",
        raise_missing_dependency,
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        _load_latest_database_version()

    assert exc_info.value.name == missing_module


def _dgdr(
    planner: PlannerConfig | None = None,
    model: str = "Qwen/Qwen3-32B",
    mocker_enabled: bool = False,
    search_strategy: str = "rapid",
) -> DynamoGraphDeploymentRequestSpec:
    features = None
    if planner is not None or mocker_enabled:
        features = FeaturesSpec(
            planner=planner,
            mocker=MockerSpec(enabled=True) if mocker_enabled else None,
        )
    return DynamoGraphDeploymentRequestSpec(
        model=model,
        features=features,
        searchStrategy=search_strategy,
    )


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
        """Mocker-only consumer still gets an AIC spec so --ais-* flags can be
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

    def test_mocker_only_rapid_produces_spec(self):
        """Mocker-only rapid requests use the DGDR search strategy."""
        dgdr = _dgdr(mocker_enabled=True)
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

    def test_mocker_only_thorough_returns_none(self):
        """Mocker-only thorough requests consume generated NPZ data."""
        dgdr = _dgdr(mocker_enabled=True, search_strategy="thorough")
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

        assert spec is None


class TestInjectMockerAicArgs:
    @pytest.fixture(autouse=True)
    def _resolved_version(self, monkeypatch):
        monkeypatch.setattr(
            "dynamo.profiler.utils.dgd_generation.get_latest_database_version",
            lambda **kwargs: "current",
        )

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
        out = _inject_mocker_ais_args(args, spec, spec.prefill_pick)
        assert "--ais-perf-model" in out
        assert out[out.index("--ais-backend") + 1] == "trtllm"
        assert out[out.index("--ais-system") + 1] == "h200_sxm"
        assert out[out.index("--ais-tp-size") + 1] == "1"
        assert out[out.index("--ais-moe-tp-size") + 1] == "1"
        assert out[out.index("--ais-moe-ep-size") + 1] == "8"
        assert out[out.index("--ais-attention-dp-size") + 1] == "8"
        # trtllm is not a mocker engine_type; leave --engine-type alone.
        assert "--engine-type" not in out
        assert out[out.index("--ais-backend-version") + 1] == "current"

    def test_matches_engine_type_for_vllm(self):
        spec = self._spec("vllm")
        out = _inject_mocker_ais_args([], spec, spec.prefill_pick)
        assert out[out.index("--engine-type") + 1] == "vllm"
        assert out[out.index("--ais-backend") + 1] == "vllm"
        assert out[out.index("--ais-backend-version") + 1] == "current"

    def test_matches_engine_type_for_sglang(self):
        spec = self._spec("sglang")
        out = _inject_mocker_ais_args([], spec, spec.decode_pick)
        assert out[out.index("--engine-type") + 1] == "sglang"
        assert out[out.index("--ais-backend") + 1] == "sglang"
        assert out[out.index("--ais-backend-version") + 1] == "current"


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

    def test_ais_perf_model_threads_into_planner_config(self, monkeypatch):
        resolved_versions = []

        def resolve_backend_version(*, system, backend):
            resolved_versions.append((system, backend))
            return "0.24.0"

        monkeypatch.setattr(
            "dynamo.profiler.utils.dgd_generation.get_latest_database_version",
            resolve_backend_version,
        )
        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        )
        dgdr = _dgdr(planner=planner)
        prefill_pick = PickedParallelConfig(tp=1, dp=1)
        decode_pick = PickedParallelConfig(tp=2, dp=1)
        spec = build_ais_perf_model_spec(
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
            ais_perf_model=spec,
        )

        assert cfg.model_name == dgdr.model
        assert "aic_perf_model" not in cfg.model_dump()
        assert cfg.ais_perf_model is not None
        assert cfg.ais_perf_model.roles["decode"]["model"] == dgdr.model
        assert cfg.ais_perf_model.roles["decode"]["system"] == "h200_sxm"
        assert cfg.ais_perf_model.roles["decode"]["backend"] == "vllm"
        assert cfg.ais_perf_model.roles["decode"]["backend_version"] == "0.24.0"
        assert cfg.ais_perf_model.roles["prefill"]["tp"] == prefill_pick.tp
        assert cfg.ais_perf_model.roles["decode"]["tp"] == decode_pick.tp
        assert resolved_versions == [("h200_sxm", "vllm")]

    def test_ais_perf_model_falls_back_when_database_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            "dynamo.profiler.utils.dgd_generation.get_latest_database_version",
            lambda **_: None,
        )
        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
        )
        dgdr = _dgdr(planner=planner)

        spec = build_ais_perf_model_spec(
            dgdr,
            best_prefill_pick=PickedParallelConfig(tp=1),
            best_decode_pick=PickedParallelConfig(tp=2),
            resolved_backend="vllm",
            system="unknown_system",
        )

        assert spec is None

    def test_ais_perf_model_falls_back_when_sdk_is_unavailable(
        self, monkeypatch, caplog
    ):
        monkeypatch.setattr(
            "dynamo.profiler.utils.dgd_generation.get_latest_database_version",
            None,
        )
        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
        )
        dgdr = _dgdr(planner=planner)

        spec = build_ais_perf_model_spec(
            dgdr,
            best_prefill_pick=PickedParallelConfig(tp=1),
            best_decode_pick=PickedParallelConfig(tp=2),
            resolved_backend="vllm",
            system="h200_sxm",
        )

        assert spec is None
        assert "AISimulate perf model is unavailable" in caplog.text

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
    def test_ais_perf_model_skips_mode_missing_required_pick(
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

        spec = build_ais_perf_model_spec(
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
    def test_mocker_only_rapid_returns_false(self):
        """Mocker-only rapid uses AIC directly instead of profile files."""
        dgdr = _dgdr(mocker_enabled=True)
        assert needs_profile_data(dgdr) is False

    def test_mocker_only_thorough_returns_true(self):
        """Mocker-only thorough still consumes generated profile files."""
        dgdr = _dgdr(mocker_enabled=True, search_strategy="thorough")
        assert needs_profile_data(dgdr) is True

    def test_rapid_planner_only_returns_false(self):
        """Planner-only rapid: no files needed; planner will use aic_spec."""
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
        planner = PlannerConfig(
            enable_throughput_scaling=True,
            enable_load_scaling=False,
            optimization_target="sla",
            pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Thorough,
        )
        dgdr = _dgdr(planner=planner, mocker_enabled=True)
        assert needs_profile_data(dgdr) is True


def _component(name: str, component_type: str, **container_fields) -> dict:
    return {
        "name": name,
        "type": component_type,
        "podTemplate": {
            "spec": {
                "containers": [{"name": "main", **container_fields}],
            }
        },
    }


def _component_map(config: dict) -> dict[str, dict]:
    return {
        component["name"]: component
        for component in config.get("spec", {}).get("components", [])
    }


def _main_container(component: dict) -> dict:
    return component["podTemplate"]["spec"]["containers"][0]


def _benchmark_mode(component: dict) -> str | None:
    env = _main_container(component).get("env", [])
    for e in env:
        if isinstance(e, dict) and e.get("name") == "DYN_BENCHMARK_MODE":
            return e.get("value")
    return None


class TestEnableVllmBenchmarkMode:
    def test_disagg_sets_prefill_and_decode(self):
        cfg = {
            "spec": {
                "components": [
                    _component("Frontend", "frontend"),
                    _component("custom-prefill", "prefill"),
                    _component("custom-decode", "decode"),
                ]
            }
        }
        enable_vllm_benchmark_mode(cfg)
        components = _component_map(cfg)
        assert _benchmark_mode(components["custom-prefill"]) == "prefill"
        assert _benchmark_mode(components["custom-decode"]) == "decode"
        assert "env" not in _main_container(components["Frontend"])

    @pytest.mark.parametrize(
        "worker_name", ["worker", "VllmDecodeWorker", "VllmWorker", "custom-worker"]
    )
    def test_agg_resolves_worker_by_type(self, worker_name: str):
        cfg = {
            "spec": {
                "components": [
                    _component("Frontend", "frontend"),
                    _component(worker_name, "worker"),
                ]
            }
        }
        enable_vllm_benchmark_mode(cfg)
        assert _benchmark_mode(_component_map(cfg)[worker_name]) == "agg"

    def test_agg_template_sets_single_generic_worker(self):
        cfg = load_dgd_template("vllm", "agg")

        enable_vllm_benchmark_mode(cfg)

        worker = next(
            component
            for component in cfg["spec"]["components"]
            if component["type"] == "worker"
        )
        assert worker["name"] == "worker"
        assert _benchmark_mode(worker) == "agg"

    def test_real_agg_template_sets_single_worker(self):
        repository_root = Path(__file__).resolve().parents[6]
        template_path = repository_root / "examples/backends/vllm/deploy/agg.yaml"
        cfg = yaml.safe_load(template_path.read_text(encoding="utf-8"))

        components = _component_map(cfg)
        assert "worker" in components
        assert "decode" not in components

        enable_vllm_benchmark_mode(cfg)
        assert _benchmark_mode(components["worker"]) == "agg"

    def test_idempotent_replaces_existing_value(self):
        # Simulates a user override that sets DYN_BENCHMARK_MODE to an
        # incorrect role; the helper must overwrite with the canonical value.
        cfg = {
            "spec": {
                "components": [
                    _component(
                        "decode",
                        "decode",
                        env=[
                            {"name": "SOMETHING_ELSE", "value": "keep"},
                            {"name": "DYN_BENCHMARK_MODE", "value": "wrong"},
                        ],
                    )
                ]
            }
        }
        enable_vllm_benchmark_mode(cfg)
        component = _component_map(cfg)["decode"]
        env = _main_container(component)["env"]
        names = [e["name"] for e in env]
        assert names.count("DYN_BENCHMARK_MODE") == 1
        assert _benchmark_mode(component) == "decode"
        # Unrelated env vars are preserved.
        assert {"name": "SOMETHING_ELSE", "value": "keep"} in env

    def test_non_worker_components_unchanged(self):
        cfg = {
            "spec": {
                "components": [
                    _component("Frontend", "frontend"),
                    _component("Planner", "planner"),
                    _component("Gateway", "epp"),
                ]
            }
        }
        enable_vllm_benchmark_mode(cfg)
        for component in cfg["spec"]["components"]:
            assert _benchmark_mode(component) is None

    def test_preserves_unrelated_component_keys(self):
        cfg = {
            "spec": {
                "components": [
                    _component(
                        "prefill",
                        "prefill",
                        image="nvcr.io/foo:1.0",
                        args=["--model-path", "x"],
                    )
                ]
            }
        }
        enable_vllm_benchmark_mode(cfg)
        component = _component_map(cfg)["prefill"]
        mc = _main_container(component)
        assert mc["image"] == "nvcr.io/foo:1.0"
        assert mc["args"] == ["--model-path", "x"]
        assert _benchmark_mode(component) == "prefill"


def test_profiler_preserves_explicit_estimator_config(tmp_path):
    planner = PlannerConfig(
        mode="decode",
        optimization_target="sla",
        ais_perf_model={
            "roles": {
                "decode": {
                    "model": "Qwen/Qwen3-32B",
                    "system": "h200_sxm",
                    "backend": "vllm",
                    "estimation_mode": "fpm_regression",
                    "systems_paths": [str(tmp_path)],
                    "estimator_config": {"fpm_regression": {"min_observations": 30}},
                }
            }
        },
    )
    dgdr = _dgdr(planner=planner)
    pick = PickedParallelConfig(tp=1)
    generated = build_ais_perf_model_spec(
        dgdr,
        best_prefill_pick=None,
        best_decode_pick=pick,
        resolved_backend="vllm",
        system="h200_sxm",
    )
    result = _build_planner_config(dgdr, None, pick, ais_perf_model=generated)
    reloaded = PlannerConfig.model_validate(result.model_dump(mode="json"))
    assert reloaded.ais_perf_model == planner.ais_perf_model
    assert generated is not planner.ais_perf_model


@pytest.mark.parametrize(
    "configured, pick, backend, system, conflict",
    [
        (
            {"model": "different-model"},
            PickedParallelConfig(),
            "vllm",
            "h200_sxm",
            "model",
        ),
        ({}, PickedParallelConfig(tp=4), "vllm", "h200_sxm", "tp"),
        ({}, PickedParallelConfig(), "sglang", "h200_sxm", "backend"),
        ({}, PickedParallelConfig(), "vllm", "b200_sxm", "system"),
        ({}, PickedParallelConfig(pp=2), "vllm", "h200_sxm", "pp"),
        ({}, PickedParallelConfig(dp=2), "vllm", "h200_sxm", "attention_dp"),
        (
            {"tp": 2, "moe_tp_size": 2, "moe_ep_size": 1},
            PickedParallelConfig(tp=2, moe_tp=1, moe_ep=2),
            "vllm",
            "h200_sxm",
            "moe_tp_size",
        ),
    ],
)
def test_profiler_rejects_estimator_identity_that_disagrees_with_deployment(
    configured, pick, backend, system, conflict
):
    planner = PlannerConfig(
        mode="decode",
        optimization_target="sla",
        ais_perf_model={
            "roles": {
                "decode": {
                    "model": "Qwen/Qwen3-32B",
                    "system": "h200_sxm",
                    "backend": "vllm",
                    "estimation_mode": "fpm_regression",
                    **configured,
                }
            }
        },
    )
    with pytest.raises(ValueError, match=rf"roles\.decode\.{conflict}=.*conflicts"):
        build_ais_perf_model_spec(
            _dgdr(planner=planner),
            best_prefill_pick=None,
            best_decode_pick=pick,
            resolved_backend=backend,
            system=system,
        )
