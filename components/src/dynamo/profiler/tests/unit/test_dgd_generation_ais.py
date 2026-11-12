# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AISimulate and AIConfigurator interpolation in profiler DGD generation."""

import json
import logging
from pathlib import Path

import pytest
import yaml

from dynamo.common.ais_perf_model import ENV_VAR
from dynamo.mocker.args import parse_args
from dynamo.mocker.config import load_mocker_engine_args, performance_config
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
    generate_mocker_config,
)
from dynamo.profiler.utils.dgd_template import load_dgd_template
from dynamo.profiler.utils.dgdr_v1beta1_types import (
    DynamoGraphDeploymentRequestSpec,
    FeaturesSpec,
    MockerSpec,
    OverridesSpec,
)
from dynamo.profiler.utils.profile_common import needs_profile_data

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_aic_import_treats_missing_root_package_as_optional(monkeypatch):
    def raise_missing_root(_):
        raise ModuleNotFoundError(name="aisimulate_core")

    monkeypatch.setattr(
        "dynamo.profiler.utils.dgd_generation.importlib.import_module",
        raise_missing_root,
    )

    assert _load_latest_database_version() is None


@pytest.mark.parametrize(
    "missing_module",
    ["aisimulate_core.sdk.operations.attention", "unrelated_dependency"],
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
    override: dict | None = None,
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
        overrides=OverridesSpec(dgd=override) if override is not None else None,
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
        config = json.loads(out[out.index("--ais-perf-model") + 1])
        assert config["backend"] == "trtllm"
        assert config["system"] == "h200_sxm"
        assert config["tp"] == 1
        assert config["moe_tp_size"] == 1
        assert config["moe_ep_size"] == 8
        assert config["attention_dp"] == 8
        assert config["backend_version"] == "current"
        old_flags = (
            "--ais-backend",
            "--ais-system",
            "--ais-tp-size",
            "--ais-moe-",
            "--ais-attention-dp-size",
        )
        assert not [token for token in out if token.startswith(old_flags)]
        # trtllm is not a mocker engine_type; leave --engine-type alone.
        assert "--engine-type" not in out

    def test_matches_engine_type_for_vllm(self):
        spec = self._spec("vllm")
        out = _inject_mocker_ais_args([], spec, spec.prefill_pick)
        config = json.loads(out[out.index("--ais-perf-model") + 1])
        assert out[out.index("--engine-type") + 1] == "vllm"
        assert config["backend"] == "vllm"
        assert config["backend_version"] == "current"
        assert config["model"] == "Qwen/Qwen3-235B"

    def test_matches_engine_type_for_sglang(self):
        spec = self._spec("sglang")
        out = _inject_mocker_ais_args([], spec, spec.decode_pick)
        config = json.loads(out[out.index("--ais-perf-model") + 1])
        assert out[out.index("--engine-type") + 1] == "sglang"
        assert config["backend"] == "sglang"
        assert config["backend_version"] == "current"
        assert config["model"] == "Qwen/Qwen3-235B"

    @pytest.mark.parametrize("explicit_capacity", [False, True])
    def test_dense_tp_pick_starts_with_canonical_estimator(self, explicit_capacity):
        # Only packaged model metadata/performance tables are used; no weights.
        from aisimulate_core.sdk import RustForwardPassPerfModel

        from dynamo.mocker.args import parse_args
        from dynamo.mocker.config import build_mocker_engine_args

        pick = PickedParallelConfig(tp=2)
        spec = self._spec("vllm").model_copy(
            update={
                "hf_id": "Qwen/Qwen3-32B",
                "prefill_pick": pick,
                "decode_pick": pick,
            }
        )
        args = ["--model-path", spec.hf_id, "--disaggregation-mode", "prefill"]
        if explicit_capacity:
            args.extend(["--num-gpu-blocks-override", "1024"])
        engine = build_mocker_engine_args(
            parse_args(_inject_mocker_ais_args(args, spec, pick))
        )
        assert engine["engine"]["num_gpu_blocks"] > 0
        perf_config = engine["engine"]["timing_model"]["config"]
        assert perf_config["tp"] == 2
        assert perf_config["moe_tp_size"] is None
        model = RustForwardPassPerfModel.best_available(perf_config)
        assert model.diagnostics()["readiness"] == "ready"


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

    @pytest.mark.parametrize("model_name", [None, "", "served-alias"])
    def test_planner_model_name_preserves_override(self, model_name):
        planner = PlannerConfig(model_name=model_name)
        dgdr = _dgdr(planner=planner)
        cfg = _build_planner_config(dgdr, None, None)
        assert cfg.model_name == (model_name or dgdr.model)
        assert planner.model_name == model_name

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
        entries = {entry["worker_type"]: entry for entry in cfg.ais_perf_model}
        assert entries["decode"]["model"] == dgdr.model
        assert entries["decode"]["system"] == "h200_sxm"
        assert entries["decode"]["backend"] == "vllm"
        assert entries["decode"]["backend_version"] == "0.24.0"
        assert entries["prefill"]["tp"] == prefill_pick.tp
        assert entries["decode"]["tp"] == decode_pick.tp
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


def _written_planner(mode: str, ais_perf_model) -> PlannerConfig:
    return PlannerConfig(
        mode=mode, optimization_target="sla", ais_perf_model=ais_perf_model
    )


def _checked_value(planner: PlannerConfig, prefill_pick=None, decode_pick=None):
    return build_ais_perf_model_spec(
        _dgdr(planner=planner),
        best_prefill_pick=prefill_pick,
        best_decode_pick=decode_pick,
        resolved_backend="vllm",
        system="h200_sxm",
    )


@pytest.mark.parametrize(
    "written,conflict",
    [
        pytest.param(
            {"model": "Qwen/Qwen3-235B-A22B"},
            "model='Qwen/Qwen3-235B-A22B' conflicts with selected deployment "
            "model='Qwen/Qwen3-32B'",
            id="model",
        ),
        pytest.param(
            {"system": "b200_sxm"},
            "system='b200_sxm' conflicts with selected deployment system='h200_sxm'",
            id="system",
        ),
        pytest.param(
            {"system": "b200_sxm", "model": "Qwen/Qwen3-235B-A22B"},
            "model='Qwen/Qwen3-235B-A22B' conflicts with selected deployment "
            "model='Qwen/Qwen3-32B'",
            id="model-and-system",
        ),
        pytest.param(
            {"backend": "sglang"},
            "backend='sglang' conflicts with selected deployment backend='vllm'",
            id="backend",
        ),
        pytest.param(
            {"tp": 8}, "tp=8 conflicts with selected deployment tp=2", id="tp"
        ),
        pytest.param(
            {"pp": 2}, "pp=2 conflicts with selected deployment pp=1", id="pp"
        ),
        pytest.param(
            {"attention_dp": 4},
            "attention_dp=4 conflicts with selected deployment attention_dp=1",
            id="attention-dp",
        ),
        pytest.param(
            {"moe_tp_size": 2},
            "moe_tp_size=2 conflicts with selected deployment moe_tp_size=1",
            id="moe-tp-size",
        ),
        pytest.param(
            {"moe_ep_size": 8},
            "moe_ep_size=8 conflicts with selected deployment moe_ep_size=1",
            id="moe-ep-size",
        ),
    ],
)
def test_entry_with_from_is_refused_for_a_key_it_writes(tmp_path, written, conflict):
    entry = {"from": str(tmp_path / "missing.yaml"), **written}
    planner = _written_planner("decode", entry)

    with pytest.raises(ValueError) as caught:
        _checked_value(planner, decode_pick=PickedParallelConfig(tp=2))

    assert type(caught.value) is ValueError
    assert str(caught.value) == f"ais_perf_model entry for decode: {conflict}"


@pytest.mark.parametrize(
    "written",
    [
        pytest.param({}, id="from-alone"),
        pytest.param(
            {
                "model": "Qwen/Qwen3-32B",
                "system": "h200_sxm",
                "backend": "vllm",
                "worker_type": "decode",
                "tp": 2,
                "pp": 1,
                "attention_dp": 1,
                "moe_tp_size": 1,
                "moe_ep_size": 1,
            },
            id="same-values",
        ),
        pytest.param(
            {"tp": 2, "moe_tp_size": None, "moe_ep_size": None}, id="null-moe-sizes"
        ),
    ],
)
def test_entry_with_from_that_agrees_is_returned_as_written(tmp_path, written):
    # The file an entry names is read by the Planner; here it does not exist.
    entry = {"from": str(tmp_path / "missing.yaml"), **written}
    planner = _written_planner("decode", dict(entry))

    value = _checked_value(planner, decode_pick=PickedParallelConfig(tp=2))

    assert value == entry
    assert value is not planner.ais_perf_model


@pytest.mark.parametrize(
    "prefill_keys,decode_tp,conflict",
    [
        pytest.param(
            {},
            8,
            "ais_perf_model entry for decode: tp=8 conflicts with selected "
            "deployment tp=2",
            id="entry-without-from",
        ),
        pytest.param(
            {"tp": 8},
            2,
            "ais_perf_model entry for prefill: tp=8 conflicts with selected "
            "deployment tp=2",
            id="entry-with-from",
        ),
    ],
)
def test_list_reports_the_entry_that_conflicts(
    tmp_path, prefill_keys, decode_tp, conflict
):
    planner = _written_planner(
        "disagg",
        [
            {
                "from": str(tmp_path / "missing.yaml"),
                "worker_type": "prefill",
                **prefill_keys,
            },
            {
                "model": "Qwen/Qwen3-32B",
                "system": "h200_sxm",
                "backend": "vllm",
                "worker_type": "decode",
                "tp": decode_tp,
            },
        ],
    )
    pick = PickedParallelConfig(tp=2)

    with pytest.raises(ValueError) as caught:
        _checked_value(planner, prefill_pick=pick, decode_pick=pick)

    assert type(caught.value) is ValueError
    assert str(caught.value) == conflict


WORKERS = ("prefill", "decode")
OVERRIDE_SHAPES = ["services", "components"]

# Each generated worker's arguments from its first AIS argument on, in both forms.
NEW_AIS_ARGUMENTS = [
    "--ais-perf-model",
    '{"attention_dp":1,"backend":"vllm","backend_version":"0.24.0",'
    '"model":"Qwen/Qwen3-32B","system":"h200_sxm","tp":2}',
    "--engine-type",
    "vllm",
]
OLD_AIS_ARGUMENTS = [
    "--ais-perf-model",
    "--ais-backend",
    "vllm",
    "--ais-backend-version",
    "0.24.0",
    "--ais-system",
    "h200_sxm",
    "--ais-tp-size",
    "2",
    "--ais-attention-dp-size",
    "1",
    "--engine-type",
    "vllm",
]

# Arguments an override adds to a worker.
MOCKER_COMMAND = ["python3", "-m", "dynamo.mocker"]
ENGINE_FILE = ["--extra-engine-args", "engine.json"]
MODEL_PATH = ["--model-path", "Qwen/Qwen3-235B-A22B"]
OLD_FLAG = ["--ais-system", "h100_sxm"]
VALUE = [
    "--ais-perf-model",
    json.dumps(
        {"model": "Qwen/Qwen3-32B", "system": "h100_sxm", "backend": "vllm", "tp": 4}
    ),
]

# What the mocker makes of a worker's command line: of the generated one, of
# one where the override's value comes last, and of one with an old flag.
GENERATED_READ = {
    "model_path": "Qwen/Qwen3-32B",
    "max_num_seqs": 256,
    "perf_model": ("Qwen/Qwen3-32B", "h200_sxm", 2),
}
REPLACED_READ = {**GENERATED_READ, "perf_model": ("Qwen/Qwen3-32B", "h100_sxm", 4)}
OLD_FLAG_ERROR = {"error": "--ais-perf-model cannot be combined with --ais-system"}


@pytest.fixture
def mocker_inputs(monkeypatch, tmp_path):
    """Pin what the generated arguments and the mocker parser read from outside."""
    monkeypatch.setattr(
        "dynamo.profiler.utils.dgd_generation.get_latest_database_version",
        lambda **kwargs: "0.24.0",
    )
    monkeypatch.delenv(ENV_VAR, raising=False)
    # ENGINE_FILE names this file.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "engine.json").write_text(json.dumps({"engine": {"max_num_seqs": 9}}))


def _worker_override(shape: str, containers: dict[str, dict]) -> dict:
    """Return a deployment override that adds container fields to the named workers.

    The override engine appends the ``args`` of a ``spec.services`` override
    to the generated ones. For ``spec.components`` it does so only with
    ``$patch: {args: append}``, which every such container gets here; without
    it the override's list would replace the generated one.
    """
    if shape == "services":
        return {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoGraphDeployment",
            "spec": {
                "services": {
                    name: {"extraPodSpec": {"mainContainer": fields}}
                    for name, fields in containers.items()
                }
            },
        }
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "spec": {
            "components": [
                {
                    "name": name,
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    **(
                                        {"$patch": {"args": "append"}}
                                        if "args" in fields
                                        else {}
                                    ),
                                    **fields,
                                }
                            ]
                        }
                    },
                }
                for name, fields in containers.items()
            ]
        },
    }


def _mocker_workers(override: dict | None = None) -> dict[str, list[str]]:
    """Return the argument list of each generated mocker worker."""
    pick = PickedParallelConfig(tp=2)
    interpolation = AICInterpolationSpec(
        hf_id="Qwen/Qwen3-32B",
        system="h200_sxm",
        backend="vllm",
        isl=1000,
        osl=100,
        sweep_max_context_length=4096,
        prefill_interpolation_granularity=8,
        decode_interpolation_granularity=4,
        prefill_pick=pick,
        decode_pick=pick,
    )
    config = generate_mocker_config(
        _dgdr(mocker_enabled=True, override=override), aic_spec=interpolation
    )
    return {
        name: _main_container(component)["args"]
        for name, component in _component_map(config).items()
        if name in WORKERS
    }


def _ais_arguments(workers: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return each worker's arguments from its first AIS argument on."""
    return {
        name: arguments[
            next(
                index
                for index, token in enumerate(arguments)
                if token.startswith(("--ais-", "--aic-"))
            ) :
        ]
        for name, arguments in workers.items()
    }


def _mocker_reads(arguments: list[str], capsys) -> dict:
    """Return what the mocker makes of one worker's command line, or its error."""
    try:
        args = parse_args(arguments)
    except SystemExit:
        return {"error": capsys.readouterr().err.rsplit(": error: ", 1)[1].strip()}
    config = load_mocker_engine_args(args)
    perf_model = performance_config(config)
    return {
        "model_path": args.model_path,
        "max_num_seqs": config["engine"]["max_num_seqs"],
        "perf_model": (perf_model["model"], perf_model["system"], perf_model["tp"]),
    }


def _old_arguments_warning(names: str) -> str:
    return (
        f"overrides.dgd sets {names}: the generated mocker workers keep the old AIS "
        "arguments (--ais-perf-model without a value plus --ais-* flags), which are "
        "deprecated and will be removed in v1.9; to end this, write --ais-perf-model "
        "VALUE in the override and remove its old --ais-* arguments"
    )


@pytest.mark.parametrize("shape", OVERRIDE_SHAPES)
@pytest.mark.parametrize(
    "added,reads",
    [
        pytest.param(
            {"decode": ENGINE_FILE + VALUE},
            {"prefill": GENERATED_READ, "decode": {**REPLACED_READ, "max_num_seqs": 9}},
            id="engine-args-same-worker",
        ),
        pytest.param(
            {"prefill": ENGINE_FILE, "decode": VALUE},
            {"prefill": {**GENERATED_READ, "max_num_seqs": 9}, "decode": REPLACED_READ},
            id="engine-args-other-worker",
        ),
        pytest.param(
            {"decode": MODEL_PATH + VALUE},
            {
                "prefill": GENERATED_READ,
                "decode": {**REPLACED_READ, "model_path": "Qwen/Qwen3-235B-A22B"},
            },
            id="model-path-same-worker",
        ),
        pytest.param(
            {"prefill": MODEL_PATH, "decode": VALUE},
            {
                "prefill": {**GENERATED_READ, "model_path": "Qwen/Qwen3-235B-A22B"},
                "decode": REPLACED_READ,
            },
            id="model-path-other-worker",
        ),
        pytest.param(
            {"decode": OLD_FLAG + VALUE},
            {"prefill": GENERATED_READ, "decode": OLD_FLAG_ERROR},
            id="old-flag-same-worker",
        ),
        pytest.param(
            {"prefill": OLD_FLAG, "decode": VALUE},
            {"prefill": OLD_FLAG_ERROR, "decode": REPLACED_READ},
            id="old-flag-other-worker",
        ),
    ],
)
def test_override_with_a_value_keeps_new_arguments_for_every_worker(
    mocker_inputs, caplog, capsys, added, reads, shape
):
    override = _worker_override(
        shape, {name: {"args": arguments} for name, arguments in added.items()}
    )

    with caplog.at_level(logging.WARNING):
        workers = _mocker_workers(override)

    assert caplog.records == []
    assert _ais_arguments(workers) == dict.fromkeys(WORKERS, NEW_AIS_ARGUMENTS)
    assert {
        name: _mocker_reads(workers[name] + added.get(name, []), capsys)
        for name in WORKERS
    } == reads


@pytest.mark.parametrize("shape", OVERRIDE_SHAPES)
@pytest.mark.parametrize(
    "container,names",
    [
        pytest.param({"args": ENGINE_FILE}, "--extra-engine-args", id="engine-args"),
        pytest.param({"args": MODEL_PATH}, "--model-path", id="model-path"),
        pytest.param({"args": OLD_FLAG}, "--ais-system", id="old-flag"),
        pytest.param({"args": ["--ais-next", "2"]}, "--ais-next", id="ambiguous-nextn"),
        pytest.param({"args": ["--aic-m", "5"]}, "--aic-m", id="ambiguous-mtp-seed"),
        pytest.param(
            {"command": MOCKER_COMMAND + OLD_FLAG}, "--ais-system", id="command"
        ),
        pytest.param(
            {"args": MODEL_PATH, "command": MOCKER_COMMAND + OLD_FLAG},
            "--ais-system and --model-path",
            id="command-then-args",
        ),
    ],
)
def test_override_without_a_value_keeps_old_arguments_for_every_worker(
    mocker_inputs, caplog, container, names, shape
):
    override = _worker_override(shape, {"decode": container})

    with caplog.at_level(logging.WARNING):
        workers = _mocker_workers(override)

    assert _ais_arguments(workers) == dict.fromkeys(WORKERS, OLD_AIS_ARGUMENTS)
    assert [record.getMessage() for record in caplog.records] == [
        _old_arguments_warning(names)
    ]


@pytest.mark.parametrize("shape", OVERRIDE_SHAPES)
@pytest.mark.parametrize(
    "containers",
    [
        pytest.param({"decode": {"command": MOCKER_COMMAND}}, id="no-ais-argument"),
        pytest.param(
            {"decode": {"command": " ".join(MOCKER_COMMAND + OLD_FLAG)}},
            id="command-as-text",
        ),
        pytest.param(
            {"decode": {"command": ["sh", "-c", " ".join(MOCKER_COMMAND + OLD_FLAG)]}},
            id="shell-script",
        ),
    ],
)
def test_override_command_without_trigger_keeps_new_arguments(
    mocker_inputs, caplog, containers, shape
):
    with caplog.at_level(logging.WARNING):
        workers = _mocker_workers(_worker_override(shape, containers))

    assert _ais_arguments(workers) == dict.fromkeys(WORKERS, NEW_AIS_ARGUMENTS)
    assert caplog.records == []


@pytest.mark.parametrize("shape", OVERRIDE_SHAPES)
@pytest.mark.parametrize(
    "containers",
    [
        pytest.param({"decode": {"command": MOCKER_COMMAND + VALUE}}, id="value"),
        pytest.param(
            {"decode": {"command": [*MOCKER_COMMAND, "=".join(VALUE)]}},
            id="equals-sign",
        ),
        pytest.param(
            {"decode": {"command": [*MOCKER_COMMAND, "--ais-perf-m", VALUE[1]]}},
            id="abbreviated",
        ),
        pytest.param(
            {"decode": {"command": MOCKER_COMMAND + VALUE, "args": OLD_FLAG}},
            id="value-then-old-flag",
        ),
        pytest.param(
            {
                "prefill": {"command": MOCKER_COMMAND + OLD_FLAG},
                "decode": {"command": MOCKER_COMMAND + VALUE},
            },
            id="old-flag-then-value",
        ),
        pytest.param(
            {"prefill": {"args": VALUE}, "decode": {"command": MOCKER_COMMAND + VALUE}},
            id="args-value-then-command-value",
        ),
        pytest.param(
            {"decode": {"command": MOCKER_COMMAND + VALUE, "args": VALUE}},
            id="value-in-both-lists",
        ),
    ],
)
def test_override_value_in_command_is_refused(mocker_inputs, caplog, containers, shape):
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError) as error:
        _mocker_workers(_worker_override(shape, containers))

    assert str(error.value) == (
        "overrides.dgd gives --ais-perf-model a value in a container's command "
        "list; a generated mocker worker would use its generated value instead, "
        "because the generated args come after the command. Write --ais-perf-model "
        "VALUE in the container's args list"
    )
    assert caplog.records == []


def test_value_before_generated_arguments_is_not_used(mocker_inputs, capsys):
    """A worker reads a value typed in ``command`` first and the generated one last."""
    workers = _mocker_workers()

    assert _mocker_reads(VALUE + workers["decode"], capsys) == GENERATED_READ


@pytest.mark.parametrize("shape", OVERRIDE_SHAPES)
def test_override_value_in_command_is_accepted_without_generated_ais_arguments(
    mocker_inputs, shape
):
    override = _worker_override(shape, {"decode": {"command": MOCKER_COMMAND + VALUE}})

    config = generate_mocker_config(_dgdr(mocker_enabled=True, override=override))

    assert [
        token
        for name, component in _component_map(config).items()
        if name in WORKERS
        for token in _main_container(component)["args"]
        if token.startswith("--ais-")
    ] == []


@pytest.mark.parametrize("joined", [False, True], ids=["two-tokens", "equals-sign"])
@pytest.mark.parametrize(
    "option,value,attribute,parsed",
    [
        pytest.param(f"--{family}-{name}", *row, id=f"{family}-{name}")
        for family in ("ais", "aic")
        for name, *row in (
            ("mt", "5", "ais_mtp_seed", 5),
            ("mtp-se", "5", "ais_mtp_seed", 5),
            ("nextn-", "0.5", "ais_nextn_accept_rates", "0.5"),
            ("nextn-accept-r", "0.5", "ais_nextn_accept_rates", "0.5"),
        )
    ],
)
def test_abbreviated_option_outside_perf_model_keeps_new_arguments(
    mocker_inputs, caplog, option, value, attribute, parsed, joined
):
    added = [f"{option}={value}"] if joined else [option, value]

    with caplog.at_level(logging.WARNING):
        workers = _mocker_workers(
            _worker_override("services", {"decode": {"args": added}})
        )

    assert _ais_arguments(workers) == dict.fromkeys(WORKERS, NEW_AIS_ARGUMENTS)
    assert caplog.records == []
    assert getattr(parse_args(workers["decode"] + added), attribute) == parsed
