# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rapid.py private helper functions.

Tests _run_naive_fallback and _run_default_sim in isolation; AIC simulation
helpers (_run_autoscale_sim) require the full AIC stack and are covered by
the end-to-end test suite.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from dynamo.profiler.rapid import _run_default_sim, _run_naive_fallback
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        FeaturesSpec,
        HardwareSpec,
        MockerSpec,
        ModelCacheSpec,
        SLASpec,
        WorkloadSpec,
    )
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_dgdr(**overrides) -> DynamoGraphDeploymentRequestSpec:
    base = dict(
        model="Qwen/Qwen3-32B",
        backend="vllm",
        image="nvcr.io/nvidia/ai-dynamo/dynamo-frontend:latest",
        hardware=HardwareSpec(gpuSku="l40s", totalGpus=4, numGpusPerNode=4),
        workload=WorkloadSpec(isl=4000, osl=1000),
        sla=SLASpec(ttft=2000.0, itl=50.0),
    )
    base.update(overrides)
    return DynamoGraphDeploymentRequestSpec(**base)


def _fake_modifier(update_image_return=None):
    m = MagicMock()
    m.update_image.return_value = update_image_return or {"kind": "DGD"}
    m.update_model_from_pvc.return_value = {"kind": "DGD"}
    return m


# ---------------------------------------------------------------------------
# _run_naive_fallback
# ---------------------------------------------------------------------------


class TestRunNaiveFallback:
    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_returns_expected_structure(self):
        """Result always has the four required keys with zeroed latencies."""
        dgdr = _make_dgdr()
        with patch(
            "dynamo.profiler.rapid.generate_naive_config",
            return_value={"artifacts": {}},
        ):
            result = _run_naive_fallback(dgdr, "Qwen/Qwen3-32B", 4, "l40s", "vllm")

        assert set(result) >= {
            "best_config_df",
            "best_latencies",
            "dgd_config",
            "chosen_exp",
        }
        assert result["best_latencies"] == {
            "ttft": 0.0,
            "tpot": 0.0,
            "request_latency": 0.0,
        }
        assert result["chosen_exp"] == "agg"
        assert isinstance(result["best_config_df"], pd.DataFrame)
        assert result["best_config_df"].empty

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_empty_artifacts_yields_none_dgd_config(self):
        """No k8s_deploy.yaml in artifacts → dgd_config is None."""
        dgdr = _make_dgdr()
        with patch(
            "dynamo.profiler.rapid.generate_naive_config",
            return_value={"artifacts": {}},
        ):
            result = _run_naive_fallback(dgdr, "Qwen/Qwen3-32B", 4, "l40s", "vllm")
        assert result["dgd_config"] is None

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_with_pvc_calls_update_model_from_pvc(self):
        """When modelCache.pvcName is set, update_model_from_pvc is called."""
        dgdr = _make_dgdr(
            modelCache=ModelCacheSpec(
                pvcName="model-cache",
                pvcModelPath="/model/qwen",
                pvcMountPath="/opt/model-cache",
            )
        )
        fake_modifier = _fake_modifier()
        with (
            patch(
                "dynamo.profiler.rapid.generate_naive_config",
                return_value={"artifacts": {"k8s_deploy.yaml": "kind: DGD"}},
            ),
            patch("dynamo.profiler.rapid.CONFIG_MODIFIERS", {"vllm": fake_modifier}),
        ):
            _run_naive_fallback(dgdr, "Qwen/Qwen3-32B", 4, "l40s", "vllm")

        fake_modifier.update_model_from_pvc.assert_called_once()

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_without_pvc_skips_update_model_from_pvc(self):
        """When no modelCache, update_model_from_pvc is not called."""
        dgdr = _make_dgdr()
        fake_modifier = _fake_modifier()
        with (
            patch(
                "dynamo.profiler.rapid.generate_naive_config",
                return_value={"artifacts": {"k8s_deploy.yaml": "kind: DGD"}},
            ),
            patch("dynamo.profiler.rapid.CONFIG_MODIFIERS", {"vllm": fake_modifier}),
        ):
            _run_naive_fallback(dgdr, "Qwen/Qwen3-32B", 4, "l40s", "vllm")

        fake_modifier.update_model_from_pvc.assert_not_called()


# ---------------------------------------------------------------------------
# _run_default_sim
# ---------------------------------------------------------------------------


class TestRunDefaultSim:
    def _execute_return(self, chosen="disagg", ttft=100.0, tpot=10.0):
        """Build a fake _execute_task_configs return value."""
        best_df = pd.DataFrame([{"tp(p)": 1}])
        latencies = {"ttft": ttft, "tpot": tpot, "request_latency": 0.0}
        return chosen, {chosen: best_df}, None, None, {chosen: latencies}

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_returns_required_keys(self):
        dgdr = _make_dgdr()
        with (
            patch("dynamo.profiler.rapid.build_default_task_configs", return_value={}),
            patch(
                "dynamo.profiler.rapid._execute_task_configs",
                return_value=self._execute_return(),
            ),
            patch(
                "dynamo.profiler.rapid._generate_dgd_from_pick",
                return_value={"kind": "DGD"},
            ),
        ):
            result = _run_default_sim(
                dgdr,
                "Qwen/Qwen3-32B",
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

        assert set(result) >= {
            "best_config_df",
            "best_latencies",
            "dgd_config",
            "chosen_exp",
            "task_configs",
        }
        assert result["chosen_exp"] == "disagg"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_load_match_passes_load_kwargs(self):
        """load_match picking mode forwards rate/concurrency/max_gpus to execute."""
        dgdr = _make_dgdr(workload=WorkloadSpec(isl=4000, osl=1000, requestRate=5.0))
        captured: dict = {}

        def fake_execute(task_configs, mode, top_n, **kwargs):
            captured.update(kwargs)
            return self._execute_return()

        with (
            patch("dynamo.profiler.rapid.build_default_task_configs", return_value={}),
            patch(
                "dynamo.profiler.rapid._execute_task_configs", side_effect=fake_execute
            ),
            patch("dynamo.profiler.rapid._generate_dgd_from_pick", return_value=None),
        ):
            _run_default_sim(
                dgdr,
                "Qwen/Qwen3-32B",
                "h200_sxm",
                "trtllm",
                8,
                4000,
                1000,
                2000.0,
                50.0,
                None,
                "load_match",
            )

        assert "target_request_rate" in captured
        assert captured["target_request_rate"] == 5.0
        assert captured["max_total_gpus"] == 8

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_default_mode_passes_no_load_kwargs(self):
        """default picking mode does not forward load-match kwargs."""
        dgdr = _make_dgdr()
        captured: dict = {}

        def fake_execute(task_configs, mode, top_n, **kwargs):
            captured.update(kwargs)
            return self._execute_return()

        with (
            patch("dynamo.profiler.rapid.build_default_task_configs", return_value={}),
            patch(
                "dynamo.profiler.rapid._execute_task_configs", side_effect=fake_execute
            ),
            patch("dynamo.profiler.rapid._generate_dgd_from_pick", return_value=None),
        ):
            _run_default_sim(
                dgdr,
                "Qwen/Qwen3-32B",
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

        assert "target_request_rate" not in captured
        assert "max_total_gpus" not in captured

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_latencies_extracted_from_chosen_exp(self):
        """best_latencies come from the chosen experiment's entry."""
        dgdr = _make_dgdr()
        with (
            patch("dynamo.profiler.rapid.build_default_task_configs", return_value={}),
            patch(
                "dynamo.profiler.rapid._execute_task_configs",
                return_value=self._execute_return(ttft=123.0, tpot=7.0),
            ),
            patch("dynamo.profiler.rapid._generate_dgd_from_pick", return_value=None),
        ):
            result = _run_default_sim(
                dgdr,
                "Qwen/Qwen3-32B",
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

        assert result["best_latencies"]["ttft"] == 123.0
        assert result["best_latencies"]["tpot"] == 7.0


# ---------------------------------------------------------------------------
# Force-disagg when interpolation data is needed
# ---------------------------------------------------------------------------


class TestRunDefaultSimForceDisagg:
    """When AIC picks an aggregated config but the DGDR requires interpolation
    data (mocker or throughput-scaling), _run_default_sim must override the
    selection to the best available disaggregated config."""

    def _call_default_sim(self, dgdr, execute_return_value):
        with (
            patch("dynamo.profiler.rapid.build_default_task_configs", return_value={}),
            patch(
                "dynamo.profiler.rapid._execute_task_configs",
                return_value=execute_return_value,
            ),
            patch("dynamo.profiler.rapid._generate_dgd_from_pick", return_value=None),
        ):
            return _run_default_sim(
                dgdr,
                "Qwen/Qwen3-32B",
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

    def _both_configs(self, chosen="agg"):
        """Return value where both agg and disagg configs are available."""
        agg_df = pd.DataFrame([{"tp(p)": 1}])
        disagg_df = pd.DataFrame([{"tp(p)": 1}])
        latencies = {"ttft": 100.0, "tpot": 10.0, "request_latency": 0.0}
        return (
            chosen,
            {"agg": agg_df, "disagg": disagg_df},
            None,
            None,
            {"agg": latencies, "disagg": latencies},
        )

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_mocker_enabled_agg_picked_overrides_to_disagg(self):
        """When mocker is enabled and AIC picks agg, chosen is overridden to disagg."""
        dgdr = _make_dgdr(features=FeaturesSpec(mocker=MockerSpec(enabled=True)))
        result = self._call_default_sim(dgdr, self._both_configs(chosen="agg"))
        assert result["chosen_exp"] == "disagg"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_no_profile_data_needed_agg_pick_preserved(self):
        """When no interpolation data is needed, an agg pick is kept as-is."""
        dgdr = _make_dgdr()  # no mocker, no throughput scaling
        result = self._call_default_sim(dgdr, self._both_configs(chosen="agg"))
        assert result["chosen_exp"] == "agg"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_mocker_enabled_disagg_picked_unchanged(self):
        """When mocker is enabled but AIC already picks disagg, no override happens."""
        dgdr = _make_dgdr(features=FeaturesSpec(mocker=MockerSpec(enabled=True)))
        result = self._call_default_sim(dgdr, self._both_configs(chosen="disagg"))
        assert result["chosen_exp"] == "disagg"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_mocker_enabled_agg_only_available_keeps_agg(self):
        """When mocker is enabled, agg is picked, and no disagg config exists, keep agg."""
        dgdr = _make_dgdr(features=FeaturesSpec(mocker=MockerSpec(enabled=True)))
        agg_df = pd.DataFrame([{"tp(p)": 1}])
        latencies = {"ttft": 100.0, "tpot": 10.0, "request_latency": 0.0}
        agg_only = ("agg", {"agg": agg_df}, None, None, {"agg": latencies})
        result = self._call_default_sim(dgdr, agg_only)
        assert result["chosen_exp"] == "agg"
