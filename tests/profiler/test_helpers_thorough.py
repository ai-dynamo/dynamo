# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for thorough.py's _pick_thorough_best_config helper.

Benchmarking helpers (_benchmark_prefill_candidates, _benchmark_decode_candidates)
require live K8s deployments and are covered by the mocked end-to-end tests
in test_profile_sla_dgdr.py.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from dynamo.profiler.thorough import _pick_thorough_best_config
    from dynamo.profiler.utils.aic_dataframe import build_decode_row, build_prefill_row
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        HardwareSpec,
        SLASpec,
        WorkloadSpec,
    )
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)

pytestmark = [pytest.mark.profiler]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_dgdr(**overrides) -> DynamoGraphDeploymentRequestSpec:
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


def _stub_dfs():
    """Minimal prefill/decode DataFrames that satisfy pick function inputs.

    Uses build_prefill_row / build_decode_row so the DataFrames contain all
    columns expected by _build_disagg_summary_dict (called via
    build_disagg_df_from_static in load_match / default paths).
    """
    prefill_row = build_prefill_row(
        model="Qwen/Qwen3-32B",
        isl=4000,
        osl=1000,
        ttft=50.0,
        tp=1,
        pp=1,
        dp=1,
        moe_tp=1,
        moe_ep=1,
        backend="trtllm",
        system="h200_sxm",
    )
    decode_row = build_decode_row(
        tpot=10.0,
        thpt_per_gpu=100.0,
        num_request=1,
        num_gpus=1,
        osl=1000,
        tp=1,
        pp=1,
        dp=1,
        moe_tp=1,
        moe_ep=1,
        backend="trtllm",
        system="h200_sxm",
    )
    prefill_df = pd.DataFrame([prefill_row])
    decode_df = pd.DataFrame([decode_row])
    return prefill_df, decode_df


def _mock_result():
    return {
        "best_config_df": pd.DataFrame(),
        "best_latencies": {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0},
    }


# ---------------------------------------------------------------------------
# _pick_thorough_best_config
# ---------------------------------------------------------------------------


class TestPickThoroughBestConfig:
    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_autoscale_calls_pick_autoscale(self):
        """autoscale mode delegates to pick_autoscale with ttft/tpot targets."""
        prefill_df, decode_df = _stub_dfs()
        dgdr = _make_dgdr()
        mock_result = _mock_result()

        with patch(
            "dynamo.profiler.thorough.pick_autoscale", return_value=mock_result
        ) as mock_pick:
            result = _pick_thorough_best_config(
                prefill_df,
                decode_df,
                "autoscale",
                2000.0,
                50.0,
                None,
                8,
                dgdr,
            )

        mock_pick.assert_called_once_with(prefill_df, decode_df, 2000.0, 50.0)
        assert result is mock_result

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_load_match_uses_request_latency_when_set(self):
        """load_match passes target_request_latency when request_latency is provided."""
        prefill_df, decode_df = _stub_dfs()
        dgdr = _make_dgdr(workload=WorkloadSpec(isl=4000, osl=1000, requestRate=5.0))

        with patch(
            "dynamo.profiler.thorough.pick_load_match", return_value=_mock_result()
        ) as mock_pick:
            _pick_thorough_best_config(
                prefill_df,
                decode_df,
                "load_match",
                2000.0,
                50.0,
                35000.0,
                8,
                dgdr,
            )

        kwargs = mock_pick.call_args.kwargs
        assert kwargs["target_request_latency"] == 35000.0
        assert "target_tpot" not in kwargs
        assert kwargs["target_request_rate"] == 5.0
        assert kwargs["max_total_gpus"] == 8

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_load_match_falls_back_to_target_tpot(self):
        """load_match passes target_tpot when no request_latency."""
        prefill_df, decode_df = _stub_dfs()
        dgdr = _make_dgdr()

        with patch(
            "dynamo.profiler.thorough.pick_load_match", return_value=_mock_result()
        ) as mock_pick:
            _pick_thorough_best_config(
                prefill_df,
                decode_df,
                "load_match",
                2000.0,
                50.0,
                None,
                8,
                dgdr,
            )

        kwargs = mock_pick.call_args.kwargs
        assert kwargs["target_tpot"] == 50.0
        assert "target_request_latency" not in kwargs

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_default_uses_request_latency_when_set(self):
        """default mode passes target_request_latency when provided."""
        prefill_df, decode_df = _stub_dfs()
        dgdr = _make_dgdr()

        with patch(
            "dynamo.profiler.thorough.pick_default", return_value=_mock_result()
        ) as mock_pick:
            _pick_thorough_best_config(
                prefill_df,
                decode_df,
                "default",
                2000.0,
                50.0,
                35000.0,
                8,
                dgdr,
            )

        kwargs = mock_pick.call_args.kwargs
        assert kwargs["target_request_latency"] == 35000.0
        assert kwargs["total_gpus"] == 8
        assert kwargs["serving_mode"] == "disagg"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_default_falls_back_to_target_tpot(self):
        """default mode passes target_tpot when no request_latency."""
        prefill_df, decode_df = _stub_dfs()
        dgdr = _make_dgdr()

        with patch(
            "dynamo.profiler.thorough.pick_default", return_value=_mock_result()
        ) as mock_pick:
            _pick_thorough_best_config(
                prefill_df,
                decode_df,
                "default",
                2000.0,
                50.0,
                None,
                8,
                dgdr,
            )

        kwargs = mock_pick.call_args.kwargs
        assert kwargs["target_tpot"] == 50.0
        assert "target_request_latency" not in kwargs

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_load_match_omits_workload_kwargs_when_no_workload(self):
        """When dgdr.workload has no rate/concurrency, those kwargs are absent."""
        prefill_df, decode_df = _stub_dfs()
        dgdr = _make_dgdr()  # no requestRate or concurrency

        with patch(
            "dynamo.profiler.thorough.pick_load_match", return_value=_mock_result()
        ) as mock_pick:
            _pick_thorough_best_config(
                prefill_df,
                decode_df,
                "load_match",
                2000.0,
                50.0,
                None,
                0,
                dgdr,
            )

        kwargs = mock_pick.call_args.kwargs
        assert "target_request_rate" not in kwargs
        assert "max_total_gpus" not in kwargs
