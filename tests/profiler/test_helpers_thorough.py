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
    from dynamo.profiler.thorough import (
        _estimate_dense_size_mib,
        _pick_thorough_best_config,
        _prune_infeasible_candidates,
    )
    from dynamo.profiler.utils.aic_dataframe import build_decode_row, build_prefill_row
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        HardwareSpec,
        SLASpec,
        WorkloadSpec,
    )
    from dynamo.profiler.utils.model_info import ModelInfo
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


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


# ---------------------------------------------------------------------------
# Helpers for VRAM-pruning tests
# ---------------------------------------------------------------------------


def _make_model_info(
    model_size_mib: float,
    is_moe: bool = False,
    num_experts: int = 0,
    hidden_size: int = 0,
    num_hidden_layers: int = 0,
    vocab_size: int = 0,
    intermediate_size: int = 0,
) -> ModelInfo:
    return ModelInfo(
        model_size=model_size_mib,
        architecture="DeepseekV3ForCausalLM" if is_moe else "LlamaForCausalLM",
        is_moe=is_moe,
        num_experts=num_experts or None,
        hidden_size=hidden_size or None,
        num_hidden_layers=num_hidden_layers or None,
        vocab_size=vocab_size or None,
        intermediate_size=intermediate_size or None,
    )


class _FakeCandidate:
    """Minimal stand-in for an AIC profiling candidate."""

    def __init__(self, tp: int, moe_ep: int = 1, pp: int = 1, moe_tp: int = 1):
        self.tp = tp
        self.moe_ep = moe_ep
        self.pp = pp
        self.moe_tp = moe_tp


# ---------------------------------------------------------------------------
# _estimate_dense_size_mib
# ---------------------------------------------------------------------------


class TestEstimateDenseSizeMib:
    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_non_moe_returns_full_model_size(self):
        """Non-MoE: dense size == total model size."""
        info = _make_model_info(model_size_mib=100_000.0, is_moe=False)
        assert _estimate_dense_size_mib(info) == 100_000.0

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_moe_missing_hidden_size_falls_back_to_full(self):
        """MoE without hidden_size: conservatively returns full model size."""
        info = _make_model_info(
            model_size_mib=1_310_720.0, is_moe=True, num_experts=256
        )
        # hidden_size and num_hidden_layers are None → fallback
        assert _estimate_dense_size_mib(info) == 1_310_720.0

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_moe_dense_fraction_less_than_total(self):
        """MoE with architecture info: dense estimate is a fraction of total."""
        # DeepSeek-R1 approximate numbers
        info = _make_model_info(
            model_size_mib=1_310_720.0,
            is_moe=True,
            num_experts=256,
            hidden_size=7168,
            num_hidden_layers=61,
            vocab_size=129_280,
            intermediate_size=2048,
        )
        dense = _estimate_dense_size_mib(info)
        assert dense < info.model_size, "Dense estimate must be less than total"
        assert dense > 0, "Dense estimate must be positive"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_moe_no_intermediate_size_still_works(self):
        """MoE without intermediate_size: expert_params=0, dense=full model."""
        info = _make_model_info(
            model_size_mib=1_310_720.0,
            is_moe=True,
            num_experts=256,
            hidden_size=7168,
            num_hidden_layers=61,
            vocab_size=129_280,
            intermediate_size=0,  # missing → expert_params = 0
        )
        # When expert_params == 0, dense_fraction = 1.0, so returns full size.
        assert _estimate_dense_size_mib(info) == info.model_size


# ---------------------------------------------------------------------------
# _prune_infeasible_candidates  (TC-3.4 regression)
# ---------------------------------------------------------------------------


class TestPruneInfeasibleCandidates:
    """Regression tests for TC-3.4: DeepSeek-R1 (671B MoE) on B200 (180 GiB)."""

    # B200 VRAM in MiB
    B200_VRAM_MIB = 180 * 1024  # 184_320

    # Approximate DeepSeek-R1 BF16 model size in MiB (~1280 GiB)
    DSR1_SIZE_MIB = 1_310_720.0

    def _dsr1_info(self) -> ModelInfo:
        return _make_model_info(
            model_size_mib=self.DSR1_SIZE_MIB,
            is_moe=True,
            num_experts=256,
            hidden_size=7168,
            num_hidden_layers=61,
            vocab_size=129_280,
            intermediate_size=2048,
        )

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_tp1_ep8_pruned_on_b200(self):
        """tp=1, ep=8 (OOM case) must be pruned on 180 GiB B200."""
        info = self._dsr1_info()
        candidates = [_FakeCandidate(tp=1, moe_ep=8)]
        result = _prune_infeasible_candidates(
            candidates, info, self.B200_VRAM_MIB, "prefill"
        )
        assert result == [], "tp=1,ep=8 should be pruned (OOM on B200)"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_tp8_ep8_kept_on_b200(self):
        """tp=8, ep=8 (the only viable 8-GPU candidate) must survive pruning."""
        info = self._dsr1_info()
        candidate = _FakeCandidate(tp=8, moe_ep=8)
        result = _prune_infeasible_candidates(
            [candidate], info, self.B200_VRAM_MIB, "prefill"
        )
        assert result == [candidate], "tp=8,ep=8 should survive VRAM check on B200"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_tp1_ep16_kept_on_b200(self):
        """tp=1, ep=16 distributes experts enough to fit on B200."""
        info = self._dsr1_info()
        candidate = _FakeCandidate(tp=1, moe_ep=16)
        result = _prune_infeasible_candidates(
            [candidate], info, self.B200_VRAM_MIB, "prefill"
        )
        assert result == [
            candidate
        ], "tp=1,ep=16 should fit on B200 (experts split 16 ways)"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_mixed_candidates_only_feasible_survive(self):
        """Only memory-feasible candidates survive; infeasible ones are pruned."""
        info = self._dsr1_info()
        tp1_ep8 = _FakeCandidate(tp=1, moe_ep=8)
        tp1_ep16 = _FakeCandidate(tp=1, moe_ep=16)
        tp8_ep8 = _FakeCandidate(tp=8, moe_ep=8)

        result = _prune_infeasible_candidates(
            [tp1_ep8, tp1_ep16, tp8_ep8], info, self.B200_VRAM_MIB, "prefill"
        )
        assert tp1_ep8 not in result, "tp=1,ep=8 must be pruned"
        assert tp1_ep16 in result, "tp=1,ep=16 must survive"
        assert tp8_ep8 in result, "tp=8,ep=8 must survive"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_no_vram_info_skips_pruning(self):
        """When vram_mib=0, no candidates are pruned (check is skipped)."""
        info = self._dsr1_info()
        candidates = [
            _FakeCandidate(tp=1, moe_ep=1),
            _FakeCandidate(tp=1, moe_ep=8),
        ]
        result = _prune_infeasible_candidates(candidates, info, 0.0, "prefill")
        assert result == candidates, "vram_mib=0 must disable pruning"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_non_moe_pruning_uses_full_model_size(self):
        """For a non-MoE model, the full model size is used (no EP benefit)."""
        # 200 GiB model on a 100 GiB GPU
        info = _make_model_info(
            model_size_mib=200 * 1024, is_moe=False, hidden_size=4096
        )
        vram_mib = 100 * 1024  # 100 GiB

        # tp=1: 200 GiB > 90 GiB usable → pruned
        tp1 = _FakeCandidate(tp=1)
        # tp=2: 100 GiB == 90 GiB usable → still pruned (not ≤)
        tp2 = _FakeCandidate(tp=2)
        # tp=4: 50 GiB < 90 GiB → feasible
        tp4 = _FakeCandidate(tp=4)

        result = _prune_infeasible_candidates([tp1, tp2, tp4], info, vram_mib, "decode")
        assert tp1 not in result
        assert tp2 not in result
        assert tp4 in result

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_pp_shards_dense_weights(self):
        """pp>1 reduces per-GPU dense-weight pressure, potentially saving a candidate."""
        info = self._dsr1_info()
        # tp=1, pp=1, ep=8 is pruned (OOM baseline)
        tp1_pp1 = _FakeCandidate(tp=1, moe_ep=8, pp=1)
        # tp=1, pp=2, ep=8: dense split across 2 pipeline stages → half the dense per GPU
        tp1_pp2 = _FakeCandidate(tp=1, moe_ep=8, pp=2)
        result = _prune_infeasible_candidates(
            [tp1_pp1, tp1_pp2], info, self.B200_VRAM_MIB, "prefill"
        )
        assert tp1_pp1 not in result, "tp=1,pp=1,ep=8 must be pruned"
        assert tp1_pp2 in result, "tp=1,pp=2,ep=8 should survive (pp halves dense load)"

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    def test_moe_tp_shards_expert_weights(self):
        """moe_tp>1 increases expert sharding, reducing per-GPU expert pressure."""
        info = self._dsr1_info()
        # tp=1, ep=8, moe_tp=1 → pruned
        baseline = _FakeCandidate(tp=1, moe_ep=8, moe_tp=1)
        # tp=1, ep=8, moe_tp=2 → expert shards = tp*pp*moe_tp*ep = 1*1*2*8 = 16
        moe_tp2 = _FakeCandidate(tp=1, moe_ep=8, moe_tp=2)
        result = _prune_infeasible_candidates(
            [baseline, moe_tp2], info, self.B200_VRAM_MIB, "prefill"
        )
        assert baseline not in result, "tp=1,ep=8,moe_tp=1 must be pruned"
        assert (
            moe_tp2 in result
        ), "tp=1,ep=8,moe_tp=2 should fit (experts split 16 ways)"
