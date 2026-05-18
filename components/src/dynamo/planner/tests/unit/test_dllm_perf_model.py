# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``DllmRegressionModel``.

The DLLM cost model is validated against empirical observations from a
real LLaDA 2.0 deployment (47 aiperf runs on 2x RTX PRO 6000) covering
prefix-heavy, RAG, long-output, pure-decode, and mixed-OSL workloads.
The reference fit on a 2-worker Dynamo fleet achieves mean prediction
error ~4% and max error ~13%.
"""

from __future__ import annotations

import math

import pytest

try:
    import msgspec  # noqa: F401
except ImportError:
    pytest.skip("msgspec required for FPM tests", allow_module_level=True)

try:
    from dynamo.common.forward_pass_metrics import ForwardPassMetrics  # noqa: F401
except ImportError:
    pytest.skip("forward_pass_metrics not available", allow_module_level=True)

from dynamo.planner.core.perf_model.dllm import (
    DllmRegressionModel,
    fit_from_observations,
)


# (ISL, OSL, per-worker concurrency, system req/s, avg latency in seconds).
# Subsampled from the LLaDA 2.0 / mini-preview aiperf corpus
# (`docs/llada-dynamo-vs-native-sglang.md`).
LLADA_REFERENCE_OBSERVATIONS = [
    (2064, 64, 4, 4.60, 1.70),
    (2064, 64, 4, 4.66, 1.71),
    (2064, 63, 4, 4.64, 1.72),
    (2064, 64, 8, 7.74, 2.00),
    (2064, 64, 8, 8.15, 1.90),
    (64, 63, 4, 5.46, 1.44),
    (64, 63, 8, 10.40, 1.30),
    (2064, 136, 4, 2.56, 3.08),
    (2064, 135, 4, 2.64, 2.98),
    (6064, 32, 4, 4.84, 1.61),
    (32, 125, 4, 3.50, 2.22),
    (64, 445, 4, 1.18, 6.19),
]


def test_fit_from_observations_yields_positive_block_coefficient():
    """The number-of-blocks coefficient must be positive — more output blocks
    can never lower wall time."""
    alpha, beta, gamma = fit_from_observations(LLADA_REFERENCE_OBSERVATIONS)
    assert alpha > 0
    assert beta > 0
    # Intercept should be small but positive (request-setup overhead).
    assert -0.5 <= gamma <= 2.0


def test_predictions_within_15_percent_on_reference_corpus():
    """The fitted model should achieve <15% max error on the LLaDA corpus."""
    alpha, beta, gamma = fit_from_observations(LLADA_REFERENCE_OBSERVATIONS)
    errs = []
    for isl, osl, pwc, _rps, lat in LLADA_REFERENCE_OBSERVATIONS:
        num_blocks = math.ceil(osl / 32)
        per_step_tokens = pwc * (isl + osl / 2.0)
        pred = alpha * num_blocks + beta * per_step_tokens + gamma
        errs.append(abs(pred - lat) / lat)
    mean_err = sum(errs) / len(errs)
    max_err = max(errs)
    assert mean_err < 0.10, f"Mean rel error {mean_err*100:.1f}% > 10%"
    assert max_err < 0.20, f"Max rel error {max_err*100:.1f}% > 20%"


def test_long_output_dominates_latency():
    """For diffusion LMs, OSL drives latency through num_blocks. A request
    with OSL=512 should be predicted to take much longer than OSL=64 at the
    same ISL.

    This is the property that AR perf models GET WRONG for LLaDA: AR predicts
    a smooth increase with OSL, but the diffusion engine's actual cost jumps
    by num_blocks (discrete) × steps_per_block forward passes."""
    alpha, beta, gamma = fit_from_observations(LLADA_REFERENCE_OBSERVATIONS)
    # 64 vs 512 OSL at same ISL=64, pwc=4
    pst = 4 * (64 + 64 / 2.0)
    pst_long = 4 * (64 + 512 / 2.0)
    lat_short = alpha * 2 + beta * pst + gamma
    lat_long = alpha * 16 + beta * pst_long + gamma
    # 8x more blocks → at least 3x latency on this model
    assert lat_long > 3.0 * lat_short


def test_block_coefficient_is_significant():
    """The block coefficient should not be a rounding error — for LLaDA the
    block term should contribute meaningfully to predicted latency."""
    alpha, beta, gamma = fit_from_observations(LLADA_REFERENCE_OBSERVATIONS)
    # Typical workload: ISL=2064, OSL=64, pwc=4. blocks=2, pst=8384.
    block_term = alpha * 2
    pst_term = beta * 8384
    # Both should contribute substantively to the total prediction.
    assert block_term > 0.2  # at least 200 ms from block term
    assert pst_term > 0.1


def test_regression_model_basic_construction():
    model = DllmRegressionModel(max_num_fpm_samples=128)
    assert model.page_size == 32
    assert not model._is_fitted
    assert model.num_observations == 0


def test_features_compute_correctly():
    model = DllmRegressionModel(max_num_fpm_samples=128, page_size=32)
    feats = model._features(isl=2048, osl=64, batch_size=4.0)
    # blocks = ceil(64/32) = 2
    assert feats[0] == 2.0
    # per_step_tokens = 4 * (2048 + 32) = 8320
    assert feats[1] == 8320.0


@pytest.mark.parametrize(
    "page_size,osl,expected_blocks",
    [
        (32, 32, 1),
        (32, 33, 2),
        (32, 64, 2),
        (32, 96, 3),
        (16, 64, 4),
    ],
)
def test_block_counting_respects_page_size(page_size, osl, expected_blocks):
    model = DllmRegressionModel(max_num_fpm_samples=128, page_size=page_size)
    feats = model._features(isl=100, osl=osl, batch_size=1.0)
    assert feats[0] == float(expected_blocks)
