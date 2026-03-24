# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test module to trigger model predownloading before any other tests run.

This module uses pytest-order to ensure the predownload_models fixture
is executed at the very beginning of the test session, downloading all
required models before other tests attempt to use them.

The tests use parametrization with pytest.param marks to match different
CI marker expressions (e.g., "vllm and e2e and gpu_1").
"""

import pytest


@pytest.mark.order("first")
@pytest.mark.pre_merge
@pytest.mark.parametrize(
    "predownload_models_variant",
    [
        pytest.param(
            "predownload_models_vllm_gpu1",
            marks=[pytest.mark.vllm, pytest.mark.e2e, pytest.mark.gpu_1],
        ),
        pytest.param(
            "predownload_models_sglang_gpu1",
            marks=[pytest.mark.sglang, pytest.mark.e2e, pytest.mark.gpu_1],
        ),
        pytest.param(
            "predownload_models_trtllm_gpu1",
            marks=[pytest.mark.trtllm, pytest.mark.e2e, pytest.mark.gpu_1],
        ),
        pytest.param(
            "predownload_models_vllm_gpu2",
            marks=[pytest.mark.vllm, pytest.mark.e2e, pytest.mark.gpu_2],
        ),
        pytest.param(
            "predownload_models_sglang_gpu2",
            marks=[pytest.mark.sglang, pytest.mark.e2e, pytest.mark.gpu_2],
        ),
        pytest.param(
            "predownload_models_trtllm_gpu2",
            marks=[pytest.mark.trtllm, pytest.mark.e2e, pytest.mark.gpu_2],
        ),
    ],
)
def test_predownload_models(predownload_models_variant, predownload_models):
    """Trigger the predownload_models fixture to download all required models.

    This test runs before any other tests in the session due to pytest.mark.order("first").
    By downloading models upfront, we ensure that model download time is not counted
    against individual test timeouts. This prevents tests from failing due to slow
    network downloads rather than actual test failures.

    The actual download logic is handled by the predownload_models session-scoped fixture
    defined in conftest.py.

    The parametrization ensures this test is selected by any CI marker expression
    that matches framework-specific E2E tests (e.g., "vllm and e2e and gpu_1").
    """
    # The fixture handles the download - this test just ensures it runs first
    pass
