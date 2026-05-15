#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KVBM integration tests for models with sliding window attention (SWA).

Validates core KVBM functionality with SWA models (e.g. Gemma 3) which
alternate between local sliding window and global attention layers. This
exercises different KV cache offload/onboard paths compared to models with
uniform full attention.

Each test runs twice — once for KVBM v1 (``kvbm.vllm_integration.connector``)
and once for KVBM v2 (``kvbm.v2.vllm.connector`` with the NIXL UCX backend),
selected via the indirect ``llm_server_kvbm`` fixture parametrize.

Tests:
1. Offload/Onboard: Request offloads to CPU, cache reset, re-request triggers onboarding
2. Eviction: GPU cache fills, blocks evicted, later retrieved without corruption
3. Determinism: Responses remain identical across offload/onboard/eviction cycles
"""

import pytest

from .common import llm_server_kvbm  # noqa: F401
from .common import (
    AELDORA_STORY,
    DeterminismTester,
    assert_deterministic,
    check_kvbm_metrics,
    print_phase,
    print_test_header,
    reset_cache,
)

# SWA model for these tests (Gemma 3 alternates local/global attention layers).
SWA_MODEL = "google/gemma-3-1b-it"

# Test configuration
MIN_OFFLOAD_BLOCKS = 6  # SWA has fewer full-attention blocks than uniform models
MAX_TOKENS = 15  # Max tokens to generate in test responses

# Test markers
pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.vllm,
    pytest.mark.nightly,
]


# Fixtures
@pytest.fixture(scope="function")
def tester(llm_server_kvbm):  # noqa: F811
    """Create tester bound to the KVBM-enabled server."""
    return DeterminismTester(
        base_url=llm_server_kvbm.base_url,
        model_id=SWA_MODEL,
        server_type=llm_server_kvbm.server_type,
    )


# Tests
@pytest.mark.parametrize(
    "llm_server_kvbm",
    [
        pytest.param({"model": SWA_MODEL}, id="v1"),
        pytest.param({"model": SWA_MODEL, "v2": True}, id="v2"),
    ],
    indirect=True,
)
@pytest.mark.timeout(170)
def test_swa_offload_and_onboard(tester, llm_server_kvbm):  # noqa: F811
    """
    Test offload → cache reset → onboard cycle with SWA model.

    Validates that:
    - Initial request triggers offload to CPU cache
    - Cache reset clears GPU cache
    - Repeated request triggers onboard from CPU to GPU
    - Responses are deterministic across the cycle
    """
    print_test_header("SWA OFFLOAD AND ONBOARD TEST")

    prompt = AELDORA_STORY[:400]

    # Phase 1: Initial request triggers offload
    print_phase(1, "Initial request (expect offload to CPU)")
    print(f"Sending request: {prompt[:80]}...")

    response_1 = tester.make_request(prompt, max_tokens=MAX_TOKENS)
    print(f"Response 1: {response_1}")

    metrics = check_kvbm_metrics("Phase 1", llm_server_kvbm.metrics_port)
    assert (
        metrics["kvbm_offload_blocks_d2h"] > 0
    ), "Phase 1: No blocks offloaded. KVBM may not be triggering offloads."
    assert (
        metrics["kvbm_onboard_blocks_h2d"] == 0
    ), f"Phase 1: Expected 0 onboarded blocks, got {metrics['kvbm_onboard_blocks_h2d']}"
    print(f"✓ Phase 1: {metrics['kvbm_offload_blocks_d2h']} blocks offloaded")

    # Phase 2: Reset GPU cache
    print_phase(2, "Clean up GPU cache")
    reset_cache(llm_server_kvbm.base_url)

    # Phase 3: Repeated request triggers onboard
    print_phase(3, "Re-send same request (expect onboard from CPU)")
    print(f"Sending same request: {prompt[:80]}...")

    response_2 = tester.make_request(prompt, max_tokens=MAX_TOKENS)
    print(f"Response 2: {response_2}")

    metrics = check_kvbm_metrics("Phase 3", llm_server_kvbm.metrics_port)
    assert (
        metrics["kvbm_onboard_blocks_h2d"] > 0
    ), "Phase 3: No blocks onboarded. Expected CPU→GPU transfer after cache reset."
    print(f"✓ Phase 3: {metrics['kvbm_onboard_blocks_h2d']} blocks onboarded from CPU")

    # Verify determinism
    print_test_header("DETERMINISM VERIFICATION")
    assert_deterministic(
        response_1,
        response_2,
        test_name="SWA Offload/Onboard",
        label1="Initial response",
        label2="After cache reset",
    )

    print("\n=== TEST PASSED ===")


@pytest.mark.parametrize(
    "llm_server_kvbm",
    [
        pytest.param(
            {"cpu_blocks": 200, "gpu_blocks": 20, "model": SWA_MODEL}, id="v1"
        ),
        pytest.param(
            {"cpu_blocks": 200, "gpu_blocks": 20, "model": SWA_MODEL, "v2": True},
            id="v2",
        ),
    ],
    indirect=True,
)
@pytest.mark.timeout(170)
def test_swa_gpu_cache_eviction(tester, llm_server_kvbm):  # noqa: F811
    """
    Test GPU cache eviction mechanics with SWA model.

    Validates that:
    - Multiple requests fill GPU cache causing eviction
    - Evicted blocks can be retrieved from CPU cache via onboarding
    - Metrics correctly reflect offload and onboard operations
    """
    print_test_header("SWA GPU CACHE EVICTION TEST")
    print(f"GPU blocks: {llm_server_kvbm.gpu_cache_blocks}")
    print(f"CPU blocks: {llm_server_kvbm.cpu_cache_blocks}")

    prompt_1 = AELDORA_STORY
    prompt_2 = (
        "Read the following entry from the ancient scrolls of Aeloria: " + AELDORA_STORY
    )

    # Phase 1: First request triggers offload
    print_phase(1, "Send first request")
    print(f"Prompt 1: {prompt_1[:80]}...")

    tester.make_request(prompt_1, max_tokens=MAX_TOKENS)

    metrics_p1 = check_kvbm_metrics("Phase 1", llm_server_kvbm.metrics_port)
    assert metrics_p1["kvbm_offload_blocks_d2h"] >= MIN_OFFLOAD_BLOCKS, (
        f"Phase 1: Expected >= {MIN_OFFLOAD_BLOCKS} blocks offloaded, "
        f"got {metrics_p1['kvbm_offload_blocks_d2h']}"
    )
    assert (
        metrics_p1["kvbm_onboard_blocks_h2d"] == 0
    ), f"Phase 1: Expected 0 onboarded, got {metrics_p1['kvbm_onboard_blocks_h2d']}"
    print(f"✓ Phase 1: {metrics_p1['kvbm_offload_blocks_d2h']} blocks offloaded")

    # Phase 2: Second request may evict first from GPU
    print_phase(2, "Send second request (may evict first from GPU)")
    print(f"Prompt 2: {prompt_2[:80]}...")

    tester.make_request(prompt_2, max_tokens=MAX_TOKENS)

    metrics_p2 = check_kvbm_metrics("Phase 2", llm_server_kvbm.metrics_port)
    assert (
        metrics_p2["kvbm_offload_blocks_d2h"] > metrics_p1["kvbm_offload_blocks_d2h"]
    ), (
        f"Phase 2: Expected additional offloads, got {metrics_p2['kvbm_offload_blocks_d2h']} "
        f"(was {metrics_p1['kvbm_offload_blocks_d2h']})"
    )
    additional_offloads = (
        metrics_p2["kvbm_offload_blocks_d2h"] - metrics_p1["kvbm_offload_blocks_d2h"]
    )
    print(f"✓ Phase 2: {additional_offloads} additional blocks offloaded")

    # Phase 3: Re-request first prompt (should onboard from CPU)
    print_phase(3, "Re-request first prompt (verify onboarding)")
    print(f"Re-sending Prompt 1: {prompt_1[:80]}...")

    tester.make_request(prompt_1, max_tokens=MAX_TOKENS)

    metrics_p3 = check_kvbm_metrics("Phase 3", llm_server_kvbm.metrics_port)
    assert (
        metrics_p3["kvbm_onboard_blocks_h2d"] > 0
    ), "Phase 3: No blocks onboarded. Expected CPU→GPU retrieval after eviction."
    print(f"✓ Phase 3: {metrics_p3['kvbm_onboard_blocks_h2d']} blocks onboarded")
    print("✓ Eviction mechanics verified: offload → eviction → onboard")

    print("\n=== TEST PASSED ===")


@pytest.mark.parametrize(
    "llm_server_kvbm",
    [
        pytest.param(
            {"cpu_blocks": 200, "gpu_blocks": 20, "model": SWA_MODEL}, id="v1"
        ),
        pytest.param(
            {"cpu_blocks": 200, "gpu_blocks": 20, "model": SWA_MODEL, "v2": True},
            id="v2",
        ),
    ],
    indirect=True,
)
@pytest.mark.timeout(160)
def test_swa_onboarding_determinism(tester, llm_server_kvbm):  # noqa: F811
    """
    Test onboarding determinism under eviction scenario with SWA model.

    Validates that:
    - Multiple onboarding cycles produce deterministic results
    - Responses are consistent when blocks are onboarded multiple times
    - Tests onboarded vs onboarded (not initial vs onboarded)
    """
    print_test_header("SWA ONBOARDING DETERMINISM TEST")
    print(f"GPU blocks: {llm_server_kvbm.gpu_cache_blocks}")
    print(f"CPU blocks: {llm_server_kvbm.cpu_cache_blocks}")

    prompt_1 = AELDORA_STORY
    prompt_2 = (
        "Read the following entry from the ancient scrolls of Aeloria: " + AELDORA_STORY
    )

    # Phase 1: First request triggers offload
    print_phase(1, "Send first request")
    print(f"Prompt 1: {prompt_1[:80]}...")
    tester.make_request(prompt_1, max_tokens=MAX_TOKENS)
    check_kvbm_metrics("Phase 1", llm_server_kvbm.metrics_port)

    # Phase 2: Second request (may evict first from GPU)
    print_phase(2, "Send second request (may evict first from GPU)")
    print(f"Prompt 2: {prompt_2[:80]}...")
    tester.make_request(prompt_2, max_tokens=MAX_TOKENS)
    check_kvbm_metrics("Phase 2", llm_server_kvbm.metrics_port)

    # Phase 3: Re-request prompt 1 (first onboard cycle)
    print_phase(3, "Re-request Prompt 1 (first onboard cycle)")
    print(f"Re-sending Prompt 1: {prompt_1[:80]}...")
    response_1_first_onboard = tester.make_request(prompt_1, max_tokens=MAX_TOKENS)
    print(f"Response 1 (first onboard): {response_1_first_onboard}")
    check_kvbm_metrics("Phase 3", llm_server_kvbm.metrics_port)

    # Phase 4: Re-request prompt 2 (first onboard cycle)
    print_phase(4, "Re-request Prompt 2 (first onboard cycle)")
    print(f"Re-sending Prompt 2: {prompt_2[:80]}...")
    response_2_first_onboard = tester.make_request(prompt_2, max_tokens=MAX_TOKENS)
    print(f"Response 2 (first onboard): {response_2_first_onboard}")
    check_kvbm_metrics("Phase 4", llm_server_kvbm.metrics_port)

    # Phase 5: Re-request prompt 1 (second onboard cycle)
    print_phase(5, "Re-request Prompt 1 (second onboard cycle)")
    print(f"Re-sending Prompt 1 (third time): {prompt_1[:80]}...")
    response_1_second_onboard = tester.make_request(prompt_1, max_tokens=MAX_TOKENS)
    print(f"Response 1 (second onboard): {response_1_second_onboard}")
    check_kvbm_metrics("Phase 5", llm_server_kvbm.metrics_port)

    # Phase 6: Re-request prompt 2 (second onboard cycle)
    print_phase(6, "Re-request Prompt 2 (second onboard cycle)")
    print(f"Re-sending Prompt 2 (third time): {prompt_2[:80]}...")
    response_2_second_onboard = tester.make_request(prompt_2, max_tokens=MAX_TOKENS)
    print(f"Response 2 (second onboard): {response_2_second_onboard}")
    check_kvbm_metrics("Phase 6", llm_server_kvbm.metrics_port)

    # Verify determinism between onboarded requests
    print_test_header("DETERMINISM VERIFICATION")
    print("\nComparing Prompt 1: First onboard vs Second onboard")
    assert_deterministic(
        response_1_first_onboard,
        response_1_second_onboard,
        test_name="SWA Prompt 1 onboarding determinism",
        label1="First onboard (Phase 3)",
        label2="Second onboard (Phase 5)",
    )

    print("\nComparing Prompt 2: First onboard vs Second onboard")
    assert_deterministic(
        response_2_first_onboard,
        response_2_second_onboard,
        test_name="SWA Prompt 2 onboarding determinism",
        label1="First onboard (Phase 4)",
        label2="Second onboard (Phase 6)",
    )

    print("\n=== TEST PASSED ===")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
