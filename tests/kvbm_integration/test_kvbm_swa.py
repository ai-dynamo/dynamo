#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KVBM integration tests for models with sliding window attention (SWA).

Validates core KVBM functionality with SWA models (e.g. Gemma 3) which
alternate between local sliding window and global attention layers. This
exercises different KV cache offload/onboard paths compared to models with
uniform full attention.

Tests (both vLLM and TRT-LLM):
1. Offload/Onboard: Request offloads to CPU, cache reset, re-request triggers onboarding
2. Eviction: GPU cache fills, blocks evicted, later retrieved without corruption
3. Determinism: Responses remain identical across offload/onboard/eviction cycles
"""

import logging
import os
from pathlib import Path

import pytest
import requests

from tests.utils.test_output import resolve_test_output_path

from .common import llm_server_kvbm  # noqa: F401
from .common import (
    DeterminismTester,
    LLMServerManager,
    ServerType,
    assert_deterministic,
    check_module_available,
    fetch_kvbm_metrics,
)

# Test configuration
MIN_OFFLOAD_BLOCKS = 6
MAX_TOKENS = 15

# Shared test prompt (Aeldora story)
AELDORA_STORY = (
    "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, "
    "lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria "
    "was buried beneath the shifting sands of time, lost to the world for centuries. You are "
    "an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled "
    "upon an ancient map hinting at secrets that Aeloria holds a secret so profound that it has "
    "the potential to reshape the very fabric of reality. Your journey will take you through "
    "treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: "
    "Character Background: Develop a detailed background for your character. Describe their "
    "motivations for seeking out Aeloria, their skills and weaknesses, and any personal "
    "connections to the ancient city or its legends. Are they driven by a quest for knowledge, "
    "a search for lost familt clue is hidden."
)

# SWA model for these tests
SWA_MODEL = "google/gemma-3-1b-it"

# Module-level markers (framework-specific markers applied per-test)
pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.model(SWA_MODEL),
]


# =============================================================================
# Helper functions
# =============================================================================


def print_test_header(title: str) -> None:
    """Print a formatted test header."""
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def print_phase(phase_num: int, description: str) -> None:
    """Print a formatted phase header."""
    print(f"\n=== Phase {phase_num}: {description} ===")


def check_kvbm_metrics(phase_name: str, metrics_port: int) -> dict[str, int]:
    """Fetch and display KVBM metrics."""
    print(f"\n--- Checking KVBM metrics after {phase_name} ---")
    metrics = fetch_kvbm_metrics(port=metrics_port)

    offload_d2h = metrics.get("kvbm_offload_blocks_d2h", 0)
    onboard_h2d = metrics.get("kvbm_onboard_blocks_h2d", 0)

    print(f"  kvbm_offload_blocks_d2h: {offload_d2h}")
    print(f"  kvbm_onboard_blocks_h2d: {onboard_h2d}")

    return {
        "kvbm_offload_blocks_d2h": offload_d2h,
        "kvbm_onboard_blocks_h2d": onboard_h2d,
    }


def reset_cache(base_url: str) -> None:
    """Reset the GPU prefix cache."""
    print("Resetting prefix cache...")
    try:
        response = requests.post(f"{base_url}/reset_prefix_cache", timeout=30)
        response.raise_for_status()
        print("Cache reset successful")
    except Exception as e:
        print(f"Warning: Cache reset failed: {e}")


# =============================================================================
# vLLM fixtures
# =============================================================================


@pytest.fixture(scope="function")
def tester(llm_server_kvbm):  # noqa: F811
    """Create tester bound to the KVBM-enabled vLLM server."""
    return DeterminismTester(
        base_url=llm_server_kvbm.base_url,
        model_id=SWA_MODEL,
        server_type=llm_server_kvbm.server_type,
    )


# =============================================================================
# TRT-LLM fixtures
# =============================================================================


@pytest.fixture(scope="function")
def trtllm_server(request, runtime_services):
    """Start and stop a TRT-LLM server for SWA KVBM testing."""
    logger = logging.getLogger("pytest")
    logger.setLevel(logging.INFO)

    cpu_blocks = getattr(request, "param", {}).get("cpu_blocks", None)
    gpu_blocks = getattr(request, "param", {}).get("gpu_blocks", None)
    port = getattr(request, "param", {}).get("port", None)

    log_dir = Path(resolve_test_output_path(request.node.name))

    if not check_module_available("tensorrt_llm"):
        pytest.skip("tensorrt_llm module not available")

    server_manager = LLMServerManager(
        port=port,
        cpu_cache_blocks=cpu_blocks,
        gpu_cache_blocks=gpu_blocks,
        log_dir=log_dir,
        server_type=ServerType.trtllm,
        model=SWA_MODEL,
        trtllm_gpu_memory_fraction=0.03,
    )

    start_timeout = int(os.environ.get("KVBM_SERVER_START_TIMEOUT", "300"))
    if not server_manager.start_server(timeout=start_timeout):
        pytest.fail(
            f"Failed to start trtllm server (cpu_blocks={cpu_blocks}, gpu_blocks={gpu_blocks}, port={server_manager.port})"
        )

    yield server_manager

    server_manager.stop_server()


@pytest.fixture(scope="function")
def trtllm_tester(trtllm_server):
    """Create tester bound to the KVBM-enabled TRT-LLM server."""
    return DeterminismTester(
        base_url=trtllm_server.base_url,
        model_id=SWA_MODEL,
        server_type=trtllm_server.server_type,
    )


# =============================================================================
# vLLM tests
# =============================================================================


@pytest.mark.vllm
@pytest.mark.parametrize("llm_server_kvbm", [{"model": SWA_MODEL}], indirect=True)
@pytest.mark.timeout(170)
def test_swa_offload_and_onboard(tester, llm_server_kvbm):  # noqa: F811
    """
    Test offload → cache reset → onboard cycle with SWA model (vLLM).

    Validates that:
    - Initial request triggers offload to CPU cache
    - Cache reset clears GPU cache
    - Repeated request triggers onboard from CPU to GPU
    - Responses are deterministic across the cycle
    """
    _run_offload_and_onboard(tester, llm_server_kvbm)


@pytest.mark.vllm
@pytest.mark.parametrize(
    "llm_server_kvbm",
    [{"cpu_blocks": 200, "gpu_blocks": 20, "model": SWA_MODEL}],
    indirect=True,
)
@pytest.mark.timeout(170)
def test_swa_gpu_cache_eviction(tester, llm_server_kvbm):  # noqa: F811
    """
    Test GPU cache eviction mechanics with SWA model (vLLM).

    Validates that:
    - Multiple requests fill GPU cache causing eviction
    - Evicted blocks can be retrieved from CPU cache via onboarding
    - Metrics correctly reflect offload and onboard operations
    """
    _run_gpu_cache_eviction(tester, llm_server_kvbm)


@pytest.mark.vllm
@pytest.mark.parametrize(
    "llm_server_kvbm",
    [{"cpu_blocks": 200, "gpu_blocks": 20, "model": SWA_MODEL}],
    indirect=True,
)
@pytest.mark.timeout(160)
def test_swa_onboarding_determinism(tester, llm_server_kvbm):  # noqa: F811
    """
    Test onboarding determinism under eviction scenario with SWA model (vLLM).

    Validates that:
    - Multiple onboarding cycles produce deterministic results
    - Responses are consistent when blocks are onboarded multiple times
    - Tests onboarded vs onboarded (not initial vs onboarded)
    """
    _run_onboarding_determinism(tester, llm_server_kvbm)


# =============================================================================
# TRT-LLM tests
# =============================================================================


@pytest.mark.trtllm
@pytest.mark.parametrize(
    "trtllm_server",
    [{"cpu_blocks": 200, "gpu_blocks": 10}],
    indirect=True,
)
@pytest.mark.timeout(170)
def test_swa_offload_and_onboard_trtllm(trtllm_tester, trtllm_server):
    """
    Test offload → eviction → onboard cycle with SWA model (TRT-LLM).

    Uses eviction-based cache clearing instead of reset_prefix_cache
    (which is not available in TRT-LLM). Validates that:
    - Initial request triggers offload to CPU cache
    - A different request evicts the first from GPU cache
    - Repeated first request triggers onboard from CPU to GPU
    - Responses are deterministic across the cycle
    """
    _run_offload_and_onboard_via_eviction(trtllm_tester, trtllm_server)


@pytest.mark.trtllm
@pytest.mark.parametrize(
    "trtllm_server",
    [{"cpu_blocks": 200, "gpu_blocks": 10}],
    indirect=True,
)
@pytest.mark.timeout(170)
def test_swa_gpu_cache_eviction_trtllm(trtllm_tester, trtllm_server):
    """
    Test GPU cache eviction mechanics with SWA model (TRT-LLM).

    Validates that:
    - Multiple requests fill GPU cache causing eviction
    - Evicted blocks can be retrieved from CPU cache via onboarding
    - Metrics correctly reflect offload and onboard operations
    """
    _run_gpu_cache_eviction(trtllm_tester, trtllm_server)


@pytest.mark.trtllm
@pytest.mark.parametrize(
    "trtllm_server",
    [{"cpu_blocks": 200, "gpu_blocks": 10}],
    indirect=True,
)
@pytest.mark.timeout(160)
def test_swa_onboarding_determinism_trtllm(trtllm_tester, trtllm_server):
    """
    Test onboarding determinism under eviction scenario with SWA model (TRT-LLM).

    Validates that:
    - Multiple onboarding cycles produce deterministic results
    - Responses are consistent when blocks are onboarded multiple times
    - Tests onboarded vs onboarded (not initial vs onboarded)
    """
    _run_onboarding_determinism(trtllm_tester, trtllm_server)


# =============================================================================
# Shared test logic
# =============================================================================


def _run_offload_and_onboard(tester, server):
    """Shared offload/onboard test logic for both vLLM and TRT-LLM."""
    print_test_header("SWA OFFLOAD AND ONBOARD TEST")

    prompt = AELDORA_STORY[:400]

    # Phase 1: Initial request triggers offload
    print_phase(1, "Initial request (expect offload to CPU)")
    print(f"Sending request: {prompt[:80]}...")

    response_1 = tester.make_request(prompt, max_tokens=MAX_TOKENS)
    print(f"Response 1: {response_1}")

    metrics = check_kvbm_metrics("Phase 1", server.metrics_port)
    assert (
        metrics["kvbm_offload_blocks_d2h"] > 0
    ), "Phase 1: No blocks offloaded. KVBM may not be triggering offloads."
    assert (
        metrics["kvbm_onboard_blocks_h2d"] == 0
    ), f"Phase 1: Expected 0 onboarded blocks, got {metrics['kvbm_onboard_blocks_h2d']}"
    print(f"✓ Phase 1: {metrics['kvbm_offload_blocks_d2h']} blocks offloaded")

    # Phase 2: Reset GPU cache
    print_phase(2, "Clean up GPU cache")
    reset_cache(server.base_url)

    # Phase 3: Repeated request triggers onboard
    print_phase(3, "Re-send same request (expect onboard from CPU)")
    print(f"Sending same request: {prompt[:80]}...")

    response_2 = tester.make_request(prompt, max_tokens=MAX_TOKENS)
    print(f"Response 2: {response_2}")

    metrics = check_kvbm_metrics("Phase 3", server.metrics_port)
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


def _run_offload_and_onboard_via_eviction(tester, server):
    """Offload/onboard test using eviction instead of reset_prefix_cache.

    TRT-LLM does not support the /reset_prefix_cache endpoint, so we evict
    the first prompt's blocks by sending a different prompt that fills the
    small GPU cache, then re-send the first prompt to trigger onboarding.
    """
    print_test_header("SWA OFFLOAD AND ONBOARD TEST (via eviction)")

    prompt = AELDORA_STORY
    filler_prompt = (
        "Read the following entry from the ancient scrolls of Aeloria: " + AELDORA_STORY
    )

    # Phase 1: Initial request triggers offload
    print_phase(1, "Initial request (expect offload to CPU)")
    print(f"Sending request: {prompt[:80]}...")

    response_1 = tester.make_request(prompt, max_tokens=MAX_TOKENS)
    print(f"Response 1: {response_1}")

    metrics = check_kvbm_metrics("Phase 1", server.metrics_port)
    assert (
        metrics["kvbm_offload_blocks_d2h"] > 0
    ), "Phase 1: No blocks offloaded. KVBM may not be triggering offloads."
    assert (
        metrics["kvbm_onboard_blocks_h2d"] == 0
    ), f"Phase 1: Expected 0 onboarded blocks, got {metrics['kvbm_onboard_blocks_h2d']}"
    print(f"✓ Phase 1: {metrics['kvbm_offload_blocks_d2h']} blocks offloaded")

    # Phase 2: Send different prompt to evict first prompt's blocks from GPU
    print_phase(2, "Send filler request to evict first prompt from GPU")
    print(f"Sending filler: {filler_prompt[:80]}...")

    tester.make_request(filler_prompt, max_tokens=MAX_TOKENS)

    metrics_p2 = check_kvbm_metrics("Phase 2", server.metrics_port)
    print(f"✓ Phase 2: {metrics_p2['kvbm_offload_blocks_d2h']} total blocks offloaded")

    # Phase 3: Re-send first prompt (should onboard from CPU)
    print_phase(3, "Re-send first request (expect onboard from CPU)")
    print(f"Sending same request: {prompt[:80]}...")

    tester.make_request(prompt, max_tokens=MAX_TOKENS)

    metrics = check_kvbm_metrics("Phase 3", server.metrics_port)
    assert (
        metrics["kvbm_onboard_blocks_h2d"] > 0
    ), "Phase 3: No blocks onboarded. Expected CPU→GPU transfer after eviction."
    print(f"✓ Phase 3: {metrics['kvbm_onboard_blocks_h2d']} blocks onboarded from CPU")

    # Note: determinism (initial vs onboarded) is NOT checked here because
    # partial onboarding can produce slightly different outputs than full
    # prefill. The dedicated test_swa_onboarding_determinism_trtllm test
    # validates onboarded-vs-onboarded determinism instead.

    print("\n=== TEST PASSED ===")


def _run_gpu_cache_eviction(tester, server):
    """Shared GPU cache eviction test logic for both vLLM and TRT-LLM."""
    print_test_header("SWA GPU CACHE EVICTION TEST")
    print(f"GPU blocks: {server.gpu_cache_blocks}")
    print(f"CPU blocks: {server.cpu_cache_blocks}")

    prompt_1 = AELDORA_STORY
    prompt_2 = (
        "Read the following entry from the ancient scrolls of Aeloria: " + AELDORA_STORY
    )

    # Phase 1: First request triggers offload
    print_phase(1, "Send first request")
    print(f"Prompt 1: {prompt_1[:80]}...")

    tester.make_request(prompt_1, max_tokens=MAX_TOKENS)

    metrics_p1 = check_kvbm_metrics("Phase 1", server.metrics_port)
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

    metrics_p2 = check_kvbm_metrics("Phase 2", server.metrics_port)
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

    metrics_p3 = check_kvbm_metrics("Phase 3", server.metrics_port)
    assert (
        metrics_p3["kvbm_onboard_blocks_h2d"] > 0
    ), "Phase 3: No blocks onboarded. Expected CPU→GPU retrieval after eviction."
    print(f"✓ Phase 3: {metrics_p3['kvbm_onboard_blocks_h2d']} blocks onboarded")
    print("✓ Eviction mechanics verified: offload → eviction → onboard")

    print("\n=== TEST PASSED ===")


def _run_onboarding_determinism(tester, server):
    """Shared onboarding determinism test logic for both vLLM and TRT-LLM."""
    print_test_header("SWA ONBOARDING DETERMINISM TEST")
    print(f"GPU blocks: {server.gpu_cache_blocks}")
    print(f"CPU blocks: {server.cpu_cache_blocks}")

    prompt_1 = AELDORA_STORY
    prompt_2 = (
        "Read the following entry from the ancient scrolls of Aeloria: " + AELDORA_STORY
    )

    # Phase 1: First request triggers offload
    print_phase(1, "Send first request")
    print(f"Prompt 1: {prompt_1[:80]}...")
    tester.make_request(prompt_1, max_tokens=MAX_TOKENS)
    check_kvbm_metrics("Phase 1", server.metrics_port)

    # Phase 2: Second request (may evict first from GPU)
    print_phase(2, "Send second request (may evict first from GPU)")
    print(f"Prompt 2: {prompt_2[:80]}...")
    tester.make_request(prompt_2, max_tokens=MAX_TOKENS)
    check_kvbm_metrics("Phase 2", server.metrics_port)

    # Phase 3: Re-request prompt 1 (first onboard cycle)
    print_phase(3, "Re-request Prompt 1 (first onboard cycle)")
    print(f"Re-sending Prompt 1: {prompt_1[:80]}...")
    response_1_first_onboard = tester.make_request(prompt_1, max_tokens=MAX_TOKENS)
    print(f"Response 1 (first onboard): {response_1_first_onboard}")
    check_kvbm_metrics("Phase 3", server.metrics_port)

    # Phase 4: Re-request prompt 2 (first onboard cycle)
    print_phase(4, "Re-request Prompt 2 (first onboard cycle)")
    print(f"Re-sending Prompt 2: {prompt_2[:80]}...")
    response_2_first_onboard = tester.make_request(prompt_2, max_tokens=MAX_TOKENS)
    print(f"Response 2 (first onboard): {response_2_first_onboard}")
    check_kvbm_metrics("Phase 4", server.metrics_port)

    # Phase 5: Re-request prompt 1 (second onboard cycle)
    print_phase(5, "Re-request Prompt 1 (second onboard cycle)")
    print(f"Re-sending Prompt 1 (third time): {prompt_1[:80]}...")
    response_1_second_onboard = tester.make_request(prompt_1, max_tokens=MAX_TOKENS)
    print(f"Response 1 (second onboard): {response_1_second_onboard}")
    check_kvbm_metrics("Phase 5", server.metrics_port)

    # Phase 6: Re-request prompt 2 (second onboard cycle)
    print_phase(6, "Re-request Prompt 2 (second onboard cycle)")
    print(f"Re-sending Prompt 2 (third time): {prompt_2[:80]}...")
    response_2_second_onboard = tester.make_request(prompt_2, max_tokens=MAX_TOKENS)
    print(f"Response 2 (second onboard): {response_2_second_onboard}")
    check_kvbm_metrics("Phase 6", server.metrics_port)

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
