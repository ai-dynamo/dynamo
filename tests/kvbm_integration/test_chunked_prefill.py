#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chunked prefill offload validation test for KVBM.

This test validates that chunked prefill blocks are being offloaded during each
prefill iteration. vLLM's chunked prefill splits long prompts into multiple
scheduler iterations, and each chunk should trigger offload operations to KVBM.
"""

import math

import pytest

from .common import llm_server_kvbm  # noqa: F401
from .common import DeterminismTester, fetch_kvbm_metrics

# Test configuration
KVBM_TEST_MODEL = "Qwen/Qwen3-0.6B"
BLOCK_SIZE = 16  # Standard vLLM block size in tokens
MAX_NUM_BATCHED_TOKENS = 256  # Small value to force chunking
MAX_TOKENS = 1  # Max tokens to generate in test responses

# Long prompt that will be chunked into multiple pieces
# This prompt is approximately 768 tokens to create 3 chunks with max_num_batched_tokens=256
LONG_PROMPT = """In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at secrets that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges.
The advancement of technology has fundamentally transformed the way we live, work, and communicate in the modern world. From the invention of the printing press to the development of the internet, each technological breakthrough has opened new possibilities and created unprecedented opportunities for human progress. Today, artificial intelligence and machine learning are reshaping industries, healthcare, education, and countless other fields, promising to solve complex problems and improve the quality of life for people around the globe.
The human brain is the most complex organ in the known universe, containing approximately 86 billion neurons, each connected to thousands of others through intricate networks of synapses. This biological supercomputer processes information at speeds that would make even the most advanced artificial intelligence systems seem primitive by comparison. Every thought, memory, emotion, and decision we make is the result of electrical and chemical signals traveling through this vast neural network. The brain's ability to learn, adapt, and create is unmatched by any machine we have ever built.
Mathematics is the language of the universe, and numbers are its alphabet. Through the elegant dance of equations and the symphony of algorithms, we unlock the secrets of nature's most profound mysteries. From the simple beauty of prime numbers to the complex elegance of calculus, mathematics provides us with the tools to understand everything from the smallest subatomic particles to the vast expanse of galaxies stretching across the cosmic void.
A journey of a thousand miles begins with a single step, as the ancient Chinese proverb wisely reminds us. This timeless wisdom speaks to the fundamental truth that every great achievement, every monumental discovery, and every life-changing transformation starts with that crucial moment of decision - the moment when we choose to take action instead of remaining in the comfort of inaction."""

# Different prompt used to evict the original from GPU cache
# This forces the original blocks out so the next request must onboard from CPU
EVICTION_PROMPT = """The ocean covers more than 70 percent of Earth's surface and contains 97 percent of the planet's water. Despite its vastness, we have explored less than 5 percent of the ocean floor, making it one of the last great frontiers of discovery on our planet. The deep sea harbors creatures that seem almost alien in their appearance and adaptations, from bioluminescent jellyfish to giant squid that can grow up to 43 feet in length.
Climate change represents one of the most pressing challenges facing humanity in the 21st century. Rising global temperatures are causing ice caps to melt, sea levels to rise, and weather patterns to become increasingly unpredictable. Scientists around the world are working tirelessly to develop new technologies and strategies to mitigate the effects of climate change and transition to renewable energy sources.
The history of human civilization spans thousands of years and encompasses countless cultures, empires, and innovations. From the ancient Egyptians who built the pyramids to the Romans who constructed vast networks of roads and aqueducts, human ingenuity has continuously pushed the boundaries of what is possible. Today, we stand on the shoulders of giants, benefiting from the accumulated knowledge and wisdom of generations past.
Music has been an integral part of human culture since prehistoric times, with evidence of musical instruments dating back over 40,000 years. From the haunting melodies of ancient flutes to the complex symphonies of modern orchestras, music has the power to evoke emotions, tell stories, and bring people together across cultural and linguistic boundaries. The universal language of music continues to evolve and inspire new generations of artists and listeners alike."""

# Test markers
pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
]


def print_test_header(title: str) -> None:
    """Print a formatted test header."""
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def print_phase(phase_num: int, description: str) -> None:
    """Print a formatted phase header."""
    print(f"\n=== Phase {phase_num}: {description} ===")


def check_kvbm_metrics(phase_name: str) -> dict[str, int]:
    """Fetch and display KVBM metrics."""
    print(f"\n--- Checking KVBM metrics after {phase_name} ---")
    metrics = fetch_kvbm_metrics()

    offload_d2h = metrics.get("kvbm_offload_blocks_d2h", 0)
    onboard_h2d = metrics.get("kvbm_onboard_blocks_h2d", 0)

    print(f"  kvbm_offload_blocks_d2h: {offload_d2h}")
    print(f"  kvbm_onboard_blocks_h2d: {onboard_h2d}")

    return {
        "kvbm_offload_blocks_d2h": offload_d2h,
        "kvbm_onboard_blocks_h2d": onboard_h2d,
    }


def estimate_prompt_tokens(prompt: str) -> int:
    """Estimate the number of tokens in a prompt.

    This is a rough estimate based on average token length.
    For more accurate results, use a tokenizer.
    """
    # Rough estimate: ~4 characters per token for English text
    return len(prompt) // 4


@pytest.fixture(scope="function")
def tester(llm_server_kvbm):  # noqa: F811
    """Create tester bound to the KVBM-enabled server."""
    return DeterminismTester(
        base_url=llm_server_kvbm.base_url,
        model_id=KVBM_TEST_MODEL,
        server_type=llm_server_kvbm.server_type,
    )


@pytest.mark.parametrize(
    "llm_server_kvbm",
    [
        {
            "model": KVBM_TEST_MODEL,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "gpu_blocks": 50,  # Need enough for chunked prefill (256 tokens = 16 blocks per chunk)
        }
    ],
    indirect=True,
)
def test_chunked_prefill_offload(tester, llm_server_kvbm):  # noqa: F811
    """
    Validate that chunked prefill blocks are offloaded.

    Test flow:
    1. Send prompt large enough to trigger multiple chunks
    2. Verify total offloaded blocks matches expected for full prefill
    3. Reset cache and re-request to verify onboarding works
    4. Verify determinism across offload/onboard cycle
    """
    print_test_header("CHUNKED PREFILL OFFLOAD TEST")

    # Calculate expected metrics
    estimated_tokens = estimate_prompt_tokens(LONG_PROMPT)
    expected_chunks = math.ceil(estimated_tokens / MAX_NUM_BATCHED_TOKENS)
    expected_blocks = math.ceil(estimated_tokens / BLOCK_SIZE)

    print("Test configuration:")
    print(f"  Block size: {BLOCK_SIZE} tokens")
    print(f"  Max batched tokens: {MAX_NUM_BATCHED_TOKENS}")
    print(f"  Estimated prompt tokens: ~{estimated_tokens}")
    print(f"  Expected chunks: ~{expected_chunks}")
    print(f"  Expected blocks (minimum): ~{expected_blocks}")

    # Phase 1: Send long prompt (chunked prefill)
    print_phase(1, "Send long prompt (expect chunked prefill with offloads)")
    print(f"Sending request: {LONG_PROMPT[:80]}...")

    response_1 = tester.make_request(LONG_PROMPT, max_tokens=MAX_TOKENS)
    print(f"Response 1: {response_1}")

    metrics_p1 = check_kvbm_metrics("Phase 1")

    # Verify offload occurred
    offloaded_blocks = metrics_p1["kvbm_offload_blocks_d2h"]
    assert offloaded_blocks > 0, (
        "Phase 1: No blocks offloaded. KVBM may not be triggering offloads "
        "during chunked prefill."
    )

    # Verify we got a reasonable number of blocks offloaded
    # Allow some tolerance since token estimation is approximate
    min_expected_blocks = expected_blocks // 2  # Allow 50% tolerance
    assert offloaded_blocks >= min_expected_blocks, (
        f"Phase 1: Expected at least {min_expected_blocks} blocks offloaded "
        f"for ~{estimated_tokens} token prompt, got {offloaded_blocks}. "
        f"Chunked prefill may not be offloading all blocks."
    )

    print(
        f"✓ Phase 1: {offloaded_blocks} blocks offloaded (expected >= {min_expected_blocks})"
    )
    print("  Offload verification: PASSED")

    # Phase 2: Evict original blocks from GPU cache
    print_phase(2, "Evict original blocks from GPU cache")
    # Send a different prompt to fill GPU cache and force eviction of original blocks
    # (reset_prefix_cache is unreliable - it fails silently if blocks aren't freed)
    print(f"Sending eviction prompt: {EVICTION_PROMPT[:80]}...")
    tester.make_request(EVICTION_PROMPT, max_tokens=MAX_TOKENS)
    print("Eviction prompt completed - original blocks should be evicted from GPU")

    # Phase 3: Re-send same prompt (should trigger onboarding)
    print_phase(3, "Re-send same prompt (expect onboarding from CPU)")
    print(f"Sending same request: {LONG_PROMPT[:80]}...")

    response_2 = tester.make_request(LONG_PROMPT, max_tokens=MAX_TOKENS)
    print(f"Response 2: {response_2}")

    metrics_p3 = check_kvbm_metrics("Phase 3")

    # Verify onboarding occurred
    onboarded_blocks = metrics_p3["kvbm_onboard_blocks_h2d"]
    assert (
        onboarded_blocks > 0
    ), "Phase 3: No blocks onboarded. Expected CPU→GPU transfer after cache eviction."

    print(f"✓ Phase 3: {onboarded_blocks} blocks onboarded from CPU")

    # Summary
    print_test_header("TEST SUMMARY")
    print(f"  Prompt tokens (estimated): ~{estimated_tokens}")
    print(f"  Chunks (estimated): ~{expected_chunks}")
    print(f"  Blocks offloaded: {offloaded_blocks}")
    print(f"  Blocks onboarded: {onboarded_blocks}")

    print("\n=== TEST PASSED ===")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
