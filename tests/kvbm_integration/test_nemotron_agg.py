#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
E2E aggregated test for Nemotron hybrid (Mamba + Transformer) models with KVBM.

Verifies that the DynamoConnector correctly handles hybrid models whose KV
caches include both standard attention Tensors and Mamba list[Tensor] layers:

  1. CPU offloading works (blocks move GPU → CPU).
  2. Cache hits restore correctly (blocks onboard CPU → GPU on repeated prefix).
  3. Output is deterministic across offload/onboard cycles.

Default model: nvidia/Nemotron-H-30B-A3B-BF16  (override via KVBM_NEMOTRON_MODEL_ID)
For faster local iteration use KVBM_NEMOTRON_MODEL_ID=nvidia/Llama-3_3-Nemotron-Super-49B-v1

    KVBM_NEMOTRON_MODEL_ID=nvidia/Llama-3_3-Nemotron-Super-49B-v1 \
        pytest tests/kvbm_integration/test_nemotron_agg.py -v -s
"""

import os

import pytest

from .common import DeterminismTester, fetch_kvbm_metrics, llm_server_kvbm  # noqa: F401

NEMOTRON_MODEL = os.environ.get(
    "KVBM_NEMOTRON_MODEL_ID", "nvidia/Nemotron-H-30B-A3B-BF16"
)

BLOCK_SIZE = 16
MAX_TOKENS = 10

# Prompt long enough to produce multiple KV cache blocks for meaningful
# offload testing, but short enough to keep test runtime reasonable.
PROMPT = (
    "In the heart of Eldoria, an ancient land of boundless magic and mysterious "
    "creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge "
    "and power, Aeloria was buried beneath the shifting sands of time, lost to the "
    "world for centuries. You are an intrepid explorer, known for your unparalleled "
    "curiosity and courage, who has stumbled upon an ancient map hinting at secrets "
    "that Aeloria holds a secret so profound that it has the potential to reshape the "
    "very fabric of reality. Your journey will take you through treacherous deserts, "
    "enchanted forests, and across perilous mountain ranges."
)

# Distinct prompt used to evict the original blocks from the GPU cache so that
# the next request for PROMPT must onboard from CPU.
EVICTION_PROMPT = (
    "The ocean covers more than 70 percent of Earth's surface and contains 97 "
    "percent of the planet's water. Despite its vastness, we have explored less "
    "than 5 percent of the ocean floor, making it one of the last great frontiers "
    "of discovery on our planet. The deep sea harbors creatures that seem almost "
    "alien in their appearance and adaptations, from bioluminescent jellyfish to "
    "giant squid that can grow up to 43 feet in length. Climate change represents "
    "one of the most pressing challenges facing humanity in the 21st century. Rising "
    "global temperatures are causing ice caps to melt, sea levels to rise, and weather "
    "patterns to become increasingly unpredictable."
)

pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.nightly,
    pytest.mark.slow,
]


@pytest.fixture(scope="function")
def tester(llm_server_kvbm):  # noqa: F811
    """Create tester bound to the KVBM-enabled Nemotron server."""
    return DeterminismTester(
        base_url=llm_server_kvbm.base_url,
        model_id=NEMOTRON_MODEL,
        server_type=llm_server_kvbm.server_type,
    )


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def _check_metrics(phase: str, metrics_port: int) -> dict[str, int]:
    """Fetch and display KVBM metrics."""
    print(f"\n--- KVBM metrics after {phase} ---")
    metrics = fetch_kvbm_metrics(port=metrics_port)
    offload = metrics.get("kvbm_offload_blocks_d2h", 0)
    onboard = metrics.get("kvbm_onboard_blocks_h2d", 0)
    print(f"  offload_blocks_d2h: {offload}")
    print(f"  onboard_blocks_h2d: {onboard}")
    return {"kvbm_offload_blocks_d2h": offload, "kvbm_onboard_blocks_h2d": onboard}


@pytest.mark.parametrize(
    "llm_server_kvbm",
    [
        {
            "model": NEMOTRON_MODEL,
            "cpu_blocks": int(os.environ.get("KVBM_NEMOTRON_CPU_BLOCKS", "200")),
            "gpu_blocks": int(os.environ.get("KVBM_NEMOTRON_GPU_BLOCKS", "50")),
        }
    ],
    indirect=True,
)
@pytest.mark.timeout(600)
def test_nemotron_offload_onboard_cycle(tester, llm_server_kvbm):  # noqa: F811
    """Verify KVBM offload and onboard for a Nemotron hybrid model.

    1. Send PROMPT → blocks should be offloaded GPU → CPU.
    2. Send EVICTION_PROMPT → push original blocks out of GPU cache.
    3. Re-send PROMPT → blocks should be onboarded CPU → GPU (cache hit).
    4. Verify response matches the first request (determinism).
    """
    _print_header("NEMOTRON HYBRID MODEL — OFFLOAD / ONBOARD CYCLE")

    print(f"Model: {NEMOTRON_MODEL}")
    print(f"GPU blocks: {llm_server_kvbm.gpu_cache_blocks}")
    print(f"CPU blocks: {llm_server_kvbm.cpu_cache_blocks}")

    # Phase 1 — initial request triggers offload
    print("\n=== Phase 1: initial request (expect offload) ===")
    response_1 = tester.make_request(PROMPT, max_tokens=MAX_TOKENS)
    print(f"Response 1: {response_1}")

    m1 = _check_metrics("Phase 1 (initial request)", llm_server_kvbm.metrics_port)
    assert m1["kvbm_offload_blocks_d2h"] > 0, (
        "Phase 1: no blocks offloaded — KVBM offload did not trigger for the "
        "Nemotron hybrid model. Check that list[Tensor] Mamba caches are "
        "normalised to raw_tensor and active_save_layers is set correctly."
    )
    print(f"✓ Phase 1: {m1['kvbm_offload_blocks_d2h']} blocks offloaded")

    # Phase 2 — evict original blocks from GPU
    print("\n=== Phase 2: send eviction prompt ===")
    tester.make_request(EVICTION_PROMPT, max_tokens=MAX_TOKENS)
    _check_metrics("Phase 2 (eviction)", llm_server_kvbm.metrics_port)
    print("✓ Phase 2: eviction prompt completed")

    # Phase 3 — re-send PROMPT, expect onboard (cache hit)
    print("\n=== Phase 3: repeat original prompt (expect onboard) ===")
    response_2 = tester.make_request(PROMPT, max_tokens=MAX_TOKENS)
    print(f"Response 2: {response_2}")

    m3 = _check_metrics("Phase 3 (onboard)", llm_server_kvbm.metrics_port)
    assert m3["kvbm_onboard_blocks_h2d"] > 0, (
        "Phase 3: no blocks onboarded — cache hit restoration failed for the "
        "Nemotron hybrid model."
    )
    print(f"✓ Phase 3: {m3['kvbm_onboard_blocks_h2d']} blocks onboarded")

    # Phase 4 — determinism check
    print("\n=== Phase 4: determinism check ===")
    assert response_1 == response_2, (
        f"Responses differ across offload/onboard cycle.\n"
        f"  Response 1: {response_1}\n"
        f"  Response 2: {response_2}"
    )
    print("✓ Phase 4: responses are deterministic across offload/onboard")

    _print_header("TEST PASSED")


@pytest.mark.parametrize(
    "llm_server_kvbm",
    [
        {
            "model": NEMOTRON_MODEL,
            "cpu_blocks": int(os.environ.get("KVBM_NEMOTRON_CPU_BLOCKS", "200")),
            "gpu_blocks": int(os.environ.get("KVBM_NEMOTRON_GPU_BLOCKS", "50")),
        }
    ],
    indirect=True,
)
@pytest.mark.timeout(600)
def test_nemotron_repeated_prefix_determinism(tester, llm_server_kvbm):  # noqa: F811
    """Verify determinism over multiple offload/onboard cycles.

    Sends the same prompt several times with eviction in between.  All
    responses must be identical, proving that Mamba + attention KV state
    is correctly preserved through the CPU cache.
    """
    _print_header("NEMOTRON HYBRID MODEL — REPEATED PREFIX DETERMINISM")

    num_cycles = int(os.environ.get("KVBM_NEMOTRON_CYCLES", "3"))
    print(f"Model: {NEMOTRON_MODEL}")
    print(f"Cycles: {num_cycles}")

    baseline: str | None = None
    for cycle in range(1, num_cycles + 1):
        print(f"\n--- cycle {cycle}/{num_cycles} ---")

        response = tester.make_request(PROMPT, max_tokens=MAX_TOKENS)
        print(f"  response: {response}")

        if baseline is None:
            baseline = response
        else:
            assert response == baseline, (
                f"Cycle {cycle}: response differs from baseline.\n"
                f"  Baseline: {baseline}\n"
                f"  Got:      {response}"
            )

        # Evict between cycles (except after the last one)
        if cycle < num_cycles:
            tester.make_request(EVICTION_PROMPT, max_tokens=MAX_TOKENS)

    metrics = _check_metrics("all cycles", llm_server_kvbm.metrics_port)
    assert metrics["kvbm_offload_blocks_d2h"] > 0, "No offload activity during test"
    assert metrics["kvbm_onboard_blocks_h2d"] > 0, "No onboard activity during test"

    print(
        f"\n✓ All {num_cycles} responses identical — deterministic "
        "across offload/onboard cycles"
    )
    _print_header("TEST PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
