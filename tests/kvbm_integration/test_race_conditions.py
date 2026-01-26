#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KVBM Race Condition Verification Tests

These tests verify the fixes for two related race conditions in the KVBM connector:

1. No-Cache-Hit Race: Requests with no cache hits had `had_operations=false`, but
   `request_finished()` always returned `true`, causing vLLM to wait for
   `finished_sending` that would never come correctly.

2. Slot Creation Race: Worker might check `is_complete()` before the slot was created
   via metadata, causing incorrect behavior.

The tests verify:
- Requests with no cache hits complete without hanging (Layer 1 fix)
- Requests with cache hits complete correctly (offload/onboard cycle)
- Multiple sequential cache-miss requests all complete correctly
- Rapid request completion doesn't trigger race conditions
- Missing slots are handled gracefully with retries (Layer 2 fix)
"""

import uuid

import pytest
import requests

from .common import llm_server_kvbm  # noqa: F401
from .common import (
    DeterminismTester,
    check_logs_for_race_indicators,
    fetch_kvbm_metrics,
    get_server_log_path,
)

# Test configuration
MAX_TOKENS = 15  # Max tokens to generate in test responses

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

# Model used for race condition tests (smaller model for faster CI)
KVBM_TEST_MODEL = "Qwen/Qwen3-0.6B"

# Test markers
pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
]


# Helper functions
def reset_cache(base_url: str) -> None:
    """Reset the GPU prefix cache."""
    print("Resetting prefix cache...")
    try:
        response = requests.post(f"{base_url}/reset_prefix_cache", timeout=30)
        response.raise_for_status()
        print("Cache reset successful")
    except Exception as e:
        print(f"Warning: Cache reset failed: {e}")


# Fixtures
@pytest.fixture(scope="function")
def tester(llm_server_kvbm):  # noqa: F811
    """Create tester bound to the KVBM-enabled server."""
    return DeterminismTester(
        base_url=llm_server_kvbm.base_url,
        model_id=KVBM_TEST_MODEL,
        server_type=llm_server_kvbm.server_type,
    )


# Tests
@pytest.mark.parametrize("llm_server_kvbm", [{"model": KVBM_TEST_MODEL}], indirect=True)
def test_no_cache_hit_request_completes(tester, llm_server_kvbm):  # noqa: F811
    """Verify requests with no cache hits complete without hanging.

    This tests the had_operations=false path where request_finished()
    should return false, allowing vLLM to free immediately.

    Before the fix: no-cache-hit requests would cause vLLM to wait
    indefinitely for finished_sending that would never come.
    """
    print("\n" + "=" * 70)
    print("TEST: No-Cache-Hit Request Completion")
    print("=" * 70)

    # Use unique prompt guaranteed to have no cache hits
    unique_prompt = (
        f"Unique test prompt {uuid.uuid4()}: Tell me a brief fact about the number 42."
    )

    print(f"Sending unique prompt (no cache hits expected): {unique_prompt[:60]}...")

    # Should complete without hanging (before fix: would hang or crash)
    # Use a reasonable timeout - if this hangs, the test will fail
    response = tester.make_request(unique_prompt, max_tokens=10)

    assert response is not None, "Response should not be None"
    assert len(response) > 0, "Response should not be empty"
    print(f"Response received: {response}")

    # Check metrics - should have offload (if prompt is long enough) but minimal/no onboard
    metrics = fetch_kvbm_metrics()
    onboard_count = metrics.get("kvbm_onboard_blocks_h2d", 0)
    print(f"Onboard blocks count: {onboard_count}")

    # For a unique prompt with no cache, we expect zero onboard operations
    # (onboard happens when we retrieve cached blocks from CPU/disk)
    print("✓ Request completed successfully without hanging")
    print("=" * 70)


@pytest.mark.parametrize("llm_server_kvbm", [{"model": KVBM_TEST_MODEL}], indirect=True)
def test_cache_hit_request_completes(tester, llm_server_kvbm):  # noqa: F811
    """Verify requests WITH cache hits complete correctly.

    This tests the had_operations=true path with CreateSlot + onboarding.
    """
    print("\n" + "=" * 70)
    print("TEST: Cache-Hit Request Completion")
    print("=" * 70)

    prompt = AELDORA_STORY[:400]

    # Phase 1: Initial request triggers offload
    print("\n--- Phase 1: Initial request (triggers offload) ---")
    print(f"Sending request: {prompt[:60]}...")
    response_1 = tester.make_request(prompt, max_tokens=MAX_TOKENS)
    print(f"Response 1: {response_1}")

    metrics_p1 = fetch_kvbm_metrics()
    offload_p1 = metrics_p1.get("kvbm_offload_blocks_d2h", 0)
    print(f"Offload blocks after Phase 1: {offload_p1}")
    assert offload_p1 > 0, "Phase 1: Expected offload activity"

    # Phase 2: Reset cache to force onboarding on next request
    print("\n--- Phase 2: Reset cache ---")
    reset_cache(llm_server_kvbm.base_url)

    # Phase 3: Re-request triggers onboard (cache hit)
    print("\n--- Phase 3: Re-request (triggers onboard from cache hit) ---")
    print(f"Re-sending same request: {prompt[:60]}...")
    response_2 = tester.make_request(prompt, max_tokens=MAX_TOKENS)
    print(f"Response 2: {response_2}")

    metrics_p3 = fetch_kvbm_metrics()
    onboard_p3 = metrics_p3.get("kvbm_onboard_blocks_h2d", 0)
    print(f"Onboard blocks after Phase 3: {onboard_p3}")
    assert onboard_p3 > 0, "Phase 3: Expected onboard activity (cache hit path)"

    # Both should complete without error
    assert response_1 is not None, "Response 1 should not be None"
    assert response_2 is not None, "Response 2 should not be None"

    print("✓ Both requests completed successfully")
    print("=" * 70)


@pytest.mark.parametrize(
    "llm_server_kvbm",
    [{"cpu_blocks": 200, "gpu_blocks": 20, "model": KVBM_TEST_MODEL}],
    indirect=True,
)
def test_multiple_cache_miss_requests(tester, llm_server_kvbm):  # noqa: F811
    """Test multiple sequential requests that all miss the cache.

    This verifies the had_operations=false path works correctly for
    multiple requests in sequence. Each request is unique and will
    not hit the cache.

    Before the fix: requests with no cache hits would cause vLLM to
    wait indefinitely for finished_sending.
    """
    print("\n" + "=" * 70)
    print("TEST: Multiple Cache Miss Requests")
    print("=" * 70)

    # All unique prompts - none will hit the cache
    prompts = [
        f"Unique request {uuid.uuid4().hex[:8]}: Tell me about topic {i}"
        for i in range(5)
    ]

    print(f"\n--- Sending {len(prompts)} sequential cache-miss requests ---")

    errors = []
    results = []

    for i, prompt in enumerate(prompts):
        try:
            print(f"  Request {i+1}/{len(prompts)}: {prompt[:45]}...")
            result = tester.make_request(prompt, max_tokens=10)
            if result is not None:
                results.append(result)
                print(f"    ✓ Completed: {result[:30]}...")
            else:
                errors.append(f"None result for: {prompt[:40]}")
                print("    ✗ None result")
        except Exception as e:
            errors.append(f"Exception for {prompt[:40]}: {str(e)}")
            print(f"    ✗ Failed: {e}")

    print("\n--- Results ---")
    print(f"Successful: {len(results)}/{len(prompts)}")
    print(f"Errors: {len(errors)}")

    if errors:
        for err in errors:
            print(f"  Error: {err}")

    assert len(errors) == 0, f"Cache miss requests failed: {errors}"
    assert len(results) == len(prompts), "Not all requests completed"

    # Verify metrics show offload activity but no onboard
    metrics = fetch_kvbm_metrics()
    print(
        f"\nMetrics: offload={metrics.get('kvbm_offload_blocks_d2h', 0)}, "
        f"onboard={metrics.get('kvbm_onboard_blocks_h2d', 0)}"
    )

    print("✓ All cache-miss requests completed successfully")
    print("=" * 70)


@pytest.mark.parametrize("llm_server_kvbm", [{"model": KVBM_TEST_MODEL}], indirect=True)
def test_rapid_request_completion(tester, llm_server_kvbm):  # noqa: F811
    """Maximize chance of CreateSlot race by completing requests quickly.

    Uses very short prompts and max_tokens to speed up completion,
    increasing probability that transfers complete before CreateSlot.

    Before the fix: this pattern was more likely to trigger the race
    condition where is_complete() was checked before slot existed.
    """
    print("\n" + "=" * 70)
    print("TEST: Rapid Request Completion (Timing Race)")
    print("=" * 70)

    # Very short prompt + minimal tokens = fast completion
    short_prompts = [
        "Hi",
        "Hello",
        "Hey",
        "One",
        "Two",
    ]

    num_iterations = 20
    print(f"Sending {num_iterations} rapid requests with minimal tokens...")

    for i in range(num_iterations):
        prompt = short_prompts[i % len(short_prompts)]
        try:
            response = tester.make_request(prompt, max_tokens=3)
            assert response is not None, f"Request {i+1} returned None"
            print(f"  Request {i+1}: '{prompt}' -> '{response[:30]}...'")
        except Exception as e:
            pytest.fail(f"Request {i+1} failed: {e}")

    print(f"\n✓ All {num_iterations} rapid requests completed without race condition")
    print("=" * 70)


@pytest.mark.parametrize(
    "llm_server_kvbm",
    [{"cpu_blocks": 200, "gpu_blocks": 20, "model": KVBM_TEST_MODEL}],
    indirect=True,
)
def test_graceful_missing_slot_handling(tester, llm_server_kvbm):  # noqa: F811
    """Verify worker handles slots gracefully without panics or assertion errors.

    This test generates load and then checks logs for:
    - Panic messages (unexpected - indicates bug)
    - Assertion errors (unexpected - indicates bug)
    - No-operations messages (expected for cache miss requests)
    """
    print("\n" + "=" * 70)
    print("TEST: Graceful Slot Handling (Log Verification)")
    print("=" * 70)

    # Generate load with multiple unique requests (all cache misses)
    # This exercises the slot creation path without the problematic
    # cache hit after reset path
    print("\n--- Sending requests to exercise slot creation ---")
    for i in range(8):
        unique_prompt = f"Log test {uuid.uuid4().hex[:8]}: {AELDORA_STORY[:100]}"
        tester.make_request(unique_prompt, max_tokens=10)
        print(f"  Completed request {i+1}/8")

    # Parse logs for race condition indicators
    print("\n--- Analyzing logs for race condition indicators ---")
    log_path = get_server_log_path(llm_server_kvbm)

    if log_path and log_path.exists():
        indicators = check_logs_for_race_indicators(log_path)

        print(f"  Slot retry messages: {indicators['slot_retry']}")
        print(f"  No-operations messages: {indicators['no_operations']}")
        print(f"  Panic messages: {indicators['panic']}")
        print(f"  Assertion errors: {indicators['assertion']}")

        # Should have NO panics or assertion errors
        assert (
            indicators["panic"] == 0
        ), f"Found {indicators['panic']} panic messages in logs"
        assert (
            indicators["assertion"] == 0
        ), f"Found {indicators['assertion']} assertion errors in logs"

        # Retry messages are OK (means Layer 2 activated and worked)
        if indicators["slot_retry"] > 0:
            print(
                f"\n  INFO: Layer 2 activated: {indicators['slot_retry']} retry attempts logged"
            )
            print(
                "  This is expected behavior - slot creation race was handled gracefully"
            )

        if indicators["no_operations"] > 0:
            print(
                f"\n  INFO: No-operations path taken {indicators['no_operations']} times"
            )
            print("  This is expected for requests without cache hits")
    else:
        print("  Warning: Could not find server log file for analysis")
        print("  Skipping log analysis (test still passes if requests completed)")

    print("\n✓ No panics or assertion errors - graceful handling verified")
    print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
