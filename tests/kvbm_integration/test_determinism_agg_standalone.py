#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone determinism tests for KVBM in aggregated mode.

These tests validate that KVBM works correctly as a standalone library
integrated directly with `vllm serve` or `trtllm-serve` commands.

This is a lighter integration test that proves KVBM works with these
frameworks without requiring the full Dynamo infrastructure.

For robust determinism testing with Dynamo frontend/worker architecture,
see test_determinism_agg.py.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from .common import TestDeterminism as BaseTestDeterminism

# Register fixtures from common.py
pytest_plugins = ["tests.kvbm_integration.common"]

# Test markers to align with repository conventions
pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.gpu_1,
    pytest.mark.nightly,
]


# =============================================================================
# Test Classes
# =============================================================================


class TestDeterminismAggStandalone(BaseTestDeterminism):
    """Test class for determinism validation using standalone vllm serve / trtllm-serve.

    These tests validate that KVBM works correctly as a standalone library
    integrated directly with vllm serve or trtllm-serve commands.
    """

    @pytest.mark.parametrize(
        "standalone_llm_server",
        [
            {"cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "10000"))},
        ],
        indirect=True,
    )
    def test_determinism_standalone_with_cache_reset(
        self, standalone_tester, standalone_llm_server, runtime_services
    ):
        """Test determinism with standalone vllm serve / trtllm-serve."""
        super().base_test_determinism_with_cache_reset(
            standalone_tester, standalone_llm_server, runtime_services
        )

    @pytest.mark.parametrize(
        "standalone_llm_server",
        [
            {"cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "10000"))},
        ],
        indirect=True,
    )
    @pytest.mark.kvbm_v2
    def test_determinism_standalone_with_cache_reset_v2(
        self, standalone_tester, standalone_llm_server, runtime_services, monkeypatch
    ):
        """Test determinism with standalone server and V2 transfer."""
        monkeypatch.setenv("DYN_KVBM_USE_V2_TRANSFER_EXPERIMENTAL", "1")
        super().base_test_determinism_with_cache_reset(
            standalone_tester, standalone_llm_server, runtime_services
        )


# =============================================================================
# Legacy Test Class (Backward Compatibility)
# =============================================================================


class TestDeterminismAgg(BaseTestDeterminism):
    """Legacy test class for determinism validation.

    These tests use standalone mode for backward compatibility.
    Consider using TestDeterminismAggStandalone instead.
    """

    @pytest.mark.parametrize(
        "llm_server",
        [
            {"cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "10000"))},
        ],
        indirect=True,
    )
    def test_determinism_agg_with_cache_reset(
        self, tester, llm_server, runtime_services
    ):
        """Test determinism across cache reset: run test with warmup, reset cache, run again without warmup."""
        super().base_test_determinism_with_cache_reset(
            tester, llm_server, runtime_services
        )

    @pytest.mark.parametrize(
        "llm_server",
        [
            {"cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "10000"))},
        ],
        indirect=True,
    )
    @pytest.mark.kvbm_v2
    def test_determinism_agg_with_cache_reset_v2(
        self, tester, llm_server, runtime_services, monkeypatch
    ):
        """Test determinism across cache reset: run test with warmup, reset cache, run again without warmup."""
        monkeypatch.setenv("DYN_KVBM_USE_V2_TRANSFER_EXPERIMENTAL", "1")
        super().base_test_determinism_with_cache_reset(
            tester, llm_server, runtime_services
        )

    @pytest.mark.parametrize(
        "llm_server",
        [
            {"cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "20000"))},
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "num_concurrent",
        [int(x) for x in os.environ.get("KVBM_CONCURRENT_REQUESTS", "3").split(",")],
    )
    @pytest.mark.parametrize(
        "max_tokens",
        [int(x) for x in os.environ.get("KVBM_MAX_TOKENS", "10").split(",")],
    )
    @pytest.mark.parametrize(
        "num_prompts",
        [int(x) for x in os.environ.get("KVBM_IFEVAL_PROMPTS", "120").split(",")],
    )
    @pytest.mark.skip(reason="Flaky test: DIS-665")
    def test_concurrent_determinism_with_ifeval(
        self,
        tester,
        llm_server,
        runtime_services,
        num_concurrent,
        max_tokens,
        num_prompts,
    ):
        """Simple concurrent determinism test: send IFEval prompts concurrently, with cache reset."""
        print("\n" + "=" * 70)
        print("CONCURRENT DETERMINISM TEST WITH IFEVAL")
        print("=" * 70)

        # Override max_tokens for this test iteration
        original_max_tokens = os.environ.get("KVBM_MAX_TOKENS")
        os.environ["KVBM_MAX_TOKENS"] = str(max_tokens)
        print(
            f"Using KVBM_MAX_TOKENS={max_tokens} (parametrized, original: {original_max_tokens or '48'})"
        )

        # Configuration comes from parametrize
        print(
            f"Configuration: {num_concurrent} concurrent requests, {max_tokens} max tokens"
        )

        # Load IFEval prompts
        ifeval_prompts = tester.download_ifeval_dataset()
        if not ifeval_prompts:
            pytest.skip("IFEval dataset not available")

        # Use parametrized number of IFEval prompts
        test_prompts = ifeval_prompts[:num_prompts]
        print(
            f"Using {len(test_prompts)} IFEval prompts for concurrent testing (parametrized: {num_prompts})"
        )
        print(f"Concurrency level: {num_concurrent} simultaneous requests")

        # Show sample prompts
        print("\nSample prompts:")
        for i, prompt in enumerate(test_prompts[:3]):
            print(f"  {i+1}. {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        if len(test_prompts) > 3:
            print(f"  ... and {len(test_prompts) - 3} more")

        def run_concurrent_test(phase_name, do_warmup=False):
            """Run one phase of concurrent testing."""
            print(f"\n=== {phase_name} ===")

            if do_warmup:
                # KV Cache warmup - send ALL test prompts to compute KV caches
                print(
                    f"Warming up KV caches with all {len(test_prompts)} test prompts..."
                )
                warmup_failed = 0

                for i, prompt in enumerate(test_prompts):
                    if i % 5 == 0 or i == len(test_prompts) - 1:
                        print(f"  Warmup progress: {i+1}/{len(test_prompts)}")

                    try:
                        tester.make_request(prompt)
                    except Exception as e:
                        warmup_failed += 1
                        if warmup_failed <= 3:
                            print(f"    Warmup failed for prompt {i}: {e}")

                if warmup_failed > 0:
                    print(
                        f"Warmup completed with {warmup_failed} failures out of {len(test_prompts)} prompts"
                    )
                else:
                    print(
                        f"Warmup completed successfully - all {len(test_prompts)} KV caches computed"
                    )

                # Wait for transfers to complete
                time.sleep(10)
            else:
                print("Skipping warmup (already done in previous phase)")

            # Run concurrent requests
            print(
                f"Sending {len(test_prompts)} requests with {num_concurrent} max concurrent..."
            )
            start_time = time.time()

            def make_request_wrapper(prompt_and_idx):
                idx, prompt = prompt_and_idx
                try:
                    response = tester.make_request(prompt)
                    return {
                        "idx": idx,
                        "prompt": prompt,
                        "response": response,
                        "success": True,
                    }
                except Exception as e:
                    return {
                        "idx": idx,
                        "prompt": prompt,
                        "error": str(e),
                        "success": False,
                    }

            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                results = list(
                    executor.map(make_request_wrapper, enumerate(test_prompts))
                )

            elapsed = time.time() - start_time
            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]

            print(
                f"Completed in {elapsed:.2f}s - Success: {len(successful)}, Failed: {len(failed)}"
            )

            if failed:
                for fail in failed[:3]:
                    print(f"  Failed: {fail['error']}")

            return successful

        # Phase 1: Before cache reset
        results_before = run_concurrent_test(
            "PHASE 1: BEFORE CACHE RESET", do_warmup=True
        )

        # Reset cache
        print("\n" + "=" * 50)
        print("RESETTING CACHE")
        print("=" * 50)
        tester.reset_prefix_cache()

        # Phase 2: After cache reset
        results_after = run_concurrent_test("PHASE 2: AFTER CACHE RESET")

        # Compare results between phases
        print("\n" + "=" * 70)
        print("DETERMINISM ANALYSIS")
        print("=" * 70)

        before_responses = {r["idx"]: r["response"] for r in results_before}
        after_responses = {r["idx"]: r["response"] for r in results_after}

        deterministic_count = 0
        total_compared = 0

        for idx in before_responses:
            if idx in after_responses:
                total_compared += 1
                before_resp = before_responses[idx]
                after_resp = after_responses[idx]

                if before_resp == after_resp:
                    deterministic_count += 1
                    print(f"   Prompt {idx}: DETERMINISTIC")
                else:
                    print(f"   Prompt {idx}: NON-DETERMINISTIC")
                    print(f"     Before: {before_resp}")
                    print(f"     After:  {after_resp}")

        # Final assessment
        success_rate = deterministic_count / total_compared if total_compared > 0 else 0
        print("\n=== FINAL RESULT ===")
        print(f"Prompts compared: {total_compared}")
        print(f"Deterministic: {deterministic_count}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Concurrent requests: {num_concurrent}")

        # Restore original max_tokens setting
        if original_max_tokens is not None:
            os.environ["KVBM_MAX_TOKENS"] = original_max_tokens
        else:
            os.environ.pop("KVBM_MAX_TOKENS", None)

        assert (
            success_rate == 1.0
        ), f"Determinism failed: {deterministic_count}/{total_compared} prompts deterministic"


if __name__ == "__main__":
    # Allow running as script
    pytest.main([__file__, "-v", "-s"])
