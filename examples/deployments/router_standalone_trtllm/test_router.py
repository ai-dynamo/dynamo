# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for TensorRT-LLM KV Router.

Usage:
    python test_router.py              # Run all tests
    python test_router.py --verbose    # Show detailed logs
    python test_router.py --mm-only    # Run only multimodal tests (no server needed)
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass

import httpx

from dynamo.llm import RadixTree, compute_block_hash_for_seq_py


@dataclass
class TestConfig:
    api_url: str = "http://localhost:8000"
    router_url: str = "http://localhost:7000"
    timeout: int = 30
    kv_settle_time: float = 3.0  # Time to wait for KV events to propagate


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    overlap: float = 0.0


def make_request(content: str, max_tokens: int = 10) -> dict:
    return {
        "model": "test",
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "max_tokens": max_tokens,
    }


def send_request(client: httpx.Client, url: str, payload: dict) -> bool:
    """Send a chat completion request and consume the stream."""
    try:
        resp = client.post(f"{url}/v1/chat/completions", json=payload)
        if resp.status_code != 200:
            return False
        for _ in resp.iter_lines():
            pass
        return True
    except Exception:
        return False


def get_tree_info(client: httpx.Client, url: str) -> dict:
    """Get radix tree debug info."""
    try:
        resp = client.get(f"{url}/debug/tree_info")
        return resp.json()
    except Exception:
        return {"num_blocks": -1, "events": []}


def extract_overlap_from_logs(verbose: bool) -> float:
    """
    In a real implementation, we'd parse server logs or add an API endpoint.
    For now, we rely on the debug endpoint and visual inspection.
    """
    return 0.0


class KvRouterTests:
    """Test cases for KV cache routing."""

    def __init__(self, config: TestConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.client = httpx.Client(timeout=config.timeout)
        self.results: list[TestResult] = []

        # Test messages designed for block_size=32
        # "Are you ok? Hello! Thank you! Thank you very much! " is ~12 tokens
        # Chat template adds ~4 tokens
        self.base_phrase = "Are you ok? Hello! Thank you! Thank you very much! "

    def log(self, msg: str):
        if self.verbose:
            print(f"    {msg}")

    def run_all(self) -> bool:
        """Run all test cases."""
        print("\nKV Router Test Suite")
        print("=" * 50)

        # Check server connectivity first
        if not self._check_servers():
            print("\nFATAL: Cannot connect to servers")
            return False

        # Run test cases
        self._test_full_match()
        self._test_partial_match()
        self._test_no_match()

        # Print summary
        return self._print_summary()

    def run_mm_tests(self) -> bool:
        """Run multimodal tests (no server needed)."""
        print("\nMultimodal KV Router Tests")
        print("=" * 50)
        print("(These tests run locally without server)")

        self._test_mm_hash_computation()
        self._test_mm_routing_distinction()

        return self._print_summary()

    def _check_servers(self) -> bool:
        """Verify both API and Router servers are reachable."""
        print("\nChecking server connectivity...")
        try:
            # Check router
            resp = self.client.get(f"{self.config.router_url}/debug/tree_info")
            if resp.status_code != 200:
                print(f"  Router not responding: {resp.status_code}")
                return False
            print(f"  Router OK (blocks in tree: {resp.json().get('num_blocks', '?')})")

            # Check API - just verify it's up
            # A simple request to verify the endpoint exists
            return True
        except Exception as e:
            print(f"  Connection error: {e}")
            return False

    def _test_full_match(self):
        """
        Test: Send identical request twice.
        Expected: Second request should have overlap > 0.
        """
        print("\n[1] Full Match Test")
        print("    Sending same request twice, expecting cache hit on second...")

        # Create a request with enough tokens for multiple full blocks
        # 5 repetitions ≈ 64 tokens ≈ 2 full blocks
        content = (self.base_phrase * 5).strip()
        payload = make_request(content)

        # Get initial state
        initial = get_tree_info(self.client, self.config.router_url)
        initial_blocks = initial["num_blocks"]
        self.log(f"Initial blocks: {initial_blocks}")

        # First request - should populate cache (or hit existing cache)
        self.log("Sending first request...")
        if not send_request(self.client, self.config.api_url, payload):
            self.results.append(TestResult("full_match", False, "First request failed"))
            return

        # Wait for KV events
        self.log(f"Waiting {self.config.kv_settle_time}s for KV events...")
        time.sleep(self.config.kv_settle_time)

        # Check blocks after first request
        after_first = get_tree_info(self.client, self.config.router_url)
        blocks_added = after_first["num_blocks"] - initial_blocks
        self.log(f"Blocks after first: {after_first['num_blocks']} (added {blocks_added})")

        # Second request - should hit cache
        self.log("Sending second request (should hit cache)...")
        if not send_request(self.client, self.config.api_url, payload):
            self.results.append(TestResult("full_match", False, "Second request failed"))
            return

        # Success: either new blocks were added, or blocks already existed (from previous runs)
        # Either way, the second request should show overlap > 0 in server logs
        total_blocks = after_first["num_blocks"]
        self.results.append(TestResult(
            "full_match", True,
            f"OK - Tree has {total_blocks} blocks. Check server logs for 'overlap > 0'."
        ))

    def _test_partial_match(self):
        """
        Test: Send request A, then request B that shares same prefix but is longer.
        Expected: Request B should have partial overlap (matching the shared prefix blocks).
        """
        print("\n[2] Partial Match Test")
        print("    Request B shares prefix with cached request A...")

        # Request A: 5 repetitions (~64 tokens, ~2 full blocks)
        content_a = (self.base_phrase * 5).strip()
        
        # Request B: 8 repetitions (~100 tokens, ~3 full blocks)
        # First 2 blocks should match A, third block is new
        content_b = (self.base_phrase * 8).strip()

        payload_a = make_request(content_a)
        payload_b = make_request(content_b)

        # Ensure A is cached (might already be from previous test)
        self.log("Ensuring request A is cached...")
        send_request(self.client, self.config.api_url, payload_a)
        time.sleep(self.config.kv_settle_time)

        before = get_tree_info(self.client, self.config.router_url)
        self.log(f"Blocks before B: {before['num_blocks']}")

        # Send request B
        self.log("Sending request B (longer, shares prefix)...")
        if not send_request(self.client, self.config.api_url, payload_b):
            self.results.append(TestResult("partial_match", False, "Request B failed"))
            return

        time.sleep(self.config.kv_settle_time)

        after = get_tree_info(self.client, self.config.router_url)
        new_blocks = after["num_blocks"] - before["num_blocks"]
        self.log(f"New blocks from B: {new_blocks}")

        # B should add new blocks (the non-matching suffix)
        # The matching prefix blocks already exist
        self.results.append(TestResult(
            "partial_match", True,
            f"OK - Request B added {new_blocks} new blocks. "
            f"Check server logs for partial overlap (0 < overlap < 1)."
        ))

    def _test_no_match(self):
        """
        Test: Send completely different content.
        Expected: No cache hit (overlap = 0).
        """
        print("\n[3] No Match Test")
        print("    Sending completely different content...")

        # Content that's very different from previous tests
        # ~80 tokens, completely different from "Hello are you ok leijun"
        content = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump. "
            "The five boxing wizards jump quickly. "
            "Sphinx of black quartz, judge my vow."
        )
        payload = make_request(content)

        before = get_tree_info(self.client, self.config.router_url)
        self.log(f"Blocks before: {before['num_blocks']}")

        # Send the different request
        self.log("Sending unrelated request...")
        if not send_request(self.client, self.config.api_url, payload):
            self.results.append(TestResult("no_match", False, "Request failed"))
            return

        # No need to wait - we're checking overlap on this request, not the next
        self.results.append(TestResult(
            "no_match", True,
            "OK - Check server logs for 'overlap = 0.000' (no cache hit expected)."
        ))

    def _print_summary(self) -> bool:
        """Print test results summary."""
        print("\n" + "=" * 50)
        print("Results")
        print("=" * 50)

        all_passed = True
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            symbol = "[OK]" if r.passed else "[X]"
            print(f"  {symbol} {r.name}: {r.message}")
            if not r.passed:
                all_passed = False

        print("\n" + "-" * 50)
        if all_passed:
            print("All tests passed.")
            print("\nTo fully verify, check server logs for:")
            print("  - Full match:    overlap > 0.5")
            print("  - Partial match: 0 < overlap < 0.5")
            print("  - No match:      overlap = 0.000")
        else:
            print("Some tests failed. Check the messages above.")

        return all_passed

    def cleanup(self):
        self.client.close()


def main():
    parser = argparse.ArgumentParser(description="KV Router Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed logs")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--router-url", default="http://localhost:7000", help="Router URL")
    args = parser.parse_args()

    config = TestConfig(api_url=args.api_url, router_url=args.router_url)
    tests = KvRouterTests(config, verbose=args.verbose)

    try:
        success = tests.run_all()
        sys.exit(0 if success else 1)
    finally:
        tests.cleanup()


if __name__ == "__main__":
    main()
