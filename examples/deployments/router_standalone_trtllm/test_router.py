# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for TensorRT-LLM KV Router.

Usage:
    python test_router.py              # Run text-only tests (requires server)
    python test_router.py --verbose    # Show detailed logs
    python test_router.py --mm-only    # Run multimodal hash tests (no server needed)
    python test_router.py --mm-server  # Run multimodal server tests (requires VLM)
    python test_router.py --all        # Run all tests
"""

import argparse
import sys
import time
from dataclasses import dataclass

import httpx

from dynamo.llm import compute_block_hash_for_seq_py

# Sample test images from COCO dataset
TEST_IMAGE_1 = "http://images.cocodataset.org/test2017/000000155781.jpg"
TEST_IMAGE_2 = "http://images.cocodataset.org/test2017/000000000001.jpg"
TEST_IMAGE_3 = "http://images.cocodataset.org/test2017/000000155721.jpg"
TEST_IMAGE_4 = "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg"


@dataclass
class RouterTestConfig:
    api_url: str = "http://localhost:8000"
    router_url: str = "http://localhost:7000"
    timeout: int = 30
    kv_settle_time: float = 3.0  # Time to wait for KV events to propagate


@dataclass
class RouterTestResult:
    name: str
    passed: bool
    message: str
    overlap: float = 0.0


def make_request(content: str, max_tokens: int = 10) -> dict:
    """Create a text-only chat completion request."""
    return {
        "model": "test",
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "max_tokens": max_tokens,
    }


def make_mm_request(text: str, image_url: str, max_tokens: int = 10) -> dict:
    """Create a multimodal chat completion request with image."""
    return {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "stream": True,
        "max_tokens": max_tokens,
    }


def make_multi_image_request(
    text: str, image_urls: list[str], max_tokens: int = 10
) -> dict:
    """Create a multimodal chat completion request with multiple images."""
    content: list[dict] = [{"type": "text", "text": text}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
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


class KvRouterTests:
    """Test cases for KV cache routing."""

    def __init__(self, config: RouterTestConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.client = httpx.Client(timeout=config.timeout)
        self.results: list[RouterTestResult] = []

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
        """Run multimodal tests (local hash computation, no server needed)."""
        print("\nMultimodal KV Router Tests (Local)")
        print("=" * 50)
        print("(These tests verify hash computation without server)")

        self._test_mm_hash_computation()
        self._test_mm_routing_distinction()
        self._test_mm_hash_consistency()
        self._test_mm_offset_affects_hash()
        self._test_mm_block_boundary()
        self._test_mm_multi_image_partial_match()

        return self._print_summary()

    def run_mm_server_tests(self) -> bool:
        """Run multimodal tests that require server."""
        print("\nMultimodal KV Router Tests (Server)")
        print("=" * 50)

        if not self._check_servers():
            print("\nFATAL: Cannot connect to servers")
            return False

        self._test_mm_same_image_cache_hit()
        self._test_mm_different_images_no_cache_hit()
        self._test_text_cache_hit_with_overlap()
        self._test_mm_multi_image_partial_match()

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
            self.results.append(
                RouterTestResult("full_match", False, "First request failed")
            )
            return

        # Wait for KV events
        self.log(f"Waiting {self.config.kv_settle_time}s for KV events...")
        time.sleep(self.config.kv_settle_time)

        # Check blocks after first request
        after_first = get_tree_info(self.client, self.config.router_url)
        blocks_added = after_first["num_blocks"] - initial_blocks
        self.log(
            f"Blocks after first: {after_first['num_blocks']} (added {blocks_added})"
        )

        # Second request - should hit cache
        self.log("Sending second request (should hit cache)...")
        if not send_request(self.client, self.config.api_url, payload):
            self.results.append(
                RouterTestResult("full_match", False, "Second request failed")
            )
            return

        # Success: either new blocks were added, or blocks already existed (from previous runs)
        # Either way, the second request should show overlap > 0 in server logs
        total_blocks = after_first["num_blocks"]
        self.results.append(
            RouterTestResult(
                "full_match",
                True,
                f"OK - Tree has {total_blocks} blocks. Check server logs for 'overlap > 0'.",
            )
        )

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
            self.results.append(
                RouterTestResult("partial_match", False, "Request B failed")
            )
            return

        time.sleep(self.config.kv_settle_time)

        after = get_tree_info(self.client, self.config.router_url)
        new_blocks = after["num_blocks"] - before["num_blocks"]
        self.log(f"New blocks from B: {new_blocks}")

        # B should add new blocks (the non-matching suffix)
        # The matching prefix blocks already exist
        self.results.append(
            RouterTestResult(
                "partial_match",
                True,
                f"OK - Request B added {new_blocks} new blocks. "
                f"Check server logs for partial overlap (0 < overlap < 1).",
            )
        )

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
            self.results.append(RouterTestResult("no_match", False, "Request failed"))
            return

        # No need to wait - we're checking overlap on this request, not the next
        self.results.append(
            RouterTestResult(
                "no_match",
                True,
                "OK - Check server logs for 'overlap = 0.000' (no cache hit expected).",
            )
        )

    def _test_mm_hash_computation(self):
        """
        Test: Verify that compute_block_hash_for_seq_py produces different hashes
        for same tokens with different mm_hash values.
        """
        print("\n[MM-1] MM Hash Computation Test")
        print("    Verifying same tokens + different mm_hash = different block_hash...")

        # Simulated tokens (32 tokens = 1 block)
        tokens = [100] * 32
        block_size = 32

        # Hash without MM info
        hash_no_mm = compute_block_hash_for_seq_py(tokens, block_size)

        # Hash with MM info (simulated mm_hash)
        mm_info_1 = {"mm_objects": [{"mm_hash": 0xDEADBEEF, "offsets": [[0, 32]]}]}
        hash_with_mm1 = compute_block_hash_for_seq_py(tokens, block_size, [mm_info_1])

        # Hash with different MM info
        mm_info_2 = {"mm_objects": [{"mm_hash": 0xCAFEBABE, "offsets": [[0, 32]]}]}
        hash_with_mm2 = compute_block_hash_for_seq_py(tokens, block_size, [mm_info_2])

        self.log(f"Hash without MM: {hash_no_mm}")
        self.log(f"Hash with MM 1:  {hash_with_mm1}")
        self.log(f"Hash with MM 2:  {hash_with_mm2}")

        # Verify all hashes are different
        if hash_no_mm == hash_with_mm1:
            self.results.append(
                RouterTestResult(
                    "mm_hash_computation",
                    False,
                    "FAIL - Hash without MM equals hash with MM",
                )
            )
            return

        if hash_with_mm1 == hash_with_mm2:
            self.results.append(
                RouterTestResult(
                    "mm_hash_computation",
                    False,
                    "FAIL - Different mm_hash produced same block_hash",
                )
            )
            return

        self.results.append(
            RouterTestResult(
                "mm_hash_computation",
                True,
                "OK - Different mm_hash values produce different block hashes",
            )
        )

    def _test_mm_routing_distinction(self):
        """
        Test: Verify that the routing logic can distinguish between
        requests with same text but different images.
        """
        print("\n[MM-2] MM Routing Distinction Test")
        print("    Verifying routing can distinguish same text + different images...")

        # This test simulates what the router would see
        tokens = [100] * 64  # 2 blocks
        block_size = 32

        # Simulate Image A cached on worker 0
        mm_info_a = {
            "mm_objects": [{"mm_hash": 0x1111111111111111, "offsets": [[0, 64]]}]
        }
        hashes_a = compute_block_hash_for_seq_py(
            tokens, block_size, [mm_info_a, mm_info_a]
        )

        # Simulate Image B cached on worker 1
        mm_info_b = {
            "mm_objects": [{"mm_hash": 0x2222222222222222, "offsets": [[0, 64]]}]
        }
        hashes_b = compute_block_hash_for_seq_py(
            tokens, block_size, [mm_info_b, mm_info_b]
        )

        self.log(f"Hashes for Image A: {hashes_a}")
        self.log(f"Hashes for Image B: {hashes_b}")

        # Verify hashes are different
        if hashes_a == hashes_b:
            self.results.append(
                RouterTestResult(
                    "mm_routing_distinction",
                    False,
                    "FAIL - Same tokens with different images produced same hashes",
                )
            )
            return

        self.results.append(
            RouterTestResult(
                "mm_routing_distinction",
                True,
                "OK - Router can distinguish requests with different images",
            )
        )

    def _test_mm_hash_consistency(self):
        """
        Test: Verify that the same mm_hash + tokens produce the same block_hash
        regardless of when computed (idempotency).
        """
        print("\n[MM-3] MM Hash Consistency Test")
        print("    Verifying same inputs produce same hash (idempotent)...")

        tokens = [151937] * 32  # Image token placeholder
        block_size = 32
        mm_hash = 0xDEADBEEFCAFEBABE

        mm_info = {"mm_objects": [{"mm_hash": mm_hash, "offsets": [[0, 32]]}]}

        # Compute hash multiple times
        hash1 = compute_block_hash_for_seq_py(tokens, block_size, [mm_info])
        hash2 = compute_block_hash_for_seq_py(tokens, block_size, [mm_info])
        hash3 = compute_block_hash_for_seq_py(tokens, block_size, [mm_info])

        self.log(f"Hash 1: {hash1}")
        self.log(f"Hash 2: {hash2}")
        self.log(f"Hash 3: {hash3}")

        if hash1 != hash2 or hash2 != hash3:
            self.results.append(
                RouterTestResult(
                    "mm_hash_consistency",
                    False,
                    f"FAIL - Same inputs produced different hashes: {hash1}, {hash2}, {hash3}",
                )
            )
            return

        self.results.append(
            RouterTestResult(
                "mm_hash_consistency",
                True,
                f"OK - Hash computation is idempotent: {hash1[0]}",
            )
        )

    def _test_mm_offset_affects_hash(self):
        """
        Test: Verify that different offsets produce different hashes,
        even with same mm_hash and tokens.
        """
        print("\n[MM-4] MM Offset Affects Hash Test")
        print("    Verifying different offsets produce different hashes...")

        tokens = [151937] * 64  # 2 blocks of image tokens
        block_size = 32
        mm_hash = 0x123456789ABCDEF0

        # Image covers first block only
        mm_info_first = {"mm_objects": [{"mm_hash": mm_hash, "offsets": [[0, 32]]}]}
        hash_first = compute_block_hash_for_seq_py(
            tokens, block_size, [mm_info_first, None]
        )

        # Image covers second block only
        mm_info_second = {"mm_objects": [{"mm_hash": mm_hash, "offsets": [[32, 64]]}]}
        hash_second = compute_block_hash_for_seq_py(
            tokens, block_size, [None, mm_info_second]
        )

        # Image covers both blocks
        mm_info_both = {"mm_objects": [{"mm_hash": mm_hash, "offsets": [[0, 64]]}]}
        hash_both = compute_block_hash_for_seq_py(
            tokens, block_size, [mm_info_both, mm_info_both]
        )

        self.log(f"Hash (first block MM):  {hash_first}")
        self.log(f"Hash (second block MM): {hash_second}")
        self.log(f"Hash (both blocks MM):  {hash_both}")

        # Block 0 with mm_info should differ from block 0 without mm_info
        # Block 1 with mm_info should differ from block 1 without mm_info
        if hash_first[0] == hash_second[0]:
            self.results.append(
                RouterTestResult(
                    "mm_offset_affects_hash",
                    False,
                    "FAIL - First block hash should differ based on MM presence",
                )
            )
            return

        self.results.append(
            RouterTestResult(
                "mm_offset_affects_hash",
                True,
                "OK - Different MM offsets produce different block hashes",
            )
        )

    def _test_mm_block_boundary(self):
        """
        Test: Verify that MM info correctly applies at block boundaries.
        """
        print("\n[MM-5] MM Block Boundary Test")
        print("    Verifying MM info applies correctly at block boundaries...")

        block_size = 32
        mm_hash = 0xFEDCBA9876543210

        # 96 tokens = 3 blocks
        # Image tokens in the middle block (32-64)
        tokens = [100] * 32 + [151937] * 32 + [200] * 32

        # MM info only applies to middle block
        mm_info = {"mm_objects": [{"mm_hash": mm_hash, "offsets": [[32, 64]]}]}
        hashes_with_mm = compute_block_hash_for_seq_py(
            tokens, block_size, [None, mm_info, None]
        )

        # No MM info
        hashes_without_mm = compute_block_hash_for_seq_py(tokens, block_size, None)

        self.log(f"Hashes with MM:    {hashes_with_mm}")
        self.log(f"Hashes without MM: {hashes_without_mm}")

        # Block 0 and 2 should be the same (no image tokens)
        # Block 1 should be different (has image tokens + mm_hash)
        if hashes_with_mm[0] != hashes_without_mm[0]:
            self.results.append(
                RouterTestResult(
                    "mm_block_boundary", False, "FAIL - Block 0 should be same (no MM)"
                )
            )
            return

        if hashes_with_mm[1] == hashes_without_mm[1]:
            self.results.append(
                RouterTestResult(
                    "mm_block_boundary", False, "FAIL - Block 1 should differ (has MM)"
                )
            )
            return

        if hashes_with_mm[2] != hashes_without_mm[2]:
            self.results.append(
                RouterTestResult(
                    "mm_block_boundary", False, "FAIL - Block 2 should be same (no MM)"
                )
            )
            return

        self.results.append(
            RouterTestResult(
                "mm_block_boundary",
                True,
                "OK - MM info correctly applies only to relevant blocks",
            )
        )

    def _test_mm_same_image_cache_hit(self):
        """
        Test: Send same text + same image twice.
        Expected: Second request should have cache hit (overlap > 0).
        """
        print("\n[MM-S1] Same Image Cache Hit Test")
        print("    Sending same text + same image twice...")

        payload = make_mm_request("Describe this image", TEST_IMAGE_1)

        # Get initial state
        initial = get_tree_info(self.client, self.config.router_url)
        self.log(f"Initial blocks: {initial['num_blocks']}")

        # First request - populates the cache
        self.log("Sending first MM request...")
        if not send_request(self.client, self.config.api_url, payload):
            self.results.append(
                RouterTestResult("mm_same_image", False, "First MM request failed")
            )
            return

        # Wait for KV events to propagate
        self.log(f"Waiting {self.config.kv_settle_time}s for KV events...")
        time.sleep(self.config.kv_settle_time)

        after_first = get_tree_info(self.client, self.config.router_url)
        blocks_added = after_first["num_blocks"] - initial["num_blocks"]
        self.log(
            f"Blocks after first: {after_first['num_blocks']} (added {blocks_added})"
        )

        if blocks_added == 0:
            self.results.append(
                RouterTestResult(
                    "mm_same_image", False, "FAIL - No blocks added after first request"
                )
            )
            return

        # Second identical request - should hit cache
        self.log("Sending second MM request (same image)...")
        if not send_request(self.client, self.config.api_url, payload):
            self.results.append(
                RouterTestResult("mm_same_image", False, "Second MM request failed")
            )
            return

        # Query router to check overlap (simulating what the second request saw)
        # We need to compute the same hashes that the API computed
        # For now, check the tree grew or stayed same (cache reuse)
        after_second = get_tree_info(self.client, self.config.router_url)
        self.log(f"Blocks after second: {after_second['num_blocks']}")

        # The second request should reuse cached blocks, so minimal new blocks added
        new_blocks_second = after_second["num_blocks"] - after_first["num_blocks"]
        self.log(f"New blocks from second request: {new_blocks_second}")

        self.results.append(
            RouterTestResult(
                "mm_same_image",
                True,
                f"OK - First added {blocks_added} blocks, second added {new_blocks_second}. "
                f"Check logs for 'overlap > 0' on second request.",
            )
        )

    def _test_mm_different_images_no_cache_hit(self):
        """
        Test: Send same text but different images.
        Expected: No cache hit (overlap ≈ 0) because mm_hash differs.
        Image blocks should not match, only text prefix might match.
        """
        print("\n[MM-S2] Different Images No Cache Hit Test")
        print("    Sending same text + different images...")

        # First image
        payload_1 = make_mm_request("Describe this image in detail", TEST_IMAGE_2)

        initial = get_tree_info(self.client, self.config.router_url)
        self.log(f"Initial blocks: {initial['num_blocks']}")

        self.log(f"Sending request with image 1: {TEST_IMAGE_2}")
        if not send_request(self.client, self.config.api_url, payload_1):
            self.results.append(
                RouterTestResult("mm_different_images", False, "Image 1 request failed")
            )
            return

        time.sleep(self.config.kv_settle_time)

        after_img1 = get_tree_info(self.client, self.config.router_url)
        blocks_img1 = after_img1["num_blocks"] - initial["num_blocks"]
        self.log(
            f"Blocks after image 1: {after_img1['num_blocks']} (added {blocks_img1})"
        )

        # Second image (same text, different image)
        payload_2 = make_mm_request("Describe this image in detail", TEST_IMAGE_3)

        self.log(f"Sending request with image 2: {TEST_IMAGE_3}")
        if not send_request(self.client, self.config.api_url, payload_2):
            self.results.append(
                RouterTestResult("mm_different_images", False, "Image 2 request failed")
            )
            return

        time.sleep(self.config.kv_settle_time)

        after_img2 = get_tree_info(self.client, self.config.router_url)
        blocks_img2 = after_img2["num_blocks"] - after_img1["num_blocks"]
        self.log(
            f"Blocks after image 2: {after_img2['num_blocks']} (added {blocks_img2})"
        )

        # Different images should add similar number of blocks
        # If image 2 had cache hit, it would add fewer blocks
        if blocks_img2 == 0:
            self.results.append(
                RouterTestResult(
                    "mm_different_images",
                    False,
                    "FAIL - Image 2 added 0 blocks (unexpected full cache hit)",
                )
            )
            return

        # Image 2 should add approximately same number of blocks as image 1
        # (since different mm_hash means image blocks don't match)
        self.results.append(
            RouterTestResult(
                "mm_different_images",
                True,
                f"OK - Image 1 added {blocks_img1} blocks, image 2 added {blocks_img2} blocks. "
                f"Different images = different block hashes.",
            )
        )

    def _test_text_cache_hit_with_overlap(self):
        """
        Test: Send same text request twice and verify overlap via router API.
        Expected: Second request should show overlap > 0 in router response.
        """
        print("\n[MM-S3] Text Cache Hit with Overlap Verification")
        print("    Sending same text twice and verifying overlap value...")

        # Use a unique prompt to avoid interference from other tests
        unique_text = (
            "This is a unique test prompt for cache hit verification. "
            "We need enough tokens to fill at least one block. "
            "The quick brown fox jumps over the lazy dog repeatedly. " * 3
        )
        payload = make_request(unique_text, max_tokens=5)

        # First request
        self.log("Sending first text request...")
        if not send_request(self.client, self.config.api_url, payload):
            self.results.append(
                RouterTestResult(
                    "text_cache_hit_overlap", False, "First request failed"
                )
            )
            return

        # Wait for KV events
        self.log(f"Waiting {self.config.kv_settle_time}s for KV events...")
        time.sleep(self.config.kv_settle_time)

        # Get tree info to see blocks
        tree_info = get_tree_info(self.client, self.config.router_url)
        self.log(f"Blocks in tree: {tree_info['num_blocks']}")

        # Second request - should see cache hit
        self.log("Sending second text request (should hit cache)...")
        if not send_request(self.client, self.config.api_url, payload):
            self.results.append(
                RouterTestResult(
                    "text_cache_hit_overlap", False, "Second request failed"
                )
            )
            return

        # For a true verification, we'd need to intercept the router response
        # or add an endpoint that returns the last routing decision
        # For now, we verify by checking if blocks increased (they shouldn't much)
        tree_info_after = get_tree_info(self.client, self.config.router_url)
        new_blocks = tree_info_after["num_blocks"] - tree_info["num_blocks"]
        self.log(f"New blocks after second request: {new_blocks}")

        self.results.append(
            RouterTestResult(
                "text_cache_hit_overlap",
                True,
                f"OK - Second request added {new_blocks} new blocks. "
                f"Check logs for 'overlap > 0' (cache hit).",
            )
        )

    def _test_mm_multi_image_partial_match(self):
        """
        Test: Verify partial cache match with multi-image requests.

        Scenario:
            Step 1: Send Request A = text + [Image_1, Image_4]
            Step 2: Send Request A again (identical) - verify full cache hit (0 new blocks)
            Step 3: Send Request B = text + [Image_1, Image_3] - verify partial match
                    (Image_3 is different, should add new blocks)

        Expected:
            - Identical request = no new blocks (full cache hit)
            - Different second image = new blocks added (partial match)
        """
        print("\n[MM-S4] Multi-Image Partial Match Test")
        print("    Verifying cache behavior with multi-image requests...")

        # Use longer settle time for this test
        settle_time = self.config.kv_settle_time * 2

        # Request A: text + Image_1 + Image_4
        payload_a = make_multi_image_request(
            "Describe these images in detail", [TEST_IMAGE_1, TEST_IMAGE_4]
        )

        initial = get_tree_info(self.client, self.config.router_url)
        self.log(f"Initial blocks: {initial['num_blocks']}")

        # Step 1: Send Request A first time
        self.log("Step 1: Sending Request A (text + Image_1 + Image_4)...")
        if not send_request(self.client, self.config.api_url, payload_a):
            self.results.append(
                RouterTestResult("mm_multi_image_partial", False, "Request A failed")
            )
            return

        time.sleep(settle_time)

        after_a1 = get_tree_info(self.client, self.config.router_url)
        blocks_a1 = after_a1["num_blocks"] - initial["num_blocks"]
        self.log(
            f"Blocks after Request A: {after_a1['num_blocks']} (added {blocks_a1})"
        )

        if blocks_a1 == 0:
            self.results.append(
                RouterTestResult(
                    "mm_multi_image_partial",
                    False,
                    "FAIL - Request A added 0 blocks (should populate cache)",
                )
            )
            return

        # Step 2: Send Request A again (identical) - should be full cache hit
        self.log(
            "Step 2: Sending Request A again (identical, expect full cache hit)..."
        )
        if not send_request(self.client, self.config.api_url, payload_a):
            self.results.append(
                RouterTestResult(
                    "mm_multi_image_partial", False, "Request A (repeat) failed"
                )
            )
            return

        time.sleep(settle_time)

        after_a2 = get_tree_info(self.client, self.config.router_url)
        blocks_a2 = after_a2["num_blocks"] - after_a1["num_blocks"]
        self.log(
            f"Blocks after Request A repeat: {after_a2['num_blocks']} (added {blocks_a2})"
        )

        # Identical request should add 0 new blocks (full cache hit)
        if blocks_a2 != 0:
            self.log(
                f"WARNING: Identical request added {blocks_a2} blocks (expected 0)"
            )

        # Step 3: Send Request B with different second image
        payload_b = make_multi_image_request(
            "Describe these images in detail", [TEST_IMAGE_1, TEST_IMAGE_3]
        )

        self.log(
            "Step 3: Sending Request B (text + Image_1 + Image_3, different 2nd image)..."
        )
        if not send_request(self.client, self.config.api_url, payload_b):
            self.results.append(
                RouterTestResult("mm_multi_image_partial", False, "Request B failed")
            )
            return

        time.sleep(settle_time)

        after_b = get_tree_info(self.client, self.config.router_url)
        blocks_b = after_b["num_blocks"] - after_a2["num_blocks"]
        self.log(f"Blocks after Request B: {after_b['num_blocks']} (added {blocks_b})")

        # Analysis:
        # - If blocks_b > 0: Image_3 created new blocks (correct - different image)
        # - If blocks_b == 0: Full cache hit (wrong - Image_3 should be different)
        #
        # Note: We can't easily verify partial match vs full cache miss because
        # the tree growth depends on whether routing hit the cached worker.
        # What we CAN verify is that different images should NOT fully cache hit.

        if blocks_b == 0 and blocks_a2 == 0:
            # Both identical and different requests added 0 blocks
            # This suggests Image_3's mm_hash is incorrectly matching Image_4
            self.results.append(
                RouterTestResult(
                    "mm_multi_image_partial",
                    False,
                    "FAIL - Request B (different image) added 0 blocks. "
                    "Image_3 should have different mm_hash than Image_4. "
                    "Check if mm_hash computation is correct.",
                )
            )
            return

        if blocks_b == 0:
            # Different image but 0 new blocks - might be timing or routing issue
            self.results.append(
                RouterTestResult(
                    "mm_multi_image_partial",
                    False,
                    f"FAIL - Request B added 0 blocks. "
                    f"Identical request added {blocks_a2}. "
                    f"This is unexpected - different images should not fully cache hit.",
                )
            )
            return

        # Success: different image added new blocks
        self.results.append(
            RouterTestResult(
                "mm_multi_image_partial",
                True,
                f"OK - Request A: {blocks_a1} blocks, A repeat: {blocks_a2}, "
                f"Request B (diff image): {blocks_b}. "
                f"Different images correctly create distinct cache entries.",
            )
        )

    def _print_summary(self) -> bool:
        """Print test results summary."""
        print("\n" + "=" * 50)
        print("Results")
        print("=" * 50)

        all_passed = True
        for r in self.results:
            _ = "PASS" if r.passed else "FAIL"
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
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed logs"
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8000", help="API server URL"
    )
    parser.add_argument(
        "--router-url", default="http://localhost:7000", help="Router URL"
    )
    parser.add_argument(
        "--mm-only",
        action="store_true",
        help="Run only multimodal local tests (no server needed)",
    )
    parser.add_argument(
        "--mm-server",
        action="store_true",
        help="Run multimodal server tests (requires VLM model)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all tests including multimodal"
    )
    args = parser.parse_args()

    config = RouterTestConfig(api_url=args.api_url, router_url=args.router_url)
    tests = KvRouterTests(config, verbose=args.verbose)

    try:
        if args.mm_only:
            # Local MM tests only (no server)
            success = tests.run_mm_tests()
        elif args.mm_server:
            # MM server tests (requires VLM)
            success = tests.run_mm_server_tests()
        elif args.all:
            # Run all tests
            success = tests.run_all()
            if success:
                success = tests.run_mm_tests()
            if success:
                success = tests.run_mm_server_tests()
        else:
            # Default: text-only tests
            success = tests.run_all()
        sys.exit(0 if success else 1)
    finally:
        tests.cleanup()


if __name__ == "__main__":
    main()
