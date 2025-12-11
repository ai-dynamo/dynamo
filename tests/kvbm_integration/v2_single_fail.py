#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Isolated failing test cases for KVBM V2 determinism.

This test isolates the 7 specific failing Shakespeare indices from v2_standalone.py:
- Shakespeare[51], [81], [85], [87], [89], [93], [95]

The failure pattern:
- These indices fail after cache reset when 100 prompts have been warmed up
- Using MAX_TOKENS=24 produces the most stable/reproducible failures

This test:
1. Warms up 100 Shakespeare prompts (to fill cache like the original test)
2. Gets responses for the 7 target indices
3. Resets cache
4. Gets responses for the same 7 indices (no warmup)
5. Compares - should match but doesn't for some indices

Usage:
    python tests/kvbm_integration/v2_single_fail.py
"""

import os
import sys
from pathlib import Path

import requests


BASE_URL = os.environ.get("KVBM_SERVER_URL", "http://localhost:8000")
MODEL_ID = os.environ.get("KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
MAX_TOKENS = int(os.environ.get("KVBM_MAX_TOKENS", "48"))
WORD_COUNT = 200

# The failing indices from v2_standalone.py with KVBM_MAX_TOKENS=24
FAILING_INDICES = [51, 81, 85, 87, 89, 93, 95]
#FAILING_INDICES = [51]

# Warmup 100 prompts (matches original test)
NUM_CACHE_PROMPTS = 100


def download_shakespeare():
    """Download Shakespeare text if needed."""
    shakespeare_file = Path("t8.shakespeare.txt")
    if not shakespeare_file.exists():
        print("Downloading Shakespeare text...")
        import urllib.request
        url = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"
        urllib.request.urlretrieve(url, shakespeare_file)
    return shakespeare_file


def get_shakespeare_content(shakespeare_file: Path, idx: int) -> str:
    """Get Shakespeare content for index (each index is WORD_COUNT words apart)."""
    with open(shakespeare_file, "r", encoding="utf-8") as f:
        words = f.read().split()
    start = idx * WORD_COUNT
    end = min(start + WORD_COUNT, len(words))
    return " ".join(words[start:end])


def make_request(content: str) -> str:
    """Make a request and return the response content."""
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "temperature": 0.0,
        "seed": 42,
        "max_completion_tokens": MAX_TOKENS,
        "top_p": 0.0001,
    }
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def reset_cache():
    """Reset the prefix cache."""
    resp = requests.post(f"{BASE_URL}/reset_prefix_cache", timeout=30)
    resp.raise_for_status()


def check_health() -> bool:
    """Check if server is healthy."""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def main():
    print("=" * 70)
    print("ISOLATED FAILING TEST - 7 Shakespeare Indices")
    print("=" * 70)
    print(f"Server:     {BASE_URL}")
    print(f"Model:      {MODEL_ID}")
    print(f"Max tokens: {MAX_TOKENS}")
    
    if not check_health():
        print("ERROR: Server not healthy")
        return 1
    
    shakespeare_file = download_shakespeare()
    
    # Get all cache prompts (100 Shakespeare sequences)
    print(f"Loading {NUM_CACHE_PROMPTS} Shakespeare prompts...")
    cache_prompts = [get_shakespeare_content(shakespeare_file, i) for i in range(NUM_CACHE_PROMPTS)]


    FAILING_INDICES = [i for i in range(NUM_CACHE_PROMPTS)]

    print(f"Targets:    Shakespeare{FAILING_INDICES}")
    print()

    # Get the target prompts we're testing
    target_prompts = {idx: get_shakespeare_content(shakespeare_file, idx) for idx in FAILING_INDICES}
    
    # Start fresh
    print("\nStep 1: Reset cache")
    reset_cache()
    
    # Warmup ALL 100 prompts (fill the cache like the original test)
    print(f"Step 2: Warmup {NUM_CACHE_PROMPTS} prompts (fill cache)")
    for i, p in enumerate(cache_prompts):
        make_request(p)
        if (i + 1) % 20 == 0:
            print(f"  Warmup {i + 1}/{NUM_CACHE_PROMPTS}")
    
    # Get responses for target indices (with full cache)
    print(f"\nStep 3: Get responses for {len(FAILING_INDICES)} target indices (cache full)")
    phase1_responses = {}
    for i, idx in enumerate(FAILING_INDICES):
        response = make_request(target_prompts[idx])
        phase1_responses[idx] = response
        # print(f"  Shakespeare[{idx}]: {response[:50]}...")
        if (i + 1) % 20 == 0:
            print(f"  G1 {i + 1}/{len(FAILING_INDICES)}")
    
    # Reset cache
    print("\nStep 4: Reset cache")
    reset_cache()
    
    # Get responses again (cache empty, no warmup)
    print(f"\nStep 5: Get responses again (cache empty, NO warmup)")
    phase2_responses = {}
    for i, idx in enumerate(FAILING_INDICES):
        response = make_request(target_prompts[idx])
        phase2_responses[idx] = response
        # print(f"  Shakespeare[{idx}]: {response[:50]}...")
        if (i + 1) % 20 == 0:
            print(f"  G2 {i + 1}/{len(FAILING_INDICES)}")
    
    # Compare
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    passed = []
    failed = []
    
    for idx in FAILING_INDICES:
        if phase1_responses[idx] == phase2_responses[idx]:
            passed.append(idx)
            print(f"  Shakespeare[{idx}]: ✓ MATCH")
        else:
            failed.append(idx)
            print(f"  Shakespeare[{idx}]: ✗ MISMATCH")
    
    print()
    print(f"Passed: {len(passed)}/{len(FAILING_INDICES)} {passed}")
    print(f"Failed: {len(failed)}/{len(FAILING_INDICES)} {failed}")
    
    if failed:
        print()
        print("=" * 70)
        print("MISMATCH DETAILS")
        print("=" * 70)
        for idx in failed:
            print(f"\nShakespeare[{idx}]:")
            print(f"  Phase 1 (cache full):")
            print(f"    {phase1_responses[idx]}")
            print(f"  Phase 2 (cache empty):")
            print(f"    {phase2_responses[idx]}")
    
    if len(failed) > 0:
        print(f"\n✗ FAIL: {len(failed)} indices failed")
        return 1
    else:
        print("\n✓ PASS: All responses match!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
