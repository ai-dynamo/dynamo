# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test cases for MM-aware KV routing.

Tests various scenarios:
1. Same images - should have high overlap
2. Swapped image order - should have partial/no overlap (different mm_hash positions)
3. Different images - should have no overlap
4. Different number of images - should have partial overlap
5. Text-only requests - baseline
"""

import argparse
import json
import time
import requests

# Test images (using stable public URLs from picsum.photos - seeded for consistency)
# Different sizes to test various token counts
IMAGES = {
    "cat": "https://picsum.photos/seed/cat123/200/200",      # Small square
    "dog": "https://picsum.photos/seed/dog456/400/300",      # Medium landscape
    "bird": "https://picsum.photos/seed/bird789/300/500",    # Tall portrait
    "flower": "https://picsum.photos/seed/flower012/600/400", # Large landscape
}


def make_request(base_url: str, images: list[str], text: str = "Describe the image(s) briefly.") -> dict:
    """Make a chat completion request with images."""
    content = []

    # Add images first
    for img_url in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": img_url}
        })

    # Add text
    content.append({
        "type": "text",
        "text": text
    })

    payload = {
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 50,
        "stream": False
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120
        )
        if response.status_code != 200:
            print(f"  ERROR: HTTP {response.status_code}: {response.text[:200]}")
            return {"error": response.text}
        return response.json()
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"error": str(e)}


def make_text_request(base_url: str, text: str) -> dict:
    """Make a text-only chat completion request."""
    payload = {
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 50,
        "stream": False
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        if response.status_code != 200:
            print(f"  ERROR: HTTP {response.status_code}: {response.text[:200]}")
            return {"error": response.text}
        return response.json()
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"error": str(e)}


def run_test(name: str, base_url: str, delay: float = 2.0):
    """Run a named test with delay between requests."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)
    time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Test MM-aware KV routing")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for API")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between tests (seconds)")
    args = parser.parse_args()

    base_url = args.base_url
    delay = args.delay

    print("="*60)
    print("MM-AWARE KV ROUTING TEST SUITE")
    print("="*60)
    print(f"Base URL: {base_url}")
    print(f"Delay between requests: {delay}s")
    print("\nWatch the mm_router logs for [ROUTING] messages to see overlap scores.\n")

    # Test 1: Same single image twice
    run_test("1. Same single image twice (expect HIGH overlap on 2nd request)", base_url, delay)
    print("Request 1a: cat image")
    make_request(base_url, [IMAGES["cat"]])
    time.sleep(delay)
    print("Request 1b: cat image again")
    make_request(base_url, [IMAGES["cat"]])

    # Test 2: Different single image
    run_test("2. Different single image (expect LOW/NO overlap)", base_url, delay)
    print("Request 2: dog image (different from cat)")
    make_request(base_url, [IMAGES["dog"]])

    # Test 3: Same two images twice
    run_test("3. Same two images twice (expect HIGH overlap on 2nd request)", base_url, delay)
    print("Request 3a: cat + dog")
    make_request(base_url, [IMAGES["cat"], IMAGES["dog"]])
    time.sleep(delay)
    print("Request 3b: cat + dog again")
    make_request(base_url, [IMAGES["cat"], IMAGES["dog"]])

    # Test 4: Swapped image order
    run_test("4. Swapped image order (expect PARTIAL overlap - text prefix matches)", base_url, delay)
    print("Request 4: dog + cat (swapped order)")
    make_request(base_url, [IMAGES["dog"], IMAGES["cat"]])

    # Test 5: Subset of images
    run_test("5. Subset of images (expect PARTIAL overlap - first image matches)", base_url, delay)
    print("Request 5: cat only (subset of cat+dog)")
    make_request(base_url, [IMAGES["cat"]])

    # Test 6: Superset of images
    run_test("6. Superset of images (expect PARTIAL overlap - prefix matches)", base_url, delay)
    print("Request 6: cat + dog + bird (superset)")
    make_request(base_url, [IMAGES["cat"], IMAGES["dog"], IMAGES["bird"]])

    # Test 7: Completely different images
    run_test("7. Completely different images (expect LOW/NO overlap)", base_url, delay)
    print("Request 7: bird + flower (completely different)")
    make_request(base_url, [IMAGES["bird"], IMAGES["flower"]])

    # Test 8: Same image different text
    run_test("8. Same image, different text (expect HIGH overlap - image tokens dominate)", base_url, delay)
    print("Request 8a: cat with text 'What is this?'")
    make_request(base_url, [IMAGES["cat"]], text="What is this?")
    time.sleep(delay)
    print("Request 8b: cat with text 'Describe this animal in detail.'")
    make_request(base_url, [IMAGES["cat"]], text="Describe this animal in detail.")

    # Test 9: Same three images twice
    run_test("9. Same three images twice (expect HIGH overlap)", base_url, delay)
    print("Request 9a: cat + dog + bird")
    make_request(base_url, [IMAGES["cat"], IMAGES["dog"], IMAGES["bird"]])
    time.sleep(delay)
    print("Request 9b: cat + dog + bird again")
    make_request(base_url, [IMAGES["cat"], IMAGES["dog"], IMAGES["bird"]])

    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)
    print("\nCheck the mm_router logs above for [ROUTING] messages.")
    print("Expected results:")
    print("  - Same images: HIGH overlap (90%+)")
    print("  - Swapped order: PARTIAL overlap (text prefix only)")
    print("  - Different images: LOW/NO overlap")
    print("  - Subset/superset: PARTIAL overlap")


if __name__ == "__main__":
    main()
