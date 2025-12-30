#!/usr/bin/env python3
"""Test script to reproduce DYN-1650: thinking_mode cannot be disabled in Dynamo+TRT-LLM"""

import argparse
import json
import time

import requests


def test_thinking_mode(
    base_url="http://0.0.0.0:8000", model_name="qwen", enable_thinking=False
):
    """Test thinking_mode with specified settings"""

    endpoint = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Please introduce Hangzhou in 10 words"},
        ],
        "chat_template_args": {"enable_thinking": enable_thinking},
        "max_tokens": 100,
        "temperature": 0,
    }

    print(f"\n{'='*60}")
    print(f"Testing with enable_thinking={enable_thinking}")
    print(f"URL: {endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print(f"{'='*60}")

    try:
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        print(f"\nResponse status: {response.status_code}")
        print(f"Full response: {json.dumps(result, indent=2)}")

        # Check if response contains thinking tags
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]

            # Check for thinking-related tags
            has_thinking = "<thinking>" in content
            has_reasoning = "<reasoning>" in content
            has_content_tags = "<content>" in content

            print("\n--- Analysis ---")
            print(f"Contains <thinking> tag: {has_thinking}")
            print(f"Contains <reasoning> tag: {has_reasoning}")
            print(f"Contains <content> tag: {has_content_tags}")
            print(f"Content: {content}")

            return {
                "enable_thinking": enable_thinking,
                "has_thinking": has_thinking,
                "has_reasoning": has_reasoning,
                "has_content_tags": has_content_tags,
                "content": content,
            }

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding response: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test thinking_mode in Dynamo/TRT-LLM")
    parser.add_argument(
        "--url", default="http://0.0.0.0:8000", help="Base URL for the API"
    )
    parser.add_argument("--model", default="qwen", help="Model name to use")
    parser.add_argument(
        "--framework",
        default="dynamo",
        help="Framework being tested (dynamo or trtllm)",
    )

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# Testing thinking_mode with {args.framework.upper()}")
    print(f"# Model: {args.model}")
    print(f"# URL: {args.url}")
    print(f"{'#'*60}")

    # Test with thinking_mode disabled
    result_disabled = test_thinking_mode(args.url, args.model, enable_thinking=False)

    time.sleep(2)  # Small delay between tests

    # Test with thinking_mode enabled
    result_enabled = test_thinking_mode(args.url, args.model, enable_thinking=True)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Framework: {args.framework.upper()}")

    if result_disabled:
        print("\nWith enable_thinking=False:")
        print(f"  - Has thinking tags: {result_disabled['has_thinking']}")
        print(f"  - Has reasoning tags: {result_disabled['has_reasoning']}")
        print(f"  - Has content tags: {result_disabled['has_content_tags']}")

    if result_enabled:
        print("\nWith enable_thinking=True:")
        print(f"  - Has thinking tags: {result_enabled['has_thinking']}")
        print(f"  - Has reasoning tags: {result_enabled['has_reasoning']}")
        print(f"  - Has content tags: {result_enabled['has_content_tags']}")

    # Check for the issue
    if result_disabled and (
        result_disabled["has_thinking"] or result_disabled["has_reasoning"]
    ):
        print("\n⚠️  ISSUE DETECTED: thinking_mode tags present even when disabled!")
        print(
            "This matches the DYN-1650 issue where thinking_mode cannot be properly disabled in Dynamo+TRT-LLM"
        )
    else:
        print(
            "\n✓ No issue detected: thinking_mode properly respects enable_thinking setting"
        )


if __name__ == "__main__":
    main()
