#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test script for Universal X-Request-Id Support in Dynamo

This script demonstrates the automatic X-Request-Id handling provided by
Dynamo SDK's built-in request tracing functionality.

Usage:
    python test_universal_tracing.py
"""

import asyncio
import uuid

import aiohttp


async def test_with_custom_request_id(base_url: str = "http://localhost:8080"):
    """Test with custom X-Request-Id header."""

    custom_request_id = f"test-universal-{uuid.uuid4()}"

    headers = {"Content-Type": "application/json", "X-Request-Id": custom_request_id}

    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [
            {"role": "user", "content": "Test universal request tracing with custom ID"}
        ],
        "max_tokens": 50,
        "stream": False,
    }

    print(f"🚀 Testing with custom X-Request-Id: {custom_request_id}")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/v1/chat/completions", headers=headers, json=payload
        ) as response:
            response_headers = dict(response.headers)
            await response.json()

            print(f"📥 Response status: {response.status}")
            returned_id = response_headers.get("X-Request-Id")
            print(f"📋 Returned X-Request-Id: {returned_id}")

            if returned_id == custom_request_id:
                print("✅ Custom request ID successfully preserved and echoed back!")
                return True
            else:
                print(
                    f"❌ Request ID mismatch. Expected: {custom_request_id}, Got: {returned_id}"
                )
                return False


async def test_without_request_id(base_url: str = "http://localhost:8080"):
    """Test without X-Request-Id header (should auto-generate)."""

    headers = {
        "Content-Type": "application/json"
        # No X-Request-Id header
    }

    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [
            {"role": "user", "content": "Test universal request tracing without ID"}
        ],
        "max_tokens": 50,
        "stream": False,
    }

    print("🚀 Testing without X-Request-Id header (should auto-generate)")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/v1/chat/completions", headers=headers, json=payload
        ) as response:
            response_headers = dict(response.headers)
            await response.json()

            print(f"📥 Response status: {response.status}")
            generated_id = response_headers.get("X-Request-Id")
            print(f"📋 Generated X-Request-Id: {generated_id}")

            if generated_id:
                try:
                    uuid.UUID(generated_id)
                    print("✅ Request ID successfully auto-generated and is valid UUID!")
                    return True
                except ValueError:
                    print("❌ Generated request ID is not valid UUID format")
                    return False
            else:
                print("❌ No request ID generated")
                return False


async def test_completions_endpoint(base_url: str = "http://localhost:8080"):
    """Test completions endpoint with X-Request-Id."""

    custom_request_id = f"completion-test-{uuid.uuid4()}"

    headers = {"Content-Type": "application/json", "X-Request-Id": custom_request_id}

    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "prompt": "Universal request tracing is",
        "max_tokens": 30,
        "stream": False,
    }

    print(f"🚀 Testing completions endpoint with X-Request-Id: {custom_request_id}")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/v1/completions", headers=headers, json=payload
        ) as response:
            response_headers = dict(response.headers)
            await response.json()

            print(f"📥 Response status: {response.status}")
            returned_id = response_headers.get("X-Request-Id")
            print(f"📋 Returned X-Request-Id: {returned_id}")

            if returned_id == custom_request_id:
                print("✅ Completions endpoint request ID successfully preserved!")
                return True
            else:
                print(
                    f"❌ Completions request ID mismatch. Expected: {custom_request_id}, Got: {returned_id}"
                )
                return False


async def test_health_endpoint(base_url: str = "http://localhost:8080"):
    """Test health endpoint with X-Request-Id."""

    custom_request_id = f"health-test-{uuid.uuid4()}"

    headers = {"X-Request-Id": custom_request_id}

    print(f"🚀 Testing health endpoint with X-Request-Id: {custom_request_id}")

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/health", headers=headers) as response:
            response_headers = dict(response.headers)
            await response.json()

            print(f"📥 Response status: {response.status}")
            returned_id = response_headers.get("X-Request-Id")
            print(f"📋 Returned X-Request-Id: {returned_id}")

            if returned_id == custom_request_id:
                print("✅ Health endpoint request ID successfully preserved!")
                return True
            else:
                print(
                    f"❌ Health request ID mismatch. Expected: {custom_request_id}, Got: {returned_id}"
                )
                return False


async def main():
    """Run all universal request tracing tests."""
    print("🧪 Testing Universal X-Request-Id Support in Dynamo")
    print("=" * 60)
    print("This demonstrates the automatic X-Request-Id handling provided by:")
    print("  • @auto_trace_endpoints decorator for frontends")
    print("  • RequestTracingMixin for processors")
    print("  • Built-in Dynamo SDK request tracing")
    print("=" * 60)

    base_url = "http://localhost:8080"

    tests = [
        ("Custom X-Request-Id (Chat Completions)", test_with_custom_request_id),
        ("Auto-generated X-Request-Id", test_without_request_id),
        ("Custom X-Request-Id (Completions)", test_completions_endpoint),
        ("Custom X-Request-Id (Health Check)", test_health_endpoint),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n📝 Test: {test_name}")
        print("-" * 50)
        try:
            result = await test_func(base_url)
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))

    print("\n🎯 Test Results Summary")
    print("=" * 60)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n📊 Overall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! Universal request tracing is working correctly!")
        print("\n💡 Benefits demonstrated:")
        print("   • Zero-configuration X-Request-Id support")
        print("   • Automatic header extraction and response injection")
        print("   • Request ID propagation across all components")
        print("   • Consistent behavior across all endpoints")
        print("   • Compatible with OpenAI API standards")
    else:
        print(
            f"\n⚠️  {len(results) - passed} test(s) failed. Please check the implementation."
        )


if __name__ == "__main__":
    asyncio.run(main())
