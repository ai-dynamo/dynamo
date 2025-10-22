#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test script for semantic routing PoC

set -e

BASE_URL="${BASE_URL:-http://localhost:8999}"

echo "=== Testing Semantic Router PoC ==="
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Reasoning query with router alias (should route to DeepSeek)
echo "Test 1: Reasoning query with model:'router' (X-Dynamo-Routing: auto)"
echo "Expected: Route to deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
echo "---"
curl -s "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Routing: auto' \
  -d '{
    "model":"router",
    "messages":[{"role":"user","content":"Prove that the sum of first n odd numbers is n^2. Think step by step and explain your reasoning."}],
    "max_tokens":128,
    "stream":false
  }' | jq -r '.model // "ERROR: No response"'
echo ""

# Test 2: Simple factoid with router alias (should route to Llama)
echo "Test 2: Simple factoid with model:'router' (X-Dynamo-Routing: auto)"
echo "Expected: Route to meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "---"
curl -s "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Routing: auto' \
  -d '{
    "model":"router",
    "messages":[{"role":"user","content":"What is the capital of Spain?"}],
    "max_tokens":64,
    "stream":false
  }' | jq -r '.model // "ERROR: No response"'
echo ""

# Test 3: Code query with router alias (should route to DeepSeek based on config)
echo "Test 3: Code query with model:'router' (X-Dynamo-Routing: auto)"
echo "Expected: Route to deepseek-ai/DeepSeekCoderV2-Lite-Instruct (if available) or another model"
echo "---"
curl -s "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Routing: auto' \
  -d '{
    "model":"router",
    "messages":[{"role":"user","content":"Write a Python function to compute the nth Fibonacci number using dynamic programming."}],
    "max_tokens":128,
    "stream":false
  }' | jq -r '.model // "ERROR: No response"'
echo ""

# Test 4: Shadow mode - explicit model with shadow routing
echo "Test 4: Shadow mode - explicit model (X-Dynamo-Routing: shadow)"
echo "Expected: Use meta-llama/Meta-Llama-3.1-8B-Instruct (no override)"
echo "---"
RESPONSE=$(curl -i -s "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Routing: shadow' \
  -d '{
    "model":"meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages":[{"role":"user","content":"Prove that the sum of first n odd numbers is n^2."}],
    "max_tokens":64,
    "stream":false
  }')
echo "$RESPONSE" | grep -i "x-route" || echo "(No routing headers found - may need to implement)"
echo "$RESPONSE" | tail -n 1 | jq -r '.model // "ERROR: No response"'
echo ""

# Test 5: Auto mode with explicit model (should NOT override, but track shadow)
echo "Test 5: Auto mode with explicit model (X-Dynamo-Routing: auto)"
echo "Expected: Use meta-llama/Meta-Llama-3.1-8B-Instruct (no override)"
echo "---"
curl -s "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Routing: auto' \
  -d '{
    "model":"meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages":[{"role":"user","content":"What is 2+2?"}],
    "max_tokens":32,
    "stream":false
  }' | jq -r '.model // "ERROR: No response"'
echo ""

echo "=== Tests Complete ==="
echo ""
echo "Check Prometheus metrics at: $BASE_URL/metrics"
echo "Look for: semantic_route_decisions_total and semantic_classifier_latency_ms"

