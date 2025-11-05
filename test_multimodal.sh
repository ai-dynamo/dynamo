#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test script for multimodal image support in vLLM backend
# Tests HTTP URLs, base64 data URLs, and mixed scenarios

set -e

API_URL="${API_URL:-http://localhost:8000/v1/chat/completions}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"

echo "========================================================================"
echo "Multimodal Image Support Test Suite"
echo "========================================================================"
echo "API: $API_URL"
echo "Model: $MODEL"
echo ""

# Test 1: Text-only request (baseline)
echo "========================================================================"
echo "TEST 1: Text-only request (no images)"
echo "========================================================================"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": \"What is 2 + 2?\"
    }],
    \"max_tokens\": 50,
    \"temperature\": 0.0,
    \"stream\": false
  }" | jq -r '.choices[0].message.content // .error // .'

echo ""
echo "✅ Text-only request complete"
echo ""

# Test 2: HTTP URL image
echo "========================================================================"
echo "TEST 2: HTTP URL image"
echo "========================================================================"
echo "Image: http://images.cocodataset.org/test2017/000000155781.jpg"
echo ""

curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
          }
        }
      ]
    }],
    "max_tokens": 100,
    "temperature": 0.0,
    "stream": false
  }' | jq -r '.choices[0].message.content // .error // .'

echo ""
echo "✅ HTTP URL request complete"
echo ""

# Test 3: Base64 data URL
echo "========================================================================"
echo "TEST 3: Base64 data URL"
echo "========================================================================"
echo "Using minimal 1x1 PNG encoded as base64"
echo ""

# Minimal 1x1 grayscale PNG (67 bytes)
TINY_PNG_B64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNoAAAAggCBd81ytgAAAABJRU5ErkJggg=="

curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What do you see in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,'"$TINY_PNG_B64"'"
          }
        }
      ]
    }],
    "max_tokens": 100,
    "temperature": 0.0,
    "stream": false
  }' | jq -r '.choices[0].message.content // .error // .'

echo ""
echo "✅ Base64 data URL request complete"
echo ""

# Test 4: Mixed URLs (HTTP + base64)
echo "========================================================================"
echo "TEST 4: Mixed URLs (HTTP + base64 in same request)"
echo "========================================================================"
echo "Image 1: HTTP URL (COCO bus)"
echo "Image 2: Base64 data URL (1x1 PNG)"
echo ""

curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe what you see in these images"},
        {
          "type": "image_url",
          "image_url": {
            "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,'"$TINY_PNG_B64"'"
          }
        }
      ]
    }],
    "max_tokens": 150,
    "temperature": 0.0,
    "stream": false
  }' | jq -r '.choices[0].message.content // .error // .'

echo ""
echo "✅ Mixed URLs request complete"
echo ""

echo "========================================================================"
echo "All tests complete!"
echo "========================================================================"

