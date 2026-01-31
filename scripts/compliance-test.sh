#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# OpenResponses Compliance Test Script
# Replicates the exact 6 tests from https://www.openresponses.org/compliance
#
# Usage:
#   ./scripts/compliance-test.sh [BASE_URL] [MODEL]
#
# Defaults:
#   BASE_URL=http://localhost:8000/v1
#   MODEL=Qwen/Qwen3-0.6B

set -euo pipefail

BASE_URL="${1:-http://localhost:8000/v1}"
MODEL="${2:-Qwen/Qwen3-0.6B}"

PASSED=0
FAILED=0
ERRORS=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  OpenResponses Compliance Tests${NC}"
echo -e "${CYAN}  Base URL: ${BASE_URL}${NC}"
echo -e "${CYAN}  Model:    ${MODEL}${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Helper: check a JSON field exists and is non-null
jq_check() {
    local json="$1"
    local expr="$2"
    echo "$json" | python3 -c "
import sys, json
data = json.load(sys.stdin)
result = $expr
print('true' if result else 'false')
" 2>/dev/null
}

# Helper: validate response schema (checks required fields from OpenAI Responses spec)
validate_response_schema() {
    local json="$1"
    local test_name="$2"
    local errors=""

    errors=$(echo "$json" | python3 -c "
import sys, json
data = json.load(sys.stdin)
errors = []

# Required string fields
for field in ['id', 'object', 'status', 'model']:
    if field not in data or not isinstance(data[field], str):
        errors.append(f'Missing or invalid required field: {field}')

# object must be 'response'
if data.get('object') != 'response':
    errors.append(f'Expected object=response, got {data.get(\"object\")}')

# created_at must be integer
if 'created_at' not in data or not isinstance(data['created_at'], (int, float)):
    errors.append('Missing or invalid created_at (expected integer)')

# output must be array
if 'output' not in data or not isinstance(data['output'], list):
    errors.append('Missing or invalid output (expected array)')

for e in errors:
    print(e)
" 2>/dev/null)

    if [ -n "$errors" ]; then
        echo "$errors"
        return 1
    fi
    return 0
}

# Helper: validate output items have required fields
validate_output_items() {
    local json="$1"
    echo "$json" | python3 -c "
import sys, json
data = json.load(sys.stdin)
errors = []
for i, item in enumerate(data.get('output', [])):
    if 'type' not in item:
        errors.append(f'output[{i}]: missing type field')
    if 'id' not in item:
        errors.append(f'output[{i}]: missing id field')
    if item.get('type') == 'message':
        if 'content' not in item or not isinstance(item['content'], list):
            errors.append(f'output[{i}]: message missing content array')
        if 'role' not in item:
            errors.append(f'output[{i}]: message missing role')
        if 'status' not in item:
            errors.append(f'output[{i}]: message missing status')
    elif item.get('type') == 'function_call':
        for field in ['call_id', 'name', 'arguments', 'status']:
            if field not in item:
                errors.append(f'output[{i}]: function_call missing {field}')
for e in errors:
    print(e)
" 2>/dev/null
}

run_test() {
    local test_id="$1"
    local test_name="$2"
    local description="$3"
    local payload="$4"
    local streaming="${5:-false}"
    local validators="$6"

    echo -e "${YELLOW}[$test_id] $test_name${NC}"
    echo -e "  $description"

    local start_time
    start_time=$(date +%s%N)

    if [ "$streaming" = "true" ]; then
        # Streaming test
        local raw_response
        raw_response=$(timeout 30 curl -sN "${BASE_URL}/responses" \
            -H 'Content-Type: application/json' \
            -d "$payload" 2>&1) || true

        local duration=$(( ($(date +%s%N) - start_time) / 1000000 ))

        if [ -z "$raw_response" ]; then
            echo -e "  ${RED}FAIL${NC} - No response received (${duration}ms)"
            FAILED=$((FAILED + 1))
            ERRORS+=("[$test_id] No response received")
            echo ""
            return
        fi

        # Parse SSE events
        local event_count=0
        local schema_errors=""
        local final_response=""
        local all_events=""

        all_events="$raw_response"
        event_count=$(echo "$raw_response" | grep -c "^event:" || true)

        # Extract final response from response.completed event
        final_response=$(echo "$raw_response" | python3 -c "
import sys, json
lines = sys.stdin.read().split('\n')
current_event = ''
current_data = ''
for line in lines:
    if line.startswith('event:'):
        current_event = line[6:].strip()
    elif line.startswith('data:'):
        current_data = line[5:].strip()
    elif line.strip() == '' and current_data:
        if current_event in ('response.completed', 'response.failed'):
            try:
                parsed = json.loads(current_data)
                if 'response' in parsed:
                    print(json.dumps(parsed['response']))
            except:
                pass
        current_event = ''
        current_data = ''
" 2>/dev/null)

        local test_errors=()

        # Validator: streamingEvents
        if [[ "$validators" == *"streamingEvents"* ]]; then
            if [ "$event_count" -eq 0 ]; then
                test_errors+=("No streaming events received")
            else
                echo -e "  streamingEvents: ${GREEN}PASS${NC} ($event_count events)"
            fi
        fi

        # Validator: streamingSchema - check required event types present
        if [[ "$validators" == *"streamingSchema"* ]]; then
            local missing_events=""
            for evt_type in "response.created" "response.in_progress" "response.completed"; do
                if ! echo "$raw_response" | grep -q "^event: $evt_type"; then
                    missing_events="$missing_events $evt_type"
                fi
            done

            # Validate each SSE event's JSON structure
            schema_errors=$(echo "$raw_response" | python3 -c "
import sys, json
lines = sys.stdin.read().split('\n')
errors = []
current_event = ''
current_data = ''
valid_event_types = {
    'response.created', 'response.queued', 'response.in_progress',
    'response.completed', 'response.failed', 'response.incomplete',
    'response.output_item.added', 'response.output_item.done',
    'response.content_part.added', 'response.content_part.done',
    'response.output_text.delta', 'response.output_text.done',
    'response.refusal.delta', 'response.refusal.done',
    'response.function_call_arguments.delta', 'response.function_call_arguments.done',
    'response.reasoning_summary_part.added', 'response.reasoning_summary_part.done',
    'response.reasoning.delta', 'response.reasoning.done',
    'response.reasoning_summary.delta', 'response.reasoning_summary.done',
    'response.output_text_annotation.added',
    'error',
}
for line in lines:
    if line.startswith('event:'):
        current_event = line[6:].strip()
    elif line.startswith('data:'):
        current_data = line[5:].strip()
    elif line.strip() == '' and current_data:
        if current_data != '[DONE]':
            try:
                parsed = json.loads(current_data)
                evt_type = parsed.get('type', 'unknown')
                if evt_type not in valid_event_types:
                    errors.append(f'Unknown event type: {evt_type}')
                if 'sequence_number' not in parsed:
                    errors.append(f'{evt_type}: missing sequence_number')
            except json.JSONDecodeError as e:
                errors.append(f'Invalid JSON in event data: {e}')
        current_event = ''
        current_data = ''
for e in errors:
    print(e)
" 2>/dev/null)

            if [ -n "$missing_events" ]; then
                test_errors+=("Missing required events:$missing_events")
            fi
            if [ -n "$schema_errors" ]; then
                test_errors+=("$schema_errors")
            fi
            if [ -z "$missing_events" ] && [ -z "$schema_errors" ]; then
                echo -e "  streamingSchema: ${GREEN}PASS${NC}"
            fi
        fi

        # Validator: completedStatus on final response
        if [[ "$validators" == *"completedStatus"* ]]; then
            if [ -z "$final_response" ]; then
                test_errors+=("No response.completed event found")
            else
                local status
                status=$(echo "$final_response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
                if [ "$status" = "completed" ]; then
                    echo -e "  completedStatus: ${GREEN}PASS${NC}"
                else
                    test_errors+=("Expected status completed, got: $status")
                fi
            fi
        fi

        if [ ${#test_errors[@]} -eq 0 ]; then
            echo -e "  ${GREEN}PASS${NC} (${duration}ms, $event_count events)"
            PASSED=$((PASSED + 1))
        else
            echo -e "  ${RED}FAIL${NC} (${duration}ms)"
            for err in "${test_errors[@]}"; do
                echo -e "    ${RED}- $err${NC}"
                ERRORS+=("[$test_id] $err")
            done
            FAILED=$((FAILED + 1))
        fi
    else
        # Non-streaming test
        local response
        response=$(timeout 30 curl -s "${BASE_URL}/responses" \
            -H 'Content-Type: application/json' \
            -d "$payload" 2>&1) || true

        local duration=$(( ($(date +%s%N) - start_time) / 1000000 ))

        if [ -z "$response" ]; then
            echo -e "  ${RED}FAIL${NC} - No response received (${duration}ms)"
            FAILED=$((FAILED + 1))
            ERRORS+=("[$test_id] No response received")
            echo ""
            return
        fi

        # Check for HTTP error
        local is_error
        is_error=$(echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('true' if 'error' in d and isinstance(d['error'], str) else 'false')
    print('true' if d.get('code') or d.get('type') == 'Bad Request' else 'false')
except: print('true')
" 2>/dev/null | head -1)

        if [ "$is_error" = "true" ]; then
            echo -e "  ${RED}FAIL${NC} - HTTP error (${duration}ms)"
            echo -e "    ${RED}Response: $(echo "$response" | head -c 200)${NC}"
            FAILED=$((FAILED + 1))
            ERRORS+=("[$test_id] HTTP error: $(echo "$response" | head -c 100)")
            echo ""
            return
        fi

        local test_errors=()

        # Schema validation
        local schema_errs
        schema_errs=$(validate_response_schema "$response" "$test_id") || true
        if [ -n "$schema_errs" ]; then
            test_errors+=("Schema: $schema_errs")
        fi

        # Output item validation
        local item_errs
        item_errs=$(validate_output_items "$response") || true
        if [ -n "$item_errs" ]; then
            test_errors+=("Items: $item_errs")
        fi

        # Validator: hasOutput
        if [[ "$validators" == *"hasOutput"* ]]; then
            local has_output
            has_output=$(jq_check "$response" "len(data.get('output', [])) > 0")
            if [ "$has_output" = "true" ]; then
                echo -e "  hasOutput: ${GREEN}PASS${NC}"
            else
                test_errors+=("Response has no output items")
            fi
        fi

        # Validator: completedStatus
        if [[ "$validators" == *"completedStatus"* ]]; then
            local status
            status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
            if [ "$status" = "completed" ]; then
                echo -e "  completedStatus: ${GREEN}PASS${NC}"
            else
                test_errors+=("Expected status completed, got: $status")
            fi
        fi

        # Validator: hasOutputType(function_call)
        if [[ "$validators" == *"hasOutputType:function_call"* ]]; then
            local has_fc
            has_fc=$(jq_check "$response" "any(item.get('type') == 'function_call' for item in data.get('output', []))")
            if [ "$has_fc" = "true" ]; then
                echo -e "  hasOutputType(function_call): ${GREEN}PASS${NC}"
            else
                test_errors+=("Expected output item of type function_call but none found")
            fi
        fi

        if [ ${#test_errors[@]} -eq 0 ]; then
            echo -e "  ${GREEN}PASS${NC} (${duration}ms)"
            PASSED=$((PASSED + 1))
        else
            echo -e "  ${RED}FAIL${NC} (${duration}ms)"
            for err in "${test_errors[@]}"; do
                echo -e "    ${RED}- $err${NC}"
                ERRORS+=("[$test_id] $err")
            done
            FAILED=$((FAILED + 1))
            # Print raw response for debugging
            echo -e "  ${YELLOW}Response:${NC}"
            echo "$response" | python3 -m json.tool 2>/dev/null | head -30 | sed 's/^/    /'
        fi
    fi
    echo ""
}

# ============================================================
# Test 1: Basic Text Response
# ============================================================
read -r -d '' PAYLOAD_1 << 'EOF' || true
{
  "model": "MODEL_PLACEHOLDER",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": "Say hello in exactly 3 words."
    }
  ]
}
EOF
PAYLOAD_1="${PAYLOAD_1//MODEL_PLACEHOLDER/$MODEL}"

run_test "1" "Basic Text Response" \
    "Simple user message, validates ResponseResource schema" \
    "$PAYLOAD_1" \
    "false" \
    "hasOutput,completedStatus"

# ============================================================
# Test 2: Streaming Response
# ============================================================
read -r -d '' PAYLOAD_2 << 'EOF' || true
{
  "model": "MODEL_PLACEHOLDER",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": "Count from 1 to 5."
    }
  ],
  "stream": true
}
EOF
PAYLOAD_2="${PAYLOAD_2//MODEL_PLACEHOLDER/$MODEL}"

run_test "2" "Streaming Response" \
    "Validates SSE streaming events and final response" \
    "$PAYLOAD_2" \
    "true" \
    "streamingEvents,streamingSchema,completedStatus"

# ============================================================
# Test 3: System Prompt
# ============================================================
read -r -d '' PAYLOAD_3 << 'EOF' || true
{
  "model": "MODEL_PLACEHOLDER",
  "input": [
    {
      "type": "message",
      "role": "system",
      "content": "You are a pirate. Always respond in pirate speak."
    },
    {
      "type": "message",
      "role": "user",
      "content": "Say hello."
    }
  ]
}
EOF
PAYLOAD_3="${PAYLOAD_3//MODEL_PLACEHOLDER/$MODEL}"

run_test "3" "System Prompt" \
    "Include system role message in input" \
    "$PAYLOAD_3" \
    "false" \
    "hasOutput,completedStatus"

# ============================================================
# Test 4: Tool Calling
# ============================================================
read -r -d '' PAYLOAD_4 << 'EOF' || true
{
  "model": "MODEL_PLACEHOLDER",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": "What's the weather like in San Francisco?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "name": "get_weather",
      "description": "Get the current weather for a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          }
        },
        "required": ["location"]
      }
    }
  ]
}
EOF
PAYLOAD_4="${PAYLOAD_4//MODEL_PLACEHOLDER/$MODEL}"

run_test "4" "Tool Calling" \
    "Define a function tool and verify function_call output" \
    "$PAYLOAD_4" \
    "false" \
    "hasOutput,hasOutputType:function_call"

# ============================================================
# Test 5: Image Input
# ============================================================
read -r -d '' PAYLOAD_5 << 'EOF' || true
{
  "model": "MODEL_PLACEHOLDER",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "What do you see in this image? Answer in one sentence."
        },
        {
          "type": "input_image",
          "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAABmklEQVR42tyWAaTyUBzFew/eG4AHz+MBSAHKBiJRGFKwIgQQJKLUIioBIhCAiCAAEizAQIAECaASqFFJq84nudjnaqvuPnxzgP9xfrq5938csPn7PwHTKSoViCIEAYEAMhmoKsU2mUCWEQqB5xEMIp/HaGQG2G6RSuH9HQ7H34rFrtPbdz4jl6PbwmEsl3QA1mt4vcRKk8dz9eg6IpF7tt9fzGY0gCgafFRFo5Blc5vLhf3eCOj1yNhM5GRMVK0aATxPZoz09YXjkQDmczJgquGQAPp9WwCNBgG027YACgUC6HRsAZRKBDAY2AJoNv/ZnwzA6WScznG3p4UAymXGAEkyXrTFAh8fLAGqagQAyGaZpYsi7bHTNPz8MEj//LxuFPo+UBS8vb0KaLXubrRa7aX0RMLCykwmn0z3+XA4WACcTpCkh9MFAZpmuVXo+mO/w+/HZvNgbblcUCxaSo/Hyck80Yu6XXDcvfVZr79cvMZjuN2U9O9vKAqjZrfbIZ0mV4TUi9Xqz6jddNy//7+e3n8Fhf/Llo2kxi8AQyGRoDkmAhAAAAAASUVORK5CYII="
        }
      ]
    }
  ]
}
EOF
PAYLOAD_5="${PAYLOAD_5//MODEL_PLACEHOLDER/$MODEL}"

run_test "5" "Image Input" \
    "Send image URL in user content" \
    "$PAYLOAD_5" \
    "false" \
    "hasOutput,completedStatus"

# ============================================================
# Test 6: Multi-turn Conversation
# ============================================================
read -r -d '' PAYLOAD_6 << 'EOF' || true
{
  "model": "MODEL_PLACEHOLDER",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": "My name is Alice."
    },
    {
      "type": "message",
      "role": "assistant",
      "content": "Hello Alice! Nice to meet you. How can I help you today?"
    },
    {
      "type": "message",
      "role": "user",
      "content": "What is my name?"
    }
  ]
}
EOF
PAYLOAD_6="${PAYLOAD_6//MODEL_PLACEHOLDER/$MODEL}"

run_test "6" "Multi-turn Conversation" \
    "Send assistant + user messages as conversation history" \
    "$PAYLOAD_6" \
    "false" \
    "hasOutput,completedStatus"

# ============================================================
# Summary
# ============================================================
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Results: ${GREEN}${PASSED} passed${NC}, ${RED}${FAILED} failed${NC} / $((PASSED + FAILED)) total"
echo -e "${CYAN}========================================${NC}"

if [ ${#ERRORS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failures:${NC}"
    for err in "${ERRORS[@]}"; do
        echo -e "  ${RED}- $err${NC}"
    done
fi

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}All compliance tests passed.${NC}"
    exit 0
else
    echo -e "${RED}Some compliance tests failed.${NC}"
    exit 1
fi
