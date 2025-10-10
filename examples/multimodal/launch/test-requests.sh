#!/bin/bash
################################################################################
# 🧪 Comprehensive test suite for multimodal chat completions API
# Tests various combinations of inputs (text, image) with and without tool calling
#
# Test Coverage:
# 1️⃣ 🖼️💬🔧 Image + Text + Tool Calling
# 2️⃣ 💬🔧 Text + Tool Calling
# 3️⃣ 🖼️💬 Image + Text (No Tools)
# 4️⃣ 💬 Text Only
# 5️⃣ 🖼️ Image Only
# 6️⃣ 💬🔧📡 Text + Tool Calling (STREAMING)
# 7️⃣ 🖼️💬📡 Image + Text (STREAMING)
# 8️⃣ 💬📡 Text Only (STREAMING)
################################################################################

# API Configuration
BASE_URL="http://localhost:8000/v1/chat/completions"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

make_request() {
    local payload="$1"
    local test_name="$2"
    
    echo "================================================================================"
    echo "🧪 TEST: $test_name"
    echo "================================================================================"
    
    # Make the request and capture response
    http_code=$(curl -s -w "%{http_code}" -o /tmp/response.json \
        -X POST "$BASE_URL" \
        -H "Content-Type: application/json" \
        -d "$payload")
    
    # Check if request was successful
    if [ "$http_code" -eq 200 ]; then
        echo -e "${GREEN}✅ Status Code: $http_code${NC}"
        echo ""
        
        # Check if response is streaming (SSE format with "data:" lines)
        if head -n 1 /tmp/response.json | grep -q "^data:"; then
            echo "📥 Response (STREAMING - SSE format):"
            echo ""
            
            # Parse streaming response
            echo "📡 Stream Chunks:"
            chunk_count=0
            accumulated_content=""
            
            while IFS= read -r line; do
                if [[ "$line" == data:* ]]; then
                    # Remove "data: " prefix
                    json_data="${line#data: }"
                    
                    if [[ "$json_data" == "[DONE]" ]]; then
                        echo -e "${BLUE}  [DONE]${NC}"
                        break
                    fi
                    
                    chunk_count=$((chunk_count + 1))
                    
                    if command -v jq &> /dev/null; then
                        # Extract delta content if present
                        delta_content=$(echo "$json_data" | jq -r '.choices[0].delta.content // empty' 2>/dev/null)
                        if [ -n "$delta_content" ]; then
                            accumulated_content="${accumulated_content}${delta_content}"
                            echo -n "$delta_content"
                        fi
                        
                        # Check for tool calls in final chunk
                        tool_calls=$(echo "$json_data" | jq -r '.choices[0].delta.tool_calls // empty' 2>/dev/null)
                        if [ -n "$tool_calls" ] && [ "$tool_calls" != "null" ]; then
                            echo ""
                            echo ""
                            echo "🛠️  Tool Calls Detected:"
                            echo "$json_data" | jq -r '.choices[0].delta.tool_calls[]? | "  🔧 Function: " + .function.name + "\n     📋 Arguments: " + .function.arguments' 2>/dev/null
                        fi
                    fi
                fi
            done < /tmp/response.json
            
            echo ""
            echo ""
            echo "📊 Summary:"
            echo "  Total chunks: $chunk_count"
            if [ -n "$accumulated_content" ]; then
                content_length=${#accumulated_content}
                echo "  Content length: $content_length characters"
                echo ""
                echo "📝 Full Content:"
                echo "$accumulated_content"
            fi
        else
            # Non-streaming response
            echo "📥 Response:"
            
            # Pretty print the response
            if command -v jq &> /dev/null; then
                cat /tmp/response.json | jq '.'
                
                # Extract and display key information
                echo ""
                echo "💬 Message Summary"
                role=$(cat /tmp/response.json | jq -r '.choices[0].message.role // "N/A"')
                echo "Role: $role"
                
                # Display content if present
                content=$(cat /tmp/response.json | jq -r '.choices[0].message.content // ""')
                if [ -n "$content" ] && [ "$content" != "null" ]; then
                    content_preview=$(echo "$content" | head -c 200)
                    if [ ${#content} -gt 200 ]; then
                        echo "📝 Content: ${content_preview}..."
                    else
                        echo "📝 Content: $content"
                    fi
                fi
                
                # Display tool calls if present
                tool_calls=$(cat /tmp/response.json | jq -r '.choices[0].message.tool_calls // empty')
                if [ -n "$tool_calls" ] && [ "$tool_calls" != "null" ]; then
                    echo ""
                    echo "🛠️  Tool Calls"
                    cat /tmp/response.json | jq -r '.choices[0].message.tool_calls[] | "  [" + (.id // "N/A") + "] 🔧 Function: " + .function.name + "\n      📋 Arguments: " + .function.arguments'
                fi
            else
                # Fallback if jq is not available
                cat /tmp/response.json
            fi
        fi
        
        echo ""
        echo -e "${GREEN}✅ Test '$test_name' PASSED${NC}"
    else
        echo -e "${RED}❌ HTTP Error${NC}"
        echo -e "${RED}❌ Status Code: $http_code${NC}"
        echo -e "${RED}❌ Response:${NC}"
        cat /tmp/response.json
        echo ""
        echo -e "${RED}❌ Test '$test_name' FAILED${NC}"
    fi
    
    echo ""
}

################################################################################
# Test 1: Image + Text + Tool Calling (general)
################################################################################
test_1_image_text_tool_calling() {
    local payload=$(cat <<'EOF'
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe what you see in this image in detail."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
          }
        }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "describe_image",
        "description": "Provides detailed description of objects and scenes in an image",
        "parameters": {
          "type": "object",
          "properties": {
            "objects": {
              "type": "array",
              "items": {"type": "string"},
              "description": "List of objects detected in the image"
            },
            "scene": {
              "type": "string",
              "description": "Overall scene description"
            }
          },
          "required": ["objects", "scene"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "max_tokens": 1024
}
EOF
)
    make_request "$payload" "1️⃣ 🖼️ 💬 🔧 Image + Text + Tool Calling"
}

################################################################################
# Test 2: Text + Tool Calling
################################################################################
test_2_text_tool_calling() {
    local payload=$(cat <<'EOF'
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "What is the weather like in San Francisco?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "description": "The temperature unit to use"
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "max_tokens": 512
}
EOF
)
    make_request "$payload" "2️⃣ 💬 🔧 Text + Tool Calling"
}

################################################################################
# Test 3: Image + Text (no tools)
################################################################################
test_3_image_text() {
    local payload=$(cat <<'EOF'
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
          }
        }
      ]
    }
  ],
  "max_tokens": 300,
  "temperature": 0.0,
  "stream": false
}
EOF
)
    make_request "$payload" "3️⃣ 🖼️ 💬 Image + Text"
}

################################################################################
# Test 4: Text Only
################################################################################
test_4_text_only() {
    local payload=$(cat <<'EOF'
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "Explain the concept of neural networks in simple terms."
    }
  ],
  "max_tokens": 300,
  "temperature": 0.0,
  "stream": false
}
EOF
)
    make_request "$payload" "4️⃣ 💬 Text Only"
}

################################################################################
# Test 5: Image Only
################################################################################
test_5_image_only() {
    local payload=$(cat <<'EOF'
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
          }
        }
      ]
    }
  ],
  "max_tokens": 300,
  "temperature": 0.0,
  "stream": false
}
EOF
)
    make_request "$payload" "5️⃣ 🖼️ Image Only"
}

################################################################################
# Test 6: Text + Tool Calling (STREAMING)
################################################################################
test_6_text_tool_calling_stream() {
    local payload=$(cat <<'EOF'
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "What is the weather like in San Francisco?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "description": "The temperature unit to use"
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "max_tokens": 512,
  "stream": true
}
EOF
)
    make_request "$payload" "6️⃣ 💬 🔧 📡 Text + Tool Calling (STREAMING)"
}

################################################################################
# Test 7: Image + Text (STREAMING)
################################################################################
test_7_image_text_stream() {
    local payload=$(cat <<'EOF'
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
          }
        }
      ]
    }
  ],
  "max_tokens": 300,
  "temperature": 0.0,
  "stream": true
}
EOF
)
    make_request "$payload" "7️⃣ 🖼️ 💬 📡 Image + Text (STREAMING)"
}

################################################################################
# Test 8: Text Only (STREAMING)
################################################################################
test_8_text_only_stream() {
    local payload=$(cat <<'EOF'
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "Explain the concept of neural networks in simple terms."
    }
  ],
  "max_tokens": 300,
  "temperature": 0.0,
  "stream": true
}
EOF
)
    make_request "$payload" "8️⃣ 💬 📡 Text Only (STREAMING)"
}

################################################################################
# Main Test Runner
################################################################################
main() {
    echo ""
    echo "================================================================================"
    echo "🚀 COMPREHENSIVE MULTIMODAL API TEST SUITE 🚀"
    echo "================================================================================"
    echo ""
    
    # Check for jq
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}⚠️  Warning: 'jq' is not installed. Output formatting will be limited.${NC}"
        echo -e "${YELLOW}   Install with: sudo apt-get install jq (Debian/Ubuntu) or brew install jq (macOS)${NC}"
        echo ""
    fi
    
    # Run all tests
    test_1_image_text_tool_calling
    test_2_text_tool_calling
    test_3_image_text
    test_4_text_only
    test_5_image_only
    test_6_text_tool_calling_stream
    test_7_image_text_stream
    test_8_text_only_stream
    
    # Cleanup
    rm -f /tmp/response.json
    
    echo "================================================================================"
    echo "🎉 ALL TESTS COMPLETED 🎉"
    echo "================================================================================"
}

# Run main function
main