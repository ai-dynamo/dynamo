<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running gpt-oss-120b Disaggregated with vLLM

Dynamo supports disaggregated serving of gpt-oss-120b with vLLM. This guide demonstrates how to deploy gpt-oss-120b using disaggregated prefill/decode serving on a single H100 node with 8 GPUs, running 1 prefill worker on 4 GPUs and 1 decode worker on 4 GPUs.

## Overview

This deployment uses disaggregated serving in vLLM where:
- **Prefill Worker**: Processes input prompts efficiently using 4 GPUs with tensor parallelism
- **Decode Worker**: Generates output tokens using 4 GPUs, optimized for token generation throughput
- **Frontend**: Provides OpenAI-compatible API endpoint with round-robin routing

## Prerequisites

This guide assumes readers already knows how to deploy Dynamo disaggregated serving with vLLM as illustrated in [README.md](/docs/backends/vllm/README.md)

## Instructions

### 1. Launch the Deployment

Note that GPT-OSS is a reasoning model with tool calling support. To
ensure the response is being processed correctly, the worker should be
launched with proper `--dyn-reasoning-parser` and `--dyn-tool-call-parser`.

**Start frontend**
```bash
python3 -m dynamo.frontend --http-port 8000 &
```

**Run decode worker**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  python -m dynamo.vllm \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --dyn-reasoning-parser gpt_oss \
  --dyn-tool-call-parser harmony
```

**Run prefill workers**
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7  python -m dynamo.vllm \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --is-prefill-worker \
  --dyn-reasoning-parser gpt_oss \
  --dyn-tool-call-parser harmony
```

### 2. Verify the Deployment is Ready

Poll the `/health` endpoint to verify that both the prefill and decode worker endpoints have started:
```
curl http://localhost:8000/health
```

Make sure that both of the `generate` endpoints are available before sending an inference request:
```
{
  "status": "healthy",
  "endpoints": [
    "dyn://dynamo.backend.generate"
  ],
  "instances": [
    {
      "component": "backend",
      "endpoint": "generate",
      "namespace": "dynamo",
      "instance_id": 7587889712474989333,
      "transport": {
        "nats_tcp": "dynamo_backend.generate-694d997dbae9a315"
      }
    },
    {
      "component": "prefill",
      "endpoint": "generate",
      "namespace": "dynamo",
      "instance_id": 7587889712474989350,
      "transport": {
        "nats_tcp": "dynamo_prefill.generate-694d997dbae9a326"
      }
    },
    ...
  ]
}
```

If only one worker endpoint is listed, the other may still be starting up. Monitor the worker logs to track startup progress.

### 3. Test the Deployment

Send a test request to verify the deployment:

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "input": "Explain the concept of disaggregated serving in LLM inference in 3 sentences.",
    "max_output_tokens": 200,
    "stream": false
  }'
```

The server exposes a standard OpenAI-compatible API endpoint that accepts JSON requests. You can adjust parameters like `max_tokens`, `temperature`, and others according to your needs.

### 4. Reasoning and Tool Calling

Dynamo has supported reasoning and tool calling in OpenAI Chat Completion endpoint. A typical workflow for application built on top of Dynamo
is that the application has a set of tools to aid the assistant provide accurate answer, and it is ususally
multi-turn as it involves tool selection and generation based on the tool result. Below is an example
of sending multi-round requests to complete a user query with reasoning and tool calling:

**Application setup (pseudocode)**
```Python
# The tool defined by the application
def get_system_health():
    for component in system.components:
        if not component.health():
            return False
    return True

# The JSON representation of the declaration in ChatCompletion tool style
tool_choice = '{
  "type": "function",
  "function": {
    "name": "get_system_health",
    "description": "Returns the current health status of the LLM runtime—use before critical operations to verify the service is live.",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  }
}'

# On user query, perform below workflow.
def user_query(app_request):
    # first round
    # create chat completion with prompt and tool choice
    request = ...
    response = send(request)

    if response["finish_reason"] == "tool_calls":
        # second round
        function, params = parse_tool_call(response)
        function_result = function(params)
        # create request with prompt, assistant response, and function result
        request = ...
        response = send(request)
    return app_response(response)
```


**First request with tools**
```bash
curl localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '
{
  "model": "openai/gpt-oss-120b",
  "messages": [
    {
      "role": "user",
      "content": "Hey, quick check: is everything up and running?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_system_health",
        "description": "Returns the current health status of the LLM runtime—use before critical operations to verify the service is live.",
        "parameters": {
          "type": "object",
          "properties": {}
        }
      }
    }
  ],
  "response_format": {
    "type": "text"
  },
  "stream": false,
  "max_tokens": 300
}'
```
**First response with tool choice**
```JSON
{
  "id": "chatcmpl-d1c12219-6298-4c83-a6e3-4e7cef16e1a9",
  "choices": [
    {
      "index": 0,
      "message": {
        "tool_calls": [
          {
            "id": "call-1",
            "type": "function",
            "function": {
              "name": "get_system_health",
              "arguments": "{}"
            }
          }
        ],
        "role": "assistant",
        "reasoning_content": "We need to check system health. Use function."
      },
      "finish_reason": "tool_calls"
    }
  ],
  "created": 1758758741,
  "model": "openai/gpt-oss-120b",
  "object": "chat.completion",
  "usage": null
}
```
**Second request with tool calling result**
```bash
curl localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '
{
  "model": "openai/gpt-oss-120b",
  "messages": [
    {
      "role": "user",
      "content": "Hey, quick check: is everything up and running?"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call-1",
          "type": "function",
          "function": {
            "name": "get_system_health",
            "arguments": "{}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call-1",
      "content": "{\"status\":\"ok\",\"uptime_seconds\":372045}"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_system_health",
        "description": "Returns the current health status of the LLM runtime—use before critical operations to verify the service is live.",
        "parameters": {
          "type": "object",
          "properties": {}
        }
      }
    }
  ],
  "response_format": {
    "type": "text"
  },
  "stream": false,
  "max_tokens": 300
}'
```
**Second response with final message**
```JSON
{
  "id": "chatcmpl-9ebfe64a-68b9-4c1d-9742-644cf770ad0e",
  "choices": [
    {
      "index": 0,
      "message": {
        "content": "All systems are green—everything’s up and running smoothly! 🚀 Let me know if you need anything else.",
        "role": "assistant",
        "reasoning_content": "The user asks: \"Hey, quick check: is everything up and running?\" We have just checked system health, it's ok. Provide friendly response confirming everything's up."
      },
      "finish_reason": "stop"
    }
  ],
  "created": 1758758853,
  "model": "openai/gpt-oss-120b",
  "object": "chat.completion",
  "usage": null
}
```