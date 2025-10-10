#!/usr/bin/env python3
"""
Generate test data for parsers e2e tests.

This script captures streaming responses from an LLM server and saves them
in the format expected by lib/llm/tests/test_parsers_e2e.rs

Usage (cd lib/llm/tests/scripts/):
    python generate_parsers_e2e_test_data.py \
        --model "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5" \
        --port 8000

    python generate_parsers_e2e_test_data.py --model "Qwen/Qwen3-0.6B" --no-tools

    python generate_parsers_e2e_test_data.py \
        --model "openai/gpt-oss-20b"
"""

import argparse
import json
import os
import uuid
from pathlib import Path
from openai import OpenAI


def get_tools():
    """Return standard weather tool definition."""
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g., 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location", "unit"]
            }
        }
    }]


def capture_streaming_response(client, model_name, messages, tools=None, **params):
    """
    Capture a streaming response from the LLM server.

    Returns a tuple of (request_id, normal_content, reasoning_content, tool_calls, data_chunks)
    """
    # Prepare API call parameters
    api_params = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        **params
    }

    if tools:
        api_params["tools"] = tools
        api_params["tool_choice"] = "auto"

    # Make the streaming API call
    stream = client.chat.completions.create(**api_params)

    # Capture all chunks
    data_chunks = []
    request_id = None
    normal_content = ""
    reasoning_content = ""
    tool_calls_dict = {}

    for chunk in stream:
        # Extract request ID from first chunk
        if request_id is None and hasattr(chunk, 'id'):
            request_id = chunk.id

        # Convert chunk to dict for storage
        chunk_dict = chunk.model_dump()

        # Process choices
        if chunk_dict.get("choices"):
            for choice in chunk_dict["choices"]:
                delta = choice.get("delta", {})

                # Accumulate normal content
                if delta.get("content"):
                    normal_content += delta["content"]

                # Accumulate reasoning content
                if delta.get("reasoning_content"):
                    reasoning_content += delta["reasoning_content"]

                # Accumulate tool calls
                if delta.get("tool_calls"):
                    for tool_call in delta["tool_calls"]:
                        index = tool_call.get("index", 0)
                        if index not in tool_calls_dict:
                            tool_calls_dict[index] = {
                                "index": index,
                                "id": tool_call.get("id", ""),
                                "type": tool_call.get("type", "function"),
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            }

                        if tool_call.get("id"):
                            tool_calls_dict[index]["id"] = tool_call["id"]

                        if tool_call.get("type"):
                            tool_calls_dict[index]["type"] = tool_call["type"]

                        if tool_call.get("function"):
                            func = tool_call["function"]
                            if func.get("name"):
                                tool_calls_dict[index]["function"]["name"] = func["name"]
                            if func.get("arguments"):
                                tool_calls_dict[index]["function"]["arguments"] += func["arguments"]

        # Store the chunk in the format expected by tests
        data_chunks.append({
            "data": chunk_dict
        })

    # Convert tool_calls_dict to sorted list
    tool_calls = [tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())]

    # Use a generated UUID if no request_id was captured
    if not request_id:
        request_id = str(uuid.uuid4())

    return request_id, normal_content, reasoning_content, tool_calls, data_chunks


def save_test_data(output_dir, request_id, normal_content,
                   reasoning_content, tool_calls, data_chunks, has_tools):
    """Save captured test data to JSON file."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    suffix = "tool" if has_tools and tool_calls else "no-tool"
    # Use first 8 characters of UUID for filename
    short_uuid = request_id.split("-")[0] if "-" in request_id else request_id[:8]
    filename = f"chat_completion_stream_{short_uuid}-{suffix}.json"
    file_path = output_path / filename

    # Prepare output data
    output_data = {
        "request_id": request_id,
        "normal_content": normal_content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
        "data": data_chunks
    }

    # Write to file with pretty printing
    with open(file_path, "w") as f:
        json.dump(output_data, f, indent=2)

    return file_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate test data for LLM parser e2e tests"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
        help="Model name to use (default: nvidia/Llama-3_3-Nemotron-Super-49B-v1_5)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for the LLM server (default: 8000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for the LLM server (default: localhost)"
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable tool calling for this test"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom prompt to use (default depends on --no-tools)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: same directory as script)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate (default: 1000)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for generation (default: 0.6)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.6,
        help="Top-p for generation (default: 0.6)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to same directory as script
        output_dir = Path(__file__).parent

    # Initialize OpenAI client
    base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key="dummy")

    # Prepare messages
    if args.prompt:
        prompt = args.prompt
    elif args.no_tools:
        prompt = "Tell me a short joke about programming"
    else:
        prompt = "what is the weather like in Tokyo in Celsius?"

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt}
    ]

    # Get tools if needed
    tools = None if args.no_tools else get_tools()

    print(f"Connecting to {base_url}")
    print(f"Model: {args.model}")
    print(f"Using tools: {not args.no_tools}")
    print(f"Prompt: {prompt}")
    print()

    # Capture streaming response
    print("Capturing streaming response...")
    request_id, normal_content, reasoning_content, tool_calls, data_chunks = capture_streaming_response(
        client=client,
        model_name=args.model,
        messages=messages,
        tools=tools,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print(f"Request ID: {request_id}")
    print(f"Captured {len(data_chunks)} chunks")
    print(f"Normal content length: {len(normal_content)} chars")
    print(f"Reasoning content length: {len(reasoning_content)} chars")
    print(f"Tool calls: {len(tool_calls)}")
    if tool_calls:
        for tc in tool_calls:
            print(f"  - {tc['function']['name']}: {tc['function']['arguments']}")
    print()

    # Save to file
    file_path = save_test_data(
        output_dir=output_dir,
        request_id=request_id,
        normal_content=normal_content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        data_chunks=data_chunks,
        has_tools=not args.no_tools
    )

    print(f"✓ Test data saved to: {file_path}")
    print(f"  Relative path: {file_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
