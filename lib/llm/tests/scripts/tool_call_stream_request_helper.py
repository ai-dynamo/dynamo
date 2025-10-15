#!/usr/bin/env python3
"""
Helper script to send streaming requests to an LLM server.
In the future we can extend this script to generate json for stream test data.

Usage (cd lib/llm/tests/scripts/):
    python tool_call_stream_request_helper.py \
        --model "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5" \
        --port 8000

    python tool_call_stream_request_helper.py --model "Qwen/Qwen3-0.6B" --no-tools

    python tool_call_stream_request_helper.py \
        --model "openai/gpt-oss-20b"
"""

import argparse
from openai import OpenAI
from pathlib import Path

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


def send_streaming_request(client, model_name, messages, tools=None, **params):
    """Send a streaming request to the LLM server."""
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

    # Process the stream
    chunk_count = 0
    for chunk in stream:
        chunk_count += 1

    return chunk_count


def main():
    parser = argparse.ArgumentParser(
        description="Send a streaming request to an LLM server"
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

    # Send streaming request
    print("Sending streaming request...")
    chunk_count = send_streaming_request(
        client=client,
        model_name=args.model,
        messages=messages,
        tools=tools,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print(f"✓ Request completed successfully")
    print(f"  Received {chunk_count} chunks")


if __name__ == "__main__":
    main()
