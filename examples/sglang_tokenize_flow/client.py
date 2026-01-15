# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example client demonstrating the tokenize -> generate_tokens -> detokenize flow.

This client:
1. Prepares a chat completion request with messages
2. Calls the tokenize dynamo endpoint to get token_ids
3. Prepares a PreprocessedRequest with the token_ids
4. Calls the generate_tokens endpoint (for token input) to get tokens back
5. Calls the detokenize dynamo endpoint to get text back

Usage:
    # Terminal 1: Start the sglang backend
    DYN_SYSTEM_PORT=9090 python -m dynamo.sglang --model Qwen/Qwen3-0.6B

    # Terminal 2: Run the client
    python examples/sglang_tokenize_flow/client.py

Note: The sglang backend exposes these endpoints:
- 'generate' for text input (ModelInput.Text)
- 'generate_tokens' for token input (ModelInput.Tokens)
- 'tokenize' for tokenizing chat messages
- 'detokenize' for converting token IDs back to text
"""

import asyncio

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    namespace = "dynamo"
    component_name = "backend"

    # Get endpoints for the dynamo component
    component = runtime.namespace(namespace).component(component_name)
    generate_tokens_endpoint = component.endpoint("generate_tokens")
    tokenize_endpoint = component.endpoint("tokenize")
    detokenize_endpoint = component.endpoint("detokenize")

    # Create clients for all endpoints
    generate_tokens_client = await generate_tokens_endpoint.client()
    tokenize_client = await tokenize_endpoint.client()
    detokenize_client = await detokenize_endpoint.client()

    # Wait for services to be ready
    print("Waiting for endpoints to be ready...")
    await asyncio.gather(
        generate_tokens_client.wait_for_instances(),
        tokenize_client.wait_for_instances(),
        detokenize_client.wait_for_instances(),
    )
    print("All endpoints ready!\n")

    # =========================================================================
    # Step 1: Prepare a chat completion request (OpenAI-style)
    # =========================================================================
    print("=" * 60)
    print("Step 1: Prepare chat completion request")
    print("=" * 60)

    chat_request = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
    }
    print(f"Chat request: {chat_request}\n")

    # =========================================================================
    # Step 2: Call tokenize endpoint to get token_ids
    # =========================================================================
    print("=" * 60)
    print("Step 2: Call tokenize endpoint (via Dynamo)")
    print("=" * 60)

    tokenize_stream = await tokenize_client.generate(chat_request)
    tokenize_data = None
    async for response in tokenize_stream:
        tokenize_data = response.data()

    print(f"Response: {tokenize_data}")

    if tokenize_data.get("status") != "ok":
        print(f"Error: {tokenize_data.get('message')}")
        return

    token_ids = tokenize_data["token_ids"]
    print(
        f"Token IDs ({len(token_ids)} tokens): {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}"
    )
    print()

    # =========================================================================
    # Step 3: Prepare a PreprocessedRequest with the token_ids
    # =========================================================================
    print("=" * 60)
    print("Step 3: Prepare PreprocessedRequest")
    print("=" * 60)

    # Use OpenAI-style format since tokenizer is enabled
    preprocessed_request = {
        "token_ids": token_ids,
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    print("PreprocessedRequest: {")
    print(
        f"  token_ids: [{token_ids[0]}, {token_ids[1]}, ... ({len(token_ids)} total)],"
    )
    print(f"  max_tokens: {preprocessed_request['max_tokens']},")
    print(f"  temperature: {preprocessed_request['temperature']},")
    print(f"  top_p: {preprocessed_request['top_p']}")
    print("}\n")

    # =========================================================================
    # Step 4: Call generate_tokens endpoint to get tokens back
    # =========================================================================
    print("=" * 60)
    print("Step 4: Call generate_tokens endpoint (via Dynamo)")
    print("=" * 60)

    all_output_token_ids = []
    generate_stream = await generate_tokens_client.generate(preprocessed_request)

    print("Streaming response tokens:")
    async for response in generate_stream:
        data = response.data()
        if "token_ids" in data:
            new_tokens = data["token_ids"]
            all_output_token_ids.extend(new_tokens)
            if len(new_tokens) <= 5:
                print(f"  Received {len(new_tokens)} token(s): {new_tokens}")
            else:
                print(f"  Received {len(new_tokens)} tokens: {new_tokens[:5]}...")

        if "finish_reason" in data:
            print(f"  Finish reason: {data['finish_reason']}")

    print(f"\nTotal output tokens: {len(all_output_token_ids)}")
    print(f"Output token IDs: {all_output_token_ids}\n")

    # =========================================================================
    # Step 5: Call detokenize endpoint to get text back
    # =========================================================================
    print("=" * 60)
    print("Step 5: Call detokenize endpoint (via Dynamo)")
    print("=" * 60)

    detokenize_stream = await detokenize_client.generate(
        {"token_ids": all_output_token_ids}
    )
    detokenize_data = None
    async for response in detokenize_stream:
        detokenize_data = response.data()

    print(f"Response: {detokenize_data}")

    if detokenize_data.get("status") == "ok":
        generated_text = detokenize_data["text"]
        print(f"\n{'=' * 60}")
        print("Generated text:")
        print("=" * 60)
        print(generated_text)
    else:
        print(f"Error: {detokenize_data.get('message')}")

    print("\n" + "=" * 60)
    print("Flow complete!")
    print("=" * 60)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
