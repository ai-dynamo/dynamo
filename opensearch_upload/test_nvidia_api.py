#!/usr/bin/env python3
"""
Test NVIDIA API with OpenAI client format.
"""
import os

try:
    from openai import OpenAI
except ImportError:
    print("Installing openai package...")
    import subprocess
    subprocess.check_call(["pip", "install", "openai"])
    from openai import OpenAI

# Read API key
with open(os.path.expanduser("~/.claude2"), "r") as f:
    api_key = f.read().strip()

print(f"API Key loaded: {api_key[:10]}...")

# Create client with NVIDIA endpoint
client = OpenAI(
    base_url="https://inference-api.nvidia.com/v1",
    api_key=api_key
)

print("\nTesting NVIDIA API with Claude...")

# Test with Claude model
try:
    completion = client.chat.completions.create(
        model="aws/anthropic/claude-opus-4-5",
        messages=[{
            "role": "user",
            "content": "Say 'Hello, this is a test!' and nothing else."
        }],
        temperature=0.2,
        max_tokens=50,
        stream=False
    )

    print("\n✅ SUCCESS! NVIDIA API is working!")
    print(f"\nResponse: {completion.choices[0].message.content}")
    print(f"Model: {completion.model}")
    print(f"Tokens: {completion.usage.total_tokens}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
