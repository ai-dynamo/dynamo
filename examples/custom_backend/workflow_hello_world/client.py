# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send an OpenAI request to the Hello World workflow."""

import argparse
import asyncio
from typing import Any

import aiohttp

EXPECTED_TEXT = "Hello, World!"


async def request_completion(base_url: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "hello-world",
        "messages": [{"role": "user", "content": "Ignore this request."}],
        "stream": False,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            result = await response.json()

    content = result["choices"][0]["message"]["content"]
    if content != EXPECTED_TEXT:
        raise RuntimeError(f"unexpected completion: {content!r}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    print(asyncio.run(request_completion(args.base_url)))


if __name__ == "__main__":
    main()
