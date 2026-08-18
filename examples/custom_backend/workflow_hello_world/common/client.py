# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Send the same OpenAI request to either implementation."""

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
