# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise either user-ensemble implementation through the stock frontend."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import aiohttp

from examples.experimental.workflow.user_ensemble.common.config import PUBLIC_MODEL_NAME

_ONE_PIXEL_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
    "x8AAusB9Wl2nL8AAAAASUVORK5CYII="
)


async def request_completion(base_url: str) -> dict[str, Any]:
    payload = {
        "model": PUBLIC_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Based on The Hitchhiker's Guide to the Galaxy, "
                            "The Answer to"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": _ONE_PIXEL_PNG}},
                    {"type": "text", "text": " is? Answer with one number."},
                ],
            }
        ],
        "max_tokens": 8,
        "temperature": 0,
        "stream": False,
        "nvext": {"extra_fields": ["engine_data"]},
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            result = await response.json()

    text = result["choices"][0]["message"]["content"]
    if "42" not in text:
        raise RuntimeError(f"unexpected completion: {text!r}")
    scores = result["nvext"]["engine_data"]["user_ensemble"]["classifier_scores"]
    if scores.get("dummy-positive") != 1.0:
        raise RuntimeError(f"unexpected classifier scores: {scores!r}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    print(asyncio.run(request_completion(args.base_url)))


if __name__ == "__main__":
    main()
