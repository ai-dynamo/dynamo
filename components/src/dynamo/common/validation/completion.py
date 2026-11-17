# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in ``completion`` check: one chat completion must return content.

Reads ``OPENAI_BASE_URL``, which the check runner sets from the worker's
request. The model is ``DYN_VALIDATION_MODEL`` if set, and otherwise the first
one the endpoint lists on ``/models``. Exits 0 on a pass; on a failure it exits 1 with the reason on
stderr, which the worker reports on ``/health``.
"""

import json
import os
import sys
import urllib.error
import urllib.request


def served_model(base_url: str) -> str:
    """The first model the endpoint lists."""
    with urllib.request.urlopen(f"{base_url}/models", timeout=30) as response:
        return json.loads(response.read())["data"][0]["id"]


def main() -> int:
    base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    if not base_url:
        print("OPENAI_BASE_URL must be set", file=sys.stderr)
        return 1
    try:
        model = os.environ.get("DYN_VALIDATION_MODEL") or served_model(base_url)
    except Exception as error:  # noqa: BLE001 - any failure is the check's reason
        print(f"listing models failed: {error!r}", file=sys.stderr)
        return 1
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "max_tokens": 32,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            message = json.loads(response.read())["choices"][0]["message"]
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        print(f"chat completion returned HTTP {error.code}: {detail}", file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - any failure is the check's reason
        print(f"chat completion failed: {error!r}", file=sys.stderr)
        return 1
    if not (
        message.get("content")
        or message.get("reasoning_content")
        or message.get("tool_calls")
    ):
        print(f"chat completion returned an empty message: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
