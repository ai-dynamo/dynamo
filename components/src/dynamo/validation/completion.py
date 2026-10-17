# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in ``completion`` check: one chat completion must return content.

Reads ``OPENAI_BASE_URL`` and ``DYN_VALIDATION_MODEL``, which the worker sets
for every gate. Exits 0 on a pass; on a failure it exits 1 with the reason on
stderr, which the worker reports on ``/health``.
"""

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("DYN_VALIDATION_MODEL")
    if not base_url or not model:
        print("OPENAI_BASE_URL and DYN_VALIDATION_MODEL must be set", file=sys.stderr)
        return 1
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "max_tokens": 32,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
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
