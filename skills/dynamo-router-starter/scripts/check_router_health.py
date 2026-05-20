# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test a Dynamo OpenAI-compatible frontend."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def request_json(
    method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 20
) -> tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = raw
        return exc.code, body
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc.reason)}


def choose_model(models_body: Any) -> str | None:
    if not isinstance(models_body, dict):
        return None
    data = models_body.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and first.get("id"):
            return str(first["id"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument(
        "--prompt", default="Say hello from Dynamo in one short sentence."
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    result: dict[str, Any] = {"base_url": base_url, "ok": False, "checks": []}

    models_status = None
    models_body = None
    for attempt in range(1, args.retries + 1):
        models_status, models_body = request_json("GET", f"{base_url}/v1/models")
        if models_status == 200:
            break
        time.sleep(args.retry_sleep)

    model = args.model or choose_model(models_body)
    result["checks"].append(
        {"name": "models", "status": models_status, "body": models_body, "model": model}
    )

    if models_status != 200:
        print(json.dumps(result, indent=2))
        return 2

    if args.skip_chat:
        result["ok"] = True
        print(json.dumps(result, indent=2))
        return 0

    if not model:
        result["checks"].append(
            {"name": "chat", "status": "skipped", "reason": "No model discovered"}
        )
        print(json.dumps(result, indent=2))
        return 3

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
    }
    chat_status, chat_body = request_json(
        "POST", f"{base_url}/v1/chat/completions", payload
    )
    result["checks"].append({"name": "chat", "status": chat_status, "body": chat_body})
    result["ok"] = chat_status == 200
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
