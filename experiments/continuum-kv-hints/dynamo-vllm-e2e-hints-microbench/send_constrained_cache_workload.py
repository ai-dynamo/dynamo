#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send the constrained-cache request sequence used by the G1 hint matrix."""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def send(
    url: str,
    model: str,
    label: str,
    messages: list[dict[str, str]],
    headers: dict[str, str],
) -> dict[str, object]:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": 4,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    usage = payload["usage"]
    result = {
        "label": label,
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        "assistant": payload["choices"][0]["message"].get("content") or "",
    }
    print(json.dumps(result), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "baseline",
            "evict",
            "retain",
            "retain-expired",
            "retain-final",
            "shared-evict",
        ),
        required=True,
    )
    parser.add_argument("--retention-expiry-wait-seconds", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8194/v1/chat/completions")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pressure = args.case in {
        "baseline",
        "retain",
        "retain-expired",
        "retain-final",
    }
    final_resume = args.case in {"evict", "retain-final"}
    results: list[dict[str, object]] = []

    target = "alpha " * 16_000
    first_messages = [{"role": "user", "content": target}]

    if args.case == "shared-evict":
        shared_headers = {"X-Dynamo-Session-Final": "false"}
        results.append(
            send(
                args.url,
                args.model,
                "shared-a-initial",
                first_messages,
                {"X-Dynamo-Session-ID": "shared-a", **shared_headers},
            )
        )
        results.append(
            send(
                args.url,
                args.model,
                "shared-b-associate",
                first_messages,
                {"X-Dynamo-Session-ID": "shared-b", **shared_headers},
            )
        )
        results.append(
            send(
                args.url,
                args.model,
                "shared-a-final",
                first_messages,
                {
                    "X-Dynamo-Session-ID": "shared-a",
                    "X-Dynamo-Session-Final": "true",
                },
            )
        )
        results.append(
            send(
                args.url,
                args.model,
                "shared-b-post-a-evict-final-check",
                first_messages,
                {
                    "X-Dynamo-Session-ID": "shared-b",
                    "X-Dynamo-Session-Final": "true",
                },
            )
        )
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        return

    first = send(
        args.url,
        args.model,
        "target-initial",
        first_messages,
        {
            "X-Dynamo-Session-ID": "synthetic-target",
            "X-Dynamo-Session-Final": "false",
        },
    )
    results.append(first)

    if args.retention_expiry_wait_seconds:
        time.sleep(args.retention_expiry_wait_seconds)

    if pressure:
        for index, token in enumerate(("bravo", "charlie", "delta"), start=1):
            results.append(
                send(
                    args.url,
                    args.model,
                    f"distractor-{index}",
                    [{"role": "user", "content": f"{token} " * 13_000}],
                    {},
                )
            )

    resumed_messages = [
        *first_messages,
        {"role": "assistant", "content": str(first["assistant"])},
        {"role": "user", "content": "Continue from the exact prior context."},
    ]
    results.append(
        send(
            args.url,
            args.model,
            "target-resume",
            resumed_messages,
            {
                "X-Dynamo-Session-ID": "synthetic-target",
                "X-Dynamo-Session-Final": str(final_resume).lower(),
            },
        )
    )

    if final_resume:
        results.append(
            send(
                args.url,
                args.model,
                "target-post-final-replay",
                resumed_messages,
                {
                    "X-Dynamo-Session-ID": "synthetic-target",
                    "X-Dynamo-Session-Final": "false",
                },
            )
        )

    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
