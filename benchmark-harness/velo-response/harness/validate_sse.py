#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate one independently delimited SSE content event per output token."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(raw: bytes, expected_tokens: int) -> dict[str, object]:
    normalized = raw.replace(b"\r\n", b"\n")
    blocks = [block for block in normalized.split(b"\n\n") if block]
    payloads: list[bytes] = []
    malformed = 0
    for block in blocks:
        lines = [
            line[5:].lstrip()
            for line in block.splitlines()
            if line.startswith(b"data:")
        ]
        if not lines:
            malformed += 1
        else:
            payloads.append(b"\n".join(lines))
    content = terminal = done = errors = 0
    for payload in payloads:
        if payload == b"[DONE]":
            done += 1
            continue
        try:
            event = json.loads(payload)
            choice = event["choices"][0]
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError):
            errors += 1
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict) and delta.get("content") is not None:
            content += 1
        elif choice.get("finish_reason") == "length":
            terminal += 1
    valid = (
        malformed == 0
        and errors == 0
        and content == expected_tokens
        and terminal == 1
        and done == 1
        and len(payloads) == expected_tokens + 2
    )
    return {
        "valid": valid,
        "expected_output_tokens": expected_tokens,
        "independent_sse_events": len(payloads),
        "content_events": content,
        "terminal_events": terminal,
        "done_events": done,
        "malformed_blocks": malformed,
        "json_errors": errors,
        "invariant": "one output token per independently encoded SSE event",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path)
    parser.add_argument("--expected-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.response.read_bytes(), args.expected_tokens)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
