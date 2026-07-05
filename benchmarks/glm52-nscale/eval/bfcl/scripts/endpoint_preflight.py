#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-context-window", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_url = os.environ["GLM52_OPENAI_BASE_URL"].rstrip("/")
    api_key = os.getenv("GLM52_OPENAI_API_KEY", "EMPTY")
    headers = {"Authorization": f"Bearer {api_key}"}
    request = Request(f"{base_url}/models", headers=headers)
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)

    entries = [entry for entry in payload.get("data", []) if isinstance(entry, dict)]
    matching = [entry for entry in entries if entry.get("id") == args.model]
    if len(matching) != 1:
        raise SystemExit(
            f"Endpoint must advertise {args.model!r} exactly once; "
            f"available models: {[entry.get('id') for entry in entries]}"
        )
    if matching[0].get("context_window") != args.expected_context_window:
        raise SystemExit(
            f"Endpoint context_window {matching[0].get('context_window')!r} != "
            f"campaign {args.expected_context_window}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Endpoint model preflight: PASS ({args.model})")


if __name__ == "__main__":
    main()
