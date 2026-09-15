# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count content-bearing and stripped UUID image parts in AIPerf inputs.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def count_uuid_image_parts(value: Any) -> dict[str, int]:
    """Recursively count UUID-decorated image_url content parts."""
    counts = {"content": 0, "stripped": 0, "missing_uuid": 0}

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if not isinstance(item, dict):
            return
        if item.get("type") == "image_url":
            if not item.get("uuid"):
                counts["missing_uuid"] += 1
            image_url = item.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
            counts["content" if url else "stripped"] += 1
        for child in item.values():
            visit(child)

    visit(value)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, help="AIPerf inputs.json")
    parser.add_argument("--expect-content", type=int)
    parser.add_argument("--expect-stripped", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.inputs.open() as source:
        counts = count_uuid_image_parts(json.load(source))

    if counts["missing_uuid"]:
        raise ValueError(f"Found {counts['missing_uuid']} image parts without UUIDs")
    if args.expect_content is not None and counts["content"] != args.expect_content:
        raise ValueError(
            f"Expected {args.expect_content} content image parts, got {counts['content']}"
        )
    if args.expect_stripped is not None and counts["stripped"] != args.expect_stripped:
        raise ValueError(
            f"Expected {args.expect_stripped} stripped image parts, got {counts['stripped']}"
        )

    rendered = json.dumps(counts, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
