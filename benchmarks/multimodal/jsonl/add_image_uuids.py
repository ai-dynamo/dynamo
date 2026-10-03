# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add deterministic image UUIDs to an existing multimodal JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.multimodal.jsonl.generate_images import compute_image_uuid


def add_image_uuids(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Copy rows from input_path and add UUIDs parallel to every images list."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    image_count = 0
    with input_path.open() as source, output_path.open("w") as destination:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            images = row.get("images")
            if not isinstance(images, list):
                raise ValueError(f"Row {line_number} must contain an images list")
            row["image_uuids"] = [compute_image_uuid(ref) for ref in images]
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")
            row_count += 1
            image_count += len(images)
    return row_count, image_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Existing JSONL dataset")
    parser.add_argument("output", type=Path, help="UUID-enriched JSONL dataset")
    args = parser.parse_args()

    rows, images = add_image_uuids(args.input, args.output)
    print(f"Wrote {rows} rows and {images} image UUIDs to {args.output}")


if __name__ == "__main__":
    main()
