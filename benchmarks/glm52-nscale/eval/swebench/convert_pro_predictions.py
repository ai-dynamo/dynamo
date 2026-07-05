#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert mini-SWE-agent's prediction map to the public Pro evaluator schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text())
    if not isinstance(predictions, dict):
        raise TypeError(
            "mini-SWE-agent predictions must be an instance_id keyed object"
        )

    converted = []
    for instance_id in sorted(predictions):
        prediction = predictions[instance_id]
        if prediction.get("instance_id") != instance_id:
            raise ValueError(f"prediction key mismatch for {instance_id}")
        converted.append(
            {
                "instance_id": instance_id,
                "patch": prediction.get("model_patch") or "",
                "prefix": args.prefix,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2) + "\n")


if __name__ == "__main__":
    main()
