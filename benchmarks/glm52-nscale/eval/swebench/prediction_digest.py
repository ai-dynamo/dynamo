#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute a stable cache identity for an evaluator prediction set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prediction_digest(path: Path) -> str:
    predictions = json.loads(path.read_text())
    if not isinstance(predictions, dict):
        raise TypeError("preds.json must be an instance_id keyed JSON object")
    canonical = json.dumps(
        predictions, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    print(prediction_digest(args.predictions))


if __name__ == "__main__":
    main()
