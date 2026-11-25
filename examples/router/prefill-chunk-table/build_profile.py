# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish strict per-rank chunk paths from one window of passive observations."""

import argparse
import json
import os
import tempfile
from pathlib import Path


def paths_from_window(samples):
    paths = {}
    latest = {}
    for sample in samples:
        total = sample["q"]
        parts = sample["segments"]
        if sample["c"] or not 0 < len(parts) <= 32:
            continue
        end = 0
        ends = []
        for count, context in parts:
            if count <= 0 or context != end:
                break
            end += count
            ends.append(end)
        else:
            if total == end:
                path = tuple(ends)
                if total in paths and paths[total] != path:
                    paths[total] = None
                else:
                    paths[total] = path
                latest[total] = max(sample["at_ns"], latest.get(total, 0))

    valid = sorted((n for n in paths if paths[n]), key=lambda n: (-latest[n], n))
    return [list(paths[n]) for n in sorted(valid[:64])]


def publish(window, output):
    profiles = [
        {
            "worker_id": rank["worker_id"],
            "dp_rank": rank["dp_rank"],
            "step_cost_blocks": rank["step_cost_blocks"],
            "paths": paths_from_window(rank["samples"]),
        }
        for rank in window["ranks"]
    ]
    # Readers see one complete generation even if this publisher is interrupted.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        try:
            document = {
                key: window[key]
                for key in (
                    "model_name",
                    "routing_group",
                    "block_size",
                    "valid_until_unix_ms",
                )
            }
            json.dump({**document, "profiles": profiles}, handle, allow_nan=False)
        except BaseException:
            handle.close()
            temp_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    publish(json.loads(args.window.read_text(encoding="utf-8")), args.output)
