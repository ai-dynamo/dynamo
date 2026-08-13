#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calculate speculative acceptance length and rate from vLLM counter snapshots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


def counter_sum(path: Path, metric: str) -> float:
    pattern = re.compile(rf"^{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$")
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            values.append(float(match.group(1)))
    if not values:
        raise ValueError(f"{metric} was not found in {path}")
    return sum(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="Prometheus snapshot before the run")
    parser.add_argument("after", type=Path, help="Prometheus snapshot after the run")
    args = parser.parse_args()

    deltas = {
        name: counter_sum(args.after, metric) - counter_sum(args.before, metric)
        for name, metric in COUNTERS.items()
    }
    if any(value < 0 for value in deltas.values()):
        raise ValueError(
            "a counter decreased; the vLLM process restarted during the run"
        )
    if deltas["drafts"] == 0 or deltas["draft_tokens"] == 0:
        raise ValueError("no speculative decoding activity was recorded")

    acceptance_length = 1 + deltas["accepted_tokens"] / deltas["drafts"]
    acceptance_rate = deltas["accepted_tokens"] / deltas["draft_tokens"]
    print(f"drafts={deltas['drafts']:.0f}")
    print(f"draft_tokens={deltas['draft_tokens']:.0f}")
    print(f"accepted_tokens={deltas['accepted_tokens']:.0f}")
    print(f"acceptance_length={acceptance_length:.15g}")
    print(f"acceptance_rate={acceptance_rate:.15g}")


if __name__ == "__main__":
    main()
