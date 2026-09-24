#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("missing_suffix_oracle", "whole_prefix_oracle"),
        required=True,
    )
    parser.add_argument("--pressure", type=float, default=0.026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    simulation = json.loads(args.simulation.read_text())
    result = next(
        item
        for item in simulation[args.policy]
        if math.isclose(item["pressure_lost_blocks_per_block_second"], args.pressure)
    )
    entries = []
    for selected in result["selected"]:
        trace_id, kind, outer_index, inner_index = selected["trigger_source_key"]
        entries.append(
            {
                "source_trace_id": trace_id,
                "source_kind": kind,
                "source_outer_idx": outer_index,
                "source_inner_idx": inner_index,
                "ttl_ms": math.ceil(selected["ttl_seconds"] * 1000),
                "block_start": selected["block_start"],
                "block_count": selected["retained_blocks"],
                "target_source_key": selected["target_source_key"],
            }
        )
    key_fields = (
        "source_trace_id",
        "source_kind",
        "source_outer_idx",
        "source_inner_idx",
    )
    if len({tuple(entry[key] for key in key_fields) for entry in entries}) != len(
        entries
    ):
        raise ValueError("selected schedule contains duplicate trigger source keys")
    args.output.write_text(
        json.dumps(
            {
                "policy": args.policy,
                "pressure_lost_blocks_per_block_second": args.pressure,
                "capacity_blocks": simulation["capacity_blocks"],
                "predicted_net_tokens": result["predicted_net_tokens"],
                "entries": entries,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
