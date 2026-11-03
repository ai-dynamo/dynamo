# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduce aiperf's per-request records to the profiling phase's timing columns, as TSV:
start_ns end_ns ttft_ms latency_ms itl_ms osl error. The first line is
`# profiling_start_ns N`, the phase's first send.

usage: extract_records.py profile_export.jsonl > records.tsv
"""

import json
import sys


def metric(record, name):
    value = (record.get("metrics") or {}).get(name)
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, list):
        value = sum(value) / len(value) if value else None
    return value if isinstance(value, (int, float)) else None


def main():
    rows = []
    with open(sys.argv[1]) as records:
        for line in records:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = record.get("metadata") or {}
            if meta.get("phase_kind", "profiling") != "profiling":
                continue
            rows.append(
                (
                    meta.get("request_start_ns"),
                    meta.get("request_end_ns"),
                    metric(record, "time_to_first_token"),
                    metric(record, "request_latency"),
                    metric(record, "inter_token_latency"),
                    metric(record, "output_sequence_length"),
                    1 if record.get("error") else 0,
                )
            )
    starts = [row[0] for row in rows if row[0] is not None]
    print(f"# profiling_start_ns {min(starts) if starts else ''}")
    for row in rows:
        print("\t".join("" if value is None else str(value) for value in row))


if __name__ == "__main__":
    main()
