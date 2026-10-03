#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf's own reading of a live-lane input (runs in the AIPerf 0.13.0 environment).

The audit must not trust the generator's ``requests.jsonl``: it asks AIPerf how it parses
``aiperf_input.jsonl``. The file goes through AIPerf's ``MooncakeTrace`` model and
``MooncakeTraceDatasetLoader`` grouping and conversation building, the code path a real run uses,
and the result is projected the way AIPerf's timing strategies consume it:

- ``view.jsonl``: one line per request in AIPerf's conversation order, with the turn metadata the
  timing strategies read (``timestamp_ms``, ``delay_ms``), the wire payload facts (prompt length,
  digest and token range; ``max_tokens``, ``min_tokens``, ``ignore_eos``, ``stream``, usage) and the
  fixed-schedule offset ``FixedScheduleStrategy`` would compute;
- ``blocks.npy`` and ``block_offsets.npy``: chained 16-token block hashes of every prompt (the
  structure the KV router and the engine prefix cache see);
- ``lcp.json``: token longest-common-prefix lengths for requested pairs of requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from aiperf.dataset.loader._delay_cap import DelayCapTracker
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader

ENGINE_BLOCK = 16


def chained_block_hashes(tokens: np.ndarray, block: int = ENGINE_BLOCK) -> np.ndarray:
    full = len(tokens) // block
    out = np.empty(full, dtype=np.uint64)
    parent = b"\x00" * 8
    data = np.ascontiguousarray(tokens[: full * block], dtype="<u4").tobytes()
    step = 4 * block
    for i in range(full):
        parent = hashlib.blake2b(
            parent + data[i * step : (i + 1) * step], digest_size=8
        ).digest()
        out[i] = int.from_bytes(parent, "little")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--no-blocks", action="store_true")
    args = parser.parse_args(argv)

    traces = []
    with args.input.open() as handle:
        for line in handle:
            if line.strip():
                traces.append(MooncakeTrace.model_validate(json.loads(line)))
    loader = MooncakeTraceDatasetLoader.__new__(MooncakeTraceDatasetLoader)
    loader._delay_cap_tracker = DelayCapTracker(cap_seconds=None)
    grouped = loader._group_traces(traces)
    conversations = loader.convert_to_conversations(grouped)

    firsts = [
        (conv.turns[0].timestamp, ci)
        for ci, conv in enumerate(conversations)
        if conv.turns and conv.turns[0].timestamp is not None
    ]
    # FixedScheduleStrategy.setup_phase: stable sort by first-turn timestamp, auto offset = first.
    schedule = sorted(firsts, key=lambda x: x[0])
    zero = schedule[0][0] if schedule else None
    schedule_rank = {ci: rank for rank, (_, ci) in enumerate(schedule)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokens_by_key: dict[tuple[str, int], np.ndarray] = {}
    blocks: list[np.ndarray] = []
    offsets = [0]
    vocab_violations = 0
    with (args.out_dir / "view.jsonl").open("w") as out:
        index = 0
        for ci, conv in enumerate(conversations):
            for ti, turn in enumerate(conv.turns):
                meta = turn.metadata()
                payload = turn.raw_payload or {}
                prompt = np.asarray(payload.get("prompt") or [], dtype=np.int64)
                if prompt.size and (prompt.min() < 0 or prompt.max() >= 151643):
                    vocab_violations += 1
                tokens = prompt.astype(np.uint32)
                key = (conv.session_id, ti)
                tokens_by_key[key] = tokens
                offset_ms = None
                if ti == 0 and meta.timestamp_ms is not None and zero is not None:
                    # FixedScheduleStrategy._timestamp_to_perf_sec, back in milliseconds.
                    offset_ms = ((meta.timestamp_ms - zero) / 1000.0) * 1000.0
                row = {
                    "index": index,
                    "conversation_index": ci,
                    "conversation_id": conv.session_id,
                    "turn_index": ti,
                    "context_mode": None
                    if conv.context_mode is None
                    else str(conv.context_mode),
                    "timestamp_ms": meta.timestamp_ms,
                    "delay_ms": meta.delay_ms,
                    "schedule_rank": schedule_rank.get(ci) if ti == 0 else None,
                    "schedule_offset_ms": offset_ms,
                    "turn_max_tokens": turn.max_tokens,
                    "payload_keys": sorted(payload),
                    "prompt_len": int(prompt.size),
                    "prompt_sha256": hashlib.sha256(
                        np.ascontiguousarray(tokens, dtype="<u4").tobytes()
                    ).hexdigest(),
                    "prompt_min": int(prompt.min()) if prompt.size else None,
                    "prompt_max": int(prompt.max()) if prompt.size else None,
                    "max_tokens": payload.get("max_tokens"),
                    "min_tokens": payload.get("min_tokens"),
                    "ignore_eos": payload.get("ignore_eos"),
                    "stream": payload.get("stream"),
                    "include_usage": (payload.get("stream_options") or {}).get(
                        "include_usage"
                    ),
                    "model": payload.get("model"),
                    "nvext": payload.get("nvext"),
                }
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                if not args.no_blocks:
                    hashes = chained_block_hashes(tokens)
                    blocks.append(hashes)
                    offsets.append(offsets[-1] + len(hashes))
                index += 1
    if not args.no_blocks:
        np.save(
            args.out_dir / "blocks.npy",
            np.concatenate(blocks) if blocks else np.empty(0, np.uint64),
        )
        np.save(args.out_dir / "block_offsets.npy", np.asarray(offsets, dtype=np.int64))

    lcp = None
    if args.pairs is not None:
        pairs = json.loads(args.pairs.read_text())
        lcp = []
        for (sa, ta), (sb, tb) in pairs:
            a, b = tokens_by_key[(sa, ta)], tokens_by_key[(sb, tb)]
            n = min(len(a), len(b))
            diff = np.flatnonzero(a[:n] != b[:n])
            lcp.append(int(diff[0]) if diff.size else n)
        (args.out_dir / "lcp.json").write_text(json.dumps(lcp))
    summary = {
        "conversations": len(conversations),
        "requests": index,
        "vocab_violations": vocab_violations,
        "schedule_entries": len(schedule),
        "schedule_zero_ms": zero,
        "pairs": None if lcp is None else len(lcp),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
