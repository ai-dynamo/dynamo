# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentX (Weka) plays: parsing, per-play statistics and the complete-play context selection.

The source corpus is ``semianalysisai/cc-traces-weka-062126`` at revision
``23f152f6f0f9399a85901b89a6458def0ef16729``: one play per JSONL line, block size 64, local hash
scope. A play is kept whole or not at all. The selection predicate: every normal or streaming
request, recursively including explicit subagents, satisfies ``in + max(out, 1) <= limit``.
Nothing is clipped, scaled or rewritten.

Native replay derives each play's hash namespace from its relative path (``.json`` files) or from
``<file>#<line index>`` (``.jsonl`` files). Hash scope is local to a play, so the namespace only has
to separate plays; selected plays are written once to stably named files and never renamed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator

from .common import MAX_MODEL_LEN, summarize

SOURCE_DATASET = "semianalysisai/cc-traces-weka-062126"
SOURCE_REVISION = "23f152f6f0f9399a85901b89a6458def0ef16729"
SOURCE_URL = (
    "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/"
    f"{SOURCE_REVISION}/traces.jsonl"
)
SOURCE_BYTES = 1_847_151_435
SOURCE_SHA256 = "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276"


def iter_requests(entries: list[dict]) -> Iterator[dict]:
    """Normal and streaming requests in source order, descending into subagents."""
    for entry in entries:
        kind = entry["type"]
        if kind == "subagent":
            yield from iter_requests(entry["requests"])
        elif kind in ("n", "s"):
            yield entry
        else:
            raise ValueError(f"unknown Weka entry type {kind!r}")


def effective_context(request: dict) -> int:
    """Tokens a request occupies: ``in + max(out, 1)`` (the importer normalizes zero outputs)."""
    return int(request["in"]) + max(int(request["out"]), 1)


def play_fits(play: dict, limit: int = MAX_MODEL_LEN) -> bool:
    requests = list(iter_requests(play["requests"]))
    return bool(requests) and max(effective_context(r) for r in requests) <= limit


def play_file_name(row_index: int, play: dict) -> str:
    """Stable file name ``<row:04d>-<source id>.json`` (the form setup already used)."""
    source_id = str(play["id"])
    if not source_id or "/" in source_id:
        raise ValueError(f"unusable play id {source_id!r}")
    return f"{row_index:04d}-{source_id}.json"


def play_stats(play: dict) -> dict:
    requests = list(iter_requests(play["requests"]))
    subagents = [e for e in play["requests"] if e["type"] == "subagent"]
    inputs = [int(r["in"]) for r in requests]
    outputs = [int(r["out"]) for r in requests]
    starts = [float(r["t"]) for r in requests]
    ends = [float(r["t"]) + float(r.get("api_time") or 0.0) for r in requests]
    think = [
        float(r["think_time"]) for r in requests if r.get("think_time") is not None
    ]
    top_level = [e for e in play["requests"] if e["type"] in ("n", "s")]
    gaps = []
    for previous, current in zip(top_level, top_level[1:]):
        gaps.append(
            float(current["t"])
            - (float(previous["t"]) + float(previous.get("api_time") or 0.0))
        )
    block_size = int(play["block_size"])
    return {
        "source_id": str(play["id"]),
        "block_size": block_size,
        "hash_id_scope": play.get("hash_id_scope"),
        "requests": len(requests),
        "explicit_subagent_groups": len(subagents),
        "subagent_statuses": dict(
            Counter(s.get("status", "<missing>") for s in subagents)
        ),
        "models": sorted({r["model"] for r in requests}),
        "declared_models": sorted(play.get("models") or []),
        "input_tokens": sum(inputs),
        "output_tokens": sum(outputs),
        "zero_outputs": outputs.count(0),
        "max_context": max(effective_context(r) for r in requests),
        "isl": summarize(inputs),
        "osl": summarize(outputs),
        "recorded_span_s": max(ends) - min(starts),
        "sum_think_time_s": sum(think),
        "max_think_time_s": max(think) if think else None,
        "max_top_level_gap_s": max(gaps) if gaps else None,
        "missing_api_time": sum(r.get("api_time") is None for r in requests),
        "input_equals_hash_blocks_times_block_size": all(
            int(r["in"]) == len(r.get("hash_ids") or []) * block_size for r in requests
        ),
    }


def line_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_play(raw: bytes) -> dict:
    return json.loads(raw)
