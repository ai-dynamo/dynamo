# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parsers for structured router log events used by e2e tests."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

_ROUTING_MESSAGE_PATTERN = re.compile(
    r"\[ROUTING\].*with\s*(?P<overlap>\d+(?:\.\d+)?)/(?P<total>\d+)\s*blocks overlap"
)
_FIELD_PATTERN = re.compile(
    r"\b(?P<key>request_id|worker_id|dp_rank|overlap_blocks|total_blocks)="
    r"(?P<value>\"[^\"]*\"|\S+)"
)


@dataclass(frozen=True)
class RouterKvOverlapRecord:
    overlap_blocks: int
    total_blocks: int
    request_id: str | None = None
    worker_id: int | None = None
    dp_rank: int | None = None


def _context_prefix(context: str) -> str:
    return f"{context}: " if context else ""


def _block_count(value: str) -> int:
    return int(float(value.strip('"')) + 0.5)


def _int_field(fields: dict[str, str], key: str) -> int | None:
    value = fields.get(key)
    if value is None:
        return None
    return int(value.strip('"'))


def _str_field(fields: dict[str, str], key: str) -> str | None:
    value = fields.get(key)
    if value is None:
        return None
    return value.strip('"')


def _record_from_line(line: str) -> RouterKvOverlapRecord | None:
    if "[ROUTING]" not in line:
        return None

    fields = {match["key"]: match["value"] for match in _FIELD_PATTERN.finditer(line)}
    overlap = fields.get("overlap_blocks")
    total = fields.get("total_blocks")
    if overlap is not None and total is not None:
        return RouterKvOverlapRecord(
            overlap_blocks=_block_count(overlap),
            total_blocks=_block_count(total),
            request_id=_str_field(fields, "request_id"),
            worker_id=_int_field(fields, "worker_id"),
            dp_rank=_int_field(fields, "dp_rank"),
        )

    match = _ROUTING_MESSAGE_PATTERN.search(line)
    if match is None:
        return None
    return RouterKvOverlapRecord(
        overlap_blocks=_block_count(match["overlap"]),
        total_blocks=_block_count(match["total"]),
    )


def extract_router_kv_overlap_records(log_text: str) -> list[RouterKvOverlapRecord]:
    return [
        record
        for line in log_text.splitlines()
        if (record := _record_from_line(line)) is not None
    ]


def wait_for_router_kv_overlap(
    logs_provider: Callable[[], str],
    *,
    start_offset: int,
    pre_request_record_count: int,
    context: str = "",
    log_label: str = "router",
    timeout_s: float = 25.0,
) -> tuple[int, int, str]:
    deadline = time.time() + timeout_s
    last_segment = ""

    while time.time() < deadline:
        full_logs = logs_provider()
        segment = full_logs[start_offset:]
        last_segment = segment
        records = extract_router_kv_overlap_records(full_logs)
        if len(records) >= pre_request_record_count + 1:
            record = records[-1]
            return record.overlap_blocks, record.total_blocks, segment
        time.sleep(1)

    segment_records = extract_router_kv_overlap_records(last_segment)
    if segment_records:
        record = segment_records[-1]
        return record.overlap_blocks, record.total_blocks, last_segment

    raise AssertionError(
        f"{_context_prefix(context)}Expected a structured router KV overlap log "
        f"event after the request.\nRecent {log_label} logs:\n{last_segment[-4000:]}"
    )
