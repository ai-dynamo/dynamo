# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TraceRecord:
    session_id: str
    row_type: str
    timestamp_ms: int
    source_order: int
    raw: dict[str, Any]


@dataclass(frozen=True)
class ConversationEntry:
    kind: str
    rendered: str


@dataclass(frozen=True)
class ToolCallSummary:
    name: str
    normalized_id: str | None
    raw_id: str | None
    arg_size_chars: int


@dataclass
class AssistantGroupSummary:
    entries: list[ConversationEntry]
    output_length: int
    assistant_text_blocks: int
    top_level_tool_calls: list[ToolCallSummary]
    raw_task_tool_ids: list[str]
    start_ms: int
    end_ms: int


@dataclass
class TurnDraft:
    session_id: str
    export_session_id: str
    turn_index: int
    input_text: str
    output_length: int
    assistant_start_ms: int
    assistant_end_ms: int
    delay_ms: int | None
    sidecar: dict[str, Any]
