# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Hashable


def count_session_ids(jsonl_path: str | Path) -> int:
    """Count unique ``session_id`` values in a JSONL dataset.

    Used to derive ``conversation_num`` for the sweep when the user hasn't set
    it explicitly. Rows without ``session_id`` count as distinct sessions
    (matches aiperf's per-row UUID fallback in ``SingleTurnDatasetLoader``).

    For multi_turn rows (``{"type": "multi_turn", "session_id": ..., "turns": [...]}``),
    the top-level ``session_id`` is what counts.
    """
    sessions: set[str] = set()
    anon_count = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("session_id")
            if sid is None:
                anon_count += 1
            else:
                sessions.add(str(sid))
    return len(sessions) + anon_count


def count_uuid_expectations(
    jsonl_path: str | Path,
    conversation_num: int | None = None,
) -> tuple[int, int]:
    """Count image payloads expected to be present and stripped by AIPerf.

    UUIDs are deduplicated independently within each session. Only the first
    ``conversation_num`` sessions are included, matching the sequential
    sampler used by the sweep. Missing or null session IDs make a row its own
    anonymous session, using the same policy as :func:`count_session_ids`.
    """
    if conversation_num is not None and conversation_num < 0:
        raise ValueError("conversation_num must be non-negative")

    seen: dict[Hashable, set[Hashable]] = defaultdict(set)
    included_sessions: set[Hashable] = set()
    content = 0
    stripped = 0

    with open(jsonl_path) as f:
        for line_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("session_id")
            session_key: Hashable = (
                ("anonymous", line_index) if sid is None else ("session", str(sid))
            )
            if session_key not in included_sessions:
                if (
                    conversation_num is not None
                    and len(included_sessions) >= conversation_num
                ):
                    continue
                included_sessions.add(session_key)

            session_seen = seen[session_key]
            for image_uuid in row.get("image_uuids", []):
                if image_uuid in session_seen:
                    stripped += 1
                else:
                    session_seen.add(image_uuid)
                    content += 1

    return content, stripped
