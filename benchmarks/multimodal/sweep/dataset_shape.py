# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path


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
