from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from benchmarks.coding.common import canonical_json
from benchmarks.coding.hashing import RollingHasher, TokenizerWrapper
from benchmarks.coding.models import TurnDraft


def build_mooncake_rows(
    turns: Sequence[TurnDraft],
    tokenizer: TokenizerWrapper,
    block_size: int,
) -> list[dict[str, Any]]:
    hasher = RollingHasher(block_size=block_size)
    first_turn_starts = [
        turn.assistant_start_ms for turn in turns if turn.turn_index == 0
    ]
    if not first_turn_starts:
        return []
    global_trace_start_ms = min(first_turn_starts)

    rows: list[dict[str, Any]] = []
    for turn in turns:
        tokens = tokenizer.encode(turn.input_text)
        blocks = [
            tokens[index : index + block_size]
            for index in range(0, len(tokens), block_size)
        ]
        row: dict[str, Any] = {
            "session_id": turn.export_session_id,
            "input_length": len(tokens),
            "output_length": turn.output_length,
            "hash_ids": hasher.hash_token_blocks(blocks) if blocks else [],
        }
        if turn.delay_ms is None:
            row["timestamp"] = turn.assistant_start_ms - global_trace_start_ms
        else:
            row["delay"] = turn.delay_ms
        rows.append(row)
    return rows


def write_jsonl(output_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
