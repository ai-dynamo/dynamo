from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_BLOCK_SIZE = 64
DEFAULT_OUTPUT_NAME = "claude_mooncake_trace.jsonl"
SIDE_CAR_TOKEN = ".sidecar"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: str | tuple[Any, ...]) -> int:
    encoded = value.encode() if isinstance(value, str) else repr(value).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def parse_utc_timestamp_ms(value: str) -> int:
    if not value:
        raise ValueError("missing timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    return int(
        datetime.fromisoformat(normalized).astimezone(timezone.utc).timestamp() * 1000
    )


def anonymized_session_id(session_id: str) -> str:
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:12]
    return f"session_{digest}"


def sidecar_path_for(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(
            f"{output_path.stem}{SIDE_CAR_TOKEN}{output_path.suffix}"
        )
    return output_path.with_name(f"{output_path.name}{SIDE_CAR_TOKEN}.jsonl")


def dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def maybe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def maybe_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [block for block in maybe_list(content) if isinstance(block, dict)]


def flatten_block_content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type == "text":
                parts.append(str(item.get("text", "")))
                continue
            if "text" in item:
                parts.append(str(item["text"]))
                continue
            parts.append(canonical_json(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"])
        return canonical_json(value)
    return str(value)
