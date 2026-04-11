from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from benchmarks.coding.common import (
    anonymized_session_id,
    canonical_json,
    content_blocks,
    flatten_block_content_text,
    parse_utc_timestamp_ms,
)
from benchmarks.coding.hashing import TokenizerWrapper
from benchmarks.coding.models import (
    AssistantGroupSummary,
    ConversationEntry,
    ToolCallSummary,
    TraceRecord,
    TurnDraft,
)


class ToolIdNormalizer:
    def __init__(self) -> None:
        self._raw_to_normalized: dict[str, str] = {}

    def normalize(self, raw_id: str | None) -> str | None:
        if not raw_id:
            return None
        if raw_id not in self._raw_to_normalized:
            next_id = len(self._raw_to_normalized) + 1
            self._raw_to_normalized[raw_id] = f"tool_{next_id:04d}"
        return self._raw_to_normalized[raw_id]


def load_trace_records(
    trace_files: Sequence[Path],
    show_progress: bool = False,
) -> dict[str, list[TraceRecord]]:
    sessions: dict[str, list[TraceRecord]] = defaultdict(list)
    source_order = 0

    for trace_file in tqdm(
        trace_files,
        desc="Loading trace files",
        unit="file",
        disable=not show_progress,
    ):
        with trace_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {trace_file}:{line_number}: {exc.msg}"
                    ) from exc

                session_id = payload.get("sessionId")
                row_type = payload.get("type")
                timestamp_raw = payload.get("timestamp")
                if not session_id or not row_type or not timestamp_raw:
                    source_order += 1
                    continue

                try:
                    timestamp_ms = parse_utc_timestamp_ms(timestamp_raw)
                except ValueError:
                    source_order += 1
                    continue

                sessions[session_id].append(
                    TraceRecord(
                        session_id=session_id,
                        row_type=row_type,
                        timestamp_ms=timestamp_ms,
                        source_order=source_order,
                        raw=payload,
                    )
                )
                source_order += 1

    for session_records in sessions.values():
        session_records.sort(
            key=lambda record: (record.timestamp_ms, record.source_order)
        )

    return sessions


def assistant_group_key(record: TraceRecord) -> str:
    request_id = record.raw.get("requestId")
    message = record.raw.get("message")
    message_id = message.get("id") if isinstance(message, dict) else None
    return (
        request_id
        or message_id
        or record.raw.get("uuid")
        or f"row-{record.source_order}"
    )


def is_compact_boundary(record: TraceRecord) -> bool:
    return (
        record.row_type == "system" and record.raw.get("subtype") == "compact_boundary"
    )


def is_compact_summary(record: TraceRecord) -> bool:
    return record.row_type == "user" and bool(record.raw.get("isCompactSummary"))


def is_local_command_wrapper_text(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(
        (
            "<command-name>",
            "<command-message>",
            "<command-args>",
            "<local-command-caveat>",
            "<local-command-stdout>",
            "<local-command-stderr>",
        )
    )


def should_skip_user_record(record: TraceRecord) -> bool:
    if record.row_type != "user":
        return False
    if bool(record.raw.get("isMeta")):
        return True

    raw_message = record.raw.get("message")
    message = raw_message if isinstance(raw_message, dict) else {}
    blocks = content_blocks(message.get("content"))
    if not blocks:
        return False
    if any(block.get("type") != "text" for block in blocks):
        return False

    texts = [str(block.get("text", "")) for block in blocks]
    return bool(texts) and all(is_local_command_wrapper_text(text) for text in texts)


def sanitize_structure(value: Any, normalizer: ToolIdNormalizer) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {
                "id",
                "tool_use_id",
                "toolUseID",
                "parentToolUseID",
            } and isinstance(item, str):
                sanitized[key] = normalizer.normalize(item) or item
                continue
            sanitized[key] = sanitize_structure(item, normalizer)
        return sanitized
    if isinstance(value, list):
        return [sanitize_structure(item, normalizer) for item in value]
    return value


def count_trailing_tool_results(entries: Sequence[ConversationEntry]) -> int:
    count = 0
    for entry in reversed(entries):
        if entry.kind != "user_tool_result":
            break
        count += 1
    return count


def render_user_entries(
    message: dict[str, Any],
    normalizer: ToolIdNormalizer,
) -> list[ConversationEntry]:
    rendered_entries: list[ConversationEntry] = []
    for block in content_blocks(message.get("content")):
        block_type = block.get("type")
        if block_type in {"thinking", "redacted_thinking"}:
            continue
        if block_type == "text":
            text = str(block.get("text", ""))
            if text:
                rendered_entries.append(
                    ConversationEntry("user_text", f"[user] {text}")
                )
            continue
        if block_type == "tool_result":
            normalized_id = normalizer.normalize(block.get("tool_use_id"))
            content_text = flatten_block_content_text(block.get("content"))
            is_error = bool(block.get("is_error", False))
            header = f"[user_tool_result id={normalized_id or 'tool_unknown'} error={str(is_error).lower()}]"
            rendered = header if not content_text else f"{header} {content_text}"
            rendered_entries.append(ConversationEntry("user_tool_result", rendered))
            continue

        sanitized = sanitize_structure(block, normalizer)
        rendered_entries.append(
            ConversationEntry(
                "user_block",
                f"[user_block type={block_type or 'unknown'}] {canonical_json(sanitized)}",
            )
        )
    return rendered_entries


def summarize_assistant_group(
    group: Sequence[TraceRecord],
    normalizer: ToolIdNormalizer,
    tokenizer: TokenizerWrapper,
) -> AssistantGroupSummary:
    entries: list[ConversationEntry] = []
    tool_calls: list[ToolCallSummary] = []
    raw_task_tool_ids: list[str] = []
    assistant_text_blocks = 0
    output_lengths: list[int] = []

    for record in group:
        raw_message = record.raw.get("message")
        message = raw_message if isinstance(raw_message, dict) else {}
        raw_usage = message.get("usage")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        output_tokens = usage.get("output_tokens")
        if isinstance(output_tokens, int):
            output_lengths.append(output_tokens)

        for block in content_blocks(message.get("content")):
            block_type = block.get("type")
            if block_type in {"thinking", "redacted_thinking"}:
                continue
            if block_type == "text":
                text = str(block.get("text", ""))
                if text:
                    assistant_text_blocks += 1
                    entries.append(
                        ConversationEntry("assistant_text", f"[assistant] {text}")
                    )
                continue
            if block_type == "tool_use":
                raw_id = block.get("id")
                normalized_id = normalizer.normalize(raw_id)
                tool_name = str(block.get("name", "unknown"))
                args_json = canonical_json(
                    sanitize_structure(block.get("input"), normalizer)
                )
                rendered = (
                    f"[assistant_tool_use id={normalized_id or 'tool_unknown'} "
                    f"name={tool_name} args={args_json}]"
                )
                entries.append(ConversationEntry("assistant_tool_use", rendered))
                tool_calls.append(
                    ToolCallSummary(
                        name=tool_name,
                        normalized_id=normalized_id,
                        raw_id=raw_id if isinstance(raw_id, str) else None,
                        arg_size_chars=len(args_json),
                    )
                )
                if tool_name == "Task" and isinstance(raw_id, str):
                    raw_task_tool_ids.append(raw_id)
                continue

            sanitized = sanitize_structure(block, normalizer)
            rendered = (
                f"[assistant_block type={block_type or 'unknown'}] "
                f"{canonical_json(sanitized)}"
            )
            entries.append(ConversationEntry("assistant_block", rendered))

    output_length = max(output_lengths, default=-1)
    if output_length < 0:
        rendered_text = "\n".join(entry.rendered for entry in entries)
        output_length = len(tokenizer.encode(rendered_text))

    return AssistantGroupSummary(
        entries=entries,
        output_length=output_length,
        assistant_text_blocks=assistant_text_blocks,
        top_level_tool_calls=tool_calls,
        raw_task_tool_ids=raw_task_tool_ids,
        start_ms=group[0].timestamp_ms,
        end_ms=group[-1].timestamp_ms,
    )


def progress_timestamp_ms(record: TraceRecord) -> int:
    raw_data = record.raw.get("data")
    data = raw_data if isinstance(raw_data, dict) else {}
    raw_message = data.get("message")
    nested_message = raw_message if isinstance(raw_message, dict) else {}
    nested_timestamp = nested_message.get("timestamp")
    if isinstance(nested_timestamp, str):
        try:
            return parse_utc_timestamp_ms(nested_timestamp)
        except ValueError:
            return record.timestamp_ms
    return record.timestamp_ms


def build_progress_index(
    records: Sequence[TraceRecord],
) -> dict[str, list[TraceRecord]]:
    progress_index: dict[str, list[TraceRecord]] = defaultdict(list)
    for record in records:
        if "progress" not in record.row_type:
            continue
        parent_tool_use_id = record.raw.get("parentToolUseID")
        if isinstance(parent_tool_use_id, str):
            progress_index[parent_tool_use_id].append(record)

    for record_list in progress_index.values():
        record_list.sort(
            key=lambda record: (progress_timestamp_ms(record), record.source_order)
        )

    return progress_index


def aggregate_progress_metrics(
    task_tool_ids: Sequence[str],
    progress_index: dict[str, list[TraceRecord]],
    normalizer: ToolIdNormalizer,
) -> dict[str, Any]:
    relevant_records: list[TraceRecord] = []
    for task_tool_id in task_tool_ids:
        relevant_records.extend(progress_index.get(task_tool_id, []))
    relevant_records.sort(
        key=lambda record: (progress_timestamp_ms(record), record.source_order)
    )

    if not relevant_records:
        return {
            "task_parent_tool_ids": [
                normalizer.normalize(tool_id) for tool_id in task_tool_ids
            ],
            "nested_progress_event_count": 0,
            "nested_agent_count": 0,
            "nested_tool_call_count": 0,
            "nested_tool_result_count": 0,
            "nested_tool_error_count": 0,
            "nested_tool_counts": {},
            "nested_tool_names": [],
            "nested_tool_total_latency_ms": 0,
            "nested_tool_max_latency_ms": 0,
            "nested_tool_avg_latency_ms": 0,
            "nested_tool_max_parallelism": 0,
            "nested_assistant_text_blocks": 0,
            "task_duration_ms": 0,
        }

    agent_ids: set[str] = set()
    assistant_text_blocks = 0
    tool_counts: Counter[str] = Counter()
    tool_start_times: dict[str, int] = {}
    tool_intervals: list[tuple[int, int]] = []
    tool_result_count = 0
    tool_error_count = 0
    first_ts = progress_timestamp_ms(relevant_records[0])
    last_ts = progress_timestamp_ms(relevant_records[-1])

    for record in relevant_records:
        timestamp_ms = progress_timestamp_ms(record)
        raw_data = record.raw.get("data")
        data = raw_data if isinstance(raw_data, dict) else {}
        agent_id = data.get("agentId")
        if isinstance(agent_id, str):
            agent_ids.add(agent_id)

        raw_message = data.get("message")
        nested_message = raw_message if isinstance(raw_message, dict) else {}
        nested_type = nested_message.get("type")
        raw_payload = nested_message.get("message")
        nested_payload = raw_payload if isinstance(raw_payload, dict) else {}
        if nested_type == "assistant":
            for block in content_blocks(nested_payload.get("content")):
                block_type = block.get("type")
                if block_type == "text":
                    assistant_text_blocks += 1
                    continue
                if block_type != "tool_use":
                    continue
                raw_id = block.get("id")
                if not isinstance(raw_id, str):
                    continue
                tool_name = str(block.get("name", "unknown"))
                tool_counts[tool_name] += 1
                tool_start_times[raw_id] = timestamp_ms
            continue

        if nested_type != "user":
            continue

        for block in content_blocks(nested_payload.get("content")):
            if block.get("type") != "tool_result":
                continue
            raw_tool_id = block.get("tool_use_id")
            if not isinstance(raw_tool_id, str):
                continue
            start_ms = tool_start_times.pop(raw_tool_id, None)
            if start_ms is not None:
                tool_intervals.append((start_ms, timestamp_ms))
            tool_result_count += 1
            if bool(block.get("is_error", False)):
                tool_error_count += 1

    total_latency = sum(max(0, end - start) for start, end in tool_intervals)
    max_latency = max((max(0, end - start) for start, end in tool_intervals), default=0)
    avg_latency = int(total_latency / len(tool_intervals)) if tool_intervals else 0

    parallel_events: list[tuple[int, int]] = []
    for start_ms, end_ms in tool_intervals:
        parallel_events.append((start_ms, 1))
        parallel_events.append((end_ms, -1))
    parallel_events.sort(key=lambda item: (item[0], -item[1]))

    current_parallelism = 0
    max_parallelism = 0
    for _, delta in parallel_events:
        current_parallelism += delta
        if current_parallelism > max_parallelism:
            max_parallelism = current_parallelism

    return {
        "task_parent_tool_ids": [
            normalizer.normalize(tool_id) for tool_id in task_tool_ids
        ],
        "nested_progress_event_count": len(relevant_records),
        "nested_agent_count": len(agent_ids),
        "nested_tool_call_count": sum(tool_counts.values()),
        "nested_tool_result_count": tool_result_count,
        "nested_tool_error_count": tool_error_count,
        "nested_tool_counts": dict(sorted(tool_counts.items())),
        "nested_tool_names": sorted(tool_counts),
        "nested_tool_total_latency_ms": total_latency,
        "nested_tool_max_latency_ms": max_latency,
        "nested_tool_avg_latency_ms": avg_latency,
        "nested_tool_max_parallelism": max_parallelism,
        "nested_assistant_text_blocks": assistant_text_blocks,
        "task_duration_ms": max(0, last_ts - first_ts),
    }


def build_turns_for_session(
    session_id: str,
    records: Sequence[TraceRecord],
    tokenizer: TokenizerWrapper,
    preserve_session_ids: bool,
) -> list[TurnDraft]:
    export_session_id = (
        session_id if preserve_session_ids else anonymized_session_id(session_id)
    )
    progress_index = build_progress_index(records)
    normalizer = ToolIdNormalizer()
    conversation_entries: list[ConversationEntry] = []
    turns: list[TurnDraft] = []
    previous_assistant_end_ms: int | None = None
    turn_index = 0
    pending_compact_reset = False

    top_level_records = [
        record
        for record in records
        if record.row_type in {"user", "assistant", "system"}
        and not bool(record.raw.get("isSidechain"))
    ]

    cursor = 0
    while cursor < len(top_level_records):
        record = top_level_records[cursor]
        if record.row_type == "system":
            pending_compact_reset = is_compact_boundary(record)
            cursor += 1
            continue

        if record.row_type == "user":
            if should_skip_user_record(record):
                pending_compact_reset = False
                cursor += 1
                continue

            raw_message = record.raw.get("message")
            message = raw_message if isinstance(raw_message, dict) else {}
            rendered_entries = render_user_entries(message, normalizer)
            if pending_compact_reset and is_compact_summary(record):
                conversation_entries = rendered_entries
            else:
                conversation_entries.extend(rendered_entries)
            pending_compact_reset = False
            cursor += 1
            continue

        pending_compact_reset = False
        group = [record]
        group_key = assistant_group_key(record)
        cursor += 1
        while cursor < len(top_level_records):
            next_record = top_level_records[cursor]
            if next_record.row_type != "assistant":
                break
            if assistant_group_key(next_record) != group_key:
                break
            group.append(next_record)
            cursor += 1

        group_summary = summarize_assistant_group(group, normalizer, tokenizer)
        input_text = "\n".join(entry.rendered for entry in conversation_entries)
        sidecar = {
            "session_id": export_session_id,
            "turn_index": turn_index,
            "num_messages_in_context": len(conversation_entries),
            "context_shape": [entry.kind for entry in conversation_entries],
            "tool_rounds_before_answer": count_trailing_tool_results(
                conversation_entries
            ),
            "used_task_tool": any(
                tool_call.name == "Task"
                for tool_call in group_summary.top_level_tool_calls
            ),
            "assistant_text_blocks": group_summary.assistant_text_blocks,
            "top_level_tool_call_count": len(group_summary.top_level_tool_calls),
            "top_level_tool_names": [
                tool_call.name for tool_call in group_summary.top_level_tool_calls
            ],
            "top_level_tool_calls": [
                {
                    "name": tool_call.name,
                    "tool_id": tool_call.normalized_id,
                    "arg_size_chars": tool_call.arg_size_chars,
                }
                for tool_call in group_summary.top_level_tool_calls
            ],
        }
        sidecar.update(
            aggregate_progress_metrics(
                group_summary.raw_task_tool_ids,
                progress_index,
                normalizer,
            )
        )

        turns.append(
            TurnDraft(
                session_id=session_id,
                export_session_id=export_session_id,
                turn_index=turn_index,
                input_text=input_text,
                output_length=group_summary.output_length,
                assistant_start_ms=group_summary.start_ms,
                assistant_end_ms=group_summary.end_ms,
                delay_ms=(
                    None
                    if previous_assistant_end_ms is None
                    else max(0, group_summary.start_ms - previous_assistant_end_ms)
                ),
                sidecar=sidecar,
            )
        )

        conversation_entries.extend(group_summary.entries)
        previous_assistant_end_ms = group_summary.end_ms
        turn_index += 1

    return turns
