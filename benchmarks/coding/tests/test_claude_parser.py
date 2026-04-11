from benchmarks.coding.claude_parser import build_turns_for_session
from benchmarks.coding.models import TraceRecord


class StubTokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(text)]


def make_record(
    *,
    row_type: str,
    timestamp_ms: int,
    source_order: int,
    raw: dict,
) -> TraceRecord:
    return TraceRecord(
        session_id="session-1",
        row_type=row_type,
        timestamp_ms=timestamp_ms,
        source_order=source_order,
        raw=raw,
    )


def test_compact_boundary_restarts_transcript_from_summary() -> None:
    records = [
        make_record(
            row_type="user",
            timestamp_ms=1_000,
            source_order=0,
            raw={
                "type": "user",
                "message": {"role": "user", "content": "before compact"},
            },
        ),
        make_record(
            row_type="assistant",
            timestamp_ms=2_000,
            source_order=1,
            raw={
                "type": "assistant",
                "message": {
                    "id": "assistant-1",
                    "content": [{"type": "text", "text": "first answer"}],
                    "usage": {"output_tokens": 3},
                },
            },
        ),
        make_record(
            row_type="system",
            timestamp_ms=3_000,
            source_order=2,
            raw={"type": "system", "subtype": "compact_boundary"},
        ),
        make_record(
            row_type="user",
            timestamp_ms=3_001,
            source_order=3,
            raw={
                "type": "user",
                "isCompactSummary": True,
                "message": {"role": "user", "content": "compacted summary"},
            },
        ),
        make_record(
            row_type="assistant",
            timestamp_ms=4_000,
            source_order=4,
            raw={
                "type": "assistant",
                "message": {
                    "id": "assistant-2",
                    "content": [{"type": "text", "text": "second answer"}],
                    "usage": {"output_tokens": 5},
                },
            },
        ),
    ]

    turns = build_turns_for_session(
        "session-1",
        records,
        tokenizer=StubTokenizer(),
        preserve_session_ids=True,
    )

    assert [turn.input_text for turn in turns] == [
        "[user] before compact",
        "[user] compacted summary",
    ]


def test_compact_boundary_skips_local_command_noise_after_summary() -> None:
    records = [
        make_record(
            row_type="user",
            timestamp_ms=1_000,
            source_order=0,
            raw={
                "type": "user",
                "message": {"role": "user", "content": "before compact"},
            },
        ),
        make_record(
            row_type="assistant",
            timestamp_ms=2_000,
            source_order=1,
            raw={
                "type": "assistant",
                "message": {
                    "id": "assistant-1",
                    "content": [{"type": "text", "text": "first answer"}],
                    "usage": {"output_tokens": 3},
                },
            },
        ),
        make_record(
            row_type="system",
            timestamp_ms=3_000,
            source_order=2,
            raw={"type": "system", "subtype": "compact_boundary"},
        ),
        make_record(
            row_type="user",
            timestamp_ms=3_001,
            source_order=3,
            raw={
                "type": "user",
                "isCompactSummary": True,
                "message": {"role": "user", "content": "compacted summary"},
            },
        ),
        make_record(
            row_type="user",
            timestamp_ms=3_002,
            source_order=4,
            raw={
                "type": "user",
                "isMeta": True,
                "message": {
                    "role": "user",
                    "content": "<local-command-caveat>ignore me</local-command-caveat>",
                },
            },
        ),
        make_record(
            row_type="user",
            timestamp_ms=3_003,
            source_order=5,
            raw={
                "type": "user",
                "message": {
                    "role": "user",
                    "content": "<command-name>/compact</command-name>\n<command-message>compact</command-message>",
                },
            },
        ),
        make_record(
            row_type="user",
            timestamp_ms=3_004,
            source_order=6,
            raw={
                "type": "user",
                "message": {
                    "role": "user",
                    "content": "<local-command-stdout>Compacted</local-command-stdout>",
                },
            },
        ),
        make_record(
            row_type="assistant",
            timestamp_ms=4_000,
            source_order=7,
            raw={
                "type": "assistant",
                "message": {
                    "id": "assistant-2",
                    "content": [{"type": "text", "text": "second answer"}],
                    "usage": {"output_tokens": 5},
                },
            },
        ),
    ]

    turns = build_turns_for_session(
        "session-1",
        records,
        tokenizer=StubTokenizer(),
        preserve_session_ids=True,
    )

    assert [turn.input_text for turn in turns] == [
        "[user] before compact",
        "[user] compacted summary",
    ]
