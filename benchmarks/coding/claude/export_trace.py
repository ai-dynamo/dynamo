#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.coding.claude.discovery import discover_trace_files
from benchmarks.coding.claude_parser import build_turns_for_session, load_trace_records
from benchmarks.coding.common import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_TOKENIZER,
    sidecar_path_for,
)
from benchmarks.coding.hashing import TokenizerWrapper
from benchmarks.coding.mooncake import build_mooncake_rows, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover local Claude session traces and export privacy-preserving "
            "Mooncake JSONL plus a text-free structural sidecar."
        )
    )
    parser.add_argument(
        "--input-path",
        action="append",
        default=[],
        help=(
            "Optional input file or directory. If set, autodiscovery is skipped. "
            "Directories may be a Claude project trace directory, a repo root, or "
            "any directory containing session JSONL files."
        ),
    )
    parser.add_argument(
        "--output-file",
        default=str(Path.cwd() / DEFAULT_OUTPUT_NAME),
        help=(
            "Main Mooncake JSONL output path. Defaults to "
            f"{DEFAULT_OUTPUT_NAME!r} in the current working directory."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="Tokenizer name passed to transformers.AutoTokenizer.from_pretrained().",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Token block size used for rolling hash IDs.",
    )
    parser.add_argument(
        "--anonymize-session-id",
        action="store_true",
        help="Replace Claude session IDs with stable anonymized IDs in both outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")

    show_progress = True
    script_dir = Path(__file__).resolve().parent
    trace_files = discover_trace_files(args.input_path, script_dir)
    if not trace_files:
        raise FileNotFoundError("No Claude session traces found.")

    tokenizer = TokenizerWrapper(args.tokenizer)
    sessions = load_trace_records(trace_files, show_progress=show_progress)
    if not sessions:
        raise ValueError(
            "No parseable Claude session rows were found in the discovered files."
        )

    all_turns = []
    preserve_session_ids = not args.anonymize_session_id
    for session_id in tqdm(
        sorted(sessions),
        desc="Building sessions",
        unit="session",
        disable=not show_progress,
    ):
        all_turns.extend(
            build_turns_for_session(
                session_id=session_id,
                records=sessions[session_id],
                tokenizer=tokenizer,
                preserve_session_ids=preserve_session_ids,
            )
        )

    if not all_turns:
        raise ValueError(
            "No assistant turns were reconstructed from the discovered traces."
        )

    all_turns.sort(
        key=lambda turn: (
            turn.assistant_start_ms,
            turn.turn_index,
            turn.export_session_id,
        )
    )
    mooncake_rows = build_mooncake_rows(
        all_turns,
        tokenizer,
        args.block_size,
        show_progress=show_progress,
    )
    sidecar_rows = [turn.sidecar for turn in all_turns]

    output_path = Path(args.output_file).expanduser()
    sidecar_path = sidecar_path_for(output_path)
    write_jsonl(output_path, mooncake_rows)
    write_jsonl(sidecar_path, sidecar_rows)

    print(
        "Wrote "
        f"{len(mooncake_rows)} Mooncake rows across {len(sessions)} sessions to {output_path}"
    )
    print(f"Wrote {len(sidecar_rows)} sidecar rows to {sidecar_path}")
    print(f"Discovered {len(trace_files)} trace files")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
