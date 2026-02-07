# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert OpenAI-style telemetry JSONL to mooncake format.

Input: telemetry.jsonl with llm_call and tool_call events
Output: mooncake-style JSONL with agent_type and priority fields

Example output:
    {"session_id": "082e33c7-...", "agent_type": "deep_coordinator", "input_length": 2426, "output_length": 33, "hash_ids": [1, 2, 3], "priority": "HIGH"}
"""

import argparse
import json
import os
from collections import defaultdict

from aiperf.common.tokenizer import Tokenizer
from common import DEFAULT_BLOCK_SIZE, DEFAULT_TOKENIZER, texts_to_hashes_and_lengths
from tqdm import tqdm

AGENT_PREFIX_MAP = {
    "You are a Deep Research agent": "deep_coordinator",
    "Gather and synthesize comprehe": "research_worker",
    "For the given task, generate a": "research_planner",
    "Current date and time:": "shallow_agent",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert telemetry JSONL to mooncake format"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input telemetry.jsonl file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to the output mooncake-style JSONL file. Defaults to <input>_mooncake.jsonl",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer name/path for hashing (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"Block size for hash generation (default: {DEFAULT_BLOCK_SIZE})",
    )
    return parser.parse_args()


def load_and_sort(filepath: str) -> list[dict]:
    """Load telemetry JSONL, filter to llm_call events, sort by (session_id, timestamp)."""
    events = []
    with open(filepath) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("event_type") != "llm_call":
                continue
            events.append(obj)

    events.sort(key=lambda e: (e["session_id"], e["timestamp"]))
    return events


def extract_system_prompt(event: dict) -> str | None:
    """Extract the system prompt text from an llm_call event, or None if absent."""
    rp = event.get("request_payload")
    if not isinstance(rp, dict):
        return None
    for msg in rp.get("messages", []):
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    return item["text"]
            return None
        return content
    return None


def extract_first_user_message(event: dict) -> str:
    """Extract the first user message text from an llm_call event."""
    rp = event.get("request_payload")
    if not isinstance(rp, dict):
        return ""
    for msg in rp.get("messages", []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return str(content)
    return ""


def classify_agent(event: dict) -> str:
    """Identify agent type from system prompt prefix or user message content."""
    sys_prompt = extract_system_prompt(event)
    if sys_prompt:
        for prefix, agent_type in AGENT_PREFIX_MAP.items():
            if sys_prompt.startswith(prefix):
                return agent_type
        return "unknown"

    user_msg = extract_first_user_message(event)
    if "Classify the user message" in user_msg:
        return "classifier"
    if "complexity analyzer" in user_msg:
        return "complexity_analyzer"
    return "unknown"


def messages_to_text(event: dict) -> str:
    """Concatenate all message contents from request_payload.messages into a single string."""
    rp = event.get("request_payload")
    if not isinstance(rp, dict):
        return ""
    parts = []
    for msg in rp.get("messages", []):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
    return "\n".join(parts)


def get_output_tokens(event: dict) -> int:
    """Extract completion_tokens from response_payload.usage."""
    rp = event.get("response_payload")
    if not isinstance(rp, dict):
        return 0
    usage = rp.get("usage", {})
    if not isinstance(usage, dict):
        return 0
    return usage.get("completion_tokens", 0)


def convert_to_mooncake(
    events: list[dict],
    tokenizer_name: str,
    block_size: int,
) -> list[dict]:
    """Convert sorted llm_call events to mooncake format."""
    # Phase 1: classify agents
    for event in events:
        event["_agent_type"] = classify_agent(event)

    # Phase 2: collect texts for tokenization
    all_texts = []
    for event in tqdm(events, desc="Extracting messages"):
        all_texts.append(messages_to_text(event))

    # Phase 3: tokenize and hash
    print(f"Tokenizing and hashing {len(all_texts)} texts...")
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    all_hash_ids, all_input_lengths = texts_to_hashes_and_lengths(
        tokenizer, all_texts, block_size
    )

    # Phase 4: build output entries
    mooncake_data = []
    for event, input_length, hash_ids in zip(
        events, all_input_lengths, all_hash_ids, strict=True
    ):
        mooncake_data.append(
            {
                "session_id": event["session_id"],
                "agent_type": event["_agent_type"],
                "input_length": input_length,
                "output_length": get_output_tokens(event),
                "hash_ids": hash_ids,
                "priority": event.get("latency_priority", "MEDIUM"),
            }
        )

    return mooncake_data


def print_statistics(mooncake_data: list[dict]):
    """Print statistics about the converted data."""
    if not mooncake_data:
        print("No data to report statistics on.")
        return

    print("\n" + "=" * 60)
    print("CONVERSION STATISTICS")
    print("=" * 60)

    sessions = defaultdict(list)
    for entry in mooncake_data:
        sessions[entry["session_id"]].append(entry)

    print(f"\nSessions: {len(sessions)}")

    turns_per_session = [len(turns) for turns in sessions.values()]
    print(
        f"Turns per session: min={min(turns_per_session)}, "
        f"max={max(turns_per_session)}, "
        f"avg={sum(turns_per_session) / len(turns_per_session):.1f}"
    )
    print(f"Total LLM calls: {len(mooncake_data)}")

    # Agent type breakdown
    from collections import Counter

    agent_counts = Counter(e["agent_type"] for e in mooncake_data)
    print("\nAgent types:")
    for agent, count in agent_counts.most_common():
        print(f"  {agent}: {count}")

    # Priority breakdown
    priority_counts = Counter(e["priority"] for e in mooncake_data)
    print("\nPriorities:")
    for prio, count in priority_counts.most_common():
        print(f"  {prio}: {count}")

    # Token statistics
    input_lengths = [e["input_length"] for e in mooncake_data]
    output_lengths = [e["output_length"] for e in mooncake_data]

    print("\nInput Length (tokens):")
    print(f"  Min: {min(input_lengths)}")
    print(f"  Max: {max(input_lengths)}")
    print(f"  Avg: {sum(input_lengths) / len(input_lengths):.1f}")

    print("\nOutput Length (tokens):")
    print(f"  Min: {min(output_lengths)}")
    print(f"  Max: {max(output_lengths)}")
    print(f"  Avg: {sum(output_lengths) / len(output_lengths):.1f}")

    # Hash statistics
    hash_lengths = [len(e["hash_ids"]) for e in mooncake_data]
    print("\nHash IDs per entry:")
    print(f"  Min: {min(hash_lengths)}")
    print(f"  Max: {max(hash_lengths)}")
    print(f"  Avg: {sum(hash_lengths) / len(hash_lengths):.1f}")

    print("=" * 60)


def main():
    args = parse_args()

    print(f"Loading {args.input_file}...")
    events = load_and_sort(args.input_file)
    print(f"Loaded {len(events)} llm_call events")

    print(f"Using tokenizer: {args.tokenizer}")
    print(f"Block size: {args.block_size}")

    mooncake_data = convert_to_mooncake(events, args.tokenizer, args.block_size)

    print_statistics(mooncake_data)

    if args.output_file is None:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = base_name + "_mooncake.jsonl"

    with open(args.output_file, "w") as f:
        for entry in mooncake_data:
            f.write(json.dumps(entry) + "\n")

    print(f"\nSaved {len(mooncake_data)} entries to {args.output_file}")
    return 0


if __name__ == "__main__":
    exit(main())
