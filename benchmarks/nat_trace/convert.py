# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert NAT (NeMo Agent Toolkit) profiler traces to mooncake format.

Input: all_requests_profiler_traces.json from NAT profiler
Output: mooncake-style JSONL with session_id for multi-turn serialization

Example output:
    {"session_id": "conv_0", "input_length": 9176, "output_length": 142, "hash_ids": [1, 2, 3]}
    {"session_id": "conv_0", "input_length": 9500, "output_length": 98, "hash_ids": [1, 2, 3, 4]}
"""

import argparse
import json
import os
import re
from collections import defaultdict

from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.synthesis.rolling_hasher import RollingHasher
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert NAT profiler traces to mooncake format"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input all_requests_profiler_traces.json file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to the output mooncake-style jsonl file. If not provided, will use input file name with .jsonl extension",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name/path for hashing. If not provided, will try to infer from trace or use a default",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Block size for hash generation (default: 64)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Limit the number of requests (conversations) to process. If not specified, all requests are processed.",
    )
    parser.add_argument(
        "--skip-requests",
        type=int,
        default=0,
        help="Skip the first N requests (default: 0)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=500,
        help="Delay in ms to add between LLM calls within a session (simulates tool call timing). If not specified, no delay field is added.",
    )
    return parser.parse_args()


def load_json_robust(filepath: str) -> list:
    """
    Load JSON file robustly, handling potentially truncated files.

    Returns list of complete request objects.
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Warning: JSON file appears truncated, attempting partial parse...")

    # Find all request boundaries
    pattern = r'"request_number":\s*(\d+)'
    matches = list(re.finditer(pattern, content))

    if not matches:
        raise ValueError("No valid requests found in file")

    requests = []
    for i, match in enumerate(matches):
        # Find the opening brace for this request
        start = content.rfind("{", 0, match.start())

        # Find the end (next request or end of file)
        if i + 1 < len(matches):
            end = content.rfind("{", 0, matches[i + 1].start())
        else:
            # Last request - try to find closing
            end = len(content)

        chunk = content[start:end].rstrip().rstrip(",")

        try:
            req = json.loads(chunk)
            requests.append(req)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse request {match.group(1)}, skipping...")
            continue

    print(f"Successfully parsed {len(requests)} complete requests")
    return requests


def extract_llm_calls(request: dict) -> list[dict]:
    """
    Extract LLM calls from a request's intermediate_steps.

    Matches LLM_START and LLM_END events by UUID to get:
    - chat_inputs (from LLM_START.metadata)
    - prompt_tokens, completion_tokens (from LLM_END.usage_info)

    Returns list of dicts with: chat_inputs, prompt_tokens, completion_tokens, model_name
    """
    steps = request.get("intermediate_steps", [])

    # Index LLM_START events by UUID
    llm_starts = {}
    for step in steps:
        payload = step.get("payload", {})
        if payload.get("event_type") == "LLM_START":
            uuid = payload.get("UUID")
            if uuid:
                llm_starts[uuid] = payload

    # Match with LLM_END events
    llm_calls = []
    for step in steps:
        payload = step.get("payload", {})
        if payload.get("event_type") == "LLM_END":
            uuid = payload.get("UUID")
            if uuid and uuid in llm_starts:
                start_payload = llm_starts[uuid]

                # Get chat_inputs from metadata
                metadata = start_payload.get("metadata", {})
                chat_inputs = metadata.get("chat_inputs", [])

                # Get token counts from usage_info
                usage_info = payload.get("usage_info", {})
                token_usage = usage_info.get("token_usage", {})
                prompt_tokens = token_usage.get("prompt_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens", 0)

                # Get model name
                model_name = start_payload.get("name", "unknown")

                llm_calls.append(
                    {
                        "chat_inputs": chat_inputs,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "model_name": model_name,
                        "event_timestamp": payload.get("event_timestamp"),
                    }
                )

    # Sort by timestamp to preserve order
    llm_calls.sort(key=lambda x: x.get("event_timestamp", 0) or 0)

    return llm_calls


def texts_to_hashes_and_lengths(
    tokenizer: Tokenizer,
    texts: list[str],
    block_size: int,
) -> tuple[list[list[int]], list[int]]:
    """
    Convert texts to hash IDs and token lengths.

    Returns:
        Tuple of (hash_ids_list, token_lengths) where:
        - hash_ids_list: List of hash ID sequences, one per input text
        - token_lengths: List of token counts, one per input text
    """
    hasher = RollingHasher(block_size=block_size)
    hash_results: list[list[int]] = []
    length_results: list[int] = []

    for text in texts:
        tokens = tokenizer.encode(text)
        length_results.append(len(tokens))

        blocks: list[list[int]] = [
            tokens[i : i + block_size] for i in range(0, len(tokens), block_size)
        ]
        if blocks:
            hashes = hasher.hash_token_blocks(blocks)
            hash_results.append(hashes)
        else:
            hash_results.append([])

    return hash_results, length_results


def chat_inputs_to_text(chat_inputs: list) -> str:
    """
    Convert chat_inputs array to a single text string for hashing.

    Concatenates all message contents with newlines.
    """
    if not chat_inputs:
        return ""

    texts = []
    for msg in chat_inputs:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if content:
                texts.append(str(content))
        elif isinstance(msg, str):
            texts.append(msg)

    return "\n".join(texts)


def convert_to_mooncake(
    requests: list,
    tokenizer_name: str,
    block_size: int,
    skip_requests: int = 0,
    num_requests: int | None = None,
    delay: int | None = None,
) -> list[dict]:
    """
    Convert NAT requests to mooncake format.

    Args:
        requests: List of request dicts from NAT profiler
        tokenizer_name: Tokenizer name for hashing
        block_size: Block size for hash generation
        skip_requests: Number of requests to skip
        num_requests: Maximum number of requests to process
        delay: Delay in ms to add between LLM calls within a session

    Returns:
        List of mooncake-format dicts
    """
    # Apply skip and limit
    requests = requests[skip_requests:]
    if num_requests is not None:
        requests = requests[:num_requests]

    print(f"Processing {len(requests)} requests...")

    # Phase 1: Collect all texts and metadata
    all_entries = []  # List of (session_id, completion_tokens, text)

    for req in tqdm(requests, desc="Extracting LLM calls"):
        request_number = req.get("request_number", 0)
        session_id = f"conv_{request_number}"

        llm_calls = extract_llm_calls(req)

        if not llm_calls:
            print(f"Warning: No LLM calls found in request {request_number}")
            continue

        for call in llm_calls:
            # Convert chat_inputs to text for hashing
            text = chat_inputs_to_text(call["chat_inputs"])

            if not text:
                print(
                    f"Warning: Empty text in request {request_number}, skipping LLM call"
                )
                continue

            all_entries.append(
                (
                    session_id,
                    call["completion_tokens"],
                    text,
                )
            )

    if not all_entries:
        print("No valid LLM calls found")
        return []

    # Phase 2: Tokenize texts to get hash IDs and token lengths
    all_texts = [entry[2] for entry in all_entries]
    print(f"Tokenizing and hashing {len(all_texts)} texts...")

    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    all_hash_ids, all_input_lengths = texts_to_hashes_and_lengths(
        tokenizer, all_texts, block_size
    )

    # Phase 3: Build mooncake entries
    mooncake_data = []
    seen_sessions = set()
    for (session_id, completion_tokens, _), hash_ids, input_length in zip(
        all_entries, all_hash_ids, all_input_lengths, strict=True
    ):
        mooncake_entry = {
            "session_id": session_id,
            "input_length": input_length,
            "output_length": completion_tokens,
            "hash_ids": hash_ids,
        }
        # Add delay for all but the first entry in each session
        if delay is not None:
            if session_id in seen_sessions:
                mooncake_entry["delay"] = delay
            else:
                seen_sessions.add(session_id)
        mooncake_data.append(mooncake_entry)

    return mooncake_data


def infer_tokenizer(requests: list) -> str:
    """
    Try to infer tokenizer from model name in traces.

    Maps common model names to HuggingFace tokenizer paths.
    """
    model_mapping = {
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
        "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    }

    # Look for model name in first request
    for req in requests[:5]:
        llm_calls = extract_llm_calls(req)
        for call in llm_calls:
            model_name = call.get("model_name", "").lower()
            for pattern, tokenizer in model_mapping.items():
                if pattern in model_name:
                    print(f"Inferred tokenizer from model '{model_name}': {tokenizer}")
                    return tokenizer

    # Default fallback
    default = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Could not infer tokenizer, using default: {default}")
    return default


def print_statistics(mooncake_data: list):
    """Print statistics about the converted data."""
    if not mooncake_data:
        print("No data to report statistics on.")
        return

    print("\n" + "=" * 60)
    print("CONVERSION STATISTICS")
    print("=" * 60)

    # Count sessions and turns
    sessions = defaultdict(list)
    for entry in mooncake_data:
        sessions[entry["session_id"]].append(entry)

    print(f"\nSessions (conversations): {len(sessions)}")

    turns_per_session = [len(turns) for turns in sessions.values()]
    print(
        f"Turns per session: min={min(turns_per_session)}, max={max(turns_per_session)}, avg={sum(turns_per_session)/len(turns_per_session):.1f}"
    )

    print(f"Total LLM calls: {len(mooncake_data)}")

    # Token statistics
    input_lengths = [e["input_length"] for e in mooncake_data]
    output_lengths = [e["output_length"] for e in mooncake_data]

    print("\nInput Length (prompt_tokens):")
    print(f"  Min: {min(input_lengths)}")
    print(f"  Max: {max(input_lengths)}")
    print(f"  Avg: {sum(input_lengths)/len(input_lengths):.1f}")

    print("\nOutput Length (completion_tokens):")
    print(f"  Min: {min(output_lengths)}")
    print(f"  Max: {max(output_lengths)}")
    print(f"  Avg: {sum(output_lengths)/len(output_lengths):.1f}")

    # Hash statistics
    hash_lengths = [len(e["hash_ids"]) for e in mooncake_data]
    print("\nHash IDs per entry:")
    print(f"  Min: {min(hash_lengths)}")
    print(f"  Max: {max(hash_lengths)}")
    print(f"  Avg: {sum(hash_lengths)/len(hash_lengths):.1f}")

    print("=" * 60)


def main():
    args = parse_args()

    # Load the JSON file
    print(f"Loading {args.input_file}...")
    requests = load_json_robust(args.input_file)
    print(f"Loaded {len(requests)} requests")

    # Determine tokenizer
    if args.tokenizer:
        tokenizer_name = args.tokenizer
    else:
        tokenizer_name = infer_tokenizer(requests)

    print(f"Using tokenizer: {tokenizer_name}")
    print(f"Block size: {args.block_size}")

    # Convert to mooncake format
    mooncake_data = convert_to_mooncake(
        requests,
        tokenizer_name,
        args.block_size,
        skip_requests=args.skip_requests,
        num_requests=args.num_requests,
        delay=args.delay,
    )

    # Print statistics
    print_statistics(mooncake_data)

    # Determine output file
    if args.output_file is None:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = base_name + "_mooncake.jsonl"

    # Save to file
    with open(args.output_file, "w") as f:
        for entry in mooncake_data:
            f.write(json.dumps(entry) + "\n")

    print(f"\nSaved {len(mooncake_data)} entries to {args.output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
