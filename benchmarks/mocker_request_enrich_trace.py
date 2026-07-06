#!/usr/bin/env python3
"""Join NeMo-Gym DirectRequest and vLLM server traces.

The NeMo-RL rollout trace carries full prompt tokens for Dynamo replay. The
server trace carries true arrival/completion timestamps and the worker that
served the request. This script joins them by shared request ID when available,
falling back to token hashes for legacy traces, and writes an enriched
DirectRequest JSONL that Dynamo will still accept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

EMPTY_TOKEN_HASH = hashlib.sha256(b"[]").hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direct_request_jsonl", type=Path)
    parser.add_argument("server_trace_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--block-size", type=int, default=1152)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any DirectRequest row cannot be matched to a server trace row.",
    )
    parser.add_argument(
        "--no-candidate-prefixes",
        action="store_true",
        help="Skip candidate prefix-overlap fields for every observed worker.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def trace_join_key(row: dict[str, Any]) -> str | None:
    key = row.get("trace_join_key")
    if key:
        return str(key)
    prompt_hash = row.get("prompt_token_hash")
    generation_hash = row.get("generation_token_hash")
    if prompt_hash and generation_hash:
        return f"{prompt_hash}:{generation_hash}"
    return None


def request_id(row: dict[str, Any]) -> str | None:
    # DirectRequest's schema-native correlation field is ``uuid`` while the
    # server trace calls the same logical identifier ``request_id``.
    value = row.get("request_id") or row.get("uuid")
    if value:
        return str(value)
    return None


def as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def is_zero_output_row(row: dict[str, Any]) -> bool:
    generation_hash = row.get("generation_token_hash")
    if generation_hash == EMPTY_TOKEN_HASH:
        return True
    for key in (
        "output_length",
        "output_tokens",
        "completion_tokens",
        "max_output_tokens",
        "extracted_generation_tokens",
    ):
        value = as_int(row.get(key))
        if value is not None:
            return value == 0
    return False


def output_length(row: dict[str, Any]) -> int | None:
    for key in (
        "output_length",
        "output_tokens",
        "completion_tokens",
        "max_output_tokens",
        "extracted_generation_tokens",
    ):
        value = as_int(row.get(key))
        if value is not None:
            return value
    return None


def generation_output_key(row: dict[str, Any]) -> tuple[str, int] | None:
    generation_hash = row.get("generation_token_hash")
    length = output_length(row)
    if generation_hash and length is not None:
        return (str(generation_hash), length)
    return None


def as_int_token_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError):
        return None


def common_prefix_tokens(left: list[int], right: list[int], block_size: int) -> int:
    num_blocks = min(len(left), len(right)) // block_size
    matched_blocks = 0
    while matched_blocks < num_blocks:
        start = matched_blocks * block_size
        end = start + block_size
        if left[start:end] != right[start:end]:
            break
        matched_blocks += 1
    return matched_blocks * block_size


def best_prefix_tokens(
    tokens: list[int], previous_requests: list[list[int]], block_size: int
) -> int:
    return max(
        (
            common_prefix_tokens(tokens, previous_tokens, block_size)
            for previous_tokens in previous_requests
        ),
        default=0,
    )


def sort_worker_ids(worker_ids: set[str]) -> list[str]:
    def key(worker_id: str) -> tuple[int, int | str]:
        try:
            return (0, int(worker_id))
        except ValueError:
            return (1, worker_id)

    return sorted(worker_ids, key=key)


def build_server_index(
    server_rows: list[dict[str, Any]],
) -> dict[str, deque[dict[str, Any]]]:
    rows_by_key: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(
        server_rows,
        key=lambda item: item.get("arrival_timestamp_ms") or float("inf"),
    ):
        key = trace_join_key(row)
        if key:
            rows_by_key[key].append(row)
    return rows_by_key


def build_server_request_id_index(
    server_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_request_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(
        server_rows,
        key=lambda item: item.get("arrival_timestamp_ms") or float("inf"),
    ):
        key = request_id(row)
        if key:
            rows_by_request_id[key].append(row)
    return rows_by_request_id


def _successful_server_row(row: dict[str, Any]) -> bool:
    status_code = as_int(row.get("status_code"))
    status_ok = status_code is None or 200 <= status_code < 300
    return status_ok and row.get("error") in (None, "")


def select_server_attempt(
    request_row: dict[str, Any],
    server_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the server attempt that produced a logical rollout request.

    A logical request ID is stable across HTTP retries, so one DirectRequest can
    correspond to multiple server rows. Prefer the latest successful attempt,
    using token hashes to disambiguate duplicate successful responses when they
    are available.
    """
    successful_rows = [row for row in server_rows if _successful_server_row(row)]
    candidates = successful_rows or server_rows

    direct_join_key = trace_join_key(request_row)
    if direct_join_key:
        exact_rows = [
            row for row in candidates if trace_join_key(row) == direct_join_key
        ]
        if exact_rows:
            return exact_rows[-1]

    direct_generation_key = generation_output_key(request_row)
    if direct_generation_key:
        generation_rows = [
            row
            for row in candidates
            if generation_output_key(row) == direct_generation_key
        ]
        if generation_rows:
            return generation_rows[-1]

    return candidates[-1]


def build_zero_output_server_index(
    server_rows: list[dict[str, Any]],
) -> dict[str, deque[dict[str, Any]]]:
    rows_by_prompt_hash: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(
        server_rows,
        key=lambda item: item.get("arrival_timestamp_ms") or float("inf"),
    ):
        if trace_join_key(row):
            continue
        prompt_hash = row.get("prompt_token_hash")
        if prompt_hash and is_zero_output_row(row):
            rows_by_prompt_hash[str(prompt_hash)].append(row)
    return rows_by_prompt_hash


def build_generation_output_server_index(
    server_rows: list[dict[str, Any]],
) -> dict[tuple[str, int], deque[dict[str, Any]]]:
    rows_by_generation: dict[tuple[str, int], deque[dict[str, Any]]] = defaultdict(
        deque
    )
    for row in sorted(
        server_rows,
        key=lambda item: item.get("arrival_timestamp_ms") or float("inf"),
    ):
        key = generation_output_key(row)
        if key:
            rows_by_generation[key].append(row)
    return rows_by_generation


def enrich_with_server_row(
    request_row: dict[str, Any],
    server_row: dict[str, Any],
    *,
    match_method: str,
) -> dict[str, Any]:
    enriched = dict(request_row)
    for key in (
        "trace_id",
        "request_id",
        "worker_id",
        "chosen_worker",
        "worker_seed",
        "bundle_indices",
        "base_url",
        "node_ip",
        "arrival_timestamp_ms",
        "arrival_monotonic_ms",
        "completion_timestamp_ms",
        "completion_monotonic_ms",
        "duration_ms",
        "num_requests_running_at_arrival",
        "num_requests_waiting_at_arrival",
        "kv_cache_usage_perc_at_arrival",
        "num_requests_running_at_completion",
        "num_requests_waiting_at_completion",
        "kv_cache_usage_perc_at_completion",
        "status_code",
        "streaming",
    ):
        if key in server_row:
            enriched[key] = server_row[key]
    enriched["server_trace_matched"] = True
    enriched["server_trace_match_method"] = match_method
    enriched["server_prompt_token_hash"] = server_row.get("prompt_token_hash")
    enriched["server_generation_token_hash"] = server_row.get("generation_token_hash")
    enriched["server_input_length"] = server_row.get("input_length")
    enriched["server_output_length"] = server_row.get("output_length")
    enriched["server_prompt_hash_mismatch"] = (
        request_row.get("prompt_token_hash") is not None
        and server_row.get("prompt_token_hash") is not None
        and request_row.get("prompt_token_hash") != server_row.get("prompt_token_hash")
    )
    server_prompt_token_ids = as_int_token_list(server_row.get("prompt_token_ids"))
    if server_prompt_token_ids is not None:
        enriched["direct_tokens"] = enriched["tokens"]
        enriched["tokens"] = server_prompt_token_ids
        enriched["input_length"] = len(server_prompt_token_ids)
        enriched["server_prompt_tokens_used_for_replay"] = True
    else:
        enriched["server_prompt_tokens_used_for_replay"] = False
    return enriched


def add_candidate_prefix_fields(
    rows: list[dict[str, Any]], worker_ids: list[str], block_size: int
) -> None:
    history_by_step_worker: dict[str | None, dict[str, list[list[int]]]] = {}

    for row in sorted(
        rows,
        key=lambda item: item.get("arrival_timestamp_ms") or float("inf"),
    ):
        step_id = row.get("step_id")
        step_id = str(step_id) if step_id is not None else None
        history_by_worker = history_by_step_worker.setdefault(
            step_id,
            {worker_id: [] for worker_id in worker_ids},
        )

        tokens = [int(token_id) for token_id in row.get("tokens", [])]
        candidate_prefixes = {
            worker_id: best_prefix_tokens(
                tokens, history_by_worker[worker_id], block_size
            )
            for worker_id in worker_ids
        }
        chosen_worker = row.get("chosen_worker")
        chosen_worker = str(chosen_worker) if chosen_worker is not None else None
        best_worker = max(candidate_prefixes, key=candidate_prefixes.get, default=None)

        row["candidate_prefix_tokens_by_worker"] = candidate_prefixes
        row["chosen_worker_prefix_tokens"] = (
            candidate_prefixes.get(chosen_worker) if chosen_worker else None
        )
        row["best_candidate_worker"] = best_worker
        row["best_candidate_prefix_tokens"] = (
            candidate_prefixes[best_worker] if best_worker is not None else None
        )

        if chosen_worker is not None:
            history_by_worker.setdefault(chosen_worker, []).append(tokens)


def main() -> None:
    args = parse_args()
    direct_rows = load_jsonl(args.direct_request_jsonl)
    server_rows = load_jsonl(args.server_trace_jsonl)
    server_by_request_id = build_server_request_id_index(server_rows)

    enriched_rows: list[dict[str, Any] | None] = [None] * len(direct_rows)
    pending_hash_rows: list[tuple[int, dict[str, Any]]] = []
    matched_server_row_ids: set[int] = set()
    request_id_matches = 0
    request_id_retry_matches = 0
    for index, row in enumerate(direct_rows):
        key = request_id(row)
        attempts = server_by_request_id.pop(key, None) if key is not None else None
        if not attempts:
            pending_hash_rows.append((index, row))
            continue

        server_row = select_server_attempt(row, attempts)
        matched_server_row_ids.update(id(attempt) for attempt in attempts)
        request_id_matches += 1
        if len(attempts) > 1:
            request_id_retry_matches += 1
        enriched = enrich_with_server_row(
            row,
            server_row,
            match_method="request_id",
        )
        enriched["server_trace_attempt_count"] = len(attempts)
        enriched_rows[index] = enriched

    remaining_server_rows = [
        row for row in server_rows if id(row) not in matched_server_row_ids
    ]
    server_by_key = build_server_index(remaining_server_rows)

    pending_rows: list[tuple[int, dict[str, Any]]] = []
    exact_matches = 0
    for index, row in pending_hash_rows:
        key = trace_join_key(row)
        server_row = (
            server_by_key[key].popleft()
            if key in server_by_key and server_by_key[key]
            else None
        )
        if server_row is None:
            pending_rows.append((index, row))
            continue

        matched_server_row_ids.add(id(server_row))
        exact_matches += 1
        enriched_rows[index] = enrich_with_server_row(
            row,
            server_row,
            match_method="trace_join_key",
        )

    remaining_server_rows = [
        row for row in server_rows if id(row) not in matched_server_row_ids
    ]
    zero_output_server_by_prompt_hash = build_zero_output_server_index(
        remaining_server_rows
    )
    generation_output_server_by_key = build_generation_output_server_index(
        remaining_server_rows
    )

    zero_output_fallback_matches = 0
    generation_fallback_matches = 0
    ambiguous_generation_fallback_matches = 0
    unmatched = 0
    for index, row in pending_rows:
        server_row = None
        match_method = "unmatched"
        if is_zero_output_row(row):
            prompt_hash = row.get("prompt_token_hash")
            prompt_hash = str(prompt_hash) if prompt_hash else None
            if (
                prompt_hash is not None
                and prompt_hash in zero_output_server_by_prompt_hash
                and zero_output_server_by_prompt_hash[prompt_hash]
            ):
                server_row = zero_output_server_by_prompt_hash[prompt_hash].popleft()
                match_method = "prompt_hash_zero_output"
                zero_output_fallback_matches += 1

        if server_row is None:
            generation_key = generation_output_key(row)
            if (
                generation_key is not None
                and generation_key in generation_output_server_by_key
                and generation_output_server_by_key[generation_key]
            ):
                candidates = generation_output_server_by_key[generation_key]
                match_method = "generation_hash_output_length"
                if len(candidates) > 1:
                    match_method = "generation_hash_output_length_ambiguous_fifo"
                    ambiguous_generation_fallback_matches += 1
                server_row = candidates.popleft()
                generation_fallback_matches += 1

        if server_row is None:
            unmatched += 1
            enriched = dict(row)
            enriched["server_trace_matched"] = False
        else:
            enriched = enrich_with_server_row(
                row,
                server_row,
                match_method=match_method,
            )
        enriched_rows[index] = enriched

    if unmatched and args.strict:
        raise SystemExit(
            f"{unmatched} DirectRequest rows did not match server trace rows"
        )

    finalized_rows = [row for row in enriched_rows if row is not None]

    if not args.no_candidate_prefixes:
        worker_ids = {
            str(row["chosen_worker"])
            for row in finalized_rows
            if row.get("chosen_worker") is not None
        }
        add_candidate_prefix_fields(
            finalized_rows, sort_worker_ids(worker_ids), args.block_size
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in finalized_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    print(
        f"wrote {len(finalized_rows)} rows to {args.output_jsonl} "
        f"({unmatched} unmatched, "
        f"{request_id_matches} request-ID matches, "
        f"{request_id_retry_matches} request-ID matches with retries, "
        f"{exact_matches} exact matches, "
        f"{zero_output_fallback_matches} zero-output fallback matches, "
        f"{generation_fallback_matches} generation fallback matches, "
        f"{ambiguous_generation_fallback_matches} ambiguous generation fallbacks)"
    )


if __name__ == "__main__":
    main()
