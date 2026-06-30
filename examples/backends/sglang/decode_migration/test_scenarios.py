#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Live correctness scenarios for request-level decode migration."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

PROMPT = (
    "Write a numbered list of practical ways to make a distributed inference "
    "service reliable. Keep every item concise."
)


def request_json(url: str, payload: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_ready(base_url: str, model: str, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=3) as response:
                available = {
                    item.get("id") for item in (json.load(response).get("data") or [])
                }
                if response.status == 200 and model in available:
                    return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Dynamo frontend did not become ready: {last_error}")


def source_nvext() -> dict:
    return {
        "routing_constraints": {"required_taints": ["decode/fast"]},
        "extra_fields": ["completion_token_ids"],
    }


def destination_nvext() -> dict:
    return {
        "routing_constraints": {"required_taints": ["decode/slow"]},
        "extra_fields": ["completion_token_ids"],
    }


def migration_nvext(sequence_length: int) -> dict:
    return {
        "decode_migration": {
            "source": {"required_taints": ["decode/fast"]},
            "destination": {"required_taints": ["decode/slow"]},
            "trigger": {"type": "sequence_length", "tokens": sequence_length},
        },
        "extra_fields": ["completion_token_ids"],
    }


def completion(
    base_url: str,
    model: str,
    max_tokens: int,
    nvext: dict,
    prompt: str | list[int] = PROMPT,
) -> dict:
    return request_json(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "seed": 1234,
            "n": 1,
            "best_of": 1,
            "max_tokens": max_tokens,
            "nvext": nvext,
        },
    )


def completion_token_ids(response: dict) -> list[int]:
    token_ids = (response.get("nvext") or {}).get("completion_token_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise AssertionError(
            f"Response omitted nvext.completion_token_ids: {response!r}"
        )
    return token_ids


def destination_replay_token_ids(
    base_url: str,
    model: str,
    tokenizer,
    source_response: dict,
    max_tokens: int,
    handoff_output_tokens: int,
) -> list[int]:
    source_ids = completion_token_ids(source_response)
    if len(source_ids) < handoff_output_tokens:
        raise AssertionError(
            "Source baseline finished before the heterogeneous-TP replay frontier"
        )
    prefix = source_ids[:handoff_output_tokens]
    remaining = max_tokens - handoff_output_tokens
    if remaining <= 0:
        return prefix[:max_tokens]
    replay = completion(
        base_url,
        model,
        remaining,
        destination_nvext(),
        prompt=tokenizer.encode(PROMPT) + prefix,
    )
    return prefix + completion_token_ids(replay)


def cancel_stream(
    base_url: str,
    model: str,
    nvext: dict,
    chunks_to_read: int,
) -> None:
    host, port = base_url.removeprefix("http://").rsplit(":", 1)
    conn = http.client.HTTPConnection(host, int(port), timeout=120)
    conn.request(
        "POST",
        "/v1/completions",
        body=json.dumps(
            {
                "model": model,
                "prompt": PROMPT,
                "temperature": 0,
                "seed": 77,
                "n": 1,
                "best_of": 1,
                "max_tokens": 256,
                "stream": True,
                "nvext": nvext,
            }
        ),
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"Cancellation stream failed with HTTP {response.status}")
    chunks = 0
    while chunks < chunks_to_read:
        line = response.readline()
        if not line:
            raise RuntimeError("Streaming response ended before cancellation point")
        if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
            chunks += 1
    conn.close()


def log_text(log_dir: Path, filename: str) -> str:
    return (log_dir / filename).read_text(errors="replace")


def wait_for_log(log_dir: Path, filename: str, text: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if text in log_text(log_dir, filename):
            return
        time.sleep(0.25)
    raise AssertionError(f"Did not find {text!r} in {filename}")


def wait_for_log_count(
    log_dir: Path,
    filename: str,
    text: str,
    minimum: int,
    timeout: float = 30,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_text(log_dir, filename).count(text) >= minimum:
            return
        time.sleep(0.25)
    raise AssertionError(
        f"Did not find at least {minimum} occurrences of {text!r} in {filename}"
    )


def wait_for_any_log_count(
    log_dir: Path,
    filename: str,
    expectations: tuple[tuple[str, int], ...],
    timeout: float = 30,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = log_text(log_dir, filename)
        if any(text.count(needle) >= minimum for needle, minimum in expectations):
            return
        time.sleep(0.25)
    raise AssertionError(
        f"Did not find any expected log count in {filename}: {expectations!r}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--migrate-after-tokens", type=int, default=8)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument(
        "--parity-mode",
        choices=("source", "destination-replay", "migration-repeat"),
        default="source",
        help=(
            "Use strict source parity for equal TP, strict destination replay "
            "for diagnostics, or deterministic migration repeats for heterogeneous TP."
        ),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    prompt_tokens = len(tokenizer.encode(PROMPT))
    migration_sequence_length = prompt_tokens + args.migrate_after_tokens
    migration_policy = migration_nvext(migration_sequence_length)

    wait_ready(args.base_url, args.model)

    # A request that finishes before the trigger must remain on the source.
    frontend_before = log_text(args.log_dir, "frontend.log")
    short = completion(
        args.base_url,
        args.model,
        max_tokens=max(1, args.migrate_after_tokens - 2),
        nvext=migration_policy,
    )
    if not short["choices"][0]["text"]:
        raise AssertionError("Short finish-race request returned no output")
    frontend_after = log_text(args.log_dir, "frontend.log")
    if frontend_after.count(
        "decode migration trigger reached"
    ) != frontend_before.count("decode migration trigger reached"):
        raise AssertionError("Request finishing before the trigger attempted migration")

    baseline = completion(args.base_url, args.model, 64, source_nvext())
    baseline_repeat = completion(args.base_url, args.model, 64, source_nvext())
    migrated = completion(args.base_url, args.model, 64, migration_policy)
    baseline_text = baseline["choices"][0]["text"]
    baseline_ids = completion_token_ids(baseline)
    baseline_repeat_ids = completion_token_ids(baseline_repeat)
    migrated_ids = completion_token_ids(migrated)
    if baseline_repeat_ids != baseline_ids:
        raise AssertionError("Source-only deterministic baseline is not repeatable")

    handoff_output_tokens = (
        math.ceil(args.migrate_after_tokens / args.stream_interval)
        * args.stream_interval
    )
    if migrated_ids[:handoff_output_tokens] != baseline_ids[:handoff_output_tokens]:
        raise AssertionError(
            "Migrated output changed before the destination handoff frontier"
        )
    replay_common_prefix_tokens = None
    if args.parity_mode == "source":
        expected_migrated_ids = baseline_ids
    elif args.parity_mode == "destination-replay":
        expected_migrated_ids = destination_replay_token_ids(
            args.base_url,
            args.model,
            tokenizer,
            baseline,
            max_tokens=64,
            handoff_output_tokens=handoff_output_tokens,
        )
    else:
        migrated_repeat = completion(args.base_url, args.model, 64, migration_policy)
        expected_migrated_ids = completion_token_ids(migrated_repeat)
        replay_ids = destination_replay_token_ids(
            args.base_url,
            args.model,
            tokenizer,
            baseline,
            max_tokens=64,
            handoff_output_tokens=handoff_output_tokens,
        )
        replay_common_prefix_tokens = next(
            (
                index
                for index, (migrated_id, replay_id) in enumerate(
                    zip(migrated_ids, replay_ids)
                )
                if migrated_id != replay_id
            ),
            min(len(migrated_ids), len(replay_ids)),
        )
    if migrated_ids != expected_migrated_ids:
        raise AssertionError(
            "Migrated token IDs differ from the selected parity oracle:\n"
            f"mode={args.parity_mode}\n"
            f"expected={expected_migrated_ids!r}\nmigrated={migrated_ids!r}"
        )

    wait_for_log(
        args.log_dir,
        "frontend.log",
        "decode migration destination handoff started",
    )
    wait_for_log(args.log_dir, "fast.log", "Decode migration transfer completed")
    wait_for_log(args.log_dir, "slow.log", "Bound decode migration destination")

    near_boundary_baseline = completion(
        args.base_url,
        args.model,
        args.migrate_after_tokens + 1,
        source_nvext(),
    )
    near_boundary = completion(
        args.base_url,
        args.model,
        args.migrate_after_tokens + 1,
        migration_policy,
    )
    near_boundary_ids = completion_token_ids(near_boundary)
    if args.parity_mode == "source":
        expected_near_boundary_ids = completion_token_ids(near_boundary_baseline)
    elif args.parity_mode == "destination-replay":
        expected_near_boundary_ids = destination_replay_token_ids(
            args.base_url,
            args.model,
            tokenizer,
            near_boundary_baseline,
            max_tokens=args.migrate_after_tokens + 1,
            handoff_output_tokens=handoff_output_tokens,
        )
    else:
        near_boundary_repeat = completion(
            args.base_url,
            args.model,
            args.migrate_after_tokens + 1,
            migration_policy,
        )
        expected_near_boundary_ids = completion_token_ids(near_boundary_repeat)
    if near_boundary_ids != expected_near_boundary_ids:
        raise AssertionError("Request finishing immediately after handoff diverged")

    recovery_source = completion(args.base_url, args.model, 24, source_nvext())
    if args.parity_mode == "source":
        expected_recovery_ids = completion_token_ids(recovery_source)
    elif args.parity_mode == "destination-replay":
        expected_recovery_ids = destination_replay_token_ids(
            args.base_url,
            args.model,
            tokenizer,
            recovery_source,
            max_tokens=24,
            handoff_output_tokens=handoff_output_tokens,
        )
    else:
        expected_recovery_ids = completion_token_ids(
            completion(args.base_url, args.model, 24, migration_policy)
        )

    cancel_chunks = math.ceil(args.migrate_after_tokens / args.stream_interval) + 1
    frontend_cancel_before = log_text(args.log_dir, "frontend.log").count(
        "Stream closed unexpectedly; issuing cancellation"
    )
    fast_abort_before = log_text(args.log_dir, "fast.log").count(
        "Calling SGLang abort_request for Request ID"
    )
    fast_commit_before = log_text(args.log_dir, "fast.log").count(
        "action=commit status=transferred"
    )
    slow_before = log_text(args.log_dir, "slow.log")
    slow_reservations_before = slow_before.count(
        "Reserved decode migration destination"
    )
    slow_aborts_before = slow_before.count("action=abort status=aborted")
    slow_monitor_aborts_before = slow_before.count(
        "Calling SGLang abort_request for Request ID"
    )
    cancel_stream(
        args.base_url,
        args.model,
        migration_policy,
        chunks_to_read=cancel_chunks,
    )
    wait_for_log_count(
        args.log_dir,
        "frontend.log",
        "Stream closed unexpectedly; issuing cancellation",
        frontend_cancel_before + 1,
    )
    wait_for_any_log_count(
        args.log_dir,
        "fast.log",
        (
            ("Calling SGLang abort_request for Request ID", fast_abort_before + 1),
            ("action=commit status=transferred", fast_commit_before + 1),
        ),
    )
    # Allow an in-flight prepare RPC to either create and abort its reservation
    # or observe cancellation before reaching the destination.
    time.sleep(0.5)
    slow_after_cancel = log_text(args.log_dir, "slow.log")
    if slow_after_cancel.count("Reserved decode migration destination") > (
        slow_reservations_before
    ):
        wait_for_any_log_count(
            args.log_dir,
            "slow.log",
            (
                ("action=abort status=aborted", slow_aborts_before + 1),
                (
                    "Calling SGLang abort_request for Request ID",
                    slow_monitor_aborts_before + 1,
                ),
            ),
        )
    recovery_commits_before = log_text(args.log_dir, "frontend.log").count(
        "decode migration destination handoff started"
    )
    after_cancel = completion(args.base_url, args.model, 24, migration_policy)
    if completion_token_ids(after_cancel) != expected_recovery_ids:
        raise AssertionError(
            "Post-cancellation migrated output differs from its deterministic oracle"
        )
    wait_for_log_count(
        args.log_dir,
        "frontend.log",
        "decode migration destination handoff started",
        recovery_commits_before + 1,
    )

    # Two requests can cross the trigger together. Both handoffs must remain
    # independent and neither client stream may corrupt or hang.
    commits_before = log_text(args.log_dir, "frontend.log").count(
        "decode migration destination handoff started"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                completion,
                args.base_url,
                args.model,
                64,
                migration_policy,
            )
            for _ in range(2)
        ]
        concurrent_results = [future.result(timeout=180) for future in futures]
    if args.parity_mode == "source":
        concurrent_expected_ids = (baseline_ids,)
    elif args.parity_mode == "migration-repeat":
        concurrent_expected_ids = (migrated_ids,)
    else:
        concurrent_expected_ids = (migrated_ids,)
    for result in concurrent_results:
        if completion_token_ids(result) not in concurrent_expected_ids:
            raise AssertionError(
                "Concurrent migration output differs from its deterministic oracle"
            )
    wait_for_log_count(
        args.log_dir,
        "frontend.log",
        "decode migration destination handoff started",
        commits_before + 2,
    )

    for worker_log in ("fast.log", "slow.log"):
        worker_text = log_text(args.log_dir, worker_log)
        if "Scheduler hit an exception" in worker_text:
            raise AssertionError(f"Scheduler crashed during scenarios: {worker_log}")
        if "pool memory leak detected" in worker_text:
            raise AssertionError(
                f"KV pool leak detected during scenarios: {worker_log}"
            )

    print(
        json.dumps(
            {
                "status": "passed",
                "baseline_matches_migration": args.parity_mode == "source",
                "parity_mode": args.parity_mode,
                "destination_replay_common_prefix_tokens": (
                    replay_common_prefix_tokens
                ),
                "finish_before_trigger": "passed",
                "finish_immediately_after_handoff": "passed",
                "cancellation_and_recovery": "passed",
                "concurrent_triggered_requests": "passed",
                "stream_interval": args.stream_interval,
                "migration_sequence_length": migration_sequence_length,
                "generated_characters": len(baseline_text),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
