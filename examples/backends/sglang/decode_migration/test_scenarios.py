#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Live HTTP scenarios for the decode-to-decode migration prototype."""

from __future__ import annotations

import argparse
import http.client
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


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


def wait_ready(base_url: str, models: set[str], timeout: float = 240.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2) as response:
                data = json.load(response).get("data") or []
                available = {item.get("id") for item in data}
                if response.status == 200 and models <= available:
                    return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Dynamo frontend did not become ready: {last_error}")


def completion(
    base_url: str,
    model: str,
    max_tokens: int,
) -> dict:
    return request_json(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": PROMPT,
            "temperature": 0,
            "seed": 1234,
            "max_tokens": max_tokens,
        },
    )


def cancel_stream(base_url: str, model: str, chunks_to_read: int = 3) -> None:
    host_port = base_url.removeprefix("http://")
    host, port = host_port.rsplit(":", 1)
    conn = http.client.HTTPConnection(host, int(port), timeout=60)
    payload = json.dumps(
        {
            "model": model,
            "prompt": PROMPT,
            "temperature": 0,
            "seed": 77,
            "max_tokens": 256,
            "stream": True,
        }
    )
    conn.request(
        "POST",
        "/v1/completions",
        body=payload,
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


def assert_log(log_dir: Path, filename: str, text: str) -> None:
    contents = (log_dir / filename).read_text(errors="replace")
    if text not in contents:
        raise AssertionError(f"Did not find {text!r} in {filename}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--baseline-model", default="decode-migration-baseline")
    parser.add_argument("--rollback-model", default="decode-migration-rollback")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--migrate-after-tokens", type=int, default=8)
    parser.add_argument("--stream-interval", type=int, default=1)
    args = parser.parse_args()

    wait_ready(
        args.base_url,
        {args.model, args.baseline_model, args.rollback_model},
    )

    baseline = completion(args.base_url, args.baseline_model, max_tokens=64)
    baseline_repeat = completion(args.base_url, args.baseline_model, max_tokens=64)
    migrated = completion(args.base_url, args.model, max_tokens=64)
    baseline_text = baseline["choices"][0]["text"]
    baseline_repeat_text = baseline_repeat["choices"][0]["text"]
    migrated_text = migrated["choices"][0]["text"]
    if baseline_repeat_text != baseline_text:
        raise AssertionError(
            "Source-only deterministic baseline is not repeatable:\n"
            f"first={baseline_text!r}\nsecond={baseline_repeat_text!r}"
        )
    if migrated_text != baseline_text:
        raise AssertionError(
            "Migrated deterministic output differs from the source-only baseline:\n"
            f"baseline={baseline_text!r}\nmigrated={migrated_text!r}"
        )

    handoff_finish_baseline = completion(
        args.base_url,
        args.baseline_model,
        max_tokens=args.migrate_after_tokens + 1,
    )
    handoff_finish = completion(
        args.base_url,
        args.model,
        max_tokens=args.migrate_after_tokens + 1,
    )
    if handoff_finish["choices"][0]["text"] != handoff_finish_baseline["choices"][0]["text"]:
        raise AssertionError("Request finishing immediately after handoff diverged")

    rollback = completion(
        args.base_url,
        args.rollback_model,
        max_tokens=64,
    )
    if rollback["choices"][0]["text"] != baseline_text:
        raise AssertionError("Source rollback output differs from baseline")

    short = completion(args.base_url, args.model, max_tokens=4)
    if not short["choices"][0]["text"]:
        raise AssertionError("Short finish-race request returned no output")

    # The coordinator executes prepare when the generator resumes after yielding
    # the threshold chunk, so consume one more SSE event before disconnecting.
    cancel_chunks = math.ceil(args.migrate_after_tokens / args.stream_interval) + 1
    cancel_stream(args.base_url, args.model, chunks_to_read=cancel_chunks)
    time.sleep(1)
    after_cancel = completion(args.base_url, args.model, max_tokens=24)
    if not after_cancel["choices"][0]["text"]:
        raise AssertionError("Worker did not complete a request after cancellation")

    assert_log(args.log_dir, "fast.log", "Prepared decode migration")
    assert_log(args.log_dir, "fast.log", "Started decode migration transfer")
    assert_log(args.log_dir, "fast.log", "Decode migration transfer completed")
    assert_log(args.log_dir, "fast.log", "action=commit")
    assert_log(args.log_dir, "fast.log", "action=resume")
    assert_log(args.log_dir, "slow.log", "Reserved decode migration destination")
    assert_log(args.log_dir, "slow.log", "Armed decode migration destination")
    assert_log(args.log_dir, "slow.log", "action=activate")

    print(
        json.dumps(
            {
                "status": "passed",
                "baseline_matches_migration": True,
                "finish_race": "passed",
                "finish_immediately_after_handoff": "passed",
                "destination_failure_rollback": "passed",
                "cancellation_and_recovery": "passed",
                "generated_characters": len(migrated_text),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
