# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only, exact-parity session contention and retention benchmark.

Uses explicitly staged tokenizer artifacts. No model weights or services are
loaded. Setup, warm requests and admission churn are measured separately.
"""
import argparse
import json
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark_delta_tokenizer import summarize
from transformers import AutoTokenizer

from dynamo.frontend.delta_tokenizer import DeltaConfig
from dynamo.frontend.toktier_session import create_delta_tokenizer


def rss_kib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def run(args):
    reference = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    config = DeltaConfig(max_entries=args.active_sessions, max_bytes=args.max_bytes)
    manager = create_delta_tokenizer(reference, args.model_dir, "sessions", config)
    if manager is None:
        raise RuntimeError("staged tokenizer is not admitted")
    prefix = "Review this implementation carefully and explain the changes.\n" * 2048
    texts = [
        prefix + f"Conversation {index}.\n" for index in range(args.active_sessions)
    ]

    # Deliberately share the session ID across scopes to exercise isolation.
    def encode(scope, text):
        result = manager.tokenize("session", text, scope=scope, template_identity="v1")
        if result != reference.encode(text, add_special_tokens=False):
            raise AssertionError("reference token parity failed")
        stats = manager.stats()
        if (
            stats["entries"] > config.max_entries
            or stats["retained_bytes"] > config.max_bytes
        ):
            raise AssertionError("retention limit exceeded")
        return result

    rss_before = rss_kib()
    setup_start, setup_cpu = time.perf_counter(), time.process_time()
    try:
        for index, text in enumerate(texts):
            scope = f"scope-{index}"
            encode(scope, text)
            texts[index] += "Continue.\n"
            encode(scope, texts[index])
        manager.wait_for_idle(timeout=180)
        setup = {
            "wall_seconds": time.perf_counter() - setup_start,
            "cpu_seconds": time.process_time() - setup_cpu,
            "rss_kib": rss_kib(),
            "state": manager.stats(),
        }

        def conversation(index):
            samples = []
            text = texts[index]
            for turn in range(args.turns):
                text += f"Turn {turn}: inspect the next result.\n"
                started = time.perf_counter()
                result = manager.tokenize(
                    "session", text, scope=f"scope-{index}", template_identity="v1"
                )
                samples.append(time.perf_counter() - started)
                if result != reference.encode(text, add_special_tokens=False):
                    raise AssertionError("concurrent reference token parity failed")
            return samples

        started, cpu_started = time.perf_counter(), time.process_time()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            batches = list(executor.map(conversation, range(args.active_sessions)))
        wall, cpu = time.perf_counter() - started, time.process_time() - cpu_started
        samples = [sample for batch in batches for sample in batch]
        warm = summarize(samples, cpu, len(samples), manager.stats())
        warm["wall_seconds_including_parity"] = wall
        warm["throughput_including_parity_rps"] = len(samples) / wall
        warm["rss_kib"] = rss_kib()

        started, cpu_started = time.perf_counter(), time.process_time()
        for index in range(args.churn_sessions):
            encode(f"churn-{index}", prefix + str(index))
        churn = {
            "wall_seconds_including_parity": time.perf_counter() - started,
            "cpu_seconds_including_parity": time.process_time() - cpu_started,
            "exact_parity_count": args.churn_sessions,
            "rss_kib": rss_kib(),
            "state": manager.stats(),
        }
    finally:
        manager.close()
    if manager.stats()["entries"] or manager.stats()["retained_bytes"]:
        raise AssertionError("close did not release reservations")
    return {
        "kind": "tokenizer_session_contention",
        "config": vars(args),
        "rss_before_sessions_kib": rss_before,
        "setup": setup,
        "warm": warm,
        "churn": churn,
        "after_close": {"rss_kib": rss_kib(), "state": manager.stats()},
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "note": "CPU includes external parity encodes; reservations are estimates, not RSS limits.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--active-sessions", type=int, default=4)
    parser.add_argument("--churn-sessions", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--max-bytes", type=int, default=1024**3)
    args = parser.parse_args()
    if (
        min(
            args.active_sessions,
            args.churn_sessions,
            args.workers,
            args.turns,
            args.max_bytes,
        )
        <= 0
    ):
        parser.error("session counts, worker counts, turns and bytes must be positive")
    result = run(args)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
