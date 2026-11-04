# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only microbenchmark for the Python SGLang delta-tokenizer adapter.

This complements the Rust benches in lib/llm/benches and the mock-worker serving
harness in this directory. It measures no network, model engine or serving TTFT.
"""
import argparse
import json
import resource
import statistics
import time
from importlib.metadata import version
from pathlib import Path

from transformers import AutoTokenizer

from dynamo.frontend.delta_tokenizer import DeltaConfig
from dynamo.frontend.toktier_session import create_delta_tokenizer


def summarize(samples, cpu_seconds, parity, stats):
    ordered = sorted(samples)

    def quantile(fraction):
        return (
            ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))] * 1000
        )

    return {
        "requests": len(samples),
        "p50_ms": statistics.median(samples) * 1000,
        "p95_ms": quantile(0.95),
        "p99_ms": quantile(0.99),
        "throughput_rps": len(samples) / sum(samples),
        "cpu_seconds": cpu_seconds,
        "exact_parity_count": parity,
        "state": stats,
        "latency_ms_samples": [sample * 1000 for sample in samples],
    }


def run_case(
    reference, directory, target, iterations, append_repeats, mode, max_entries
):
    block = "The cat sat by the window, watching raindrops race down the glass.\n"
    block_ids = len(reference.encode(block, add_special_tokens=False))
    text = block * max(1, target // block_ids)
    base_count = len(reference.encode(text, add_special_tokens=False))
    config = DeltaConfig(
        verify_rate=1 if mode == "verified" else 0, max_entries=max_entries
    )
    manager = (
        None
        if mode == "reference"
        else create_delta_tokenizer(reference, directory, "benchmark", config)
    )
    if mode != "reference" and manager is None:
        raise RuntimeError("tokenizer artifact or installed loader is not admitted")
    scope, template = "benchmark", "rendered-text-v1"
    clock = [time.monotonic()]
    if manager is not None:
        manager.clock = lambda: clock[0]
        manager.tokenize("session", text, scope=scope, template_identity=template)
        text += "Warm-up user turn.\n"
        manager.tokenize("session", text, scope=scope, template_identity=template)
        manager.wait_for_idle()
    samples, cpu_seconds, parity = [], 0.0, 0
    paired_reference = []
    paired_reference_cpu = 0.0
    seed_wait_seconds = 0.0
    workload_cpu_start = time.process_time()
    try:
        for index in range(iterations):
            text += (
                "user turn: please inspect the result and continue.\n" * append_repeats
            )
            if mode == "invalidated":
                text = ("X" if index % 2 else "Y") + text[1:]
            elif mode == "expired":
                clock[0] += 301
                manager.expire()
            elif mode == "evicted":
                # One complete cache's worth of unrelated entries forces
                # deterministic eviction without a process-wide reset.
                for other in range(config.max_entries):
                    manager.tokenize(
                        f"evict-{index}-{other}",
                        text,
                        scope=scope,
                        template_identity=template,
                    )
                manager.wait_for_idle()
            start_cpu = time.process_time()
            start = time.perf_counter()
            if manager is None:
                actual = reference.encode(text, add_special_tokens=False)
            else:
                actual = manager.tokenize(
                    "session", text, scope=scope, template_identity=template
                )
            samples.append(time.perf_counter() - start)
            cpu_seconds += time.process_time() - start_cpu
            reference_cpu_start = time.process_time()
            reference_start = time.perf_counter()
            expected = reference.encode(text, add_special_tokens=False)
            paired_reference.append(time.perf_counter() - reference_start)
            paired_reference_cpu += time.process_time() - reference_cpu_start
            if actual != expected:
                raise AssertionError("token parity failed")
            parity += 1
            if manager is not None:
                seed_start = time.perf_counter()
                manager.wait_for_idle()
                seed_wait_seconds += time.perf_counter() - seed_start
        stats = manager.stats() if manager is not None else {}
        report = summarize(samples, cpu_seconds, parity, stats)
        report["paired_reference"] = summarize(
            paired_reference, paired_reference_cpu, parity, {}
        )
        report["initial_tokens"] = base_count
        report["append_repeats"] = append_repeats
        report["seed_wait_seconds"] = seed_wait_seconds
        report["workload_cpu_seconds_including_seeds_and_parity"] = (
            time.process_time() - workload_cpu_start
        )
        return report
    finally:
        if manager is not None:
            manager.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=[128, 32768, 65536, 131072]
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-entries", type=int, default=1)
    parser.add_argument("--append-repeats", nargs="+", type=int, default=[1, 128])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["reference", "warm", "verified", "invalidated", "expired", "evicted"],
        default=["reference", "warm", "verified", "invalidated", "expired"],
    )
    args = parser.parse_args()
    if args.iterations < 2 or any(
        value <= 0 for value in [*args.lengths, *args.append_repeats]
    ):
        parser.error("iterations must be >= 2 and workload sizes must be positive")
    reference = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True
    )
    report = {
        "versions": {
            name: version(name) for name in ["toktier", "transformers", "tokenizers"]
        },
        "kind": "tokenizer_microbenchmark",
        "results": {},
    }
    for length in args.lengths:
        for appended in args.append_repeats:
            for mode in args.modes:
                key = f"{length}:{appended}:{mode}"
                result = run_case(
                    reference,
                    str(args.model_dir),
                    length,
                    args.iterations,
                    appended,
                    mode,
                    args.max_entries,
                )
                report["results"][key] = result
                print(key, json.dumps(result), flush=True)
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
