#!/usr/bin/env python3
"""Benchmark streaming chat-audio completions against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_PROMPT = "Say the word benchmark in a clear voice."


@dataclasses.dataclass
class RequestResult:
    ok: bool
    latency_s: float
    status: int | None
    chunks: int
    response_bytes: int
    audio_samples: int
    sample_rate: int | None
    error: str | None = None


def parse_concurrency(value: str) -> list[int]:
    levels = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        level = int(raw)
        if level <= 0:
            raise argparse.ArgumentTypeError("concurrency levels must be positive")
        levels.append(level)
    if not levels:
        raise argparse.ArgumentTypeError("at least one concurrency level is required")
    return levels


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "modalities": ["audio"],
        "audio": {"format": args.audio_format, "voice": args.voice},
        "stream": True,
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
    }


def post_streaming_request(args: argparse.Namespace) -> RequestResult:
    payload = json.dumps(build_payload(args)).encode("utf-8")
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    chunks = 0
    response_bytes = 0
    audio_samples = 0
    sample_rate: int | None = None
    status: int | None = None

    try:
        with urllib.request.urlopen(
            request, timeout=args.request_timeout_s
        ) as response:
            status = response.status
            for raw_line in response:
                response_bytes += len(raw_line)
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                chunks += 1
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                audio = event.get("nvext", {}).get("audio")
                if isinstance(audio, dict):
                    audio_samples += int(audio.get("num_samples") or 0)
                    if audio.get("sample_rate"):
                        sample_rate = int(audio["sample_rate"])
        latency_s = time.perf_counter() - started
        return RequestResult(
            ok=200 <= (status or 0) < 300 and audio_samples > 0,
            latency_s=latency_s,
            status=status,
            chunks=chunks,
            response_bytes=response_bytes,
            audio_samples=audio_samples,
            sample_rate=sample_rate,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read(2048).decode("utf-8", errors="replace")
        latency_s = time.perf_counter() - started
        return RequestResult(
            ok=False,
            latency_s=latency_s,
            status=exc.code,
            chunks=chunks,
            response_bytes=response_bytes,
            audio_samples=audio_samples,
            sample_rate=sample_rate,
            error=body,
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 - include transport failures in benchmark output.
        latency_s = time.perf_counter() - started
        return RequestResult(
            ok=False,
            latency_s=latency_s,
            status=status,
            chunks=chunks,
            response_bytes=response_bytes,
            audio_samples=audio_samples,
            sample_rate=sample_rate,
            error=str(exc),
        )


def summarize_level(
    concurrency: int, request_count: int, elapsed_s: float, results: list[RequestResult]
) -> dict[str, Any]:
    successes = [result for result in results if result.ok]
    failures = [result for result in results if not result.ok]
    latencies = [result.latency_s for result in successes]
    audio_seconds = 0.0
    for result in successes:
        if result.sample_rate:
            audio_seconds += result.audio_samples / result.sample_rate

    return {
        "concurrency": concurrency,
        "requests": request_count,
        "elapsed_s": elapsed_s,
        "successes": len(successes),
        "failures": len(failures),
        "success_rps": len(successes) / elapsed_s if elapsed_s > 0 else 0.0,
        "audio_seconds_per_s": audio_seconds / elapsed_s if elapsed_s > 0 else 0.0,
        "audio_seconds_total": audio_seconds,
        "latency_avg_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p90_s": percentile(latencies, 0.90),
        "latency_p95_s": percentile(latencies, 0.95),
        "latency_max_s": max(latencies) if latencies else 0.0,
        "sample_rate": next(
            (result.sample_rate for result in successes if result.sample_rate), None
        ),
        "audio_samples_avg": (
            statistics.mean(result.audio_samples for result in successes)
            if successes
            else 0.0
        ),
        "response_bytes_avg": (
            statistics.mean(result.response_bytes for result in successes)
            if successes
            else 0.0
        ),
        "first_error": failures[0].error if failures else None,
        "statuses": sorted({result.status for result in results if result.status}),
    }


def run_level(args: argparse.Namespace, concurrency: int) -> dict[str, Any]:
    request_count = args.requests_per_level or max(
        args.min_requests, concurrency * args.requests_multiplier
    )
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(post_streaming_request, args) for _ in range(request_count)
        ]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
    elapsed_s = time.perf_counter() - started
    return summarize_level(concurrency, request_count, elapsed_s, results)


def print_summary(summary: dict[str, Any]) -> None:
    header = (
        "conc  req  ok  fail  elapsed_s  rps     audio_s/s  "
        "lat_avg  lat_p50  lat_p90  audio_samples"
    )
    print(header, flush=True)
    for level in summary["levels"]:
        print(
            f"{level['concurrency']:>4}  {level['requests']:>3}  "
            f"{level['successes']:>2}  {level['failures']:>4}  "
            f"{level['elapsed_s']:>9.2f}  {level['success_rps']:>6.3f}  "
            f"{level['audio_seconds_per_s']:>9.3f}  "
            f"{level['latency_avg_s']:>7.2f}  {level['latency_p50_s']:>7.2f}  "
            f"{level['latency_p90_s']:>7.2f}  {level['audio_samples_avg']:>13.0f}",
            flush=True,
        )
    best = summary["best"]
    print(
        "\nBest: "
        f"concurrency={best['concurrency']} "
        f"success_rps={best['success_rps']:.3f} "
        f"audio_seconds_per_s={best['audio_seconds_per_s']:.3f}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", required=True, help="Endpoint base URL, e.g. http://127.0.0.1:8000"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--voice", default="Chelsie")
    parser.add_argument("--audio-format", default="wav")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--concurrency", type=parse_concurrency, default=parse_concurrency("1,2,4,8")
    )
    parser.add_argument("--requests-per-level", type=int, default=0)
    parser.add_argument("--requests-multiplier", type=int, default=2)
    parser.add_argument("--min-requests", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    for index in range(args.warmup):
        result = post_streaming_request(args)
        if not result.ok:
            print(f"warmup {index + 1} failed: {result.error}", flush=True)
            return 1

    levels = []
    for concurrency in args.concurrency:
        print(f"running concurrency={concurrency}", flush=True)
        levels.append(run_level(args, concurrency))

    best = max(levels, key=lambda item: (item["success_rps"], -item["failures"]))
    summary = {
        "url": args.url,
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "levels": levels,
        "best": best,
    }
    print_summary(summary)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    return 0 if any(level["successes"] for level in levels) else 1


if __name__ == "__main__":
    raise SystemExit(main())
