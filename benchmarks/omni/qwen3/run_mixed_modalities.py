#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark Qwen3-Omni with text + image + audio + video in one request.

This is intentionally separate from run_sweep.sh because AIPerf's chat driver
does not currently build a single mixed-media OpenAI payload. The request shape
matches vLLM-Omni's Qwen3-Omni online serving examples:

  input:  audio_url + image_url + video_url + text
  output: modalities ["text", "audio"]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_AUDIO_URL = (
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/"
    "multimodal_asset/mary_had_lamb.ogg"
)
DEFAULT_IMAGE_URL = (
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/"
    "vision_model_images/cherry_blossom.jpg"
)
DEFAULT_VIDEO_URL = (
    "https://huggingface.co/datasets/raushan-testing-hf/videos-test/"
    "resolve/main/sample_demo_1.mp4"
)
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
USER_PROMPT = (
    "Use all provided media. In one short sentence, name what is in the image, "
    "what song is in the audio, and why the video is funny."
)


def parse_modalities(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": args.audio_url},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": args.image_url},
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": args.video_url},
                    },
                    {"type": "text", "text": args.prompt},
                ],
            },
        ],
        "modalities": parse_modalities(args.modalities),
    }
    if not args.dynamo_compatible:
        payload["sampling_params_list"] = [
            {
                "temperature": 0.4,
                "top_p": 0.9,
                "top_k": 1,
                "max_tokens": args.text_tokens,
                "seed": 42,
                "repetition_penalty": 1.05,
                "stop_token_ids": [151645],
            },
            {
                "temperature": 0.9,
                "top_k": 50,
                "max_tokens": args.talker_tokens,
                "seed": 42,
                "detokenize": False,
                "repetition_penalty": 1.05,
                "stop_token_ids": [2150],
            },
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": args.audio_tokens,
                "seed": 42,
                "detokenize": True,
                "repetition_penalty": 1.1,
            },
        ]
    if args.speaker and not args.dynamo_compatible:
        payload["speaker"] = args.speaker
    return payload


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        elapsed = time.perf_counter() - start
        parsed = json.loads(body.decode("utf-8"))
        choices = parsed.get("choices") or []
        text_chars = 0
        audio_bytes = 0
        for choice in choices:
            msg = choice.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                text_chars += len(content)
            audio = msg.get("audio") or {}
            audio_data = audio.get("data")
            if isinstance(audio_data, str):
                audio_bytes += len(audio_data)
        data = parsed.get("data") or []
        for item in data:
            if isinstance(item, dict):
                audio_data = item.get("b64_json") or item.get("data")
                if isinstance(audio_data, str):
                    audio_bytes += len(audio_data)
        return {
            "ok": True,
            "latency_s": elapsed,
            "status": 200,
            "object": parsed.get("object"),
            "choices": len(choices),
            "data_items": len(data),
            "text_chars": text_chars,
            "audio_b64_chars": audio_bytes,
        }
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "latency_s": elapsed,
            "status": exc.code,
            "error": body[:2000],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "latency_s": elapsed,
            "status": None,
            "error": repr(exc),
        }


def run_batch(
    url: str,
    payload: dict[str, Any],
    concurrency: int,
    requests: int,
    timeout: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(post_json, url, payload, timeout) for _ in range(requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def summarize(results: list[dict[str, Any]], total_s: float) -> dict[str, Any]:
    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    latencies_ms = [r["latency_s"] * 1000.0 for r in ok]
    return {
        "requests": len(results),
        "successes": len(ok),
        "failures": len(failed),
        "wall_time_s": total_s,
        "request_throughput": len(ok) / total_s if total_s > 0 else None,
        "latency_ms": {
            "avg": sum(latencies_ms) / len(latencies_ms) if latencies_ms else None,
            "p50": percentile(latencies_ms, 0.50),
            "p90": percentile(latencies_ms, 0.90),
            "p99": percentile(latencies_ms, 0.99),
        },
        "first_error": failed[0].get("error") if failed else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--modalities", default="text,audio")
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--prompt", default=USER_PROMPT)
    parser.add_argument("--audio-url", default=DEFAULT_AUDIO_URL)
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--text-tokens", type=int, default=96)
    parser.add_argument("--talker-tokens", type=int, default=512)
    parser.add_argument("--audio-tokens", type=int, default=8192)
    parser.add_argument(
        "--dynamo-compatible",
        action="store_true",
        help="Omit vLLM-Omni-only request fields rejected by Dynamo frontend validation.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = build_payload(args)
    if args.warmup:
        run_batch(args.url, payload, 1, args.warmup, args.timeout)

    start = time.perf_counter()
    results = run_batch(
        args.url, payload, args.concurrency, args.requests, args.timeout
    )
    total_s = time.perf_counter() - start
    summary = summarize(results, total_s)
    report = {
        "config": {
            "url": args.url,
            "model": args.model,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "modalities": parse_modalities(args.modalities),
            "dynamo_compatible": args.dynamo_compatible,
            "media": {
                "audio_url": args.audio_url,
                "image_url": args.image_url,
                "video_url": args.video_url,
            },
        },
        "summary": summary,
        "results": results,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
