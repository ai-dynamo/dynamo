#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark for Qwen3.5 multimodal + approximate KV router.

Self-contained: starts etcd, NATS, frontend, vLLM worker, runs benchmark, stops.

Usage:
    # Text-only benchmark
    python examples/backends/vllm/qwen35/benchmark.py --repeats 5

    # Multimodal benchmark
    python examples/backends/vllm/qwen35/benchmark.py --num-images 1 --repeats 5

    # Custom model
    python examples/backends/vllm/qwen35/benchmark.py --model Qwen/Qwen3.5-0.8B --repeats 5
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import time
from contextlib import contextmanager
from types import SimpleNamespace

import requests
from PIL import Image

from tests.conftest import EtcdServer, NatsServer
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_ports

MODEL = "Qwen/Qwen3.5-0.8B"

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]


# -- Helpers ------------------------------------------------------------------


def make_data_uri(color: tuple[int, int, int], size: int = 64) -> str:
    img = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _mock_request(name: str) -> SimpleNamespace:
    return SimpleNamespace(node=SimpleNamespace(name=name))


def _check_ready(response) -> bool:
    try:
        return (response.json() or {}).get("status") == "ready"
    except ValueError:
        return False


# -- Managed processes --------------------------------------------------------


class Qwen35VLLMWorker(ManagedProcess):
    """vLLM worker for Qwen3.5 hybrid multimodal model."""

    def __init__(self, *, system_port: int, model: str):
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_PORT"] = str(system_port)
        env["DYN_REQUEST_PLANE"] = "tcp"

        cmd = [
            "python3", "-m", "dynamo.vllm",
            "--model", model,
            "--enable-multimodal",
            "--mamba-cache-mode", "align",
            "--is-decode-worker",
            "--enforce-eager",
            "--max-model-len", "4096",
            "--max-num-seqs", "2",
        ]

        super().__init__(
            command=cmd,
            env=env,
            health_check_urls=[(f"http://localhost:{system_port}/health", _check_ready)],
            timeout=600,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m dynamo.vllm"],
            log_dir="bench_qwen35_worker",
        )


class Qwen35Frontend(ManagedProcess):
    """Frontend with approximate KV routing for hybrid models."""

    def __init__(self, *, frontend_port: int):
        env = os.environ.copy()
        env["DYN_LOG"] = "info"
        env["DYN_REQUEST_PLANE"] = "tcp"

        log_dir = "bench_qwen35_frontend"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=[
                "python3", "-m", "dynamo.frontend",
                "--http-port", str(frontend_port),
                "--router-mode", "kv",
                "--no-router-kv-events",
            ],
            env=env,
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", check_models_api)
            ],
            timeout=120,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m dynamo.frontend"],
            log_dir=log_dir,
        )


# -- Service stack ------------------------------------------------------------


@contextmanager
def start_qwen35_stack(mock_req, model: str):
    """Frontend (approx KV router) -> vLLM (hybrid multimodal)."""
    frontend_port, vllm_port = allocate_ports(count=2, start_port=10000)

    print("Starting etcd + NATS...")
    with NatsServer(mock_req, port=0) as nats, EtcdServer(mock_req, port=0) as etcd:
        os.environ["NATS_SERVER"] = f"nats://localhost:{nats.port}"
        os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd.port}"

        print(f"Starting vLLM worker ({model})...")
        with Qwen35VLLMWorker(system_port=vllm_port, model=model):
            print("Starting Frontend (approx KV router)...")
            with Qwen35Frontend(frontend_port=frontend_port):
                print(f"All services ready at http://localhost:{frontend_port}")
                yield frontend_port

        os.environ.pop("NATS_SERVER", None)
        os.environ.pop("ETCD_ENDPOINTS", None)


# -- Benchmark ----------------------------------------------------------------


def measure_request(url: str, payload: dict, timeout: int = 120) -> dict:
    t_start = time.perf_counter()
    t_first_token = None
    tokens = []

    with requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "):
                continue
            data = line_str[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    tokens.append(content)
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

    elapsed = time.perf_counter() - t_start
    ttft = (t_first_token - t_start) if t_first_token else elapsed
    return {
        "elapsed": elapsed,
        "ttft": ttft,
        "num_tokens": len(tokens),
        "response_text": "".join(tokens),
    }


def build_payload(prompt: str, image_uris: list[str], max_tokens: int, model: str) -> dict:
    if image_uris:
        content = [{"type": "text", "text": prompt}]
        for uri in image_uris:
            content.append({"type": "image_url", "image_url": {"url": uri}})
    else:
        content = prompt

    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": True,
    }


def run_scenario(url: str, payload: dict, repeats: int, wait: float, label: str):
    """Run one benchmark scenario and print results."""
    print(f"\n--- {label} ---")
    results = []
    for i in range(repeats):
        result = measure_request(url, payload)
        results.append(result)
        print(
            f"  [{i + 1}/{repeats}] {result['elapsed']:.3f}s  "
            f"TTFT={result['ttft']:.3f}s  "
            f"tokens={result['num_tokens']}  "
            f'"{result["response_text"][:80]}"'
        )
        if i < repeats - 1:
            time.sleep(wait)

    baseline_ttft = results[0]["ttft"]
    print(f"\n  {'#':<4} {'TTFT (s)':<12} {'Total (s)':<12} {'Speedup':<10}")
    print(f"  {'-' * 38}")
    for i, r in enumerate(results):
        speedup = f"{baseline_ttft / r['ttft']:.2f}x" if i > 0 and r["ttft"] > 0 else "(cold)"
        print(f"  {i + 1:<4} {r['ttft']:<12.3f} {r['elapsed']:<12.3f} {speedup:<10}")

    if len(results) > 1:
        avg_warm_ttft = sum(r["ttft"] for r in results[1:]) / (len(results) - 1)
        print(f"  {'-' * 38}")
        print(f"  Cold TTFT:     {baseline_ttft:.3f}s")
        print(f"  Warm TTFT avg: {avg_warm_ttft:.3f}s")
        if avg_warm_ttft > 0:
            print(f"  Speedup:       {baseline_ttft / avg_warm_ttft:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3.5 multimodal + approx KV router"
    )
    parser.add_argument("--prompt", default="Describe what you see.")
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--wait", type=float, default=1.0)
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()

    image_uris = [
        make_data_uri(COLORS[i % len(COLORS)], args.image_size)
        for i in range(args.num_images)
    ]

    mode = f"multimodal ({args.num_images} img)" if image_uris else "text-only"
    mock_req = _mock_request("qwen35_benchmark")

    print("=" * 60)
    print("Qwen3.5 Multimodal + Approx KV Router Benchmark")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Mode:       {mode}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Repeats:    {args.repeats}")
    print()

    with start_qwen35_stack(mock_req, args.model) as frontend_port:
        url = f"http://localhost:{frontend_port}"

        # Scenario 1: Same prompt repeated (measures prefix cache reuse)
        payload = build_payload(args.prompt, image_uris, args.max_tokens, args.model)
        run_scenario(url, payload, args.repeats, args.wait, f"{mode} — same prompt repeated")

        # Scenario 2: Different images, same text (only if multimodal)
        if image_uris:
            print()
            diff_results = []
            print(f"\n--- {mode} — different images each time ---")
            for i in range(args.repeats):
                diff_uris = [
                    make_data_uri(COLORS[(i * len(image_uris) + j) % len(COLORS)], args.image_size)
                    for j in range(args.num_images)
                ]
                diff_payload = build_payload(args.prompt, diff_uris, args.max_tokens, args.model)
                result = measure_request(url, diff_payload)
                diff_results.append(result)
                print(
                    f"  [{i + 1}/{args.repeats}] {result['elapsed']:.3f}s  "
                    f"TTFT={result['ttft']:.3f}s  "
                    f"tokens={result['num_tokens']}"
                )
                if i < args.repeats - 1:
                    time.sleep(args.wait)

            avg_ttft = sum(r["ttft"] for r in diff_results) / len(diff_results)
            print(f"\n  Avg TTFT (diff images): {avg_ttft:.3f}s")

    print("\nAll services stopped.")


if __name__ == "__main__":
    main()
