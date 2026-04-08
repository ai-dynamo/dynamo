#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KV routing evaluation for Qwen3.5 hybrid model (GDN + FullAttention + Vision).

Measures Speed-of-Light (SOL) for prefix cache / KV-aware routing: the maximum
TTFT reduction achievable on an ideal workload (100% prefix reuse).

Tests both text-only and multimodal inputs with approximate routing.

Usage:
    bash examples/backends/vllm/qwen35/launch.sh   # start the stack
    python examples/backends/vllm/qwen35/test_requests.py
    python examples/backends/vllm/qwen35/test_requests.py --repeats 10 --warmup 3
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from dataclasses import dataclass

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Qwen3.5 hybrid: block_size=544 (from GDN float32 state / Attn page alignment)
BLOCK_SIZE = 544

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_data_uri(color: tuple[int, int, int], size: int = 64) -> str:
    img = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


_PARAGRAPHS = [
    (
        "The theory of general relativity, published by Albert Einstein in 1915, "
        "describes gravity not as a force between masses, but as a curvature of "
        "spacetime caused by mass and energy. According to this theory, massive "
        "objects like stars and planets bend the fabric of spacetime around them, "
        "and other objects move along curved paths determined by this geometry. "
        "This elegant framework has been confirmed by numerous experiments, "
        "including the bending of light around the sun, gravitational time "
        "dilation measured by atomic clocks, and the detection of gravitational "
        "waves from merging black holes. "
    ),
    (
        "Quantum mechanics is a fundamental theory in physics that describes "
        "the behavior of matter and energy at the smallest scales. Unlike "
        "classical physics, quantum mechanics introduces probability as a core "
        "concept, where particles exist in superpositions of states until "
        "measured. The wave function describes the quantum state of a system "
        "and its evolution over time according to the Schrodinger equation. "
        "Key phenomena include quantum entanglement, tunneling, and the "
        "uncertainty principle formulated by Werner Heisenberg. "
    ),
    (
        "The standard model of particle physics is the theoretical framework "
        "describing three of the four known fundamental forces and classifying "
        "all known elementary particles. It describes the electromagnetic, weak, "
        "and strong nuclear interactions using gauge bosons as force carriers. "
        "The discovery of the Higgs boson at CERN in 2012 confirmed the "
        "mechanism by which particles acquire mass. Despite its success, the "
        "standard model does not incorporate gravity or explain dark matter "
        "and dark energy which make up most of the universe. "
    ),
    (
        "Thermodynamics is the branch of physics that deals with heat, work, "
        "temperature, and their relation to energy and entropy. The four laws "
        "of thermodynamics govern the behavior of these quantities and provide "
        "a quantitative description using measurable macroscopic physical "
        "quantities. The second law introduces the concept of entropy, stating "
        "that the total entropy of an isolated system can never decrease over "
        "time. This principle underlies the arrow of time and limits the "
        "efficiency of all heat engines and refrigeration cycles. "
    ),
    (
        "Cosmology is the study of the origin, evolution, and eventual fate "
        "of the universe. Modern cosmology began with the discovery of cosmic "
        "microwave background radiation in 1965 by Penzias and Wilson, "
        "providing strong evidence for the Big Bang theory. The universe is "
        "approximately 13.8 billion years old and is expanding at an "
        "accelerating rate driven by dark energy. Observations of Type Ia "
        "supernovae, galaxy clusters, and baryon acoustic oscillations "
        "continue to refine our understanding of cosmic expansion. "
    ),
    (
        "Electromagnetism describes how electrically charged particles interact "
        "with each other and with magnetic fields. James Clerk Maxwell unified "
        "electricity and magnetism into a single theory in the 1860s, showing "
        "that light itself is an electromagnetic wave. Maxwells equations "
        "predict the existence of electromagnetic radiation across a spectrum "
        "from radio waves to gamma rays. This theory underpins modern "
        "technology including wireless communication, electric motors, "
        "and medical imaging devices like MRI scanners. "
    ),
]


def make_long_text(min_tokens: int = 600, variant: int = 0) -> str:
    """Generate text that tokenizes to at least `min_tokens` tokens.

    Each `variant` produces a unique text (different paragraph content)
    so that different test scenarios don't share cache.
    """
    paragraph = _PARAGRAPHS[variant % len(_PARAGRAPHS)]
    n_repeats = (min_tokens // 70) + 2
    return (paragraph * n_repeats).strip()


@dataclass
class Result:
    elapsed: float
    ttft: float  # from nvext.timing.ttft_ms (server-side, more accurate)
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    kv_hit_rate: float | None
    response_text: str

    @property
    def cache_ratio(self) -> float:
        if self.prompt_tokens == 0:
            return 0.0
        return self.cached_tokens / self.prompt_tokens

    def summary(self, label: str = "") -> str:
        kv = f"  kv_hit={self.kv_hit_rate:.2f}" if self.kv_hit_rate is not None else ""
        cache = f"  cached={self.cached_tokens}/{self.prompt_tokens}({self.cache_ratio:.0%})"
        text = self.response_text[:60].replace("\n", " ")
        return (
            f"  {label}TTFT={self.ttft*1000:.1f}ms  total={self.elapsed:.3f}s"
            f"  prompt={self.prompt_tokens}  gen={self.completion_tokens}"
            f"{cache}{kv}  \"{text}\""
        )


def send(url: str, payload: dict, timeout: int = 120) -> Result:
    """Send non-streaming request, return structured Result."""
    p = {**payload, "stream": False}
    p.pop("stream_options", None)
    t_start = time.perf_counter()
    resp = requests.post(f"{url}/v1/chat/completions", json=p, timeout=timeout)
    elapsed = time.perf_counter() - t_start
    resp.raise_for_status()
    data = resp.json()

    usage = data.get("usage", {})
    nvext = data.get("nvext", {})
    timing = nvext.get("timing", {})
    prompt_details = usage.get("prompt_tokens_details") or {}

    content = ""
    if data.get("choices"):
        content = data["choices"][0].get("message", {}).get("content", "")

    return Result(
        elapsed=elapsed,
        ttft=timing.get("ttft_ms", 0) / 1000.0,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        cached_tokens=prompt_details.get("cached_tokens", 0),
        kv_hit_rate=timing.get("kv_hit_rate"),
        response_text=content,
    )


def build_text(model: str, prompt: str, max_tokens: int = 10) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }


def build_mm(model: str, text: str, image_uris: list[str], max_tokens: int = 10) -> dict:
    content: list[dict] = [{"type": "text", "text": text}]
    for uri in image_uris:
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_table(results: list[Result], label: str) -> None:
    """Print a table with both vLLM cache hit and router kv_hit side by side."""
    print(f"\n  {'#':<4} {'TTFT(ms)':<10} {'Speedup':<9} {'Prompt':<8} "
          f"{'vLLM cached':<16} {'Router kv_hit':<14}")
    print(f"  {'─'*65}")
    cold_ttft = results[0].ttft
    for i, r in enumerate(results):
        speedup = f"{cold_ttft / r.ttft:.2f}x" if i > 0 and r.ttft > 0 else "—"
        # vLLM cached: from usage.prompt_tokens_details.cached_tokens
        vllm_cached = f"{r.cached_tokens}/{r.prompt_tokens}({r.cache_ratio:.0%})" if r.prompt_tokens else "—"
        # Router kv_hit: from nvext.timing.kv_hit_rate
        router_kv = f"{r.kv_hit_rate:.2f}" if r.kv_hit_rate is not None else "—"
        print(f"  {i+1:<4} {r.ttft*1000:<10.1f} {speedup:<9} {r.prompt_tokens:<8} "
              f"{vllm_cached:<16} {router_kv:<14}")

    warm = results[1:]
    if warm:
        avg_ttft = sum(r.ttft for r in warm) / len(warm)
        avg_vllm_cache = sum(r.cache_ratio for r in warm) / len(warm)
        avg_router_kv = sum((r.kv_hit_rate or 0) for r in warm) / len(warm)
        print(f"  {'─'*65}")
        print(f"  Cold TTFT:       {cold_ttft*1000:.1f} ms")
        print(f"  Warm TTFT avg:   {avg_ttft*1000:.1f} ms")
        print(f"  vLLM cache avg:  {avg_vllm_cache:.0%}")
        print(f"  Router kv avg:   {avg_router_kv:.2f}")
        if avg_ttft > 0:
            print(f"  {BOLD}SOL Speedup:    {cold_ttft / avg_ttft:.2f}x{RESET}")


def run_scenario(
    url: str,
    payload: dict,
    repeats: int,
    warmup: int,
    label: str,
    wait: float = 0.3,
) -> list[Result]:
    """Run cold + warm measured repeats.

    Request #1 is genuinely cold (no prior cache for this payload).
    Warmup uses a DIFFERENT prompt to warm up the engine without populating
    cache for the target payload. Requests #2..N are warm (cache populated by #1).
    """
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")

    if warmup > 0:
        # Warmup with a DIFFERENT prompt to warm engine (JIT, memory, etc.)
        # without populating cache for the actual test payload
        warmup_payload = build_text(
            payload["model"],
            "This is a warmup request to initialize engine state. " * 20,
            max_tokens=1,
        )
        print(f"\n  Engine warmup ({warmup} requests with different prompt)...")
        for i in range(warmup):
            r = send(url, warmup_payload)
            print(f"    [{i+1}/{warmup}] TTFT={r.ttft*1000:.1f}ms (warmup, not target payload)")
            time.sleep(wait)

    print(f"\n  Measured ({repeats} requests, #1=cold, #2+=warm):")
    results = []
    for i in range(repeats):
        r = send(url, payload)
        results.append(r)
        tag = "COLD" if i == 0 else "warm"
        print(r.summary(label=f"[{i+1}/{repeats} {tag}] "))
        time.sleep(wait)

    print_table(results, label)
    return results


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


def test_text_sol(url: str, model: str, repeats: int, warmup: int) -> list[Result]:
    """SOL: Text-only, same long prompt repeated.

    Best-case scenario for prefix cache: 100% token reuse.
    Measures maximum possible TTFT reduction.
    """
    # Target ~3 full blocks (3 × 544 = 1632 tokens), variant=0
    text = make_long_text(min_tokens=1700, variant=0)
    prompt = f"Summarize in one sentence:\n\n{text}"
    payload = build_text(model, prompt, max_tokens=10)
    return run_scenario(url, payload, repeats, warmup,
                        "SOL: Text — same prompt repeated (ideal prefix reuse)")


def test_text_shared_prefix(url: str, model: str, repeats: int, warmup: int) -> list[Result]:
    """Same long prefix, different short suffix each time.

    Simulates real workload: shared system prompt / context, different user queries.
    """
    prefix = make_long_text(min_tokens=1700, variant=1)
    suffixes = [
        "What is the main idea?",
        "List three key experiments.",
        "Explain time dilation.",
        "How were gravitational waves detected?",
        "Compare this to Newton's gravity.",
        "What is spacetime curvature?",
        "How does mass affect spacetime?",
        "What did Einstein publish in 1915?",
        "Summarize in one word.",
        "Is this theory proven?",
    ]

    print(f"\n{'='*60}")
    print(f" Text — shared prefix, different suffix")
    print(f"{'='*60}")

    if warmup > 0:
        warmup_payload = build_text(model, "Engine warmup request. " * 20, max_tokens=1)
        print(f"\n  Engine warmup ({warmup} requests)...")
        for i in range(warmup):
            r = send(url, warmup_payload)
            print(f"    [{i+1}/{warmup}] TTFT={r.ttft*1000:.1f}ms (warmup)")
            time.sleep(0.3)

    print(f"\n  Measured ({repeats} requests, #1=cold, each with different suffix):")
    results = []
    for i in range(repeats):
        suffix = suffixes[i % len(suffixes)]
        p = build_text(model, f"{prefix}\n\nQuestion: {suffix}", max_tokens=10)
        r = send(url, p)
        results.append(r)
        tag = "COLD" if i == 0 else "warm"
        print(r.summary(label=f"[{i+1}/{repeats} {tag}] "))
        time.sleep(0.3)

    print_table(results, "Text — shared prefix, different suffix")
    return results


def test_text_multi_turn(url: str, model: str, repeats: int, warmup: int) -> list[Result]:
    """Simulate multi-turn conversation: each turn appends to history.

    This is the ideal case for align-mode hybrid cache.
    """
    base_context = make_long_text(min_tokens=1200, variant=2)
    turns = [
        ("user", "What is general relativity about?"),
        ("assistant", "General relativity describes gravity as spacetime curvature caused by mass and energy."),
        ("user", "How was it confirmed?"),
        ("assistant", "Through light bending, gravitational time dilation, and gravitational wave detection."),
        ("user", "Who discovered gravitational waves?"),
        ("assistant", "LIGO collaboration first detected gravitational waves in 2015."),
        ("user", "Tell me more about LIGO."),
        ("assistant", "LIGO uses laser interferometry to detect spacetime distortions smaller than a proton."),
        ("user", "What's next for gravitational wave research?"),
        ("assistant", "Space-based detectors like LISA and improved ground detectors will expand our observations."),
    ]

    print(f"\n{'='*60}")
    print(f" Multi-turn conversation (cumulative history)")
    print(f"{'='*60}")

    results = []
    messages = [{"role": "user", "content": f"Context:\n{base_context}"}]

    n_turns = min(repeats, len(turns) // 2)
    for i in range(n_turns):
        # Add user + assistant turns
        user_turn = turns[i * 2]
        messages.append({"role": user_turn[0], "content": user_turn[1]})
        if i > 0:
            # Add previous assistant response
            assistant_turn = turns[i * 2 - 1]
            messages.insert(-1, {"role": assistant_turn[0], "content": assistant_turn[1]})

        payload = {
            "model": model,
            "messages": messages.copy(),
            "max_tokens": 10,
        }
        r = send(url, payload)
        results.append(r)
        print(r.summary(label=f"  Turn {i+1}: "))
        time.sleep(0.3)

    if len(results) > 1:
        print_table(results, "Multi-turn")
    return results


def test_mm_sol(url: str, model: str, repeats: int, warmup: int) -> list[Result]:
    """SOL: Multimodal — same images + text repeated.

    Images are placed BEFORE text so vision tokens form the prefix.
    This tests whether vLLM caches vision token blocks correctly.
    """
    text = make_long_text(min_tokens=1200, variant=3)
    images = [
        make_data_uri((255, 0, 0), size=256),
        make_data_uri((0, 255, 0), size=256),
        make_data_uri((0, 0, 255), size=256),
    ]
    # Images first, then text
    content: list[dict] = []
    for uri in images:
        content.append({"type": "image_url", "image_url": {"url": uri}})
    content.append({"type": "text", "text": f"Describe each image and summarize:\n\n{text}"})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 10,
    }
    return run_scenario(url, payload, repeats, warmup,
                        "SOL: Multimodal — same images+text repeated")


def test_mm_same_images_diff_text(url: str, model: str, repeats: int, warmup: int) -> list[Result]:
    """Same images (before text), different text suffix each time.

    Images first → vision tokens form the prefix. Same images every time
    but text changes. Tests whether vision token prefix gets cached.
    """
    images = [
        make_data_uri((255, 0, 0), size=256),
        make_data_uri((0, 255, 0), size=256),
    ]
    suffix_text = make_long_text(min_tokens=800, variant=4)
    questions = [
        "What colors do you see?",
        "Describe the texture and pattern.",
        "Could these be design elements?",
        "Compare the two images.",
        "What emotions do these evoke?",
        "Rate each image 1-10.",
        "How would you use these?",
        "What color space are these?",
        "Describe in detail.",
        "Summarize in one sentence.",
    ]

    print(f"\n{'='*60}")
    print(f" Multimodal — same images first, different text suffix")
    print(f"{'='*60}")

    if warmup > 0:
        warmup_payload = build_text(model, "Engine warmup request. " * 20, max_tokens=1)
        print(f"\n  Engine warmup ({warmup} requests)...")
        for i in range(warmup):
            r = send(url, warmup_payload)
            print(f"    [{i+1}/{warmup}] TTFT={r.ttft*1000:.1f}ms (warmup)")
            time.sleep(0.3)

    results = []
    print(f"\n  Measured ({repeats} requests, #1=cold):")
    for i in range(repeats):
        q = questions[i % len(questions)]
        # Images first, then text (suffix varies)
        content: list[dict] = []
        for uri in images:
            content.append({"type": "image_url", "image_url": {"url": uri}})
        content.append({"type": "text", "text": f"{suffix_text}\n\n{q}"})
        p = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 10,
        }
        r = send(url, p)
        results.append(r)
        tag = "COLD" if i == 0 else "warm"
        print(r.summary(label=f"[{i+1}/{repeats} {tag}] "))
        time.sleep(0.3)

    print_table(results, "MM — same images first, diff text suffix")
    return results


def test_mm_diff_images_same_text(url: str, model: str, repeats: int, warmup: int) -> list[Result]:
    """Different images (before text), same text each time.

    Images first → vision tokens form the prefix. Different images each
    request, so vision prefix changes. Tests whether different images
    correctly break the cache for the vision portion.
    """
    text = f"{make_long_text(min_tokens=1200, variant=5)}\n\nDescribe each image."

    print(f"\n{'='*60}")
    print(f" Multimodal — different images first, same text after")
    print(f"{'='*60}")

    if warmup > 0:
        warmup_payload = build_text(model, "Engine warmup request. " * 20, max_tokens=1)
        print(f"\n  Engine warmup ({warmup} requests)...")
        for i in range(warmup):
            r = send(url, warmup_payload)
            print(f"    [{i+1}/{warmup}] TTFT={r.ttft*1000:.1f}ms (warmup)")
            time.sleep(0.3)

    results = []
    print(f"\n  Measured ({repeats} requests, #1=cold):")
    for i in range(repeats):
        # Different images each time, placed BEFORE text
        images = [
            make_data_uri(COLORS[i % len(COLORS)], size=256),
            make_data_uri(COLORS[(i + 1) % len(COLORS)], size=256),
        ]
        content: list[dict] = []
        for uri in images:
            content.append({"type": "image_url", "image_url": {"url": uri}})
        content.append({"type": "text", "text": text})
        p = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 10,
        }
        r = send(url, p)
        results.append(r)
        tag = "COLD" if i == 0 else f"img{i}"
        print(r.summary(label=f"[{i+1}/{repeats} {tag}] "))
        time.sleep(0.3)

    print_table(results, "MM — diff images first, same text after")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="KV routing SOL evaluation for Qwen3.5 hybrid model"
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--repeats", type=int, default=5, help="Measured requests per scenario")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup requests (populate cache)")
    parser.add_argument("--skip-mm", action="store_true", help="Skip multimodal tests")
    parser.add_argument(
        "--scenarios", nargs="+",
        choices=["text-sol", "text-prefix", "text-multi-turn",
                 "mm-sol", "mm-same-img", "mm-diff-img", "all"],
        default=["all"],
        help="Which scenarios to run",
    )
    args = parser.parse_args()

    run_all = "all" in args.scenarios

    print(f"{'='*60}")
    print(f" KV Routing SOL Evaluation — Qwen3.5 Hybrid Model")
    print(f"{'='*60}")
    print(f"  URL:     {args.url}")
    print(f"  Model:   {args.model}")
    print(f"  Repeats: {args.repeats}  Warmup: {args.warmup}")
    print(f"  Block size: {BLOCK_SIZE} tokens (hybrid GDN+Attn alignment)")

    # Check frontend
    try:
        resp = requests.get(f"{args.url}/v1/models", timeout=5)
        resp.raise_for_status()
        models = [m["id"] for m in resp.json().get("data", [])]
        print(f"  Models:  {models}")
    except Exception as e:
        print(f"\n  ERROR: Frontend not reachable — {e}")
        sys.exit(1)

    all_results: dict[str, list[Result]] = {}

    # --- Text scenarios ---

    if run_all or "text-sol" in args.scenarios:
        all_results["text-sol"] = test_text_sol(
            args.url, args.model, args.repeats, args.warmup)

    if run_all or "text-prefix" in args.scenarios:
        all_results["text-prefix"] = test_text_shared_prefix(
            args.url, args.model, args.repeats, args.warmup)

    if run_all or "text-multi-turn" in args.scenarios:
        all_results["text-multi-turn"] = test_text_multi_turn(
            args.url, args.model, args.repeats, args.warmup)

    # --- Multimodal scenarios ---

    if not args.skip_mm:
        if run_all or "mm-sol" in args.scenarios:
            all_results["mm-sol"] = test_mm_sol(
                args.url, args.model, args.repeats, args.warmup)

        if run_all or "mm-same-img" in args.scenarios:
            all_results["mm-same-img"] = test_mm_same_images_diff_text(
                args.url, args.model, args.repeats, args.warmup)

        if run_all or "mm-diff-img" in args.scenarios:
            all_results["mm-diff-img"] = test_mm_diff_images_same_text(
                args.url, args.model, args.repeats, args.warmup)

    # --- Summary ---

    print(f"\n{'='*60}")
    print(f" {BOLD}Summary{RESET}")
    print(f"{'='*60}")
    print(f"\n  {'Scenario':<25} {'Cold':<10} {'Warm':<10} {'Speedup':<9} "
          f"{'vLLM cache':<12} {'Router kv':<10}")
    print(f"  {'─'*76}")
    for name, results in all_results.items():
        if not results:
            continue
        cold = results[0].ttft
        warm_results = results[1:]
        if warm_results:
            avg_warm = sum(r.ttft for r in warm_results) / len(warm_results)
            avg_vllm = sum(r.cache_ratio for r in warm_results) / len(warm_results)
            avg_router = sum((r.kv_hit_rate or 0) for r in warm_results) / len(warm_results)
            speedup = cold / avg_warm if avg_warm > 0 else 0
            print(
                f"  {name:<25} {cold*1000:<10.1f} {avg_warm*1000:<10.1f} {speedup:<9.2f}x "
                f"{avg_vllm:<12.0%} {avg_router:<10.2f}"
            )
        else:
            print(f"  {name:<25} {cold*1000:<10.1f} {'—':<10} {'—':<9} {'—':<12} {'—':<10}")

    print()


if __name__ == "__main__":
    main()
