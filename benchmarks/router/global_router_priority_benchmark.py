#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async RL priority pool routing benchmark for the global router.

Models a NeMo-RL async GRPO workload where multiple batches of multi-turn
requests run concurrently.  Each batch contains "normal" requests and a
configurable fraction of "straggler" requests (many turns, large output per
turn).  A sliding window of ``--lag`` concurrent batches mirrors the
``max_trajectory_age_steps`` parameter in NeMo-RL async GRPO.

Two phases are executed:

1. **Baseline** — every request is sent without priority, so the global
   router's grid mapping sends everything to pool 0 (the slow mocker pool).
2. **Priority** — straggler requests carry
   ``nvext.agent_hints.priority = <priority-value>``, triggering the global
   router's ``priority_overrides`` rule to route them to pool 1 (the fast
   mocker pool).

A batch "completes" only when **all** its requests finish every turn, so
stragglers gate the batch.  When stragglers are routed to the fast pool, batch
completion time drops — that's what the benchmark measures.

Prerequisites
-------------
* Deploy ``examples/global_planner/global-planner-priority-benchmark.yaml``
  (mocker-based, no GPU needed).
* Port-forward the frontend::

      kubectl port-forward svc/gp-ctrl-frontend 8000:8000

Example
-------
::

    python global_router_priority_benchmark.py \\
        --num-batches 6 --prompts-per-batch 16 --lag 2 \\
        --straggler-fraction 0.15 --straggler-turns 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field

import aiohttp
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("async_rl_bench")

# ---------------------------------------------------------------------------
# Defaults modelled after NeMo-RL GRPO configs
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "nvidia/Llama-3.1-8B-Instruct-FP8"
DEFAULT_URL = "http://localhost:8000"

# Per-token estimate for padding (Llama-3 tokeniser ≈ 4 chars/token on
# repetitive English; "alpha " is consistently 1–2 tokens).
CHARS_PER_TOKEN = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TurnResult:
    turn_idx: int
    ttft_ms: float | None
    total_ms: float
    output_tokens_requested: int


@dataclass
class RequestResult:
    request_id: int
    is_straggler: bool
    turns: list[TurnResult] = field(default_factory=list)
    total_ms: float = 0.0


@dataclass
class BatchResult:
    batch_id: int
    completion_ms: float = 0.0
    requests: list[RequestResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------
def _pad_text(target_tokens: int) -> str:
    """Return a string that tokenises to roughly *target_tokens* tokens."""
    return "alpha " * max(target_tokens, 1)


def build_batches(
    num_batches: int,
    prompts_per_batch: int,
    generations_per_prompt: int,
    straggler_fraction: float,
    normal_turns: int,
    normal_osl_mean: int,
    normal_osl_std: int,
    straggler_turns: int,
    straggler_osl_mean: int,
    straggler_osl_std: int,
    base_isl: int,
    user_tokens_per_turn: int,
    seed: int,
) -> list[list[dict]]:
    """Build a list of batches.  Each batch is a list of request descriptors.

    Each prompt is expanded into *generations_per_prompt* independent requests,
    mirroring ``repeat_interleave(num_generations)`` in NeMo-RL
    (``async_utils.py:491``).  Every generation shares the same prompt and
    straggler label but gets its own OSL draws (rollouts diverge independently).

    A request descriptor is a dict with keys:
        turns            – number of conversation turns
        max_tokens_per_turn – list[int], sampled OSL for each turn
        is_straggler     – bool
        base_isl         – first-turn ISL (approximate tokens)
        user_tokens_per_turn – extra user tokens added each turn
        prompt_id        – int, identifies which prompt this generation belongs to
    """
    rng = np.random.RandomState(seed)
    batches: list[list[dict]] = []
    for _ in range(num_batches):
        batch: list[dict] = []
        labels = rng.random(prompts_per_batch) < straggler_fraction
        for prompt_id, is_straggler in enumerate(labels):
            if is_straggler:
                n_turns = straggler_turns
                osl_mean, osl_std = straggler_osl_mean, straggler_osl_std
            else:
                n_turns = normal_turns
                osl_mean, osl_std = normal_osl_mean, normal_osl_std
            for _ in range(generations_per_prompt):
                osl_samples = rng.normal(osl_mean, osl_std, n_turns)
                max_tokens_per_turn = [max(1, int(round(v))) for v in osl_samples]
                batch.append(
                    {
                        "turns": n_turns,
                        "max_tokens_per_turn": max_tokens_per_turn,
                        "is_straggler": bool(is_straggler),
                        "base_isl": base_isl,
                        "user_tokens_per_turn": user_tokens_per_turn,
                        "prompt_id": prompt_id,
                    }
                )
        batches.append(batch)
    return batches


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
async def _stream_chat(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    priority: int | None,
) -> tuple[str, float | None, float]:
    """Send a streaming /v1/chat/completions request.

    Returns (assistant_text, ttft_ms | None, total_ms).
    """
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if priority is not None:
        payload["nvext"] = {"agent_hints": {"priority": priority}}

    start = time.monotonic()
    first_token_at: float | None = None
    response_text = ""

    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer not-used"},
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                        response_text += content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    except Exception as exc:
        logger.warning("Request failed: %s", exc)
        return "", None, (time.monotonic() - start) * 1000

    end = time.monotonic()
    ttft_ms = (first_token_at - start) * 1000 if first_token_at else None
    return response_text, ttft_ms, (end - start) * 1000


# ---------------------------------------------------------------------------
# Request / batch runners
# ---------------------------------------------------------------------------
async def run_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    spec: dict,
    request_id: int,
    priority: int | None,
) -> RequestResult:
    """Execute all turns of a single multi-turn request sequentially."""
    result = RequestResult(request_id=request_id, is_straggler=spec["is_straggler"])
    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": _pad_text(spec["base_isl"])},
    ]

    req_start = time.monotonic()
    for turn_idx in range(spec["turns"]):
        max_tokens = spec["max_tokens_per_turn"][turn_idx]
        assistant_text, ttft_ms, total_ms = await _stream_chat(
            session, url, model, messages, max_tokens, priority
        )
        result.turns.append(
            TurnResult(
                turn_idx=turn_idx,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                output_tokens_requested=max_tokens,
            )
        )
        # Build next turn: append assistant response + new user message.
        messages.append({"role": "assistant", "content": assistant_text or "ok"})
        messages.append(
            {"role": "user", "content": _pad_text(spec["user_tokens_per_turn"])}
        )
    result.total_ms = (time.monotonic() - req_start) * 1000
    return result


async def run_batch(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    batch: list[dict],
    batch_id: int,
    priority_value: int | None,
    tag_stragglers: bool,
) -> BatchResult:
    """Run every request in a batch concurrently; return when ALL finish."""
    batch_start = time.monotonic()
    tasks = []
    for idx, spec in enumerate(batch):
        pri = priority_value if (tag_stragglers and spec["is_straggler"]) else None
        tasks.append(run_request(session, url, model, spec, idx, pri))
    request_results = await asyncio.gather(*tasks)
    elapsed = (time.monotonic() - batch_start) * 1000

    n_stragglers = sum(1 for s in batch if s["is_straggler"])
    label = "priority" if tag_stragglers else "baseline"
    logger.info(
        "Batch %d [%s] done in %.0f ms  (%d requests, %d stragglers)",
        batch_id,
        label,
        elapsed,
        len(batch),
        n_stragglers,
    )
    return BatchResult(
        batch_id=batch_id,
        completion_ms=elapsed,
        requests=list(request_results),
    )


async def run_phase(
    url: str,
    model: str,
    batches: list[list[dict]],
    lag: int,
    priority_value: int | None,
    tag_stragglers: bool,
    phase_label: str,
) -> list[BatchResult]:
    """Run all batches with FIFO-ordered sliding window of *lag* concurrent batches.

    Mirrors NeMo-RL's async GRPO sequencing: up to *lag* batches run
    concurrently, but a new batch is only admitted when the **oldest**
    in-flight batch completes (not just any batch).  This matches the
    constraint that training consumes steps in order, so only the oldest
    step's completion advances ``weight_version`` and opens a new slot
    (``grpo.py:2646-2651``, ``async_utils.py:348-351``).
    """
    logger.info(
        "=== %s: %d batches, lag=%d ===",
        phase_label,
        len(batches),
        lag,
    )
    results: list[BatchResult] = []

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        # In-flight batch futures, keyed by batch_id, in submission order.
        inflight: dict[int, asyncio.Task] = {}
        next_to_submit = 0
        next_to_retire = 0  # oldest batch we're waiting on

        # Seed the window with up to `lag` batches.
        while next_to_submit < len(batches) and len(inflight) < lag:
            bid = next_to_submit
            task = asyncio.create_task(
                run_batch(
                    session, url, model, batches[bid], bid,
                    priority_value, tag_stragglers,
                )
            )
            inflight[bid] = task
            next_to_submit += 1

        # Drain: wait for the oldest batch, then admit one new batch.
        while inflight:
            result = await inflight.pop(next_to_retire)
            results.append(result)
            next_to_retire += 1

            # Admit the next batch now that the oldest slot freed.
            if next_to_submit < len(batches):
                bid = next_to_submit
                task = asyncio.create_task(
                    run_batch(
                        session, url, model, batches[bid], bid,
                        priority_value, tag_stragglers,
                    )
                )
                inflight[bid] = task
                next_to_submit += 1

    results.sort(key=lambda r: r.batch_id)
    return results


# ---------------------------------------------------------------------------
# Metrics & reporting
# ---------------------------------------------------------------------------
def _summarise(results: list[BatchResult], label: str) -> dict:
    batch_times = [r.completion_ms for r in results]
    straggler_latencies = [
        req.total_ms
        for r in results
        for req in r.requests
        if req.is_straggler
    ]
    normal_latencies = [
        req.total_ms
        for r in results
        for req in r.requests
        if not req.is_straggler
    ]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"count": 0}
        return {
            "count": len(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "p95": float(np.percentile(vals, 95)),
            "min": min(vals),
            "max": max(vals),
        }

    return {
        "label": label,
        "batch_completion": _stats(batch_times),
        "straggler_latency": _stats(straggler_latencies),
        "normal_latency": _stats(normal_latencies),
    }


def plot_results(
    baseline: list[BatchResult],
    priority: list[BatchResult],
    output_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: batch completion times ---
    ax = axes[0]
    b_times = [r.completion_ms / 1000 for r in baseline]
    p_times = [r.completion_ms / 1000 for r in priority]
    x = np.arange(len(b_times))
    w = 0.35
    ax.bar(x - w / 2, b_times, w, label="Baseline (all pool 0)")
    ax.bar(x + w / 2, p_times, w, label="Priority (stragglers->pool 1)")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Completion time (s)")
    ax.set_title("Batch Completion Time")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: per-request latency by type ---
    ax = axes[1]
    categories = ["Normal\n(baseline)", "Normal\n(priority)",
                   "Straggler\n(baseline)", "Straggler\n(priority)"]
    data = [
        [r.total_ms for br in baseline for r in br.requests if not r.is_straggler],
        [r.total_ms for br in priority for r in br.requests if not r.is_straggler],
        [r.total_ms for br in baseline for r in br.requests if r.is_straggler],
        [r.total_ms for br in priority for r in br.requests if r.is_straggler],
    ]
    positions = [1, 2, 3.5, 4.5]
    colors = ["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"]
    for pos, d, c in zip(positions, data, colors):
        if d:
            bp = ax.boxplot(
                [d], positions=[pos], widths=0.6, patch_artist=True,
                showfliers=False,
            )
            bp["boxes"][0].set_facecolor(c)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("Request latency (ms)")
    ax.set_title("Per-Request Latency")
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 3: summary bar chart ---
    ax = axes[2]
    bs = _summarise(baseline, "baseline")
    ps = _summarise(priority, "priority")
    labels = ["Mean batch\ntime", "Straggler\nmedian", "Normal\nmedian"]
    b_vals = [
        bs["batch_completion"].get("mean", 0) / 1000,
        bs["straggler_latency"].get("median", 0) / 1000,
        bs["normal_latency"].get("median", 0) / 1000,
    ]
    p_vals = [
        ps["batch_completion"].get("mean", 0) / 1000,
        ps["straggler_latency"].get("median", 0) / 1000,
        ps["normal_latency"].get("median", 0) / 1000,
    ]
    x = np.arange(len(labels))
    ax.bar(x - w / 2, b_vals, w, label="Baseline")
    ax.bar(x + w / 2, p_vals, w, label="Priority")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time (s)")
    ax.set_title("Summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Global Router Priority Pool Routing — Async RL Workload", fontsize=13
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "async_rl_priority_results.png")
    fig.savefig(path, dpi=150)
    logger.info("Plot saved: %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Async RL priority pool routing benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Connection
    p.add_argument("--url", default=DEFAULT_URL, help="Frontend URL")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name")

    # Workload shape
    p.add_argument(
        "--num-batches", type=int, default=6,
        help="Number of RL training batches to simulate",
    )
    p.add_argument(
        "--prompts-per-batch", type=int, default=16,
        help="Prompts per batch (analogous to num_prompts_per_step)",
    )
    p.add_argument(
        "--generations-per-prompt", type=int, default=4,
        help="Independent rollouts per prompt (analogous to num_generations_per_prompt; "
             "NeMo-RL default is 32, use a smaller value to keep benchmark tractable)",
    )
    p.add_argument(
        "--lag", type=int, default=2,
        help="Max concurrent batches; maps to max_trajectory_age_steps + 1 in NeMo-RL "
             "(e.g. --lag 2 matches max_trajectory_age_steps=1)",
    )

    # Normal requests
    p.add_argument("--normal-turns", type=int, default=3, help="Turns per normal request")
    p.add_argument("--normal-osl-mean", type=int, default=200, help="Mean OSL per turn (normal)")
    p.add_argument("--normal-osl-std", type=int, default=50, help="Std-dev of OSL per turn (normal)")

    # Straggler requests
    p.add_argument(
        "--straggler-fraction", type=float, default=0.15,
        help="Fraction of requests that are stragglers",
    )
    p.add_argument("--straggler-turns", type=int, default=8, help="Turns per straggler request")
    p.add_argument("--straggler-osl-mean", type=int, default=800, help="Mean OSL per turn (straggler)")
    p.add_argument("--straggler-osl-std", type=int, default=200, help="Std-dev of OSL per turn (straggler)")

    # Token sizing
    p.add_argument("--base-isl", type=int, default=500, help="Approximate first-turn ISL in tokens")
    p.add_argument("--user-tokens-per-turn", type=int, default=100, help="New user tokens added each turn")

    # Priority
    p.add_argument(
        "--priority-value", type=int, default=50,
        help="Priority value for straggler requests (must be >= 10 for default manifest override rule)",
    )

    # Output
    p.add_argument("--output-dir", default="async_rl_priority_results", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    return p.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    batches = build_batches(
        num_batches=args.num_batches,
        prompts_per_batch=args.prompts_per_batch,
        generations_per_prompt=args.generations_per_prompt,
        straggler_fraction=args.straggler_fraction,
        normal_turns=args.normal_turns,
        normal_osl_mean=args.normal_osl_mean,
        normal_osl_std=args.normal_osl_std,
        straggler_turns=args.straggler_turns,
        straggler_osl_mean=args.straggler_osl_mean,
        straggler_osl_std=args.straggler_osl_std,
        base_isl=args.base_isl,
        user_tokens_per_turn=args.user_tokens_per_turn,
        seed=args.seed,
    )

    total_requests = sum(len(b) for b in batches)
    total_stragglers = sum(1 for b in batches for s in b if s["is_straggler"])
    logger.info(
        "Workload: %d batches, %d prompts/batch x %d generations = %d requests/batch, "
        "%d total requests (%d stragglers, %.0f%%)",
        len(batches),
        args.prompts_per_batch,
        args.generations_per_prompt,
        args.prompts_per_batch * args.generations_per_prompt,
        total_requests,
        total_stragglers,
        100 * total_stragglers / total_requests if total_requests else 0,
    )

    # Phase 1: baseline (no priority, everything → pool 0)
    baseline = await run_phase(
        args.url, args.model, batches, args.lag,
        priority_value=None, tag_stragglers=False,
        phase_label="Baseline (all pool 0 / slow)",
    )

    # Phase 2: priority (stragglers tagged → pool 1 / fast)
    priority = await run_phase(
        args.url, args.model, batches, args.lag,
        priority_value=args.priority_value, tag_stragglers=True,
        phase_label="Priority (stragglers -> pool 1 / fast)",
    )

    # Report
    bs = _summarise(baseline, "baseline")
    ps = _summarise(priority, "priority")

    logger.info("--- Results ---")
    logger.info(
        "Baseline  mean batch time: %.1f s  |  straggler median: %.1f s  |  normal median: %.1f s",
        bs["batch_completion"]["mean"] / 1000,
        bs["straggler_latency"].get("median", 0) / 1000,
        bs["normal_latency"].get("median", 0) / 1000,
    )
    logger.info(
        "Priority  mean batch time: %.1f s  |  straggler median: %.1f s  |  normal median: %.1f s",
        ps["batch_completion"]["mean"] / 1000,
        ps["straggler_latency"].get("median", 0) / 1000,
        ps["normal_latency"].get("median", 0) / 1000,
    )

    if bs["batch_completion"]["mean"] > 0:
        speedup = bs["batch_completion"]["mean"] / ps["batch_completion"]["mean"]
        logger.info("Batch completion speedup: %.2fx", speedup)

    # Save raw metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"baseline": bs, "priority": ps}, f, indent=2)
    logger.info("Metrics saved: %s", metrics_path)

    # Plot
    plot_results(baseline, priority, args.output_dir)


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
