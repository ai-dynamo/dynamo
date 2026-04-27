#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async RL priority benchmark for aggregated serving deployments.

Models a NeMo-RL async GRPO workload where multiple batches of multi-turn
requests run concurrently. Each batch contains a mix of "normal" requests, a
configurable fraction of "straggler" requests (many turns, large output per
turn), and a small number of even larger "whale" requests. A sliding window of
``--lag`` concurrent batches mirrors the
``max_trajectory_age_steps`` parameter in NeMo-RL async GRPO.

The benchmark always runs a single mixed workload:

* normal requests are sent without priority hints
* a selected subset of tail requests carries
  ``nvext.agent_hints.priority = <priority-value>``

In a single-pool deployment, those hints are effectively inert. In a
global-router deployment, tagged tail requests can route to a different pool.
The default tag policy is ``oldest_inflight_stragglers``, which tags
stragglers / whales in whichever batch is currently oldest in flight. Other
policies can instead use only observable signals such as batch age, batch tail
completion, or current estimated ISL. Because requests are multi-turn, later
turns can become priority-tagged even if earlier turns were not.

A batch "completes" only when **all** its requests finish every turn, so
tail requests gate the batch. The benchmark measures batch completion time plus
normal/straggler/whale request latency under that mixed workload.

Prerequisites
-------------
* Deploy either:
  - ``examples/global_planner/single-pool-priority-benchmark.yaml``, or
  - ``examples/global_planner/global-planner-priority-benchmark.yaml``
  (both are mocker-based and require no GPUs).
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
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

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
DEFAULT_MODEL = "nvidia/llama-3.1-8b-instruct-fp8"
DEFAULT_URL = "http://localhost:8000"

# Per-token estimate for padding (Llama-3 tokeniser ≈ 4 chars/token on
# repetitive English; "alpha " is consistently 1–2 tokens).
CHARS_PER_TOKEN = 5
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


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
    request_class: str
    is_straggler: bool
    is_whale: bool
    turns: list[TurnResult] = field(default_factory=list)
    total_ms: float = 0.0
    tagged_turns: int = 0


@dataclass
class BatchResult:
    batch_id: int
    completion_ms: float = 0.0
    requests: list[RequestResult] = field(default_factory=list)


@dataclass
class OldestInFlightTracker:
    batch_id: int = 0


@dataclass
class BatchProgressTracker:
    total_requests: int
    completed_requests: int = 0


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------
def _pad_text(target_tokens: int) -> str:
    """Return a string that tokenises to roughly *target_tokens* tokens."""
    return "alpha " * max(target_tokens, 1)


def _run_prompt_text(
    target_tokens: int,
    run_key: str,
    prompt_id: int,
    turn_idx: int,
) -> str:
    """Build prompt text that preserves intra-run sharing but isolates runs.

    A small run/prompt marker prevents repeated benchmark invocations from
    accidentally sharing warm KV state, while still letting all generations of
    the same prompt share prefixes within a run.
    """
    marker = f"run {run_key} prompt {prompt_id} turn {turn_idx} "
    marker_tokens = max(1, math.ceil(len(marker) / CHARS_PER_TOKEN))
    return marker + _pad_text(max(target_tokens - marker_tokens, 1))


def build_batches(
    num_batches: int,
    prompts_per_batch: int,
    generations_per_prompt: int,
    straggler_fraction: float,
    whale_fraction: float,
    normal_turns: int,
    normal_turns_std: float,
    normal_osl_mean: int,
    normal_osl_std: int,
    straggler_turns: int,
    straggler_turns_std: float,
    straggler_osl_mean: int,
    straggler_osl_std: int,
    whale_turns_multiplier: float,
    whale_osl_multiplier: float,
    base_isl: int,
    user_tokens_per_turn: int,
    seed: int,
) -> list[list[dict]]:
    """Build a list of batches.  Each batch is a list of request descriptors.

    Each prompt contributes *generations_per_prompt* independent requests,
    mirroring ``repeat_interleave(num_generations)`` in NeMo-RL
    (``async_utils.py:491``). Request class is assigned per request copy, not
    per prompt group, so different generations of the same prompt may diverge
    into normal / straggler / whale tails independently.

    A request descriptor is a dict with keys:
        turns            – number of conversation turns
        max_tokens_per_turn – list[int], sampled OSL for each turn
        request_class    – "normal" | "straggler" | "whale"
        is_straggler     – bool
        is_whale         – bool
        base_isl         – first-turn ISL (approximate tokens)
        user_tokens_per_turn – extra user tokens added each turn
        prompt_id        – int, identifies which prompt this generation belongs to
    """
    rng = np.random.RandomState(seed)
    total_prompts = num_batches * prompts_per_batch
    requests_per_batch = prompts_per_batch * generations_per_prompt
    total_requests = total_prompts * generations_per_prompt
    if straggler_fraction + whale_fraction > 1.0:
        raise ValueError("straggler_fraction + whale_fraction must be <= 1.0")

    num_straggler_requests = int(round(total_requests * straggler_fraction))
    num_whale_requests = int(round(total_requests * whale_fraction))
    request_classes = np.array(
        (
            ["whale"] * num_whale_requests
            + ["straggler"] * num_straggler_requests
            + ["normal"] * (
                total_requests - num_straggler_requests - num_whale_requests
            )
        ),
        dtype=object,
    )
    rng.shuffle(request_classes)

    request_specs: list[dict[str, Any]] = []
    for request_class in request_classes.tolist():
        is_whale = request_class == "whale"
        is_straggler = request_class == "straggler"
        if is_whale:
            sampled_turns = rng.normal(
                straggler_turns * whale_turns_multiplier,
                straggler_turns_std * whale_turns_multiplier,
            )
            n_turns = max(1, int(round(sampled_turns)))
            osl_mean = int(round(straggler_osl_mean * whale_osl_multiplier))
            osl_std = max(1, int(round(straggler_osl_std * whale_osl_multiplier)))
        elif is_straggler:
            sampled_turns = rng.normal(straggler_turns, straggler_turns_std)
            n_turns = max(1, int(round(sampled_turns)))
            osl_mean, osl_std = straggler_osl_mean, straggler_osl_std
        else:
            sampled_turns = rng.normal(normal_turns, normal_turns_std)
            n_turns = max(1, int(round(sampled_turns)))
            osl_mean, osl_std = normal_osl_mean, normal_osl_std

        osl_samples = rng.normal(osl_mean, osl_std, n_turns)
        max_tokens_per_turn = [max(1, int(round(v))) for v in osl_samples]
        total_work = (
            base_isl
            + user_tokens_per_turn * max(n_turns - 1, 0)
            + sum(max_tokens_per_turn)
        )
        request_specs.append(
            {
                "request_class": request_class,
                "turns": n_turns,
                "max_tokens_per_turn": max_tokens_per_turn,
                "is_straggler": bool(is_straggler),
                "is_whale": bool(is_whale),
                "base_isl": base_isl,
                "user_tokens_per_turn": user_tokens_per_turn,
                "total_work": total_work,
            }
        )

    batch_requests: list[list[dict[str, Any]]] = [[] for _ in range(num_batches)]
    batch_work = [0] * num_batches

    whale_groups = [
        spec for spec in request_specs if spec["request_class"] == "whale"
    ]
    other_groups = [
        spec for spec in request_specs if spec["request_class"] != "whale"
    ]

    # Spread whale requests across the run rather than front-loading them into
    # the initial in-flight window. A staggered batch order keeps whales out of
    # the first few batches when possible while still distributing them evenly.
    whale_groups.sort(key=lambda group: group["total_work"], reverse=True)
    if whale_groups:
        staggered_batch_order = list(range(1, num_batches, 2)) + list(
            range(0, num_batches, 2)
        )
        whale_batch_ids = [
            staggered_batch_order[idx % num_batches] for idx in range(len(whale_groups))
        ]
        for batch_id, group in zip(whale_batch_ids, whale_groups, strict=True):
            batch_requests[batch_id].append(group)
            batch_work[batch_id] += group["total_work"]

    # Keep the overall request mix fixed, but spread the remaining heavy
    # groups across batches so one run does not get an arbitrarily unlucky tail
    # batch.
    other_groups.sort(key=lambda group: group["total_work"], reverse=True)
    for group in other_groups:
        eligible_batch_ids = [
            batch_id
            for batch_id, requests in enumerate(batch_requests)
            if len(requests) < requests_per_batch
        ]
        batch_id = min(eligible_batch_ids, key=lambda idx: (batch_work[idx], idx))
        batch_requests[batch_id].append(group)
        batch_work[batch_id] += group["total_work"]

    batches: list[list[dict]] = []
    for requests in batch_requests:
        prompt_slots: list[list[dict[str, Any]]] = [[] for _ in range(prompts_per_batch)]
        prompt_slot_work = [0] * prompts_per_batch
        for request in sorted(
            requests, key=lambda spec: spec["total_work"], reverse=True
        ):
            eligible_prompt_ids = [
                prompt_id
                for prompt_id, slot in enumerate(prompt_slots)
                if len(slot) < generations_per_prompt
            ]
            prompt_id = min(
                eligible_prompt_ids,
                key=lambda idx: (len(prompt_slots[idx]), prompt_slot_work[idx], idx),
            )
            prompt_slots[prompt_id].append(request)
            prompt_slot_work[prompt_id] += request["total_work"]

        batch: list[dict] = []
        for prompt_id, slot in enumerate(prompt_slots):
            for request in slot:
                request_spec = {k: v for k, v in request.items() if k != "total_work"}
                batch.append(
                    {
                        **request_spec,
                        "prompt_id": prompt_id,
                    }
                )
        batches.append(batch)
    return batches


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
async def _stream_chat_once(
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

    end = time.monotonic()
    ttft_ms = (first_token_at - start) * 1000 if first_token_at else None
    return response_text, ttft_ms, (end - start) * 1000


def _is_retryable_request_error(exc: Exception) -> bool:
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in RETRYABLE_STATUS_CODES
    return isinstance(
        exc,
        (
            aiohttp.ClientConnectionError,
            aiohttp.ServerDisconnectedError,
            asyncio.TimeoutError,
        ),
    )


async def _stream_chat(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    priority: int | None,
    max_attempts: int = 6,
    retry_backoff_s: float = 0.5,
) -> tuple[str, float | None, float]:
    """Send a streaming /v1/chat/completions request with transient retries."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await _stream_chat_once(
                session, url, model, messages, max_tokens, priority
            )
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_request_error(exc) or attempt == max_attempts:
                break
            sleep_s = retry_backoff_s * (2 ** (attempt - 1))
            logger.warning(
                "Request attempt %d/%d failed transiently: %s; retrying in %.1fs",
                attempt,
                max_attempts,
                exc,
                sleep_s,
            )
            await asyncio.sleep(sleep_s)
    raise RuntimeError(f"Request failed after {max_attempts} attempts: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Request / batch runners
# ---------------------------------------------------------------------------
async def run_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    spec: dict,
    batch_id: int,
    request_id: int,
    priority_value: int,
    priority_tag_policy: str,
    oldest_inflight: OldestInFlightTracker,
    batch_progress: BatchProgressTracker,
    tail_promotion_fraction: float,
    priority_isl_threshold: int,
    run_key: str,
) -> RequestResult:
    """Execute all turns of a single multi-turn request sequentially."""
    result = RequestResult(
        request_id=request_id,
        request_class=spec["request_class"],
        is_straggler=spec["is_straggler"],
        is_whale=spec["is_whale"],
    )
    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": _run_prompt_text(
                spec["base_isl"], run_key, spec["prompt_id"], 0
            ),
        },
    ]

    req_start = time.monotonic()
    current_isl_tokens = spec["base_isl"]
    for turn_idx in range(spec["turns"]):
        max_tokens = spec["max_tokens_per_turn"][turn_idx]
        should_tag = _should_tag_request(
            batch_id,
            spec,
            priority_tag_policy,
            oldest_inflight.batch_id,
            batch_progress.completed_requests,
            batch_progress.total_requests,
            tail_promotion_fraction,
            current_isl_tokens,
            priority_isl_threshold,
        )
        priority = priority_value if should_tag else None
        if should_tag:
            result.tagged_turns += 1
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
            {
                "role": "user",
                "content": _run_prompt_text(
                    spec["user_tokens_per_turn"],
                    run_key,
                    spec["prompt_id"],
                    turn_idx + 1,
                ),
            }
        )
        current_isl_tokens += max_tokens + spec["user_tokens_per_turn"]
    result.total_ms = (time.monotonic() - req_start) * 1000
    batch_progress.completed_requests += 1
    return result


def _should_tag_request(
    batch_id: int,
    spec: dict,
    tag_policy: str,
    current_oldest_batch_id: int,
    completed_requests_in_batch: int,
    total_requests_in_batch: int,
    tail_promotion_fraction: float,
    current_isl_tokens: int,
    priority_isl_threshold: int,
) -> bool:
    """Return whether this request should receive a priority hint now."""
    if tag_policy == "oldest_inflight_batch":
        return batch_id == current_oldest_batch_id
    if tag_policy == "isl_threshold":
        return current_isl_tokens >= priority_isl_threshold
    if tag_policy == "batch_tail_promotion":
        if batch_id != current_oldest_batch_id:
            return False
        if total_requests_in_batch <= 0:
            return False
        threshold = math.ceil(total_requests_in_batch * tail_promotion_fraction)
        return completed_requests_in_batch >= threshold
    if tag_policy == "batch_tail_promotion_all_batches":
        if total_requests_in_batch <= 0:
            return False
        threshold = math.ceil(total_requests_in_batch * tail_promotion_fraction)
        return completed_requests_in_batch >= threshold
    if not (spec["is_straggler"] or spec["is_whale"]):
        return False
    if tag_policy == "all_stragglers":
        return True
    if tag_policy == "oldest_inflight_stragglers":
        return batch_id == current_oldest_batch_id
    if tag_policy == "earliest_batch_stragglers":
        return batch_id == 0
    raise ValueError(f"Unknown priority tag policy: {tag_policy}")


async def run_batch(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    batch: list[dict],
    batch_id: int,
    priority_value: int,
    priority_tag_policy: str,
    oldest_inflight: OldestInFlightTracker,
    tail_promotion_fraction: float,
    priority_isl_threshold: int,
    run_key: str,
) -> BatchResult:
    """Run every request in a batch concurrently; return when ALL finish."""
    batch_start = time.monotonic()
    tasks = []
    batch_progress = BatchProgressTracker(total_requests=len(batch))
    for idx, spec in enumerate(batch):
        tasks.append(
            run_request(
                session,
                url,
                model,
                spec,
                batch_id,
                idx,
                priority_value,
                priority_tag_policy,
                oldest_inflight,
                batch_progress,
                tail_promotion_fraction,
                priority_isl_threshold,
                run_key,
            )
        )
    request_results = await asyncio.gather(*tasks)
    elapsed = (time.monotonic() - batch_start) * 1000

    n_stragglers = sum(1 for s in batch if s["is_straggler"])
    n_whales = sum(1 for s in batch if s["is_whale"])
    n_tagged_requests = sum(1 for r in request_results if r.tagged_turns > 0)
    n_tagged_turns = sum(r.tagged_turns for r in request_results)
    logger.info(
        "Batch %d done in %.0f ms  (%d requests, %d stragglers, %d whales, %d tagged requests, %d tagged turns)",
        batch_id,
        elapsed,
        len(batch),
        n_stragglers,
        n_whales,
        n_tagged_requests,
        n_tagged_turns,
    )
    return BatchResult(
        batch_id=batch_id,
        completion_ms=elapsed,
        requests=list(request_results),
    )


async def run_workload(
    url: str,
    model: str,
    batches: list[list[dict]],
    lag: int,
    priority_value: int,
    priority_tag_policy: str,
    tail_promotion_fraction: float,
    priority_isl_threshold: int,
    run_key: str,
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
        "=== Tagged mixed workload: %d batches, lag=%d ===",
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
        oldest_inflight = OldestInFlightTracker(batch_id=0)

        # Seed the window with up to `lag` batches.
        while next_to_submit < len(batches) and len(inflight) < lag:
            bid = next_to_submit
            task = asyncio.create_task(
                run_batch(
                    session, url, model, batches[bid], bid,
                    priority_value, priority_tag_policy, oldest_inflight,
                    tail_promotion_fraction, priority_isl_threshold, run_key,
                )
            )
            inflight[bid] = task
            next_to_submit += 1

        # Drain: wait for the oldest batch, then admit one new batch.
        while inflight:
            result = await inflight.pop(next_to_retire)
            results.append(result)
            next_to_retire += 1
            if inflight or next_to_submit < len(batches):
                oldest_inflight.batch_id = next_to_retire

            # Admit the next batch now that the oldest slot freed.
            if next_to_submit < len(batches):
                bid = next_to_submit
                task = asyncio.create_task(
                    run_batch(
                        session, url, model, batches[bid], bid,
                        priority_value, priority_tag_policy, oldest_inflight,
                        tail_promotion_fraction, priority_isl_threshold, run_key,
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
    whale_latencies = [
        req.total_ms
        for r in results
        for req in r.requests
        if req.is_whale
    ]
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
        if req.request_class == "normal"
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
        "whale_latency": _stats(whale_latencies),
        "straggler_latency": _stats(straggler_latencies),
        "normal_latency": _stats(normal_latencies),
    }


def plot_results(results: list[BatchResult], output_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: batch completion times ---
    ax = axes[0]
    batch_times = [r.completion_ms / 1000 for r in results]
    x = np.arange(len(batch_times))
    ax.bar(x, batch_times, 0.7, color="#ff7f0e")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Completion time (s)")
    ax.set_title("Batch Completion Time")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: per-request latency by type ---
    ax = axes[1]
    categories = ["Normal", "Straggler"]
    data = [
        [r.total_ms for br in results for r in br.requests if r.request_class == "normal"],
        [r.total_ms for br in results for r in br.requests if r.is_straggler],
    ]
    colors = ["#1f77b4", "#ff7f0e"]
    whale_data = [r.total_ms for br in results for r in br.requests if r.is_whale]
    if whale_data:
        categories.append("Whale")
        data.append(whale_data)
        colors.append("#c44e52")
    positions = list(range(1, len(categories) + 1))
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
    summary = _summarise(results, "priority")
    labels = ["Mean batch\ntime"]
    values = [summary["batch_completion"].get("mean", 0) / 1000]
    bar_colors = ["#4c72b0"]
    if summary["whale_latency"].get("count", 0):
        labels.append("Whale\nmedian")
        values.append(summary["whale_latency"].get("median", 0) / 1000)
        bar_colors.append("#c44e52")
    labels.extend(["Straggler\nmedian", "Normal\nmedian"])
    values.extend(
        [
            summary["straggler_latency"].get("median", 0) / 1000,
            summary["normal_latency"].get("median", 0) / 1000,
        ]
    )
    bar_colors.extend(["#dd8452", "#55a868"])
    x = np.arange(len(labels))
    ax.bar(x, values, 0.7, color=bar_colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time (s)")
    ax.set_title("Summary")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Async RL Priority Benchmark", fontsize=13
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
        description="Async RL priority benchmark",
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
    p.add_argument("--normal-turns", type=int, default=3, help="Mean turns per normal request")
    p.add_argument("--normal-turns-std", type=float, default=0.0, help="Std-dev of turns per normal request")
    p.add_argument("--normal-osl-mean", type=int, default=200, help="Mean OSL per turn (normal)")
    p.add_argument("--normal-osl-std", type=int, default=50, help="Std-dev of OSL per turn (normal)")

    # Straggler requests
    p.add_argument(
        "--straggler-fraction", type=float, default=0.15,
        help="Fraction of requests that are stragglers",
    )
    p.add_argument("--straggler-turns", type=int, default=8, help="Mean turns per straggler request")
    p.add_argument("--straggler-turns-std", type=float, default=0.0, help="Std-dev of turns per straggler request")
    p.add_argument("--straggler-osl-mean", type=int, default=800, help="Mean OSL per turn (straggler)")
    p.add_argument("--straggler-osl-std", type=int, default=200, help="Std-dev of OSL per turn (straggler)")
    p.add_argument(
        "--whale-fraction", type=float, default=0.02,
        help="Fraction of prompts that are very large whale requests",
    )
    p.add_argument(
        "--whale-turns-multiplier", type=float, default=2.0,
        help="Multiplier applied to straggler turns when sampling whales",
    )
    p.add_argument(
        "--whale-osl-multiplier", type=float, default=2.25,
        help="Multiplier applied to straggler OSL when sampling whales",
    )

    # Token sizing
    p.add_argument("--base-isl", type=int, default=8000, help="Approximate first-turn ISL in tokens")
    p.add_argument("--user-tokens-per-turn", type=int, default=256, help="New user tokens added each turn")

    # Priority
    p.add_argument(
        "--priority-value", type=int, default=50,
        help="Priority value for tail requests (must be >= 10 for default manifest override rule)",
    )
    p.add_argument(
        "--priority-tag-policy",
        choices=[
            "oldest_inflight_stragglers",
            "oldest_inflight_batch",
            "isl_threshold",
            "batch_tail_promotion",
            "batch_tail_promotion_all_batches",
            "earliest_batch_stragglers",
            "all_stragglers",
        ],
        default="oldest_inflight_stragglers",
        help="Which tail requests receive priority hints during the run",
    )
    p.add_argument(
        "--priority-isl-threshold",
        type=int,
        default=12000,
        help="For `isl_threshold`: estimated current-turn ISL in tokens at which the request starts carrying priority hints",
    )
    p.add_argument(
        "--tail-promotion-fraction",
        type=float,
        default=0.8,
        help="For tail-promotion policies: promote unfinished requests once this fraction of the batch has completed",
    )
    p.add_argument(
        "--phase",
        choices=["both", "baseline", "priority"],
        default="priority",
        help=argparse.SUPPRESS,
    )

    # Output
    p.add_argument("--output-dir", default="async_rl_priority_results", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    return p.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    if args.phase != "priority":
        logger.warning(
            "Ignoring legacy --phase=%s; benchmark always runs the tagged mixed workload",
            args.phase,
        )
    if args.priority_tag_policy == "isl_threshold" and args.priority_isl_threshold <= 0:
        raise ValueError("--priority-isl-threshold must be > 0 for isl_threshold policy")

    batches = build_batches(
        num_batches=args.num_batches,
        prompts_per_batch=args.prompts_per_batch,
        generations_per_prompt=args.generations_per_prompt,
        straggler_fraction=args.straggler_fraction,
        whale_fraction=args.whale_fraction,
        normal_turns=args.normal_turns,
        normal_turns_std=args.normal_turns_std,
        normal_osl_mean=args.normal_osl_mean,
        normal_osl_std=args.normal_osl_std,
        straggler_turns=args.straggler_turns,
        straggler_turns_std=args.straggler_turns_std,
        straggler_osl_mean=args.straggler_osl_mean,
        straggler_osl_std=args.straggler_osl_std,
        whale_turns_multiplier=args.whale_turns_multiplier,
        whale_osl_multiplier=args.whale_osl_multiplier,
        base_isl=args.base_isl,
        user_tokens_per_turn=args.user_tokens_per_turn,
        seed=args.seed,
    )

    total_requests = sum(len(b) for b in batches)
    total_stragglers = sum(1 for b in batches for s in b if s["is_straggler"])
    total_whales = sum(1 for b in batches for s in b if s["is_whale"])
    logger.info(
        "Workload: %d batches, %d prompts/batch x %d generations = %d requests/batch, "
        "%d total requests (%d stragglers, %.0f%%, %d whales, %.0f%%)",
        len(batches),
        args.prompts_per_batch,
        args.generations_per_prompt,
        args.prompts_per_batch * args.generations_per_prompt,
        total_requests,
        total_stragglers,
        100 * total_stragglers / total_requests if total_requests else 0,
        total_whales,
        100 * total_whales / total_requests if total_requests else 0,
    )
    logger.info(
        "Priority tagging policy: %s",
        args.priority_tag_policy,
    )
    if args.priority_tag_policy == "isl_threshold":
        logger.info(
            "ISL promotion threshold: %d tokens",
            args.priority_isl_threshold,
        )
    if args.priority_tag_policy in {
        "batch_tail_promotion",
        "batch_tail_promotion_all_batches",
    }:
        logger.info(
            "Tail promotion threshold: %.0f%% batch completion",
            args.tail_promotion_fraction * 100,
        )

    results = await run_workload(
        args.url, args.model, batches, args.lag,
        priority_value=args.priority_value,
        priority_tag_policy=args.priority_tag_policy,
        tail_promotion_fraction=args.tail_promotion_fraction,
        priority_isl_threshold=args.priority_isl_threshold,
        run_key="priority",
    )

    total_tagged_requests = sum(
        1 for batch in results for req in batch.requests if req.tagged_turns > 0
    )
    total_tagged_turns = sum(req.tagged_turns for batch in results for req in batch.requests)
    metrics: dict[str, Any] = {
        "priority": _summarise(results, "priority"),
        "workload": {
            "total_requests": total_requests,
            "normal_requests": total_requests - total_stragglers - total_whales,
            "straggler_requests": total_stragglers,
            "whale_requests": total_whales,
        },
        "tagging": {
            "policy": args.priority_tag_policy,
            "priority_value": args.priority_value,
            "priority_isl_threshold": args.priority_isl_threshold,
            "tail_promotion_fraction": args.tail_promotion_fraction,
            "tagged_requests": total_tagged_requests,
            "tagged_turns": total_tagged_turns,
            "total_requests": total_requests,
        },
    }

    logger.info("--- Results ---")
    ps = metrics["priority"]
    if ps["whale_latency"].get("count", 0):
        logger.info(
            "Priority  mean batch time: %.1f s  |  whale median: %.1f s  |  "
            "straggler median: %.1f s  |  normal median: %.1f s",
            ps["batch_completion"]["mean"] / 1000,
            ps["whale_latency"].get("median", 0) / 1000,
            ps["straggler_latency"].get("median", 0) / 1000,
            ps["normal_latency"].get("median", 0) / 1000,
        )
    else:
        logger.info(
            "Priority  mean batch time: %.1f s  |  straggler median: %.1f s  |  normal median: %.1f s",
            ps["batch_completion"]["mean"] / 1000,
            ps["straggler_latency"].get("median", 0) / 1000,
            ps["normal_latency"].get("median", 0) / 1000,
        )

    # Save raw metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved: %s", metrics_path)

    plot_results(results, args.output_dir)


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
