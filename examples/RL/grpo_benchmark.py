# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified RL Benchmark Script

Handles both single-turn and multi-turn conversations uniformly.
Single-turn is just a conversation with 1 turn.

Usage:
    # From JSONL (typically 1-turn math problems)
    python rl_benchmark.py --dataset-path batch.jsonl

    # From HuggingFace (multi-turn conversations)
    python rl_benchmark.py --dataset-name nvidia/Nemotron-Instruction-Following-Chat-v1
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from openai import AsyncOpenAI


@dataclass
class Config:
    api_base: str = "http://localhost:8000/v1"
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    # Dataset - supports HF dataset OR local JSONL
    dataset_name: str = ""  # HF dataset name
    dataset_subset: str = ""  # HF dataset subset/split
    dataset_path: str = ""  # Local JSONL path (takes precedence if set)

    # Generation params
    num_prompts: int = 128
    num_generations_per_prompt: int = 8
    max_new_tokens: int = 8192
    temperature: float = 1.0
    max_concurrency: int = 1024


@dataclass
class Turn:
    """A single turn in a conversation."""

    user: str
    ground_truth: str = ""
    generated: str = ""
    reward: float = 0.0
    logprobs: List[Any] = field(default_factory=list)
    # Timing metrics
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class BenchmarkStats:
    """Accumulated benchmark statistics."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    latencies: List[float] = field(default_factory=list)

    def add(self, input_tokens: int, output_tokens: int, latency: float):
        self.total_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.latencies.append(latency)

    def summary(self, elapsed: float) -> str:
        if not self.latencies:
            return "No requests completed"

        total_tokens = self.total_input_tokens + self.total_output_tokens
        sorted_lat = sorted(self.latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p90 = sorted_lat[int(len(sorted_lat) * 0.9)]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

        return (
            f"  Requests: {self.total_requests:,}\n"
            f"  Tokens:   {self.total_input_tokens:,} in + {self.total_output_tokens:,} out = {total_tokens:,} total\n"
            f"  Throughput:\n"
            f"    {self.total_requests / elapsed:,.1f} requests/sec\n"
            f"    {self.total_output_tokens / elapsed:,.1f} output tokens/sec\n"
            f"    {total_tokens / elapsed:,.1f} total tokens/sec\n"
            f"  Latency (per request):\n"
            f"    mean: {statistics.mean(self.latencies)*1000:.1f}ms\n"
            f"    p50:  {p50*1000:.1f}ms | p90: {p90*1000:.1f}ms | p99: {p99*1000:.1f}ms\n"
            f"    min:  {min(self.latencies)*1000:.1f}ms | max: {max(self.latencies)*1000:.1f}ms"
        )


def load_data(cfg: Config) -> List[Dict]:
    """
    Load data from either JSONL or HuggingFace dataset.
    Returns a list of conversations, each with:
      - id: unique identifier
      - turns: List[Turn] (single-turn has exactly 1 turn)
    """
    if cfg.dataset_path:
        return _load_jsonl(cfg)
    elif cfg.dataset_name:
        return _load_huggingface(cfg)
    else:
        raise ValueError("Must specify either --dataset-path or --dataset-name")


def _load_jsonl(cfg: Config) -> List[Dict]:
    """Load from local JSONL file, wrapping each prompt as a 1-turn conversation."""
    if not os.path.exists(cfg.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {cfg.dataset_path}")

    print(f"Loading dataset from {cfg.dataset_path}")
    data = []

    with open(cfg.dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            prompt_text = item.get("prompt", "")
            ground_truth = item.get("answer", item.get("ground_truth", ""))

            # Wrap as a 1-turn conversation
            turn = Turn(user=prompt_text, ground_truth=ground_truth)
            data.append({"id": idx, "turns": [turn]})

    print(f"  Loaded {len(data)} prompts (1 turn each)")
    return data


def _load_huggingface(cfg: Config) -> List[Dict]:
    """Load from HuggingFace dataset, parsing multi-turn conversations."""
    from datasets import load_dataset

    print(
        f"Loading dataset {cfg.dataset_name}"
        + (f"/{cfg.dataset_subset}" if cfg.dataset_subset else "")
    )

    if cfg.dataset_subset:
        dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_subset)
    else:
        dataset = load_dataset(cfg.dataset_name, split="train")

    data = []
    for idx, entry in enumerate(dataset):
        messages = entry.get("messages", [])
        if not messages:
            continue

        # Parse alternating user/assistant turns
        turns = []
        pending_user = None

        for msg in messages:
            role, content = msg.get("role", ""), msg.get("content", "")
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user:
                turns.append(Turn(user=pending_user, ground_truth=content))
                pending_user = None

        if turns:
            data.append({"id": idx, "turns": turns})

    turn_counts = [len(d["turns"]) for d in data]
    print(
        f"  {len(data)} conversations, {min(turn_counts)}-{max(turn_counts)} turns (avg {statistics.mean(turn_counts):.1f})"
    )

    return data


async def run_rollout(
    data: List[Dict], cfg: Config
) -> tuple[List[Dict], BenchmarkStats, float]:
    """
    Run generation rollouts for all conversations.
    Each conversation is processed turn-by-turn, accumulating message history.
    """
    client = AsyncOpenAI(
        base_url=cfg.api_base,
        api_key="not-needed",
        timeout=600.0,
    )
    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    stats = BenchmarkStats()
    stats_lock = asyncio.Lock()

    async def process_conversation(entry: Dict) -> Dict:
        """Process a single conversation (works for both 1-turn and N-turn)."""
        async with semaphore:
            messages = []
            for turn in entry["turns"]:
                messages.append({"role": "user", "content": turn.user})

                req_start = time.perf_counter()
                response = await client.chat.completions.create(
                    model=cfg.model_name,
                    messages=messages,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_new_tokens,
                    logprobs=True,
                )
                req_elapsed = time.perf_counter() - req_start

                choice = response.choices[0]
                turn.generated = choice.message.content
                turn.logprobs = choice.logprobs.content if choice.logprobs else []

                # Extract token counts from response
                usage = response.usage
                turn.input_tokens = usage.prompt_tokens if usage else 0
                turn.output_tokens = usage.completion_tokens if usage else 0
                turn.latency_s = req_elapsed

                # Thread-safe stats update
                async with stats_lock:
                    stats.add(turn.input_tokens, turn.output_tokens, req_elapsed)

                messages.append({"role": "assistant", "content": turn.generated})
            return entry

    # Run all conversations in parallel with concurrency limit
    start_time = time.perf_counter()
    results = await asyncio.gather(*[process_conversation(entry) for entry in data])
    elapsed = time.perf_counter() - start_time

    total_turns = sum(len(r["turns"]) for r in results)
    print(
        f"Completed {len(results)} conversations ({total_turns} turns) in {elapsed:.2f}s"
    )
    print(stats.summary(elapsed))

    return results, stats, elapsed


async def main(cfg: Config):
    """Main entry point."""
    # Load data (handles both JSONL and HF datasets)
    data = load_data(cfg)

    # Limit to num_prompts
    if len(data) > cfg.num_prompts:
        data = data[: cfg.num_prompts]
        print(f"Limited to {cfg.num_prompts} prompts")

    # Expand for multiple generations per prompt
    expanded_data = [
        {
            "id": p["id"],
            "gen_idx": i,
            "turns": [Turn(t.user, t.ground_truth) for t in p["turns"]],
        }
        for p in data
        for i in range(cfg.num_generations_per_prompt)
    ]

    print(f"\n{'='*60}")
    print(
        f"Running: {len(data)} prompts × {cfg.num_generations_per_prompt} = {len(expanded_data)} rollouts"
    )
    print(f"{'='*60}")

    # Run rollouts
    results, stats, elapsed = await run_rollout(expanded_data, cfg)

    # Print sample results
    print(f"\n{'='*60}")
    print("Sample Results (first 3)")
    print(f"{'='*60}")
    for r in results[:3]:
        print(f"\nConversation {r['id']} (gen {r.get('gen_idx', 0)}):")
        for i, turn in enumerate(r["turns"]):
            print(f"  Turn {i+1}:")
            print(f"    User: {turn.user[:100]}...")
            print(f"    Generated: {turn.generated[:100]}...")
            print(f"    Tokens: {turn.input_tokens} in, {turn.output_tokens} out")


def parse_args() -> Config:
    """Parse command line arguments into Config."""
    parser = argparse.ArgumentParser(description="Unified RL Benchmark")

    # Dataset source (mutually exclusive-ish, path takes precedence)
    parser.add_argument(
        "--dataset-path", type=str, default="", help="Path to local JSONL dataset file"
    )
    parser.add_argument(
        "--dataset-name", type=str, default="", help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset-subset",
        type=str,
        default="",
        help="HuggingFace dataset subset/split",
    )

    # Model/API
    parser.add_argument(
        "--api-base", type=str, default="http://localhost:8000/v1", help="API base URL"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name",
    )

    # Generation params
    parser.add_argument(
        "--num-prompts", type=int, default=5, help="Number of prompts to use"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="Number of generations per prompt",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max new tokens per generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=1024, help="Max concurrent requests"
    )

    args = parser.parse_args()

    return Config(
        api_base=args.api_base,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        dataset_path=args.dataset_path,
        num_prompts=args.num_prompts,
        num_generations_per_prompt=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_concurrency=args.max_concurrency,
    )


if __name__ == "__main__":
    cfg = parse_args()
    asyncio.run(main(cfg))
