# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mock async GRPO implementation with Dynamo Runtime API.
"""

import argparse
import asyncio
import logging
import time
from typing import Dict, List

import uvloop
from components.data_loader import load_data
from components.replay_buffer import ReplayBuffer
from components.trajectory import DynamoTrajectoryCollector
from utils import Config, Trajectory

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="grpo_trainer")


def _blocking_training_step(
    trajectories: List[Trajectory],
    step: int,
) -> Dict[str, float]:
    """
    Mock training step that runs in a thread pool.
    """

    # Simulate training time (replace with actual training)
    import time as _time

    _time.sleep(0.5)

    # Compute metrics
    rewards = [t.total_reward for t in trajectories]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    # Compute generation logprob stats (for importance sampling)
    all_logprobs = []
    for t in trajectories:
        all_logprobs.extend(t.get_generation_logprobs())

    avg_logprob = sum(all_logprobs) / len(all_logprobs) if all_logprobs else 0.0

    # Compute trajectory age
    ages = [step - t.generation_weight_version for t in trajectories]
    avg_age = sum(ages) / len(ages) if ages else 0.0

    # Compute mean generation length (output tokens per turn)
    all_output_lengths = [turn.output_tokens for t in trajectories for turn in t.turns]
    mean_generation_length = (
        sum(all_output_lengths) / len(all_output_lengths) if all_output_lengths else 0.0
    )

    return {
        "step": step,
        "num_trajectories": len(trajectories),
        "avg_reward": avg_reward,
        "avg_generation_logprob": avg_logprob,
        "avg_trajectory_age": avg_age,
        "mean_generation_length": mean_generation_length,
        "total_tokens": sum(
            sum(turn.input_tokens + turn.output_tokens for turn in t.turns)
            for t in trajectories
        ),
    }


async def run_training_step(
    trajectories: List[Trajectory],
    step: int,
) -> Dict[str, float]:
    """
    Run training step in a thread pool to allow trajectory collection to continue in parallel
    """
    return await asyncio.to_thread(
        _blocking_training_step,
        trajectories,
        step,
    )


async def async_grpo_train(runtime: DistributedRuntime, cfg: Config, data: List[Dict]):
    """
    Main async GRPO training loop using Dynamo runtime.

    This demonstrates the full async GRPO workflow:
    1. Start background trajectory collection via Dynamo runtime
    2. Wait for buffer to fill
    3. Sample and train in a loop
    4. Coordinate weight updates with collector
    """
    buffer_size = (
        cfg.num_prompts_per_step
        * cfg.num_generations_per_prompt
        * cfg.max_trajectory_age_steps
        * 2  # Slack factor
    )
    replay_buffer = ReplayBuffer(max_size=buffer_size)

    # Mock reward based on output length
    def reward_fn(trajectory: Trajectory) -> float:
        total_output = sum(turn.output_tokens for turn in trajectory.turns)
        return min(1.0, total_output / cfg.max_new_tokens)

    # Initialize collector with Dynamo runtime
    collector = DynamoTrajectoryCollector(
        runtime=runtime,
        cfg=cfg,
        replay_buffer=replay_buffer,
        reward_fn=reward_fn,
        start_step=0,
    )
    collection_task = asyncio.create_task(collector.start_collection(data))

    # Training parameters
    min_trajectories = cfg.num_prompts_per_step * cfg.num_generations_per_prompt
    weight_version = 0

    logger.info("Async GRPO Training (Dynamo Runtime)")
    logger.info(f"  Namespace: {cfg.namespace}")
    logger.info(f"  Component: {cfg.component}")
    logger.info(f"  Endpoint: {cfg.endpoint}")
    logger.info(f"  Prompts per step: {cfg.num_prompts_per_step}")
    logger.info(f"  Generations per prompt: {cfg.num_generations_per_prompt}")
    logger.info(f"  Trajectories per step: {min_trajectories}")
    logger.info(f"  Max trajectory age: {cfg.max_trajectory_age_steps} steps")
    logger.info(f"  Max steps: {cfg.num_steps}")

    try:
        for step in range(cfg.num_steps):
            logger.info(f"Train step {step}")
            step_start = time.perf_counter()

            while True:
                result = replay_buffer.sample(
                    num_trajectories=min_trajectories,
                    current_weight_version=weight_version,
                    max_age_steps=cfg.max_trajectory_age_steps,
                )

                if result is not None:
                    trajectories, avg_age = result
                    break

                if time.perf_counter() - step_start > cfg.buffer_wait_timeout:
                    raise TimeoutError(
                        f"Timeout waiting for trajectories (buffer: {replay_buffer.size()})"
                    )

                await asyncio.sleep(0.1)

            rollout_elapsed = time.perf_counter() - step_start
            logger.info(
                f"{len(trajectories)} trajectories sampled in {rollout_elapsed:.1f}s (avg_age={avg_age:.2f})"
            )

            logger.info("Run training step")
            train_start = time.perf_counter()
            metrics = await run_training_step(trajectories, step)
            train_elapsed = time.perf_counter() - train_start

            # Simulated weight refit
            logger.info("[Train] Refitting weights...")
            collector.prepare_for_refit()
            await asyncio.sleep(5.0)  # Simulate delay from refit
            weight_version += 1
            collector.set_weight_version(weight_version)
            collector.resume_after_refit()

            step_elapsed = time.perf_counter() - step_start

            # Log metrics
            logger.info("[Train] Metrics:")
            logger.info(f"   Trajectory age: {metrics['avg_trajectory_age']:.2f} steps")
            logger.info(
                f"   Mean generation length: {metrics['mean_generation_length']:.1f} tokens"
            )
            logger.info(f"   Total tokens: {metrics['total_tokens']:,}")
            logger.info(
                f"   Generation throughput: {metrics['total_tokens'] / rollout_elapsed:.2f} tokens/s"
            )
            logger.info(
                f"[Train] Timing: wait={rollout_elapsed:.1f}s, train={train_elapsed:.1f}s, total={step_elapsed:.1f}s"
            )

            # Buffer stats
            buf_stats = replay_buffer.get_stats()
            logger.info(
                f"[Buffer] {buf_stats['current_size']}/{buf_stats['max_size']} "
                f"(pushed={buf_stats['total_pushed']}, sampled={buf_stats['total_sampled']}, expired={buf_stats['total_expired']})"
            )

    except KeyboardInterrupt:
        logger.info("[Train] Training interrupted by user")
    except Exception as e:
        logger.exception(f"[Train] Training error: {e}")
    finally:
        # Cleanup - give in-flight requests time to complete
        logger.info("[Train] Stopping trajectory collection...")
        collector.stop()

        # Wait briefly for in-flight requests to complete gracefully
        await asyncio.sleep(0.5)

        collection_task.cancel()
        try:
            await collection_task
        except asyncio.CancelledError:
            pass

        # Final stats
        logger.info(f"{'='*60}")
        logger.info("Final Statistics")
        logger.info(f"{'='*60}")
        logger.info(f"Collector: {collector.get_stats()}")
        logger.info(f"Buffer: {replay_buffer.get_stats()}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Async GRPO with Dynamo Runtime API",
    )

    # Dataset
    parser.add_argument(
        "--dataset-path", type=str, default="", help="Path to JSONL dataset"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="nvidia/Nemotron-Instruction-Following-Chat-v1",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset-subset", type=str, default="chat_if", help="HuggingFace subset"
    )

    # Dynamo settings
    parser.add_argument(
        "--namespace", type=str, default="dynamo", help="Dynamo namespace"
    )
    parser.add_argument(
        "--component",
        type=str,
        default="backend",
        help="Dynamo component (backend/prefill)",
    )
    parser.add_argument(
        "--endpoint", type=str, default="generate", help="Dynamo endpoint"
    )
    parser.add_argument(
        "--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    )

    # GRPO params
    parser.add_argument("--num-prompts-per-step", type=int, default=128)
    parser.add_argument("--num-generations-per-prompt", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--max-trajectory-age", type=int, default=1)

    # Generation params
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-concurrency", type=int, default=1024)

    # Runtime settings
    parser.add_argument(
        "--store-kv", type=str, default="etcd", choices=["etcd", "file", "mem"]
    )
    parser.add_argument(
        "--request-plane", type=str, default="tcp", choices=["tcp", "nats", "http"]
    )

    args = parser.parse_args()

    return Config(
        namespace=args.namespace,
        component=args.component,
        endpoint=args.endpoint,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        dataset_path=args.dataset_path,
        num_prompts_per_step=args.num_prompts_per_step,
        num_generations_per_prompt=args.num_generations_per_prompt,
        num_steps=args.num_steps,
        max_trajectory_age_steps=args.max_trajectory_age,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_concurrency=args.max_concurrency,
        store_kv=args.store_kv,
        request_plane=args.request_plane,
    )


@dynamo_worker()
async def main(runtime: DistributedRuntime):
    cfg = parse_args()
    data = load_data(cfg)
    await async_grpo_train(runtime, cfg, data)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())
