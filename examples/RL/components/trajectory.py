# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from components.replay_buffer import ReplayBuffer
from utils import Config, Trajectory, Turn

logger = logging.getLogger(__name__)


class HttpTrajectoryCollector:
    """
    Collects trajectories asynchronously via HTTP (OpenAI-compatible API).
    """

    def __init__(
        self,
        client,
        cfg: Config,
        replay_buffer: ReplayBuffer,
        reward_fn: Optional[Callable[[Trajectory], float]] = None,
        start_step: int = 0,
        num_steps: int = 1,
    ):
        self.client = client
        self.cfg = cfg
        self.replay_buffer = replay_buffer
        self.reward_fn = reward_fn or (lambda t: 0.0)

        self.running = False
        self.current_weight_version = start_step
        self.initial_weight_version = start_step
        self.max_target_weight = start_step + num_steps - 1

        # Pause/resume events
        self._manual_pause = asyncio.Event()
        self._manual_pause.set()  # Start unpaused
        self._refit_pause = asyncio.Event()
        self._refit_pause.set()  # Start unpaused

        # Track which targets are being generated
        self._generating_targets: set = set()
        self._gen_lock = asyncio.Lock()

        # Stats
        self.total_generated = 0
        self.generation_errors = 0

    def _calculate_target_weights(self, generation_weight_version: int) -> List[int]:
        """Calculate target weight versions for the generation weight version."""
        max_age = self.cfg.max_trajectory_age_steps

        if generation_weight_version == self.initial_weight_version:
            candidates = list(
                range(
                    self.initial_weight_version,
                    self.initial_weight_version + max_age + 1,
                )
            )
        else:
            candidates = [generation_weight_version + i for i in range(1, max_age + 1)]

        return [t for t in candidates if t <= self.max_target_weight]

    async def _get_next_target_for_generation(
        self, generation_weight_version: int
    ) -> Optional[int]:
        """Get the next target weight that needs generation."""
        target_weights = self._calculate_target_weights(generation_weight_version)

        async with self._gen_lock:
            existing = self.replay_buffer.get_existing_target_weights()
            for target in target_weights:
                if (
                    target > self.replay_buffer.last_target_weight_already_generated
                    and target not in self._generating_targets
                    and target not in existing
                ):
                    self._generating_targets.add(target)
                    return target

        return None

    async def _generate_single(
        self,
        prompt_data: Dict,
        gen_idx: int,
        target_weight_version: int,
    ) -> Optional[Trajectory]:
        """Generate a single trajectory."""
        try:
            messages = []
            turns = []

            for turn_data in prompt_data["turns"]:
                messages.append({"role": "user", "content": turn_data.user})

                req_start = time.perf_counter()
                response = await self.client.chat.completions.create(
                    model=self.cfg.model_name,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_new_tokens,
                    logprobs=True,
                    top_logprobs=1,
                )
                req_elapsed = time.perf_counter() - req_start

                choice = response.choices[0]
                generated_text = choice.message.content or ""

                token_logprobs = []
                if choice.logprobs and choice.logprobs.content:
                    for token_info in choice.logprobs.content:
                        if token_info.logprob is not None:
                            token_logprobs.append(token_info.logprob)

                usage = response.usage
                turn = Turn(
                    user=turn_data.user,
                    ground_truth=turn_data.ground_truth,
                    generated=generated_text,
                    logprobs=choice.logprobs.content if choice.logprobs else [],
                    token_logprobs=token_logprobs,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    latency_s=req_elapsed,
                )
                turns.append(turn)
                messages.append({"role": "assistant", "content": generated_text})

            trajectory = Trajectory(
                id=prompt_data["id"],
                gen_idx=gen_idx,
                turns=turns,
                generation_weight_version=self.current_weight_version,
                target_weight_version=target_weight_version,
            )

            # Compute reward
            trajectory.total_reward = self.reward_fn(trajectory)

            self.total_generated += 1
            return trajectory

        except Exception as e:
            logger.exception(f"[Collector] Generation error: {e}")
            self.generation_errors += 1
            return None

    async def _process_prompt_batch(
        self,
        prompts: List[Dict],
        target_weight_version: int,
    ):
        """Process a batch of prompts for a specific target weight version."""
        generation_weight = self.current_weight_version
        semaphore = asyncio.Semaphore(self.cfg.max_concurrency)

        async def bounded_generate(prompt: Dict, gen_idx: int) -> Optional[Trajectory]:
            """Generate with concurrency limiting."""
            if not self.running:
                return None
            async with semaphore:
                if not self.running:
                    return None
                return await self._generate_single(
                    prompt, gen_idx, target_weight_version
                )

        async def push_to_buffer(result: Optional[Trajectory]) -> None:
            """Push result to buffer with backoff."""
            if result is None:
                return

            backoff = 0.01
            max_retries = 100
            for _ in range(max_retries):
                status = self.replay_buffer.push(
                    result,
                    generation_weight,
                    target_weight_version,
                )
                if status == "success":
                    break
                elif status == "full":
                    if not self.running:
                        logger.warning("[Collector] Dropping trajectory")
                        break
                    await asyncio.sleep(min(backoff, 0.5))
                    backoff *= 1.5

        max_concurrent_tasks = self.cfg.max_concurrency
        active_tasks = set()

        logger.debug(
            f"[Collector] Processing batch with max {max_concurrent_tasks} concurrent tasks"
        )

        # Create work queue
        work_items = []
        for prompt in prompts:
            for gen_idx in range(self.cfg.num_generations_per_prompt):
                work_items.append((prompt, gen_idx))

        work_iter = iter(work_items)

        try:
            # Initial fill of task pool
            for _ in range(min(max_concurrent_tasks, len(work_items))):
                await self._manual_pause.wait()
                await self._refit_pause.wait()

                if not self.running:
                    break

                try:
                    prompt, gen_idx = next(work_iter)
                    task = asyncio.create_task(bounded_generate(prompt, gen_idx))
                    active_tasks.add(task)
                except StopIteration:
                    break

            # Process completions and spawn new tasks
            while active_tasks:
                # Wait for at least one task to complete
                done, active_tasks = await asyncio.wait(
                    active_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # Process completed tasks
                for task in done:
                    try:
                        result = await task
                        if result is not None:
                            await push_to_buffer(result)
                    except asyncio.CancelledError:
                        continue
                    except Exception as e:
                        logger.error(f"[Collector] Task failed: {e}")
                        continue

                # Don't spawn new tasks if stopping
                if not self.running:
                    continue

                # Spawn new tasks to maintain pool size
                while len(active_tasks) < max_concurrent_tasks:
                    await self._manual_pause.wait()
                    await self._refit_pause.wait()

                    if not self.running:
                        break

                    try:
                        prompt, gen_idx = next(work_iter)
                        task = asyncio.create_task(bounded_generate(prompt, gen_idx))
                        active_tasks.add(task)
                    except StopIteration:
                        break

        finally:
            # Release target reservation
            async with self._gen_lock:
                self._generating_targets.discard(target_weight_version)

    async def start_collection(self, data: List[Dict]):
        """Collect trajectories from the data."""
        self.running = True
        data_iter = iter(data)
        batch_size = self.cfg.num_prompts_per_step

        logger.debug(
            f"[Collector] Starting trajectory collection (batch_size={batch_size})"
        )

        try:
            while self.running:
                # Check pauses
                await self._manual_pause.wait()
                await self._refit_pause.wait()

                if not self.running:
                    break

                # Get next target weight
                target = await self._get_next_target_for_generation(
                    self.current_weight_version
                )

                if target is None:
                    # All targets generated, wait for weight update
                    await asyncio.sleep(0.1)
                    continue

                # Get next batch of prompts
                batch = []
                for _ in range(batch_size):
                    try:
                        batch.append(next(data_iter))
                    except StopIteration:
                        data_iter = iter(data)
                        batch.append(next(data_iter))

                logger.info(
                    f"[Collector] Generating for target weight {target} (current: {self.current_weight_version})"
                )
                await self._process_prompt_batch(batch, target)

        except asyncio.CancelledError:
            logger.info("[Collector] Trajectory collection cancelled")
        except Exception as e:
            logger.exception(f"[Collector] Collection error: {e}")
        finally:
            self.running = False
            logger.info("[Collector] Trajectory collection stopped")

    def set_weight_version(self, version: int):
        """Update current weight version after training step."""
        old_version = self.current_weight_version
        self.current_weight_version = version
        logger.info(f"[Collector] Weight version updated: {old_version} -> {version}")

    def pause(self):
        """Pause trajectory collection."""
        self._manual_pause.clear()
        logger.debug("[Collector] Paused")

    def resume(self):
        """Resume trajectory collection."""
        self._manual_pause.set()
        logger.debug("[Collector] Resumed")

    def prepare_for_refit(self):
        """Pause for weight refit."""
        self._refit_pause.clear()
        logger.debug("[Collector] Paused for weight refit")

    def resume_after_refit(self):
        """Resume after weight refit."""
        self._refit_pause.set()
        logger.debug("[Collector] Resumed after weight refit")

    def stop(self):
        """Stop collection."""
        self.running = False
        self._manual_pause.set()
        self._refit_pause.set()

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            "total_generated": self.total_generated,
            "generation_errors": self.generation_errors,
            "current_weight_version": self.current_weight_version,
            "running": self.running,
        }
