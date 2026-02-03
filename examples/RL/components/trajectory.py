# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from components.replay_buffer import ReplayBuffer
from utils import Config, Trajectory, Turn

from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


class DynamoTrajectoryCollector:
    """
    Collects trajectories asynchronously via Dynamo runtime API.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        cfg: Config,
        replay_buffer: ReplayBuffer,
        reward_fn: Optional[Callable[[Trajectory], float]] = None,
        start_step: int = 0,
    ):
        self.runtime = runtime
        self.cfg = cfg
        self.replay_buffer = replay_buffer
        self.reward_fn = reward_fn or (lambda t: 0.0)

        self.running = False
        self.current_weight_version = start_step
        self.initial_weight_version = start_step

        # Dynamo client
        self.client = None
        self.tokenizer = None

        # Pause/resume events
        self._manual_pause = asyncio.Event()
        self._manual_pause.set()  # Start unpaused
        self._refit_pause = asyncio.Event()
        self._refit_pause.set()  # Start unpaused

        # Track which targets are being generated
        self._generating_targets: set = set()
        self._gen_lock = asyncio.Lock()

        # Generation stats
        self.total_generated = 0
        self.generation_errors = 0

    async def _init_client(self):
        """Initialize the Dynamo runtime client."""
        logger.debug(
            f"[Collector] Connecting to {self.cfg.namespace}.{self.cfg.component}.{self.cfg.endpoint}"
        )

        # Get the worker endpoint
        endpoint = (
            self.runtime.namespace(self.cfg.namespace)
            .component(self.cfg.component)
            .endpoint(self.cfg.endpoint)
        )
        self.client = await endpoint.client()

        logger.debug("[Collector] Waiting for worker instances...")
        await self.client.wait_for_instances()

        instance_ids = self.client.instance_ids()
        logger.info(
            f"[Collector] Connected to {len(instance_ids)} worker instances: {instance_ids}"
        )

        # Load tokenizer
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
            logger.debug("[Collector] Loaded tokenizer")
        except Exception as e:
            logger.warning(f"[Collector] Could not load tokenizer: {e}")
            logger.warning(
                "[Collector] Will send text prompts (workers must handle tokenization)"
            )
            self.tokenizer = None

    def _calculate_target_weights(self, generation_weight_version: int) -> List[int]:
        """Calculate target weight versions for the generation weight version."""
        max_age = self.cfg.max_trajectory_age_steps

        if generation_weight_version == self.initial_weight_version:
            return list(
                range(
                    self.initial_weight_version,
                    self.initial_weight_version + max_age + 1,
                )
            )

        return [generation_weight_version + i for i in range(1, max_age + 1)]

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

    def _build_request(self, prompt: str) -> Dict[str, Any]:
        request = {
            "model": self.cfg.model_name,
            "sampling_options": {
                "temperature": self.cfg.temperature,
                "top_p": 1.0,
            },
            "stop_conditions": {
                "max_tokens": self.cfg.max_new_tokens,
            },
            "output_options": {
                "logprobs": 1,
            },
        }

        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode(prompt)
            request["token_ids"] = token_ids
        else:
            request["prompt"] = prompt

        return request

    async def _generate_single(
        self,
        prompt_data: Dict,
        gen_idx: int,
        target_weight_version: int,
    ) -> Optional[Trajectory]:
        """Generate a single trajectory."""
        if self.client is None:
            raise RuntimeError("Client not initialized. Call _init_client() first.")

        try:
            turns = []

            for turn_data in prompt_data["turns"]:
                generated_tokens = []
                token_logprobs = []

                req_start = time.perf_counter()
                request = self._build_request(turn_data.user)

                try:
                    stream = await self.client.generate(request)

                    async for response in stream:
                        if hasattr(response, "data"):
                            output = response.data()
                        else:
                            output = response

                        if isinstance(output, str):
                            try:
                                output = json.loads(output)
                            except json.JSONDecodeError:
                                continue
                        if not isinstance(output, dict):
                            continue

                        # Extract tokens
                        if "token_ids" in output:
                            new_tokens = output["token_ids"]
                            if isinstance(new_tokens, list):
                                generated_tokens.extend(new_tokens)

                        # Extract logprobs
                        if "log_probs" in output and output["log_probs"]:
                            log_probs = output["log_probs"]
                            if isinstance(log_probs, list):
                                token_logprobs.extend(log_probs)

                        # Check for finish
                        if output.get("finish_reason"):
                            break

                except Exception as e:
                    logger.error(f"[Collector] Stream error: {e}")
                    raise

                req_elapsed = time.perf_counter() - req_start

                if self.tokenizer is not None and generated_tokens:
                    generated_text = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                else:
                    generated_text = "".join(str(t) for t in generated_tokens)

                turn = Turn(
                    user=turn_data.user,
                    ground_truth=turn_data.ground_truth,
                    generated=generated_text,
                    token_logprobs=token_logprobs,
                    input_tokens=len(self.tokenizer.encode(turn_data.user))
                    if self.tokenizer
                    else 0,
                    output_tokens=len(generated_tokens),
                    latency_s=req_elapsed,
                )
                turns.append(turn)

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
            async with semaphore:
                return await self._generate_single(
                    prompt, gen_idx, target_weight_version
                )

        # Create all generation tasks
        tasks = []
        for prompt in prompts:
            for gen_idx in range(self.cfg.num_generations_per_prompt):
                # Check for pause before each generation during weight refit
                await self._manual_pause.wait()
                await self._refit_pause.wait()

                if not self.running:
                    for t in tasks:
                        t.cancel()
                    return

                task = asyncio.create_task(bounded_generate(prompt, gen_idx))
                tasks.append(task)

        # Process and push results immediately to buffer
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
            except asyncio.CancelledError:
                continue
            except Exception as e:
                logger.error(f"[Collector] Task failed: {e}")
                continue

            if result is None:
                continue

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
                        # Drop the trajectory if stopping and buffer is full
                        logger.warning("[Collector] Dropping trajectory")
                        break
                    await asyncio.sleep(min(backoff, 0.5))
                    backoff *= 1.5

        # Release target reservation
        async with self._gen_lock:
            self._generating_targets.discard(target_weight_version)

    async def start_collection(self, data: List[Dict]):
        """Collect trajectories from the data."""
        await self._init_client()

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
