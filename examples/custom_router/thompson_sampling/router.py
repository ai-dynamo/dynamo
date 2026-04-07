# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ..base import BaseCustomRouter, RequestLike, WorkerSelection
from .bandit import BetaBandit


WorkerKey = tuple[int, int]


@dataclass(slots=True)
class AgentState:
    """Sticky routing state keyed by agent or prefix identity."""

    last_worker: WorkerKey
    reuse_remaining: int


@dataclass(slots=True)
class CandidateScore:
    """Intermediate scoring record for one ``(worker_id, dp_rank)`` candidate."""

    worker_id: int
    dp_rank: int
    prefill_tokens: int
    decode_blocks: int
    overlap_ratio: float
    ts_sample: float
    load_score: float
    sticky_bonus: float
    switch_penalty: float
    combined: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "dp_rank": self.dp_rank,
            "potential_prefill_tokens": self.prefill_tokens,
            "potential_decode_blocks": self.decode_blocks,
            "overlap_ratio": self.overlap_ratio,
            "ts_sample": self.ts_sample,
            "load_score": self.load_score,
            "sticky_bonus": self.sticky_bonus,
            "switch_penalty": self.switch_penalty,
            "combined": self.combined,
        }


class ThompsonSamplingRouter(BaseCustomRouter):
    """Python Thompson Sampling strategy built on top of ``KvRouter``.

    Algorithm:
    1. Query ``KvRouter.get_potential_loads()`` for every worker.
    2. Approximate overlap for each worker as
       ``1 - potential_prefill_tokens / prompt_tokens``.
    3. Query ``KvRouter.best_worker()`` once and replace that candidate's overlap
       with the exact overlap derived from ``overlap_blocks``. This corrects the
       best KV candidate while the rest use the load-derived proxy.
    4. Sample a Beta bandit for each worker:
       ``sample ~ Beta(alpha, beta)``.
    5. Compute a load score:
       ``load_score = 1 / (1 + prefill_term + decode_term + reuse_term)``.
    6. Combine the terms:
       ``combined = sample * load_score + overlap_weight * overlap_ratio``
       plus a sticky bonus for the previously used worker, or a switch penalty
       for other workers when the agent still has reuse budget remaining.
    7. Pick the max-scoring worker and apply an immediate synthetic reward:
       ``reward = 0.65 * overlap_ratio + 0.35 * load_score``.
       The bandit's posterior is updated as ``alpha += reward`` and
       ``beta += 1 - reward``.

    This is intentionally a pure binding-level example. It does not reach into
    lower-level KV internals like ``RadixTree`` or ``ZmqKvEventListener``.
    """

    def __init__(
        self,
        kv_router,
        *,
        block_size: int,
        initial_reuse_budget: int = 3,
        overlap_weight: float = 1.0,
        sticky_bonus: float = 0.35,
        switch_penalty: float = 0.15,
        prefill_penalty_weight: float = 1.0,
        decode_penalty_weight: float = 0.35,
        reuse_penalty_weight: float = 0.25,
        bandit_seed: int = 7,
    ):
        super().__init__(kv_router)
        self.block_size = int(block_size)
        self.initial_reuse_budget = max(0, int(initial_reuse_budget))
        self.overlap_weight = float(overlap_weight)
        self.sticky_bonus = float(sticky_bonus)
        self.switch_penalty = float(switch_penalty)
        self.prefill_penalty_weight = float(prefill_penalty_weight)
        self.decode_penalty_weight = float(decode_penalty_weight)
        self.reuse_penalty_weight = float(reuse_penalty_weight)
        self.rng = random.Random(bandit_seed)

        self.bandits: dict[WorkerKey, BetaBandit] = {}
        self.agent_states: dict[str, AgentState] = {}

    async def select_worker(
        self,
        request: RequestLike,
        request_id: str,
    ) -> WorkerSelection:
        token_ids = self.token_ids_from_request(request)
        prompt_tokens = max(len(token_ids), 1)
        agent_id = self.extract_agent_id(request)

        loads = await self.get_worker_loads(request)
        if not loads:
            raise RuntimeError("KvRouter returned no candidate workers")

        best_worker_id, best_dp_rank, best_overlap_blocks = await self.get_best_overlap(request)
        best_exact_key = (int(best_worker_id), int(best_dp_rank))
        best_exact_overlap = min(
            1.0,
            (float(best_overlap_blocks) * float(self.block_size)) / float(prompt_tokens),
        )

        max_decode_blocks = max(
            int(load.get("potential_decode_blocks", 0)) for load in loads
        )
        max_decode_blocks = max(max_decode_blocks, 1)

        reuse_pressure = self._normalized_reuse_pressure()
        agent_state = self.agent_states.get(agent_id) if agent_id else None
        excluded_worker = self._excluded_worker(agent_state, len(loads))

        candidates: list[CandidateScore] = []
        for load in loads:
            worker_id = int(load["worker_id"])
            dp_rank = int(load["dp_rank"])
            key = (worker_id, dp_rank)
            if excluded_worker is not None and key == excluded_worker:
                continue

            prefill_tokens = int(load.get("potential_prefill_tokens", 0))
            decode_blocks = int(load.get("potential_decode_blocks", 0))

            overlap_ratio = max(
                0.0,
                1.0 - (min(prefill_tokens, prompt_tokens) / float(prompt_tokens)),
            )
            if key == best_exact_key:
                overlap_ratio = max(overlap_ratio, best_exact_overlap)

            load_score = self._compute_load_score(
                prompt_tokens=prompt_tokens,
                prefill_tokens=prefill_tokens,
                decode_blocks=decode_blocks,
                max_decode_blocks=max_decode_blocks,
                reuse_pressure=reuse_pressure.get(key, 0.0),
            )
            ts_sample = self._sample_bandit(key)
            sticky_bonus = self._sticky_bonus(agent_state, key)
            switch_penalty = self._switch_penalty(agent_state, key)
            combined = (
                (ts_sample * load_score)
                + (self.overlap_weight * overlap_ratio)
                + sticky_bonus
                - switch_penalty
            )

            candidates.append(
                CandidateScore(
                    worker_id=worker_id,
                    dp_rank=dp_rank,
                    prefill_tokens=prefill_tokens,
                    decode_blocks=decode_blocks,
                    overlap_ratio=overlap_ratio,
                    ts_sample=ts_sample,
                    load_score=load_score,
                    sticky_bonus=sticky_bonus,
                    switch_penalty=switch_penalty,
                    combined=combined,
                )
            )

        if not candidates:
            # Reuse budget can exclude the previous worker. If that eliminates every
            # candidate, fall back to the router's exact best worker.
            candidates.append(
                CandidateScore(
                    worker_id=best_exact_key[0],
                    dp_rank=best_exact_key[1],
                    prefill_tokens=prompt_tokens,
                    decode_blocks=0,
                    overlap_ratio=best_exact_overlap,
                    ts_sample=self._sample_bandit(best_exact_key),
                    load_score=1.0,
                    sticky_bonus=0.0,
                    switch_penalty=0.0,
                    combined=best_exact_overlap,
                )
            )

        chosen = max(candidates, key=lambda candidate: (candidate.combined, -candidate.worker_id, -candidate.dp_rank))
        chosen_key = (chosen.worker_id, chosen.dp_rank)

        reward = self._compute_reward(chosen)
        self.bandits.setdefault(chosen_key, BetaBandit()).observe(reward)
        self._update_agent_state(agent_id, chosen_key)

        return WorkerSelection(
            worker_id=chosen.worker_id,
            dp_rank=chosen.dp_rank,
            metadata={
                "algorithm": "thompson_sampling",
                "request_id": request_id,
                "agent_id": agent_id,
                "reward": reward,
                "best_kv_worker": {
                    "worker_id": best_exact_key[0],
                    "dp_rank": best_exact_key[1],
                    "overlap_ratio": best_exact_overlap,
                },
                "candidate_scores": [candidate.as_dict() for candidate in candidates],
                "selected": chosen.as_dict(),
            },
        )

    def _sample_bandit(self, key: WorkerKey) -> float:
        return self.bandits.setdefault(key, BetaBandit()).sample(self.rng)

    def _compute_load_score(
        self,
        *,
        prompt_tokens: int,
        prefill_tokens: int,
        decode_blocks: int,
        max_decode_blocks: int,
        reuse_pressure: float,
    ) -> float:
        prefill_ratio = min(prefill_tokens, prompt_tokens) / float(prompt_tokens)
        decode_ratio = decode_blocks / float(max_decode_blocks)
        penalty = (
            (self.prefill_penalty_weight * prefill_ratio)
            + (self.decode_penalty_weight * decode_ratio)
            + (self.reuse_penalty_weight * reuse_pressure)
        )
        return 1.0 / (1.0 + penalty)

    def _normalized_reuse_pressure(self) -> dict[WorkerKey, float]:
        raw_pressure: dict[WorkerKey, int] = {}
        for state in self.agent_states.values():
            raw_pressure[state.last_worker] = raw_pressure.get(state.last_worker, 0) + max(
                state.reuse_remaining,
                0,
            )
        max_pressure = max(raw_pressure.values(), default=0)
        if max_pressure <= 0:
            return {}
        return {
            worker_key: pressure / float(max_pressure)
            for worker_key, pressure in raw_pressure.items()
        }

    def _excluded_worker(
        self,
        agent_state: AgentState | None,
        candidate_count: int,
    ) -> WorkerKey | None:
        if agent_state is None or candidate_count <= 1:
            return None
        if agent_state.reuse_remaining > 0:
            return None
        return agent_state.last_worker

    def _sticky_bonus(self, agent_state: AgentState | None, key: WorkerKey) -> float:
        if agent_state is None or key != agent_state.last_worker:
            return 0.0
        if self.initial_reuse_budget <= 0 or agent_state.reuse_remaining <= 0:
            return 0.0
        normalized_budget = agent_state.reuse_remaining / float(self.initial_reuse_budget)
        return self.sticky_bonus * normalized_budget

    def _switch_penalty(self, agent_state: AgentState | None, key: WorkerKey) -> float:
        if agent_state is None or key == agent_state.last_worker:
            return 0.0
        if agent_state.reuse_remaining <= 0:
            return 0.0
        return self.switch_penalty

    @staticmethod
    def _compute_reward(candidate: CandidateScore) -> float:
        reward = (0.65 * candidate.overlap_ratio) + (0.35 * candidate.load_score)
        return max(0.0, min(1.0, reward))

    def _update_agent_state(self, agent_id: str | None, selected_key: WorkerKey) -> None:
        if agent_id is None:
            return

        current = self.agent_states.get(agent_id)
        if current is None or current.last_worker != selected_key:
            self.agent_states[agent_id] = AgentState(
                last_worker=selected_key,
                reuse_remaining=self.initial_reuse_budget,
            )
            return

        current.reuse_remaining = max(0, current.reuse_remaining - 1)
