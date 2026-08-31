# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrator loop tests with fake tier transports (no bindings/tokenizer)."""

from __future__ import annotations

import asyncio
import itertools

import pytest

from dynamo.squeeze_evolve.orchestrator import (
    SqueezeEvolveConfig,
    SqueezeEvolveOrchestrator,
    parse_tiers,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class FakeTier:
    """Records calls; returns a fixed boxed answer."""

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.calls = 0

    async def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        return f"reasoning... \\boxed{{{self._answer}}}"


def _run(coro):
    return asyncio.run(coro)


def _msgs(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def test_loop0_uses_most_expensive_tier_only():
    cheap, exp = FakeTier("1"), FakeTier("2")
    cfg = SqueezeEvolveConfig(
        k=2, population=4, groups=4, loops=1, confidence_percentiles=[50.0], seed=0
    )
    orch = SqueezeEvolveOrchestrator(cfg=cfg, transports=[cheap, exp])
    answer = _run(orch.run(_msgs("solve x")))
    # loops=1 -> only Loop 0: population candidates from the expensive tier.
    assert exp.calls == 4
    assert cheap.calls == 0
    assert answer == "reasoning... \\boxed{2}"  # candidates[0]


def test_uniform_diversity_routes_all_to_expensive():
    # Loop 0 fills the population from the expensive tier (all answer "2"), so in
    # Loop 1 every group has diversity 1 -> all equal -> all route to the
    # expensive tier (f <= threshold).
    cheap, exp = FakeTier("1"), FakeTier("2")
    cfg = SqueezeEvolveConfig(
        k=2, population=4, groups=4, loops=2, confidence_percentiles=[50.0], seed=0
    )
    orch = SqueezeEvolveOrchestrator(cfg=cfg, transports=[cheap, exp])
    _run(orch.run(_msgs("solve x")))
    assert exp.calls == 8 and cheap.calls == 0  # 4 (loop0) + 4 (loop1)


def test_total_generations_equals_population_plus_groups_per_loop():
    answers = itertools.cycle(["1", "2", "3", "4"])

    class VariedTier:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, messages: list[dict[str, str]]) -> str:
            self.calls += 1
            return f"\\boxed{{{next(answers)}}}"

    t0, t1 = VariedTier(), VariedTier()
    cfg = SqueezeEvolveConfig(
        k=2, population=6, groups=6, loops=3, confidence_percentiles=[50.0], seed=1
    )
    orch = SqueezeEvolveOrchestrator(cfg=cfg, transports=[t0, t1])
    _run(orch.run(_msgs("q")))
    # Loop 0 = population (6); Loops 1 & 2 = groups each (6 + 6).
    assert t0.calls + t1.calls == 6 + 6 + 6


def test_single_tier_runs_with_no_thresholds():
    only = FakeTier("9")
    cfg = SqueezeEvolveConfig(
        k=2, population=4, groups=4, loops=2, confidence_percentiles=[], seed=0
    )
    orch = SqueezeEvolveOrchestrator(cfg=cfg, transports=[only])
    answer = _run(orch.run("q"))
    assert only.calls == 8  # 4 (loop0) + 4 (loop1), all to the one tier
    assert answer == "reasoning... \\boxed{9}"


# -- tier JSON parsing ------------------------------------------------------


def test_parse_tiers_basic():
    tiers = parse_tiers(
        '[{"endpoint":"dynamo.cheap.generate","model":"Q/c"},'
        '{"endpoint":"dynamo.exp.generate","model":"Q/e","temperature":0.5}]'
    )
    assert len(tiers) == 2
    assert tiers[0].endpoint == "dynamo.cheap.generate" and tiers[0].model == "Q/c"
    assert tiers[1].temperature == 0.5
    assert tiers[0].block_size == 0  # default


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        '[{"endpoint":"a.b.c"}]',  # missing model
        '[{"endpoint":"two.parts","model":"m"}]',  # endpoint not 3-part
        '[{"endpoint":"a..b","model":"m"}]',  # empty endpoint segment
        '[{"endpoint":"a.b.c","model":"m","bogus":1}]',  # unknown key
        '{"endpoint":"a.b.c","model":"m"}',  # not a list
    ],
)
def test_parse_tiers_rejects_bad_input(raw):
    with pytest.raises(ValueError):
        parse_tiers(raw)
