# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Squeeze-Evolve component CLI parsing and config assembly.

Dynamo-native: CLI flags + env vars (no Pydantic, no YAML). Model tiers come from
a single ``--tiers`` JSON array; the shared per-tier ``KvRouter`` knobs are the
standard ``--router-*`` flags (reused via ``KvRouterArgGroup``).
"""

from __future__ import annotations

import argparse
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.aic_perf_args import (
    AicPerfArgGroup,
    AicPerfConfigBase,
)
from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.common.configuration.utils import add_argument
from dynamo.squeeze_evolve.orchestrator import SqueezeEvolveConfig, Tier, parse_tiers

_TIERS_EXAMPLE = (
    '[{"endpoint":"dynamo.cheap.generate","model":"Qwen/Qwen3-30B-A3B-Instruct-2507",'
    '"temperature":0.7,"top_p":0.8,"max_tokens":8192,"block_size":64},'
    '{"endpoint":"dynamo.expensive.generate","model":"Qwen/Qwen3-235B-A22B-Instruct-2507"}]'
)


def _nullable_int(raw: str) -> Optional[int]:
    if raw is None or str(raw).lower() in ("", "none", "null"):
        return None
    return int(raw)


class SqueezeEvolveRunConfig(KvRouterConfigBase, AicPerfConfigBase):
    """Squeeze-Evolve config: the SE knobs + the shared per-tier KvRouter knobs.

    Inherits the KvRouter/AicPerf bases directly (NOT ``DynamoRouterConfig``) so we
    reuse ``kv_router_kwargs()`` / ``aic_perf_kwargs()`` + the load-aware preset
    without ``DynamoRouterConfig``'s single-``--endpoint`` requirement — Squeeze-Evolve
    has N tier endpoints (in ``--tiers``).
    """

    namespace: str = "dynamo"
    tiers: Optional[list[Tier]] = None
    k: int = 4
    population: int = 16
    groups: int = 0  # 0 -> follow --population
    loops: int = 5
    confidence_percentiles: list[float] = [50.0]
    task: str = "math"
    seed: Optional[int] = None
    tier_concurrency: int = 32
    default_block_size: int = 64
    model_name: Optional[str] = None
    model_path: Optional[str] = None

    def to_algo_config(self) -> SqueezeEvolveConfig:
        return SqueezeEvolveConfig(
            k=self.k,
            population=self.population,
            groups=self.groups or self.population,
            loops=self.loops,
            confidence_percentiles=[float(p) for p in self.confidence_percentiles],
            task=self.task,
            seed=self.seed,
            tier_concurrency=self.tier_concurrency,
        )

    def validate(self) -> None:  # type: ignore[override]
        self.apply_load_aware_preset()  # shared KvRouter preset (KvRouterConfigBase)
        if not self.tiers:
            raise ValueError("--tiers is required (JSON array, cheapest first)")
        self.confidence_percentiles = [float(p) for p in self.confidence_percentiles]
        n = len(self.tiers)
        if n > 1 and len(self.confidence_percentiles) != n - 1:
            raise ValueError(
                f"{n} tiers require {n - 1} --confidence-percentiles, "
                f"got {len(self.confidence_percentiles)}"
            )
        if not self.model_name:
            raise ValueError("--model-name is required")
        if not 1 <= self.k <= self.population:
            raise ValueError("--k must be in [1, --population]")
        if self.population < 1 or self.loops < 1:
            raise ValueError("--population and --loops must be >= 1")
        if self.groups < 0:
            raise ValueError("--groups must be >= 0 (0 follows --population)")
        if self.tier_concurrency < 1:
            raise ValueError("--tier-concurrency must be >= 1")
        # After Loop 1 the population shrinks to --groups, so later loops select
        # --k from it; guard --groups < --k when >1 recombination loop runs.
        effective_groups = self.groups or self.population
        if self.loops > 2 and effective_groups < self.k:
            raise ValueError(
                "--groups must be >= --k when --loops > 2 "
                "(population shrinks to --groups after Loop 1)"
            )


class SqueezeEvolveArgGroup(ArgGroup):
    name = "squeeze-evolve"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("Squeeze-Evolve Options")
        add_argument(
            g,
            flag_name="--tiers",
            env_var="DYN_SQUEEZE_TIERS",
            default=None,
            arg_type=parse_tiers,
            help="JSON array of model tiers, cheapest first. Each entry: "
            "{endpoint (3-part namespace.component.endpoint), model (HF id), "
            "temperature?, top_p?, max_tokens?, block_size?, tokenizer?, "
            "trust_remote_code?}. "
            f"Example: {_TIERS_EXAMPLE}",
        )
        add_argument(
            g,
            flag_name="--model-name",
            env_var="DYN_SQUEEZE_MODEL_NAME",
            default=None,
            arg_type=str,
            help="Frontend-visible model name clients call (e.g. squeeze-evolve/aime25).",
        )
        add_argument(
            g,
            flag_name="--model-path",
            env_var="DYN_SQUEEZE_MODEL_PATH",
            default=None,
            arg_type=str,
            help="HF repo id / path for the model card used by register_model. "
            "Defaults to the most expensive tier's model.",
        )
        add_argument(
            g,
            flag_name="--namespace",
            env_var="DYN_NAMESPACE",
            default="dynamo",
            arg_type=str,
            help="Dynamo namespace to serve under (default: dynamo).",
        )
        add_argument(
            g,
            flag_name="--k",
            env_var="DYN_SQUEEZE_K",
            default=4,
            arg_type=int,
            help="Group size for selection (default: 4).",
        )
        add_argument(
            g,
            flag_name="--population",
            env_var="DYN_SQUEEZE_POPULATION",
            default=16,
            arg_type=int,
            help="Candidates per problem, generated in Loop 0 (default: 16).",
        )
        add_argument(
            g,
            flag_name="--groups",
            env_var="DYN_SQUEEZE_GROUPS",
            default=0,
            arg_type=int,
            help="Groups per loop (default: 0 = follow --population).",
        )
        add_argument(
            g,
            flag_name="--loops",
            env_var="DYN_SQUEEZE_LOOPS",
            default=5,
            arg_type=int,
            help="Evolutionary iterations including Loop 0 (default: 5).",
        )
        add_argument(
            g,
            flag_name="--confidence-percentiles",
            env_var="DYN_SQUEEZE_CONFIDENCE_PERCENTILES",
            default=[50.0],
            arg_type=float,
            nargs="+",
            help="N-1 percentile thresholds for N tiers (default: 50).",
        )
        add_argument(
            g,
            flag_name="--task",
            env_var="DYN_SQUEEZE_TASK",
            default="math",
            arg_type=str,
            choices=["math", "gpqa_diamond", "generic"],
            help="Task type for answer extraction (default: math).",
        )
        add_argument(
            g,
            flag_name="--seed",
            env_var="DYN_SQUEEZE_SEED",
            default=None,
            arg_type=_nullable_int,
            help="RNG seed. Leave unset for serving (per-request independence).",
        )
        add_argument(
            g,
            flag_name="--tier-concurrency",
            env_var="DYN_SQUEEZE_TIER_CONCURRENCY",
            default=32,
            arg_type=int,
            help="Max concurrent generations per tier (default: 32).",
        )
        add_argument(
            g,
            flag_name="--default-block-size",
            env_var="DYN_SQUEEZE_DEFAULT_BLOCK_SIZE",
            default=64,
            arg_type=int,
            help="KvRouter block size for tiers that omit block_size (default: 64).",
        )
        # Shared per-tier KvRouter knobs (--router-*) + AIC perf model flags.
        KvRouterArgGroup().add_arguments(parser)
        AicPerfArgGroup().add_arguments(parser)


def parse_args(argv: Optional[list[str]] = None) -> SqueezeEvolveRunConfig:
    parser = argparse.ArgumentParser(
        description="Dynamo Squeeze-Evolve: verifier-free evolutionary test-time "
        "scaling that routes candidate groups across model tiers, served as a "
        "chat model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    SqueezeEvolveArgGroup().add_arguments(parser)
    args = parser.parse_args(argv)
    config = SqueezeEvolveRunConfig.from_cli_args(args)
    config.validate()
    return config


__all__ = [
    "SqueezeEvolveArgGroup",
    "SqueezeEvolveRunConfig",
    "parse_args",
]
