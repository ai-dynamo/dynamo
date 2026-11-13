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

from dynamo.common.ais_perf_model import router_perf_config
from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.ais_perf_args import (
    AisPerfArgGroup,
    AisPerfConfigBase,
)
from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.common.configuration.utils import add_argument, join_names
from dynamo.squeeze_evolve.orchestrator import SqueezeEvolveConfig, Tier, parse_tiers

_TIERS_EXAMPLE = (
    '[{"endpoint":"dynamo.cheap.generate","model":"Qwen/Qwen3-30B-A3B-Instruct-2507",'
    '"temperature":0.7,"top_p":0.8,"max_tokens":8192,"block_size":64},'
    '{"endpoint":"dynamo.expensive.generate","model":"Qwen/Qwen3-235B-A22B-Instruct-2507"}]'
)
# Where squeeze_evolve reads an AIS perf model; errors and the warning name it.
_TIER_KEY = "ais_perf_model in each --tiers entry"


def _nullable_int(raw: str) -> Optional[int]:
    if raw is None or str(raw).lower() in ("", "none", "null"):
        return None
    return int(raw)


class SqueezeEvolveRunConfig(KvRouterConfigBase, AisPerfConfigBase):
    """Squeeze-Evolve config: the SE knobs + the shared per-tier KvRouter knobs.

    Inherits the KvRouter/AisPerf bases directly (NOT ``DynamoRouterConfig``) so we
    reuse ``kv_router_kwargs()`` / ``ais_perf_kwargs()`` + the load-aware preset
    without ``DynamoRouterConfig``'s single-``--endpoint`` requirement — Squeeze-Evolve
    has N tier endpoints (in ``--tiers``).

    A tier's AIS perf model is its ``ais_perf_model`` key and nothing else.
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

    def new_ais_perf_input(self) -> tuple[None, None]:
        """Return no value: ``--ais-perf-model`` and its variable are not read.

        ``validate()`` rejects the typed flag. ``DYN_AIS_PERF_MODEL`` is
        ignored: it may be set for other processes in the same environment.
        """
        return None, None

    def _tiers_with_ais_perf_model(self) -> list[int]:
        """Return the index of each tier that writes ``ais_perf_model``."""
        return [
            index
            for index, tier in enumerate(self.tiers or [])
            if tier.ais_perf_model is not None
        ]

    def ais_perf_kwargs_by_tier(self) -> list[Optional[dict]]:
        """Return the ``AisPerfConfig`` arguments of each tier, in ``--tiers`` order.

        Every item is ``None`` when the load model is not ``ais``. A tier gets
        the config of its own ``ais_perf_model``; old AIS inputs give one
        config that every tier shares. An error in the value of tier ``i``
        starts with ``--tiers[i]:``.
        """
        tiers = self.tiers or []
        if self.router_prefill_load_model != "ais":
            return [None] * len(tiers)
        if self.old_ais_perf_inputs():
            # Read with no tiers too: the errors of old inputs come first.
            return [self.ais_perf_kwargs()] * len(tiers)
        # Read before the tiers: its error is not about one tier's value.
        worker_type = self.router_worker_type()
        by_tier: list[Optional[dict]] = []
        for index, tier in enumerate(tiers):
            try:
                config = router_perf_config(tier.ais_perf_model, worker_type)
            except ValueError as error:
                raise ValueError(f"--tiers[{index}]: {error}") from error
            self._check_nextn_accept_rates(config)
            by_tier.append({"config": config})
        return by_tier

    def validate(self) -> None:  # type: ignore[override]
        self.apply_router_config()
        ais = self.router_prefill_load_model == "ais"
        if ais:
            self.check_renamed_env_vars()
        if self.ais_perf_model is not None:
            raise ValueError(
                f"--ais-perf-model is not read by squeeze_evolve; write {_TIER_KEY}"
            )
        tier_count = len(self.tiers or [])
        with_key = self._tiers_with_ais_perf_model()
        old = self.old_ais_perf_inputs()
        if with_key and not ais:
            raise ValueError(
                f"ais_perf_model in --tiers[{with_key[0]}] requires "
                "--router-prefill-load-model=ais"
            )
        if with_key and old:
            raise ValueError(
                f"ais_perf_model in --tiers[{with_key[0]}] cannot be combined "
                f"with {join_names(old)}"
            )
        if ais and not old and len(with_key) < tier_count:
            missing = [
                f"--tiers[{index}]"
                for index in range(tier_count)
                if index not in with_key
            ]
            verb = "has" if len(missing) == 1 else "have"
            raise ValueError(
                f"{join_names(missing)} {verb} no ais_perf_model; with "
                "--router-prefill-load-model=ais every tier needs one"
            )
        self.validate_ais_perf_inputs(replacement=_TIER_KEY)
        if ais:
            self.ais_perf_kwargs_by_tier()
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
            "trust_remote_code?, ais_perf_model? (AIS perf model of this tier "
            "for --router-prefill-load-model=ais: a config, a list of configs, "
            "a file path or inline JSON; --ais-perf-model and DYN_AIS_PERF_MODEL "
            "are not read here)}. "
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
        # Shared per-tier KvRouter knobs (--router-*) + AIS perf model flags.
        KvRouterArgGroup().add_arguments(parser)
        AisPerfArgGroup().add_arguments(parser)


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
