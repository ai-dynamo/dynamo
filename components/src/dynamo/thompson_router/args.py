# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thompson Router CLI parsing, config, and assembly."""

import argparse
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.utils import add_argument
from dynamo.llm import KvRouterConfig
from dynamo.router.args import (
    DynamoRouterArgGroup,
    DynamoRouterConfig,
    build_kv_router_config,
)


class ThompsonRouterConfig(DynamoRouterConfig):
    """Extends DynamoRouterConfig with Thompson Sampling parameters."""

    # Base scoring
    ts_weight: float
    thompson_temperature: float
    cold_start_threshold: float
    idle_boost: float
    queue_penalty_weight: float
    load_mod_floor: float
    beta_decay: float
    lints_lambda: float
    lints_v: float
    lints_forget_rate: float
    latency_ema_alpha: float

    # Feature toggles
    enable_softmax: bool
    enable_cold_start: bool
    enable_idle_boost: bool
    enable_load_mod_floor: bool
    enable_lints: bool
    enable_affinity: bool
    enable_switching_cost: bool
    enable_adaptive_temp: bool
    enable_adaptive_explore: bool
    enable_sticky_floor: bool

    # Feature-specific weights
    lints_weight: float
    affinity_base: float
    affinity_reuse_weight: float
    switch_base: float
    switch_reuse: float
    sticky_load_floor: float
    adaptive_temp_base: float

    # Management server
    mgmt_port: int

    def to_thompson_config(self) -> dict:
        """Convert to the dict format expected by KvThompsonRouter."""
        return {
            "kv_thompson": {
                "ts_weight": self.ts_weight,
                "temperature": self.thompson_temperature,
                "cold_start_threshold": self.cold_start_threshold,
                "idle_boost": self.idle_boost,
                "queue_penalty_weight": self.queue_penalty_weight,
                "load_mod_floor": self.load_mod_floor,
                "beta_decay": self.beta_decay,
                "lints_lambda": self.lints_lambda,
                "lints_v": self.lints_v,
                "lints_forget_rate": self.lints_forget_rate,
                "latency_ema_alpha": self.latency_ema_alpha,
                "enable_softmax": self.enable_softmax,
                "enable_cold_start": self.enable_cold_start,
                "enable_idle_boost": self.enable_idle_boost,
                "enable_load_mod_floor": self.enable_load_mod_floor,
                "enable_lints": self.enable_lints,
                "enable_affinity": self.enable_affinity,
                "enable_switching_cost": self.enable_switching_cost,
                "enable_adaptive_temp": self.enable_adaptive_temp,
                "enable_adaptive_explore": self.enable_adaptive_explore,
                "enable_sticky_floor": self.enable_sticky_floor,
                "lints_weight": self.lints_weight,
                "affinity_base": self.affinity_base,
                "affinity_reuse_weight": self.affinity_reuse_weight,
                "switch_base": self.switch_base,
                "switch_reuse": self.switch_reuse,
                "sticky_load_floor": self.sticky_load_floor,
                "adaptive_temp_base": self.adaptive_temp_base,
            }
        }


class ThompsonArgGroup(ArgGroup):
    """CLI argument group for Thompson Sampling router options."""

    name = "thompson-router"

    def add_arguments(self, parser) -> None:
        DynamoRouterArgGroup().add_arguments(parser)

        g = parser.add_argument_group("Thompson Sampling Options")

        add_argument(
            g,
            flag_name="--ts-weight",
            env_var="DYN_TS_WEIGHT",
            default=0.05,
            help="Weight for Beta-TS exploration term",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--thompson-temperature",
            env_var="DYN_THOMPSON_TEMPERATURE",
            default=1.70,
            help="Softmax temperature for worker selection",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--cold-start-threshold",
            env_var="DYN_COLD_START_THRESHOLD",
            default=0.37,
            help="Overlap threshold below which cold-start round-robin is used",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--idle-boost",
            env_var="DYN_IDLE_BOOST",
            default=0.135,
            help="Minimum effective overlap for idle workers",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--queue-penalty-weight",
            env_var="DYN_QUEUE_PENALTY_WEIGHT",
            default=2.5,
            help="Exponential queue penalty weight",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--load-mod-floor",
            env_var="DYN_LOAD_MOD_FLOOR",
            default=0.3,
            help="Minimum load modifier floor",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--beta-decay",
            env_var="DYN_BETA_DECAY",
            default=0.995,
            help="Exponential decay for Beta bandit parameters",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--lints-lambda",
            env_var="DYN_LINTS_LAMBDA",
            default=1.0,
            help="Ridge regularization for LinTS",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--lints-v",
            env_var="DYN_LINTS_V",
            default=0.25,
            help="Exploration variance for LinTS posterior sampling",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--lints-forget-rate",
            env_var="DYN_LINTS_FORGET_RATE",
            default=0.995,
            help="Exponential forgetting rate for LinTS",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--latency-ema-alpha",
            env_var="DYN_LATENCY_EMA_ALPHA",
            default=0.2,
            help="EMA smoothing factor for latency baselines",
            arg_type=float,
        )

        # Feature toggles
        fg = parser.add_argument_group("Thompson Feature Toggles")

        add_argument(
            fg,
            flag_name="--enable-softmax",
            env_var="DYN_ENABLE_SOFTMAX",
            default=False,
            help="Use softmax probabilistic selection instead of argmax",
            arg_type=bool,
        )
        add_argument(
            fg,
            flag_name="--enable-cold-start",
            env_var="DYN_ENABLE_COLD_START",
            default=False,
            help="Round-robin when max overlap is below threshold",
            arg_type=bool,
        )
        add_argument(
            fg,
            flag_name="--enable-idle-boost",
            env_var="DYN_ENABLE_IDLE_BOOST",
            default=False,
            help="Boost overlap for idle workers",
            arg_type=bool,
        )
        add_argument(
            fg,
            flag_name="--enable-load-mod-floor",
            env_var="DYN_ENABLE_LOAD_MOD_FLOOR",
            default=False,
            help="Apply minimum floor to load modifier",
            arg_type=bool,
        )
        add_argument(
            fg,
            flag_name="--enable-lints",
            env_var="DYN_ENABLE_LINTS",
            default=False,
            help="Enable LinTS contextual bandit (7-dim feature-aware)",
            arg_type=bool,
        )
        add_argument(
            fg,
            flag_name="--enable-affinity",
            env_var="DYN_ENABLE_AFFINITY",
            default=False,
            help="Enable prefix stickiness for multi-turn sessions",
            arg_type=bool,
        )
        add_argument(
            fg,
            flag_name="--enable-switching-cost",
            env_var="DYN_ENABLE_SWITCHING_COST",
            default=False,
            help="Penalty for migrating prefix to different worker",
            arg_type=bool,
        )
        add_argument(
            fg,
            flag_name="--enable-adaptive-temp",
            env_var="DYN_ENABLE_ADAPTIVE_TEMP",
            default=False,
            help="Temperature decays with session depth",
            arg_type=bool,
        )
        add_argument(
            fg,
            flag_name="--enable-adaptive-explore",
            env_var="DYN_ENABLE_ADAPTIVE_EXPLORE",
            default=False,
            help="Beta-TS weight decays with session depth",
            arg_type=bool,
        )
        add_argument(
            fg,
            flag_name="--enable-sticky-floor",
            env_var="DYN_ENABLE_STICKY_FLOOR",
            default=False,
            help="Protect sticky worker's load_mod minimum",
            arg_type=bool,
        )

        # Feature-specific weights
        wg = parser.add_argument_group("Thompson Feature Weights")

        add_argument(wg, flag_name="--lints-weight", env_var="DYN_LINTS_WEIGHT", default=-1.0, help="LinTS contribution weight (negative = tanh-wrapped)", arg_type=float)
        add_argument(wg, flag_name="--affinity-base", env_var="DYN_AFFINITY_BASE", default=0.15, help="Base affinity bonus", arg_type=float)
        add_argument(wg, flag_name="--affinity-reuse-weight", env_var="DYN_AFFINITY_REUSE_WEIGHT", default=0.02, help="Affinity reuse budget weight", arg_type=float)
        add_argument(wg, flag_name="--switch-base", env_var="DYN_SWITCH_BASE", default=0.04, help="Base switching cost penalty", arg_type=float)
        add_argument(wg, flag_name="--switch-reuse", env_var="DYN_SWITCH_REUSE", default=0.01, help="Switching cost reuse budget weight", arg_type=float)
        add_argument(wg, flag_name="--sticky-load-floor", env_var="DYN_STICKY_LOAD_FLOOR", default=0.01, help="Minimum load_mod for sticky worker", arg_type=float)
        add_argument(wg, flag_name="--adaptive-temp-base", env_var="DYN_ADAPTIVE_TEMP_BASE", default=1.0, help="Base temperature for adaptive temp", arg_type=float)

        # Management server
        mg = parser.add_argument_group("Management Server")
        add_argument(
            mg,
            flag_name="--mgmt-port",
            env_var="DYN_THOMPSON_MGMT_PORT",
            default=8084,
            help="Port for the Thompson router management HTTP server",
            arg_type=int,
        )


def parse_args(argv: Optional[list[str]] = None) -> ThompsonRouterConfig:
    """Parse command-line arguments for the Thompson router.

    Returns:
        ThompsonRouterConfig: Parsed and validated configuration.
    """
    parser = argparse.ArgumentParser(
        description="Dynamo Thompson Sampling Router: KV-aware routing with online learning",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = ThompsonArgGroup()
    group.add_arguments(parser)

    args = parser.parse_args(argv)
    config = ThompsonRouterConfig.from_cli_args(args)
    config.validate()
    return config
