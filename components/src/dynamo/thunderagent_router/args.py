# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ThunderAgent router CLI parsing and config assembly."""

from __future__ import annotations

import argparse
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.utils import add_argument
from dynamo.router.args import (
    DynamoRouterArgGroup,
    DynamoRouterConfig,
    build_aic_perf_config,
    build_kv_router_config,
)
from dynamo.thunderagent_router.router import ThunderAgentConfig


class ThunderAgentRouterConfig(DynamoRouterConfig):
    """Extends the standalone-router config with ThunderAgent scheduler params."""

    pause_threshold: float
    pause_target: float
    soft_demote_threshold: float
    soft_demote_priority_jump: float
    resume_priority_boost: float
    resume_timeout_seconds: float
    resume_hysteresis: float
    acting_token_weight: float
    acting_decay_tau_seconds: float
    scheduler_interval_seconds: float
    scheduling_disabled: bool
    model_name: Optional[str] = None
    model_path: Optional[str] = None

    def to_thunderagent_config(self) -> ThunderAgentConfig:
        return ThunderAgentConfig(
            pause_threshold=self.pause_threshold,
            pause_target=self.pause_target,
            soft_demote_threshold=self.soft_demote_threshold,
            soft_demote_priority_jump=self.soft_demote_priority_jump,
            resume_priority_boost=self.resume_priority_boost,
            resume_timeout_seconds=self.resume_timeout_seconds,
            resume_hysteresis=self.resume_hysteresis,
            acting_token_weight=self.acting_token_weight,
            acting_decay_tau_seconds=self.acting_decay_tau_seconds,
            scheduler_interval_seconds=self.scheduler_interval_seconds,
            scheduling_disabled=self.scheduling_disabled,
        )

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        if not 0.0 <= self.pause_threshold <= 1.0:
            raise ValueError("--pause-threshold must be in [0, 1]")
        if not 0.0 <= self.soft_demote_threshold <= self.pause_threshold:
            raise ValueError(
                "--soft-demote-threshold must be in [0, --pause-threshold]"
            )
        if self.scheduler_interval_seconds <= 0:
            raise ValueError("--scheduler-interval-seconds must be > 0")
        if self.resume_timeout_seconds <= 0:
            raise ValueError("--resume-timeout-seconds must be > 0")


class ThunderAgentArgGroup(ArgGroup):
    name = "thunderagent-router"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # Inherit standard router options (--endpoint, --router-block-size, KV
        # router knobs, AicPerf options).
        DynamoRouterArgGroup().add_arguments(parser)

        g = parser.add_argument_group("ThunderAgent Scheduler Options")

        add_argument(
            g,
            flag_name="--pause-threshold",
            env_var="DYN_THUNDERAGENT_PAUSE_THRESHOLD",
            default=0.95,
            help="Hard-pause when worker utilization >= this fraction of "
            "max_num_batched_tokens (default: 0.95)",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--soft-demote-threshold",
            env_var="DYN_THUNDERAGENT_SOFT_DEMOTE_THRESHOLD",
            default=0.80,
            help="Soft-demote priority when worker utilization >= this and "
            "below --pause-threshold (default: 0.80)",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--soft-demote-priority-jump",
            env_var="DYN_THUNDERAGENT_SOFT_DEMOTE_PRIORITY_JUMP",
            default=-2.0,
            help="priority_jump (seconds) applied to soft-demoted programs. "
            "Negative pushes the request later in the queue (default: -2.0)",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--resume-priority-boost",
            env_var="DYN_THUNDERAGENT_RESUME_PRIORITY_BOOST",
            default=1.0,
            help="priority_jump (seconds) added to a request that just resumed "
            "from hard pause (default: 1.0)",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--resume-timeout-seconds",
            env_var="DYN_THUNDERAGENT_RESUME_TIMEOUT_SECONDS",
            default=1800.0,
            help="Maximum wait on a paused program before a forced resume "
            "(default: 1800)",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--resume-hysteresis",
            env_var="DYN_THUNDERAGENT_RESUME_HYSTERESIS",
            default=0.10,
            help="Working-set fraction below pause_threshold required before "
            "resuming any paused program. 0.10 = resume only when "
            "working_set <= (pause_threshold - 0.10) * pool. Larger "
            "values stabilise high-cadence loads at the cost of holding "
            "programs paused longer (default: 0.10).",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--pause-target",
            env_var="DYN_THUNDERAGENT_PAUSE_TARGET",
            default=0.80,
            help="Setpoint that pause cycles drive util DOWN to. Trigger "
            "fires at --pause-threshold (e.g. 0.95) but we keep pausing "
            "until projected util reaches --pause-target (e.g. 0.80). "
            "Fixes the 'stall at 0.948' failure where pauses stopped "
            "firing the moment util slipped under threshold (default: 0.80).",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--acting-token-weight",
            env_var="DYN_THUNDERAGENT_ACTING_TOKEN_WEIGHT",
            default=1.0,
            help="Flat multiplier on token_total for ACTING programs in the "
            "PAUSE-side working set. Mirrors upstream TA's tool_coefficient. "
            "Default 1.0 = conservative on pause; lower admits more programs "
            "before pause_threshold trips.",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--acting-decay-tau-seconds",
            env_var="DYN_THUNDERAGENT_ACTING_DECAY_TAU_SECONDS",
            default=1.0,
            help="Time constant for the exponential decay applied to ACTING "
            "programs in the RESUME-side working set. Weight is "
            "2^(-(now - acting_since) / tau). With tau=1.0s, programs idle "
            "for 10s contribute ~0.001x their tokens; idle 60s contribute "
            "~0. Mirrors upstream TA's remaining_capacity_with_decay -- the "
            "decay replaces a hard TTL/GC, so a returning zombie program "
            "keeps its token_total history (default: 1.0).",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--scheduler-interval-seconds",
            env_var="DYN_THUNDERAGENT_SCHEDULER_INTERVAL_SECONDS",
            default=5.0,
            help="Period of the background pause/resume scheduler tick (default: 5.0)",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--scheduling-disabled",
            env_var="DYN_THUNDERAGENT_SCHEDULING_DISABLED",
            default=False,
            help="When set, the router records lifecycle state but does not "
            "pause / resume / soft-demote. Used as the 'TR off' arm to "
            "isolate scheduling value vs program-aware passthrough.",
            arg_type=bool,
        )
        add_argument(
            g,
            flag_name="--model-name",
            env_var="DYN_THUNDERAGENT_MODEL_NAME",
            default=None,
            help="Model name to register at the Dynamo frontend. When set the "
            "router calls register_model so the frontend dispatches "
            "requests for this model to the router (which then forwards to "
            "the worker pointed at by --endpoint). Leave unset to behave as "
            "a pure utility endpoint with no frontend registration.",
            arg_type=str,
        )
        add_argument(
            g,
            flag_name="--model-path",
            env_var="DYN_THUNDERAGENT_MODEL_PATH",
            default=None,
            help="Path or HF repo ID to load tokenizer + model card from for "
            "register_model. Defaults to --model-name; set this when the "
            "client-facing name differs from the on-disk location (e.g. "
            "served name 'zai-org/GLM-4.6-FP8' but local cache "
            "/home/nvidia/hf_cache/models/glm-4.6-fp8).",
            arg_type=str,
        )


def parse_args(argv: Optional[list[str]] = None) -> ThunderAgentRouterConfig:
    parser = argparse.ArgumentParser(
        description="Dynamo ThunderAgent Router: program-level scheduler with "
        "tool-boundary pause/resume on top of native KV-aware routing",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ThunderAgentArgGroup().add_arguments(parser)
    args = parser.parse_args(argv)
    config = ThunderAgentRouterConfig.from_cli_args(args)
    config.validate()
    return config


__all__ = [
    "ThunderAgentArgGroup",
    "ThunderAgentRouterConfig",
    "build_aic_perf_config",
    "build_kv_router_config",
    "parse_args",
]
