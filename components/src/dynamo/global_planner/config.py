# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative configuration for the GlobalPlanner.

:class:`GlobalPlannerConfig` is the single validated description of how one
GlobalPlanner process behaves. It mirrors
:class:`~dynamo.planner.config.planner_config.PlannerConfig`: the same
``from_config_arg`` contract (inline JSON string *or* a path to a JSON/YAML
file), so a DGD can hand the component one ``--config`` blob exactly the way it
already does for the local planner.

This is the *only* way to configure the component. Every setting has exactly one
place it can be set, so there is no flag-versus-file precedence to reason about
and no way for two sources to disagree.

Validation happens here rather than being discovered mid-arbitration. An
unsatisfiable band (``min > max``) or a non-positive intent TTL previously
started fine and then mis-arbitrated silently at request time; both now fail at
startup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class GlobalPlannerConfig(BaseModel):
    """Validated GlobalPlanner configuration.

    Field defaults are the authoritative ones -- the argparse layer defaults
    every flag to ``None`` so that "not passed" stays distinguishable from
    "passed the default value", and only explicitly-supplied flags override
    ``--config``.
    """

    model_config = ConfigDict(extra="forbid")

    managed_namespaces: Optional[list[str]] = Field(
        default=None,
        description=(
            "Dynamo namespaces authorized to send scale requests. None accepts "
            "every caller (implicit mode) and counts every DGD in the "
            "GlobalPlanner's Kubernetes namespace toward the budget."
        ),
    )

    environment: Literal["kubernetes"] = Field(
        default="kubernetes",
        description="Execution environment. Only 'kubernetes' is supported today.",
    )

    no_operation: bool = Field(
        default=False,
        description=(
            "Log incoming scale requests and return success without applying "
            "any scaling. Useful for exercising the end-to-end path without "
            "touching Kubernetes."
        ),
    )

    max_total_gpus: int = Field(
        default=-1,
        ge=-1,
        description=(
            "Ceiling on total GPUs across all managed pools. A hard bound -- "
            "never relaxed by pairing tolerance. 0 forbids all GPU scaling; "
            "-1 disables the ceiling."
        ),
    )

    min_total_gpus: int = Field(
        default=-1,
        ge=-1,
        description=(
            "Floor on total GPUs across all managed pools. Scale-downs that "
            "breach it are denied unless they can be paired with a pending "
            "opposite-direction intent. -1 disables the floor."
        ),
    )

    intent_cache_ttl_seconds: float = Field(
        default=360.0,
        gt=0,
        description=(
            "How long a pool's cached scale intent stays eligible as a pair "
            "partner. Should be at least 2x the slowest local planner tick so "
            "opposite-direction intents can overlap; throughput scaling ticks "
            "every 180s by default, so 360 covers two ticks."
        ),
    )

    @model_validator(mode="after")
    def _validate_budget_band(self) -> "GlobalPlannerConfig":
        """Reject a budget band no allocation can satisfy.

        Both bounds active with ``min > max`` means every request is denied on
        one edge or the other. Previously this started cleanly and surfaced only
        as a stream of rejections at request time.
        """
        if (
            self.min_total_gpus >= 0
            and self.max_total_gpus >= 0
            and self.min_total_gpus > self.max_total_gpus
        ):
            raise ValueError(
                f"min_total_gpus ({self.min_total_gpus}) exceeds max_total_gpus "
                f"({self.max_total_gpus}): no total GPU count satisfies this band"
            )
        return self

    # ------------------------------------------------------------------ #
    # Loading                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config_arg(cls, config_arg: str) -> "GlobalPlannerConfig":
        """Build a config from a ``--config`` argument.

        Auto-detects whether the argument is a path to a JSON/YAML file or an
        inline JSON string, then loads and validates it. Mirrors
        ``PlannerConfig.from_config_arg`` so both planner components accept the
        same shape of argument.
        """
        path = Path(config_arg)
        try:
            is_file = path.is_file()
        except OSError:
            # Path component too long -- an inline JSON string, not a path.
            is_file = False
        if is_file:
            return cls._load_from_file(path)

        try:
            data = json.loads(config_arg)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"--config value is neither a valid file path nor valid JSON: {e}"
            ) from e

        return cls.model_validate(data)

    @classmethod
    def _load_from_file(cls, path: Path) -> "GlobalPlannerConfig":
        suffix = path.suffix.lower()
        text = path.read_text()

        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            # Unknown suffix: try JSON first, then YAML.
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = yaml.safe_load(text)

        return cls.model_validate(data)

    # ------------------------------------------------------------------ #
    # Derived state                                                      #
    # ------------------------------------------------------------------ #

    def budget_enforcement_enabled(self) -> bool:
        """Whether either budget bound is active."""
        return self.max_total_gpus >= 0 or self.min_total_gpus >= 0

    def log_summary(self) -> None:
        """Log the effective configuration at startup."""
        logger.info(f"Environment: {self.environment}")

        if self.managed_namespaces:
            logger.info("Authorization: ENABLED")
            logger.info(f"Authorized namespaces: {self.managed_namespaces}")
        else:
            logger.info("Authorization: DISABLED (accepting all namespaces)")

        if self.no_operation:
            logger.info(
                "No-operation mode: ENABLED (scale requests will be logged, not executed)"
            )
        else:
            logger.info("No-operation mode: DISABLED")

        if self.max_total_gpus >= 0:
            logger.info(f"Max total GPUs: {self.max_total_gpus}")
        else:
            logger.info("Max total GPUs: UNLIMITED")

        if self.min_total_gpus >= 0:
            logger.info(f"Min total GPUs: {self.min_total_gpus}")
        else:
            logger.info("Min total GPUs: DISABLED")

        # Intent cache TTL governs pair freshness for BOTH floor and ceiling
        # pairing, so log it whenever either bound is active.
        if self.budget_enforcement_enabled():
            logger.info(f"Intent cache TTL seconds: {self.intent_cache_ttl_seconds}")
