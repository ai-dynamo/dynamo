# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for the cost-eval decision service."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Defaults aligned with the Planner regression model defaults at
# ``components/src/dynamo/planner/core/perf_model/base.py`` and
# ``components/src/dynamo/planner/config/defaults.py:97`` (max_num_fpm_samples=64).
DEFAULT_MIN_OBSERVATIONS = 5
DEFAULT_MAX_NUM_FPM_SAMPLES = 64
DEFAULT_FPM_POLL_INTERVAL_S = 1.0


class CostEvalConfig(BaseModel):
    """Pydantic config for the cost-eval decision service.

    Loaded from a JSON or YAML file via ``--config``. Keep this small — the
    service has few moving parts.
    """

    namespace: str = Field(
        description="Dynamo namespace the service registers its endpoint under.",
    )

    prefill_component_name: str = Field(
        default="prefill",
        description=(
            "Component name to subscribe to for prefill-pool FPM events. "
            "Combined with prefill_endpoint_name to form the FPM subscription target."
        ),
    )
    prefill_endpoint_name: str = Field(
        default="load_metrics",
        description="Endpoint name on the prefill component publishing FPM events.",
    )

    decode_component_name: str = Field(
        default="decode",
        description="Component name to subscribe to for decode-pool FPM events.",
    )
    decode_endpoint_name: str = Field(
        default="load_metrics",
        description="Endpoint name on the decode component publishing FPM events.",
    )

    # Per-pool regression hyperparameters. Defaults match Planner.
    min_observations: int = Field(
        default=DEFAULT_MIN_OBSERVATIONS,
        ge=1,
        description="Minimum FPM observations before a regression is considered warm.",
    )
    max_num_fpm_samples: int = Field(
        default=DEFAULT_MAX_NUM_FPM_SAMPLES,
        ge=1,
        description="Rolling FPM-sample window size per regression.",
    )

    # How often to drain FPM events from each subscriber and feed them into
    # the regressions. Below 5 minutes keeps the Anthropic prompt-cache
    # warm-equivalent fresh; faster than that is wasted polling unless the
    # workload is genuinely bursty. Default 1s — Planner uses a similar cadence.
    fpm_poll_interval_s: float = Field(
        default=DEFAULT_FPM_POLL_INTERVAL_S,
        gt=0.0,
        description="Seconds between FPM subscriber polls.",
    )

    # Engine config the regression's estimate_next_ttft requires. Set from
    # the cluster's engine config; not derivable from FPM alone.
    max_num_batched_tokens: int = Field(
        ge=1,
        description=(
            "Engine's max_num_batched_tokens for chunked prefill. Passed to "
            "PrefillRegressionModel.estimate_next_ttft / "
            "AggRegressionModel.estimate_next_ttft."
        ),
    )

    @classmethod
    def from_config_arg(cls, arg: str) -> "CostEvalConfig":
        """Load from either an inline JSON string or a path to JSON/YAML."""
        path = Path(arg)
        if path.is_file():
            text = path.read_text()
            if path.suffix in {".yaml", ".yml"}:
                raw: Optional[dict] = yaml.safe_load(text)
            else:
                raw = json.loads(text)
        else:
            raw = json.loads(arg)
        if not isinstance(raw, dict):
            raise ValueError(
                f"Expected mapping at config root, got {type(raw).__name__}"
            )
        return cls(**raw)
