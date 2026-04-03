# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common argument handling for Dynamo backend workers.

This module re-exports the canonical DynamoRuntimeConfig and
DynamoRuntimeArgGroup from dynamo.common.configuration.groups.runtime_args
and adds DynamoBackendConfig / DynamoBackendArgGroup for fields that are
common across all backends but not part of the runtime layer.
"""

import argparse
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.runtime_args import (  # noqa: F401
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.configuration.utils import add_argument


class DynamoBackendConfig(DynamoRuntimeConfig):
    """Configuration common to all Dynamo backends.

    Extends DynamoRuntimeConfig with fields that every backend needs
    (model identity, disaggregation, component name) but that don't
    belong in the runtime layer.

    Backend-specific fields (engine args, parallelism, etc.) should go
    on a separate object stored as ``config.extra``.
    """

    model: Optional[str] = None
    served_model_name: Optional[str] = None
    disaggregation_mode: str = "aggregated"
    component: str = "backend"
    use_kv_events: bool = False

    def get_model_name(self) -> str:
        """Get the effective model name for display and metrics."""
        return self.served_model_name or self.model or "unknown"


class DynamoBackendArgGroup(ArgGroup):
    """CLI arguments common to all Dynamo backends."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("Dynamo Backend Options")

        add_argument(
            g,
            flag_name="--model",
            env_var="DYN_MODEL",
            default=None,
            help="Model name or path.",
        )
        add_argument(
            g,
            flag_name="--served-model-name",
            env_var="DYN_SERVED_MODEL_NAME",
            default=None,
            help="Name to advertise for this model. Defaults to --model value.",
        )
        add_argument(
            g,
            flag_name="--disaggregation-mode",
            env_var="DYN_DISAGGREGATION_MODE",
            default="aggregated",
            help="Disaggregation mode for the backend worker.",
            choices=["aggregated", "prefill", "decode"],
        )
        add_argument(
            g,
            flag_name="--component",
            env_var="DYN_COMPONENT",
            default="backend",
            help="Component name for Dynamo registration.",
        )
