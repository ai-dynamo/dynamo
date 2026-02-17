#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Dynamo Global Router configuration ArgGroup."""

from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument


class DynamoGlobalRouterArgGroup(ArgGroup):
    """Global Router-specific Dynamo configuration (CLI and env)."""

    def add_arguments(self, parser) -> None:
        """Add Dynamo Global Router arguments to parser."""
        g = parser.add_argument_group("Dynamo Global Router Options")

        add_argument(
            g,
            flag_name="--config",
            env_var="DYN_GLOBAL_ROUTER_CONFIG",
            default=None,
            help="Path to the JSON configuration file defining pool namespaces and selection strategy. Must be set via CLI or env.",
            dest="config_path",
        )
        add_argument(
            g,
            flag_name="--model-name",
            env_var="DYN_GLOBAL_ROUTER_MODEL_NAME",
            default=None,
            help="Model name for registration (must match workers). Must be set via CLI or env.",
        )
        add_argument(
            g,
            flag_name="--namespace",
            env_var="DYN_NAMESPACE",
            default="dynamo",
            help="Dynamo namespace for the global router.",
        )
        add_argument(
            g,
            flag_name="--component-name",
            env_var="DYN_GLOBAL_ROUTER_COMPONENT_NAME",
            default="global_router",
            help="Component name for the global router.",
        )
        add_argument(
            g,
            flag_name="--default-ttft-target",
            env_var="DYN_GLOBAL_ROUTER_DEFAULT_TTFT_TARGET",
            default=None,
            help="Default TTFT target (ms) for prefill pool selection when SLA not present in request.",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--default-itl-target",
            env_var="DYN_GLOBAL_ROUTER_DEFAULT_ITL_TARGET",
            default=None,
            help="Default ITL target (ms) for decode pool selection when SLA not present in request.",
            arg_type=float,
        )


class DynamoGlobalRouterConfig(ConfigBase):
    """Configuration for Dynamo Global Router (CLI/env-backed)."""

    config_path: Optional[str] = None
    model_name: Optional[str] = None
    namespace: str
    component_name: str
    default_ttft_target: Optional[float] = None
    default_itl_target: Optional[float] = None

    def validate(self) -> None:
        """Require config_path and model_name to be set via CLI or env."""
        if not self.config_path or not self.config_path.strip():
            raise ValueError(
                "config_path must be set via --config or DYN_GLOBAL_ROUTER_CONFIG"
            )
        if not self.model_name or not self.model_name.strip():
            raise ValueError(
                "model_name must be set via --model-name or DYN_GLOBAL_ROUTER_MODEL_NAME"
            )
