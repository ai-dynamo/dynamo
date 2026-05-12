# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared router configuration ArgGroup.

Defines the router configuration parameters once so that both
``dynamo.frontend`` and other components can reuse them without duplication.
Field names on ``RouterConfigBase`` match the ``RouterConfig`` Python
constructor kwargs 1:1 (for the non-positional args), so ``router_kwargs()``
returns a dict that can be unpacked into
``RouterConfig(mode, kv_config, **config.router_kwargs())``.
"""

from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument

# Fields forwarded verbatim as kwargs to RouterConfig.__init__.
_ROUTER_FIELDS: tuple[str, ...] = (
    "active_decode_blocks_threshold",
    "active_prefill_tokens_threshold",
    "active_prefill_tokens_threshold_frac",
    "enforce_disagg",
)


def _nullable_float(value: str) -> Optional[float]:
    """Parse a float, or return None for the literal 'None'."""
    if value is None or value == "None":
        return None
    return float(value)


def _nullable_int(value: str) -> Optional[int]:
    """Parse an int, or return None for the literal 'None'."""
    if value is None or value == "None":
        return None
    return int(value)


class RouterConfigBase(ConfigBase):
    """Mixin carrying the shared router configuration fields."""

    router_mode: str
    min_initial_workers: int
    enforce_disagg: bool
    active_decode_blocks_threshold: Optional[float]
    active_prefill_tokens_threshold: Optional[int]
    active_prefill_tokens_threshold_frac: Optional[float]

    def router_kwargs(self) -> dict:
        """Return a dict suitable for ``RouterConfig(mode, kv_config, **kwargs)``."""
        return {f: getattr(self, f) for f in _ROUTER_FIELDS}


class RouterArgGroup(ArgGroup):
    """CLI arguments for the shared router configuration parameters."""

    def add_arguments(self, parser) -> None:
        g = parser.add_argument_group("Router Options")

        add_argument(
            g,
            flag_name="--router-mode",
            env_var="DYN_ROUTER_MODE",
            default="round-robin",
            help=(
                "How to route the request. power-of-two picks 2 random workers and "
                "routes to the one with fewer in-flight requests. least-loaded routes to "
                "the worker with the fewest active requests. device-aware-weighted routes "
                "based on worker device type (CPU/CUDA). In disaggregated prefill mode, "
                "both power-of-two and least-loaded skip bootstrap optimization and fall "
                "back to the synchronous prefill path."
            ),
            choices=[
                "round-robin",
                "random",
                "power-of-two",
                "kv",
                "direct",
                "least-loaded",
                "device-aware-weighted",
            ],
        )
        add_argument(
            g,
            flag_name="--router-min-initial-workers",
            env_var="DYN_ROUTER_MIN_INITIAL_WORKERS",
            default=0,
            help=(
                "Minimum number of workers required before router startup continues. "
                "This is exported as DYN_ROUTER_MIN_INITIAL_WORKERS so the generic "
                "push-router path and the KV router's config-ready worker gate share "
                "the same startup threshold. Set to 0 to disable the startup wait."
            ),
            arg_type=int,
            dest="min_initial_workers",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enforce-disagg",
            env_var="DYN_ENFORCE_DISAGG",
            default=False,
            dest="enforce_disagg",
            help=(
                "Strictly enforce disaggregated mode. Requests will fail if the prefill router "
                "has not activated yet (e.g., prefill workers still registering). This is stricter "
                "than the default: without this flag, requests arriving before prefill workers are "
                "discovered fall through to aggregated decode-only routing."
            ),
        )
        add_argument(
            g,
            flag_name="--active-decode-blocks-threshold",
            env_var="DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",
            default=1.0,
            help=(
                "Threshold fraction (0.0-1.0) of KV cache block utilization above which a worker "
                "is considered busy. Pass 'None' on the CLI to disable this check. Default: 1.0."
            ),
            arg_type=_nullable_float,
        )
        add_argument(
            g,
            flag_name="--active-prefill-tokens-threshold",
            env_var="DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
            default=10_000_000,
            help=(
                "Literal token count threshold for determining when a worker is considered busy "
                "based on prefill token utilization. When active prefill tokens exceed this "
                "threshold, the worker is marked as busy. Pass 'None' on the CLI to disable this "
                "check. Uses OR logic with --active-prefill-tokens-threshold-frac. Default: 10000000."
            ),
            arg_type=_nullable_int,
        )
        add_argument(
            g,
            flag_name="--active-prefill-tokens-threshold-frac",
            env_var="DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
            default=10.0,
            help=(
                "Fraction of max_num_batched_tokens for busy detection. Worker is busy when "
                "active_prefill_tokens > frac * max_num_batched_tokens. Pass 'None' on the CLI to "
                "disable this check. Uses OR logic with --active-prefill-tokens-threshold. Default: 10.0."
            ),
            arg_type=_nullable_float,
        )
