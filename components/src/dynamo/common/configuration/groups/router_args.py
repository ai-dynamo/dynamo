# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared router configuration ArgGroup.

Defines the router configuration parameters once so that both
``dynamo.frontend`` and other components can reuse them without duplication.
Active field names on ``RouterConfigBase`` match the ``RouterConfig`` Python
constructor kwargs 1:1 (for the non-positional args), so ``router_kwargs()``
returns a dict that can be unpacked into
``RouterConfig(mode, kv_config, **config.router_kwargs())``. Deprecated fields
remain parseable but are not forwarded.
"""

import argparse
import logging
import math
import os
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, nullable_float, nullable_int

logger = logging.getLogger(__name__)

# Fields forwarded verbatim as kwargs to RouterConfig.__init__.
_ROUTER_FIELDS: tuple[str, ...] = (
    "active_decode_blocks_threshold",
    "active_prefill_tokens_threshold",
    "active_prefill_tokens_threshold_frac",
    "session_affinity_ttl_secs",
    "non_cpu_to_cpu_ratio",
)

_ENFORCE_DISAGG_DEPRECATION = (
    "%s is deprecated and ignored; disaggregated routing topology and readiness "
    "are determined automatically from registered worker types"
)

_ADMISSION_CONTROL_REMOVAL_WARNING = (
    "DYN_ADMISSION_CONTROL is no longer supported and is ignored; configure "
    "DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD, DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD, "
    "and DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC directly"
)

_ADMISSION_CONTROL_FLAG_REMOVAL_WARNING = (
    "--admission-control is no longer supported and is ignored; configure "
    "--active-decode-blocks-threshold, --active-prefill-tokens-threshold, "
    "and --active-prefill-tokens-threshold-frac directly"
)


class _IgnoredAdmissionControlAction(argparse.Action):
    """Warn and store nothing, so the namespace never carries the value."""

    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(_ADMISSION_CONTROL_FLAG_REMOVAL_WARNING)


class _DeprecatedEnforceDisaggAction(argparse.BooleanOptionalAction):
    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(_ENFORCE_DISAGG_DEPRECATION, option_string)
        super().__call__(parser, namespace, values, option_string)


class RouterConfigBase(ConfigBase):
    """Mixin carrying the shared router configuration fields."""

    router_mode: str
    min_initial_workers: int
    enforce_disagg: bool
    session_affinity_ttl_secs: Optional[int]
    active_decode_blocks_threshold: Optional[float]
    active_prefill_tokens_threshold: Optional[int]
    active_prefill_tokens_threshold_frac: Optional[float]
    non_cpu_to_cpu_ratio: Optional[int]

    def router_kwargs(self) -> dict:
        """Return a dict suitable for ``RouterConfig(mode, kv_config, **kwargs)``."""
        return {f: getattr(self, f) for f in _ROUTER_FIELDS}

    def validate_rejection_thresholds(self) -> None:
        """Validate independently configured busy-worker rejection thresholds."""
        decode_threshold = self.active_decode_blocks_threshold
        if decode_threshold is not None and not (
            math.isfinite(decode_threshold) and 0.0 <= decode_threshold <= 1.0
        ):
            raise ValueError(
                "--active-decode-blocks-threshold must be between 0.0 and 1.0"
            )

        prefill_threshold = self.active_prefill_tokens_threshold
        if prefill_threshold is not None and prefill_threshold < 0:
            raise ValueError("--active-prefill-tokens-threshold must be >= 0")

        prefill_threshold_frac = self.active_prefill_tokens_threshold_frac
        if prefill_threshold_frac is not None and not (
            math.isfinite(prefill_threshold_frac) and prefill_threshold_frac >= 0.0
        ):
            raise ValueError(
                "--active-prefill-tokens-threshold-frac must be a finite value >= 0"
            )

    def validate_non_cpu_to_cpu_ratio(self) -> None:
        """Validate the optional device-aware weighted router capacity ratio."""
        ratio = self.non_cpu_to_cpu_ratio
        if ratio is not None and ratio < 1:
            raise ValueError("--encoder-non-cpu-to-cpu-ratio must be >= 1")

    def log_rejection_thresholds(self) -> None:
        """Log which independently configured rejection checks are active."""
        configured = [
            f"{flag}={value}"
            for flag, value in (
                (
                    "--active-decode-blocks-threshold",
                    self.active_decode_blocks_threshold,
                ),
                (
                    "--active-prefill-tokens-threshold",
                    self.active_prefill_tokens_threshold,
                ),
                (
                    "--active-prefill-tokens-threshold-frac",
                    self.active_prefill_tokens_threshold_frac,
                ),
            )
            if value is not None
        ]
        if configured:
            logger.info(
                "busy-worker rejection enabled by %s",
                ", ".join(configured),
            )
        else:
            logger.info(
                "busy-worker rejection disabled: no rejection threshold is configured"
            )


class RouterArgGroup(ArgGroup):
    """CLI arguments for the shared router configuration parameters."""

    def add_arguments(self, parser) -> None:
        if "DYN_ADMISSION_CONTROL" in os.environ:
            logger.warning(_ADMISSION_CONTROL_REMOVAL_WARNING)
        if "DYN_ENFORCE_DISAGG" in os.environ:
            logger.warning(_ENFORCE_DISAGG_DEPRECATION, "DYN_ENFORCE_DISAGG")

        g = parser.add_argument_group("Router Options")

        # Removed master switch, still accepted so existing launch commands
        # keep starting; warns and sets nothing on the namespace (parity with
        # the DYN_ADMISSION_CONTROL handling above). Not in _ROUTER_FIELDS.
        g.add_argument(
            "--admission-control",
            choices=("token-capacity", "none"),
            action=_IgnoredAdmissionControlAction,
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )

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
        add_argument(
            g,
            flag_name="--router-session-affinity-ttl-secs",
            env_var="DYN_ROUTER_SESSION_AFFINITY_TTL_SECS",
            default=None,
            help=(
                "Enable session affinity with this router-local idle TTL in seconds. "
                "Bindings synchronize across router replicas on a best-effort basis. "
                "Affinity is disabled when this option is omitted. "
                "This is independent of KV prediction TTL settings."
            ),
            arg_type=int,
            dest="session_affinity_ttl_secs",
        )
        g.add_argument(
            "--encoder-non-cpu-to-cpu-ratio",
            default=None,
            type=int,
            dest="non_cpu_to_cpu_ratio",
            help=(
                "Device-aware weighted routing capacity ratio between non-CPU "
                "encoder workers and CPU encoder workers. When unset, the Rust "
                "router falls back to env var DYN_ENCODER_CUDA_TO_CPU_RATIO and then 8."
            ),
        )
        add_argument(
            g,
            flag_name="--enforce-disagg",
            env_var="DYN_ENFORCE_DISAGG",
            default=False,
            dest="enforce_disagg",
            help=(
                "DEPRECATED: accepted for compatibility but ignored. Routing topology and "
                "readiness are determined from registered worker types."
            ),
            arg_type=None,
            action=_DeprecatedEnforceDisaggAction,
        )
        add_argument(
            g,
            flag_name="--active-decode-blocks-threshold",
            env_var="DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",
            default=None,
            help=(
                "Threshold fraction (0.0-1.0) of KV cache block utilization above which a worker "
                "is considered busy. Setting a numeric value enables this rejection check. "
                "Unset by default; pass 'None' to disable it."
            ),
            arg_type=nullable_float,
        )
        add_argument(
            g,
            flag_name="--active-prefill-tokens-threshold",
            env_var="DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
            default=None,
            help=(
                "Literal token count threshold for determining when a worker is considered busy "
                "based on prefill token utilization. When active prefill tokens exceed this "
                "threshold, the worker is marked as busy. Setting a numeric value enables this "
                "rejection check. Unset by default; pass 'None' to disable it. Uses OR logic "
                "with --active-prefill-tokens-threshold-frac."
            ),
            arg_type=nullable_int,
        )
        add_argument(
            g,
            flag_name="--active-prefill-tokens-threshold-frac",
            env_var="DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
            default=None,
            help=(
                "Fraction of max_num_batched_tokens for busy detection. Worker is busy when "
                "active_prefill_tokens > frac * max_num_batched_tokens. Setting a numeric value "
                "enables this rejection check. Unset by default; pass 'None' to disable it. Uses "
                "OR logic with --active-prefill-tokens-threshold."
            ),
            arg_type=nullable_float,
        )
