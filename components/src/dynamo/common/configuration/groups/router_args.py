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

import logging
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument

logger = logging.getLogger(__name__)

# Fields forwarded verbatim as kwargs to RouterConfig.__init__.
_ROUTER_FIELDS: tuple[str, ...] = (
    "active_decode_blocks_threshold",
    "active_prefill_tokens_threshold",
    "active_prefill_tokens_threshold_frac",
    "enforce_disagg",
)

# Valid values for --admission-control.
#
# - "token-capacity": apply the configured per-worker busy thresholds
#   (--active-decode-blocks-threshold, --active-prefill-tokens-threshold,
#   --active-prefill-tokens-threshold-frac).
# - "none": disable busy-worker admission checks entirely; router queueing
#   remains controlled by --router-queue-threshold.
ADMISSION_CONTROL_CHOICES: tuple[str, ...] = ("token-capacity", "none")
# Sentinel default — distinguishes "user did not pass --admission-control"
# (auto-decide based on whether any threshold flag is explicitly set)
# from "user explicitly passed --admission-control none" (treat as
# contradiction if combined with an explicit threshold flag, raise).
# Not in ADMISSION_CONTROL_CHOICES so argparse never accepts it from input.
_ADMISSION_CONTROL_AUTO: str = "_auto_"

# Production defaults for the busy thresholds, applied only when
# admission-control resolves to "token-capacity". The CLI/env defaults are
# `None` (sentinel for "user did not set this") so apply_admission_control
# can distinguish "user explicitly passed a threshold" from "argparse
# filled in a default" — and so explicit `--admission-control none` does
# not raise the contradiction error against defaults the user never set.
_DEFAULT_ACTIVE_DECODE_BLOCKS_THRESHOLD: float = 1.0
_DEFAULT_ACTIVE_PREFILL_TOKENS_THRESHOLD: int = 10_000_000
_DEFAULT_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC: float = 64.0


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
    # Sentinel default — see _ADMISSION_CONTROL_AUTO comment. After
    # apply_admission_control runs, this is always one of
    # ADMISSION_CONTROL_CHOICES.
    admission_control: str = _ADMISSION_CONTROL_AUTO

    def router_kwargs(self) -> dict:
        """Return a dict suitable for ``RouterConfig(mode, kv_config, **kwargs)``."""
        self.apply_admission_control()
        return {f: getattr(self, f) for f in _ROUTER_FIELDS}

    def apply_admission_control(self) -> None:
        """Apply the --admission-control mode to the busy thresholds.

        Three input modes:
        - `_ADMISSION_CONTROL_AUTO` (sentinel default; the user did not pass
          --admission-control and did not set DYN_ADMISSION_CONTROL): if any
          threshold flag is explicitly set, auto-promote to "token-capacity"
          so the threshold takes effect (preserves the v1.0.x / v1.1.x
          launch-config contract where setting a threshold flag implicitly
          activated admission control). Otherwise resolve to "none".
        - "token-capacity": keep the configured busy thresholds as-is.
        - "none" (explicit): clear all busy thresholds; if any threshold
          flag was also explicitly set, raise — the combination is a
          contradiction (you explicitly turned admission off while also
          configuring a threshold value).

        After this method returns, ``self.admission_control`` is always one
        of ADMISSION_CONTROL_CHOICES.
        """
        explicit_thresholds: list[str] = []
        if self.active_decode_blocks_threshold is not None:
            explicit_thresholds.append("--active-decode-blocks-threshold")
        if self.active_prefill_tokens_threshold is not None:
            explicit_thresholds.append("--active-prefill-tokens-threshold")
        if self.active_prefill_tokens_threshold_frac is not None:
            explicit_thresholds.append("--active-prefill-tokens-threshold-frac")

        if self.admission_control == _ADMISSION_CONTROL_AUTO:
            if explicit_thresholds:
                logger.info(
                    "admission-control: implicit mode resolved to 'token-capacity' "
                    "because %s was explicitly set. Pass --admission-control "
                    "token-capacity to make this explicit, or unset the "
                    "threshold(s) to keep admission control disabled.",
                    ", ".join(explicit_thresholds),
                )
                self.admission_control = "token-capacity"
            else:
                self.admission_control = "none"

        if self.admission_control not in ADMISSION_CONTROL_CHOICES:
            raise ValueError(
                f"--admission-control must be one of "
                f"{ADMISSION_CONTROL_CHOICES}, got {self.admission_control!r}"
            )

        if self.admission_control == "token-capacity":
            # Fill in production defaults for any threshold the user did not
            # explicitly set. Until now the threshold fields default to None
            # so the explicit-user check above stays accurate.
            if self.active_decode_blocks_threshold is None:
                self.active_decode_blocks_threshold = (
                    _DEFAULT_ACTIVE_DECODE_BLOCKS_THRESHOLD
                )
            if self.active_prefill_tokens_threshold is None:
                self.active_prefill_tokens_threshold = (
                    _DEFAULT_ACTIVE_PREFILL_TOKENS_THRESHOLD
                )
            if self.active_prefill_tokens_threshold_frac is None:
                self.active_prefill_tokens_threshold_frac = (
                    _DEFAULT_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC
                )
            return

        # admission_control == "none" (explicit or auto-resolved). Contradiction
        # with any explicit threshold flag — raise rather than silently
        # overriding.
        if explicit_thresholds:
            raise ValueError(
                "--admission-control none cannot be combined with explicit "
                f"{', '.join(explicit_thresholds)} — drop the threshold flag(s) "
                "to keep admission disabled, or pass --admission-control "
                "token-capacity to activate the threshold(s)."
            )
        # All thresholds remain None (their argparse default) — nothing to clear.


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
            default=None,
            help=(
                "Threshold fraction (0.0-1.0) of KV cache block utilization above which a worker "
                "is considered busy. Setting this implies --admission-control token-capacity. "
                "Pass 'None' on the CLI to disable this check. "
                "Token-capacity default: 1.0."
            ),
            arg_type=_nullable_float,
        )
        add_argument(
            g,
            flag_name="--active-prefill-tokens-threshold",
            env_var="DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
            default=None,
            help=(
                "Literal token count threshold for determining when a worker is considered busy "
                "based on prefill token utilization. When active prefill tokens exceed this "
                "threshold, the worker is marked as busy. Setting this implies "
                "--admission-control token-capacity. Pass 'None' on the CLI to disable this "
                "check. Uses OR logic with --active-prefill-tokens-threshold-frac. "
                "Token-capacity default: 10000000."
            ),
            arg_type=_nullable_int,
        )
        add_argument(
            g,
            flag_name="--active-prefill-tokens-threshold-frac",
            env_var="DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
            default=None,
            help=(
                "Fraction of max_num_batched_tokens for busy detection. Worker is busy when "
                "active_prefill_tokens > frac * max_num_batched_tokens. Setting this implies "
                "--admission-control token-capacity. Pass 'None' on the CLI to disable this "
                "check. Uses OR logic with --active-prefill-tokens-threshold. "
                "Token-capacity default: 64.0."
            ),
            arg_type=_nullable_float,
        )
        add_argument(
            g,
            flag_name="--admission-control",
            env_var="DYN_ADMISSION_CONTROL",
            default=_ADMISSION_CONTROL_AUTO,
            help=(
                "Admission control mode. 'token-capacity' enables per-worker busy "
                "checks using --active-decode-blocks-threshold, "
                "--active-prefill-tokens-threshold, and "
                "--active-prefill-tokens-threshold-frac. 'none' disables those "
                "busy checks; router queueing remains controlled by "
                "--router-queue-threshold."
            ),
            choices=list(ADMISSION_CONTROL_CHOICES),
        )
