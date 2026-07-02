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

import logging
from typing import Any, Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import (
    add_argument,
    add_negatable_bool_argument,
    nullable_float,
    nullable_int,
)

logger = logging.getLogger(__name__)

# Fields forwarded verbatim as kwargs to RouterConfig.__init__.
_ROUTER_FIELDS: tuple[str, ...] = (
    "active_decode_blocks_threshold",
    "active_prefill_tokens_threshold",
    "active_prefill_tokens_threshold_frac",
    "session_affinity_ttl_secs",
)

_ENFORCE_DISAGG_DEPRECATION = (
    "--enforce-disagg and DYN_ENFORCE_DISAGG are deprecated and ignored; "
    "routing topology and readiness are determined from registered worker types"
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
# admission-control resolves to "token-capacity" AND the user did not pass
# the corresponding flag at all.
_DEFAULT_ACTIVE_DECODE_BLOCKS_THRESHOLD: float = 1.0
_DEFAULT_ACTIVE_PREFILL_TOKENS_THRESHOLD: int = 10_000_000
_DEFAULT_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC: float = 64.0

# Sentinel default for the three threshold flags. Distinguishes three
# states that all collapse to the same Python value otherwise:
#   - `_THRESHOLD_UNSET`: user did not pass the flag and the env var is
#     unset — fill the production default in token-capacity mode.
#   - `None`: user explicitly passed `--<flag> None` (or set the env var
#     to "None") — keep the check disabled even in token-capacity mode.
#   - numeric: user-supplied value — keep as-is.
# Replaced with `None` in `apply_admission_control` before the value
# leaves the config object, so downstream consumers still see
# `Optional[float|int]`.
_THRESHOLD_UNSET: Any = object()


class RouterConfigBase(ConfigBase):
    """Mixin carrying the shared router configuration fields."""

    router_mode: str
    min_initial_workers: int
    enforce_disagg: bool
    session_affinity_ttl_secs: Optional[int]
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
        if self.enforce_disagg:
            logger.warning(_ENFORCE_DISAGG_DEPRECATION)
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
        - "token-capacity": keep configured thresholds as-is, fill
          production defaults for thresholds the user did not pass.
        - "none" (explicit): clear all busy thresholds; if any threshold
          flag was set to a *numeric* value, raise — explicit `--<flag>
          None` is consistent with admission disabled and silently kept.

        After this method returns, ``self.admission_control`` is always one
        of ADMISSION_CONTROL_CHOICES and the threshold fields are
        ``Optional[float|int]``. Calling the method again on the resolved
        state is a no-op (idempotent).
        """
        # `numeric_thresholds` is the subset that actually configures a cap.
        # The auto-promote rule and the explicit-`none` contradiction both
        # key off this — explicit `--<flag> None` is consistent with
        # admission disabled, so it never auto-promotes and never raises.
        # Restricting the contradiction check to numeric values also makes
        # this method idempotent: after the sentinel → None normalization
        # below, a subsequent call sees fields that are `None` (no longer
        # _THRESHOLD_UNSET) and correctly treats them as "no numeric cap".
        numeric_thresholds: list[str] = []
        for value, flag in (
            (
                self.active_decode_blocks_threshold,
                "--active-decode-blocks-threshold",
            ),
            (
                self.active_prefill_tokens_threshold,
                "--active-prefill-tokens-threshold",
            ),
            (
                self.active_prefill_tokens_threshold_frac,
                "--active-prefill-tokens-threshold-frac",
            ),
        ):
            if value is _THRESHOLD_UNSET or value is None:
                continue
            numeric_thresholds.append(flag)

        if self.admission_control == _ADMISSION_CONTROL_AUTO:
            if numeric_thresholds:
                logger.info(
                    "admission-control: implicit mode resolved to 'token-capacity' "
                    "because %s was set to a numeric value. Pass --admission-control "
                    "token-capacity to make this explicit, or unset the "
                    "threshold(s) to keep admission control disabled.",
                    ", ".join(numeric_thresholds),
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
            # Fill production defaults only for thresholds the user did not
            # pass at all. Explicit `None` from the user is preserved so the
            # documented "Pass 'None' on the CLI to disable this check"
            # semantic holds even in token-capacity mode.
            if self.active_decode_blocks_threshold is _THRESHOLD_UNSET:
                self.active_decode_blocks_threshold = (
                    _DEFAULT_ACTIVE_DECODE_BLOCKS_THRESHOLD
                )
            if self.active_prefill_tokens_threshold is _THRESHOLD_UNSET:
                self.active_prefill_tokens_threshold = (
                    _DEFAULT_ACTIVE_PREFILL_TOKENS_THRESHOLD
                )
            if self.active_prefill_tokens_threshold_frac is _THRESHOLD_UNSET:
                self.active_prefill_tokens_threshold_frac = (
                    _DEFAULT_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC
                )
            return

        # admission_control == "none" (explicit or auto-resolved). A numeric
        # threshold value alongside explicit `none` is a contradiction; an
        # explicit `--<flag> None` is consistent and silently kept.
        if numeric_thresholds:
            raise ValueError(
                "--admission-control none cannot be combined with explicit "
                f"{', '.join(numeric_thresholds)} — drop the threshold flag(s) "
                "to keep admission disabled, or pass --admission-control "
                "token-capacity to activate the threshold(s)."
            )
        # Sentinel → None so downstream sees the documented Optional[…] type.
        # Explicit user-passed `None` already matches; this is a no-op for those.
        if self.active_decode_blocks_threshold is _THRESHOLD_UNSET:
            self.active_decode_blocks_threshold = None
        if self.active_prefill_tokens_threshold is _THRESHOLD_UNSET:
            self.active_prefill_tokens_threshold = None
        if self.active_prefill_tokens_threshold_frac is _THRESHOLD_UNSET:
            self.active_prefill_tokens_threshold_frac = None


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
        add_argument(
            g,
            flag_name="--router-session-affinity-ttl-secs",
            env_var="DYN_ROUTER_SESSION_AFFINITY_TTL_SECS",
            default=None,
            help=(
                "Enable session affinity and set the process-local cache eviction TTL "
                "in seconds. etcd and shared FileStore use immutable distributed claims "
                "whose lifetime follows the creating frontend, not this TTL. Memory and "
                "Kubernetes discovery remain process-local. Affinity is disabled when "
                "this option is omitted."
            ),
            arg_type=int,
            dest="session_affinity_ttl_secs",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enforce-disagg",
            env_var="DYN_ENFORCE_DISAGG",
            default=False,
            dest="enforce_disagg",
            help=(
                "DEPRECATED: accepted for compatibility but ignored. Routing topology and "
                "readiness are determined from registered worker types."
            ),
        )
        add_argument(
            g,
            flag_name="--active-decode-blocks-threshold",
            env_var="DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",
            default=_THRESHOLD_UNSET,
            help=(
                "Threshold fraction (0.0-1.0) of KV cache block utilization above which a worker "
                "is considered busy. Setting this implies --admission-control token-capacity. "
                "Pass 'None' on the CLI to disable this check. "
                "Token-capacity default: 1.0."
            ),
            arg_type=nullable_float,
        )
        add_argument(
            g,
            flag_name="--active-prefill-tokens-threshold",
            env_var="DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
            default=_THRESHOLD_UNSET,
            help=(
                "Literal token count threshold for determining when a worker is considered busy "
                "based on prefill token utilization. When active prefill tokens exceed this "
                "threshold, the worker is marked as busy. Setting this implies "
                "--admission-control token-capacity. Pass 'None' on the CLI to disable this "
                "check. Uses OR logic with --active-prefill-tokens-threshold-frac. "
                "Token-capacity default: 10000000."
            ),
            arg_type=nullable_int,
        )
        add_argument(
            g,
            flag_name="--active-prefill-tokens-threshold-frac",
            env_var="DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
            default=_THRESHOLD_UNSET,
            help=(
                "Fraction of max_num_batched_tokens for busy detection. Worker is busy when "
                "active_prefill_tokens > frac * max_num_batched_tokens. Setting this implies "
                "--admission-control token-capacity. Pass 'None' on the CLI to disable this "
                "check. Uses OR logic with --active-prefill-tokens-threshold. "
                "Token-capacity default: 64.0."
            ),
            arg_type=nullable_float,
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
