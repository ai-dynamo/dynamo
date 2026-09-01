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
from typing import TYPE_CHECKING, Optional, Protocol, Sequence

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.common.configuration.utils import add_argument, nullable_float, nullable_int

if TYPE_CHECKING:
    from dynamo.llm import RouterConfig

logger = logging.getLogger(__name__)

# Fields forwarded verbatim as kwargs to RouterConfig.__init__.
_ROUTER_FIELDS: tuple[str, ...] = (
    "active_decode_blocks_threshold",
    "active_prefill_tokens_threshold",
    "active_prefill_tokens_threshold_frac",
    "session_affinity_ttl_secs",
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
    """CLI arguments for the shared router configuration parameters.

    Both arguments are required, deliberately. A caller that fell back to
    frontend-shaped defaults would give a worker ``--router-mode round-robin``,
    and since a worker's card replaces the frontend's configuration wholesale,
    that worker would silently override a frontend running any other mode.
    Requiring the choice turns "forgot to think about it" into a TypeError at
    startup instead of routing that quietly ignores the operator.

    Args:
        default_router_mode: Default for ``--router-mode``. The frontend passes
            the historical ``"round-robin"``; a worker set passes ``None`` so
            that omitting the flag advertises nothing and inherits the
            frontend's configuration.
        include_frontend_only: Whether to register arguments the frontend alone
            consumes. ``--router-min-initial-workers`` gates frontend startup
            and is not carried on the model card, so a worker registering it
            would ship a flag that does nothing.
    """

    def __init__(
        self,
        *,
        default_router_mode: Optional[str],
        include_frontend_only: bool,
    ) -> None:
        self.default_router_mode = default_router_mode
        self.include_frontend_only = include_frontend_only

    def add_arguments(self, parser) -> None:
        if "DYN_ADMISSION_CONTROL" in os.environ:
            logger.warning(_ADMISSION_CONTROL_REMOVAL_WARNING)
        if "DYN_ENFORCE_DISAGG" in os.environ:
            logger.warning(_ENFORCE_DISAGG_DEPRECATION, "DYN_ENFORCE_DISAGG")

        g = parser.add_argument_group("Router Options")

        if self.include_frontend_only:
            # Arguments the frontend alone consumes. None of them are carried on
            # a model card, so registering them on a worker would ship flags
            # that silently do nothing.
            #
            # --admission-control and --enforce-disagg are removed/deprecated
            # and accepted only so existing frontend launch commands keep
            # starting; no worker command ever passed them.
            g.add_argument(
                "--admission-control",
                choices=("token-capacity", "none"),
                action=_IgnoredAdmissionControlAction,
                default=argparse.SUPPRESS,
                help=argparse.SUPPRESS,
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
            flag_name="--router-mode",
            env_var="DYN_ROUTER_MODE",
            default=self.default_router_mode,
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


# CLI spelling -> `dynamo.llm.RouterMode` attribute name.
ROUTER_MODE_MAP: dict[str, str] = {
    "round-robin": "RoundRobin",
    "random": "Random",
    "power-of-two": "PowerOfTwoChoices",
    "kv": "KV",
    "direct": "Direct",
    "least-loaded": "LeastLoaded",
    "device-aware-weighted": "DeviceAwareWeighted",
}


class WorkerRouterConfig(RouterConfigBase, KvRouterConfigBase):
    """Router configuration a worker set advertises in its model card.

    Same composition the frontend's config uses, so the two stay in step
    without duplicating field declarations.
    """

    # Registered only for the frontend, so give them values here rather than
    # leaving the attributes absent on a worker's config object.
    min_initial_workers: int = 0
    enforce_disagg: bool = False


def add_worker_router_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the worker-side router flags.

    ``--router-mode`` defaults to ``None``: a worker that omits it advertises
    nothing and inherits the frontend's configuration.
    """
    RouterArgGroup(default_router_mode=None, include_frontend_only=False).add_arguments(
        parser
    )
    KvRouterArgGroup().add_arguments(parser)


def parse_worker_router_config(
    argv: Sequence[str],
) -> tuple[WorkerRouterConfig, list[str]]:
    """Parse the router flags out of ``argv``, returning the rest untouched.

    Backends call this between their own argument parsing and their engine's,
    so the engine parser never sees these flags.
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_worker_router_arguments(parser)
    namespace, remainder = parser.parse_known_args(list(argv))
    return WorkerRouterConfig.from_cli_args(namespace), remainder


def register_worker_router_help(parser: argparse.ArgumentParser) -> None:
    """Surface the worker router flags in ``--help``.

    They are parsed by a separate parser, so they would otherwise be invisible.
    Same display-only trick the backends use for their engine arguments;
    ``_group_actions`` is private argparse API, as it is at those call sites.
    """
    source_parser = argparse.ArgumentParser(add_help=False)
    add_worker_router_arguments(source_parser)
    group = parser.add_argument_group(
        "Router Advertisement Options. Declared in this worker's model card to "
        "override the frontend's routing for this worker set only."
    )
    for action in source_parser._actions:
        if action.option_strings:
            group._group_actions.append(action)


class RouterConfigSource(Protocol):
    """Anything carrying `RouterConfigBase` and `KvRouterConfigBase` fields.

    Both `WorkerRouterConfig` and the frontend's own config qualify, and neither
    subclasses the other, so this states the requirement structurally.
    """

    def router_kwargs(self) -> dict:
        ...

    def kv_router_kwargs(self) -> dict:
        ...


def build_router_config(
    config: Optional[RouterConfigSource],
) -> Optional["RouterConfig"]:
    """Build the ``RouterConfig`` a worker set advertises in its model card.

    ``None`` means no mode was requested, leaving ``router_config`` off the card
    so the worker inherits the frontend's configuration. A ``None`` config means
    the same thing, so a backend can pass its optional advertisement directly.
    The frontend passes its own config here too; it always has a mode, so it
    never gets ``None`` back.
    """
    if config is None:
        return None
    router_mode = getattr(config, "router_mode", None)
    if router_mode is None:
        return None

    # Imported lazily so that importing a backend's argument definitions does
    # not pull in the compiled bindings.
    from dynamo.llm import KvRouterConfig, RouterConfig, RouterMode

    try:
        mode_attr = ROUTER_MODE_MAP[router_mode]
    except KeyError as error:
        raise ValueError(
            f"unknown router mode {router_mode!r}; expected one of "
            f"{', '.join(sorted(ROUTER_MODE_MAP))}"
        ) from error

    mode = getattr(RouterMode, mode_attr)
    # Only KV routing consults KvRouterConfig; passing it for other modes would
    # imply tuning that is never read.
    kv_router_config = (
        KvRouterConfig(**config.kv_router_kwargs()) if mode == RouterMode.KV else None
    )
    return RouterConfig(mode, kv_router_config, **config.router_kwargs())
