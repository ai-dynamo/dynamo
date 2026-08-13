#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Worker-side router advertisement.

A worker set may declare in its model deployment card how the frontend should
route to it, overriding the frontend's global router configuration for that set
alone. Because the card is per worker type, this is the mechanism by which a
disaggregated deployment gives its prefill and decode tiers different routing
strategies -- for example a KV-routed prefill tier in front of a round-robin
decode tier.

Workers expose the same flags as the frontend (``--router-mode``,
``--router-kv-events``, ...), but parse them into their own
[`WorkerRouterConfig`] rather than flattening them onto the backend's config.
That separation is load-bearing: every backend already defines a
``use_kv_events`` field meaning "this worker publishes KV events", while
``--router-kv-events`` carries the opposite sense of "the router subscribes to
them". Keeping the two on separate objects means neither can silently shadow the
other, and the same holds for any field either side adds later.
"""

import argparse
from typing import TYPE_CHECKING, Optional, Sequence

from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.common.configuration.groups.router_args import (
    RouterArgGroup,
    RouterConfigBase,
)

if TYPE_CHECKING:
    from dynamo.llm import RouterConfig

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


def add_worker_router_arguments(parser) -> None:
    """Register the worker-side router flags on ``parser``.

    ``--router-mode`` defaults to ``None`` so that a worker which does not pass
    it advertises nothing and inherits the frontend's mode. Defaulting to a
    concrete mode would make every worker override the frontend on upgrade.
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


def register_worker_router_help(parser, source_parser=None) -> None:
    """Surface the worker router flags in ``parser``'s ``--help``.

    The flags are parsed by a separate parser, so they would otherwise be
    invisible to ``--help``. Mirrors how the backends already display their
    engine's arguments.
    """
    if source_parser is None:
        source_parser = argparse.ArgumentParser(add_help=False)
        add_worker_router_arguments(source_parser)
    group = parser.add_argument_group(
        "Router Advertisement Options. Declared in this worker's model card to "
        "override the frontend's routing for this worker set only."
    )
    for action in source_parser._actions:
        if action.option_strings:
            group._group_actions.append(action)


def build_router_config(config) -> Optional["RouterConfig"]:
    """Build the ``RouterConfig`` a worker set advertises in its model card.

    Returns ``None`` when no mode was requested, which leaves ``router_config``
    off the card so the worker inherits the frontend's global configuration.
    That is the behavior of every deployment that does not set ``--router-mode``.

    Accepts anything carrying `RouterConfigBase` and `KvRouterConfigBase`
    fields, so the frontend can share it.
    """
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
