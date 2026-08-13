#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Worker-side router advertisement.

A worker set may declare in its model deployment card how the frontend should
route to it, overriding the frontend's global ``--router-mode`` for that set
alone. Because the card is per worker type, this is the mechanism by which a
disaggregated deployment gives its prefill and decode tiers different routing
strategies -- for example a KV-routed prefill tier in front of a round-robin
decode tier.

Backends expose this as a ``--router-mode`` flag and pass the result to
``register_model(router_config=...)``. Omitting the flag leaves ``router_config``
off the card entirely, so the worker inherits the frontend's global mode.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dynamo.llm import RouterConfig

# CLI spelling -> `dynamo.llm.RouterMode` attribute name. Mirrors the frontend's
# --router-mode vocabulary so a worker advertises from the same set of modes.
ROUTER_MODE_MAP: dict[str, str] = {
    "round-robin": "RoundRobin",
    "random": "Random",
    "power-of-two": "PowerOfTwoChoices",
    "kv": "KV",
    "direct": "Direct",
    "least-loaded": "LeastLoaded",
    "device-aware-weighted": "DeviceAwareWeighted",
}

ROUTER_MODE_CHOICES: list[str] = sorted(ROUTER_MODE_MAP)

WORKER_ROUTER_MODE_HELP = (
    "Advertise a router mode in this worker's model deployment card, "
    "overriding the frontend's global --router-mode for this worker set only. "
    "Omit to inherit the frontend's mode. Set it separately on prefill and "
    "decode workers to give the two tiers different routing strategies, e.g. "
    "--router-mode kv on prefill in front of --router-mode round-robin decode. "
    "KV mode advertises default KV router tuning; per-worker KV tuning is not "
    "configurable from the worker side."
)


def build_worker_router_config(router_mode: Optional[str]) -> Optional["RouterConfig"]:
    """Build the ``RouterConfig`` a worker set advertises in its model card.

    Returns ``None`` when no mode was requested, which leaves ``router_config``
    off the card so the worker inherits the frontend's global mode. That is the
    behavior of every deployment that does not set the flag.
    """
    if router_mode is None:
        return None

    # Imported lazily so that merely importing a backend's argument definitions
    # does not pull in the compiled bindings.
    from dynamo.llm import KvRouterConfig, RouterConfig, RouterMode

    try:
        mode_attr = ROUTER_MODE_MAP[router_mode]
    except KeyError as error:
        raise ValueError(
            f"unknown router mode {router_mode!r}; expected one of "
            f"{', '.join(ROUTER_MODE_CHOICES)}"
        ) from error

    mode = getattr(RouterMode, mode_attr)
    # Only KV routing consults KvRouterConfig; passing it for other modes would
    # imply tuning that is never read.
    kv_router_config = KvRouterConfig() if mode == RouterMode.KV else None
    return RouterConfig(mode, kv_router_config)
