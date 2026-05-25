# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config-driven auto-detection for the prefill router handler.

Both `dynamo.vllm` and the standalone `python -m kvbm.vllm.prefill`
entrypoint call this. The signal is the rendered KVBM config the vLLM
process already carries inside its `kv_transfer_config`: when the
disagg role is `"prefill"` and a hub URL is set, the worker is meant
to participate in the prefill router and we wrap its engine.

The single source of truth for the field paths is `kvbm-hub`'s
`render.rs` `build_extra_config` / `authoritative_overlay` — both write
`leader.disagg.role` and `leader.hub.url`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional


def try_wrap_engine(vllm_config: Any, engine: Any) -> Optional[Any]:
    """Inspect `vllm_config` for the KVBM disagg-prefill + hub signal and,
    if present, build a `PrefillRouterHandler` around the live engine.

    Returns the constructed handler, or `None` if this worker is not a
    prefill participant against a hub. Decode workers and aggregated
    deployments return `None`.
    """
    extra = (
        getattr(
            getattr(vllm_config, "kv_transfer_config", None),
            "kv_connector_extra_config",
            None,
        )
        or {}
    )
    leader = extra.get("leader") or {}
    hub_url = (leader.get("hub") or {}).get("url")
    role = (leader.get("disagg") or {}).get("role")
    if not (hub_url and role == "prefill"):
        return None

    from kvbm.hub import PrefillRouterHandler, make_dispatch_lambda

    loop = asyncio.get_running_loop()
    lam = make_dispatch_lambda(engine, loop)
    return PrefillRouterHandler(lam, hub_url)
