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
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def try_wrap_engine(vllm_config: Any, engine: Any) -> Optional[Any]:
    """Inspect `vllm_config` for the KVBM disagg-prefill + hub signal and,
    if present, build a `PrefillRouterHandler` around the live engine
    with both the dispatch and calibrate handlers registered.

    Returns the constructed handler, or `None` if this worker is not a
    prefill participant against a hub. Decode workers and aggregated
    deployments return `None`.

    The calibrate handler is best-effort: if framework introspection
    fails (vLLM internals shift across versions), we log and proceed
    with dispatch-only registration. Dispatch is the load-bearing path;
    calibration is a sibling capability.
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

    from kvbm.hub import (
        PrefillRouterHandler,
        capture_calibration_defaults,
        make_calibrate_lambda,
        make_dispatch_lambda,
    )

    loop = asyncio.get_running_loop()
    dispatch_lam = make_dispatch_lambda(engine, loop)

    calibrate_lam = None
    calibration_defaults = None
    try:
        model_id = (
            getattr(getattr(vllm_config, "model_config", None), "model", None) or ""
        )
        calibration_defaults = capture_calibration_defaults(engine, model_id)
        calibrate_lam = make_calibrate_lambda(engine, loop)
    except Exception as e:
        logger.warning(
            "try_wrap_engine: skipping calibrate handler — %s: %s",
            type(e).__name__,
            e,
        )

    return PrefillRouterHandler(
        dispatch_lam,
        hub_url,
        calibrate_lambda=calibrate_lam,
        calibration_defaults=calibration_defaults,
    )
