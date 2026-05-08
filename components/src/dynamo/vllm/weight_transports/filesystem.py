# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filesystem weight transport (Phase 1).

Equivalent of the existing ``update_weights_from_path`` route, but reachable
through the unified :class:`WeightTransport` Protocol so the same wire shape
covers full-FT and LoRA, and so future backends slot in alongside.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .base import (
    EngineAdapter,
    InitCtx,
    InitResult,
    TransportState,
    UpdateResult,
    UpdateWeightsRequest,
    WeightTransport,
)

logger = logging.getLogger(__name__)


class FilesystemTransport(WeightTransport):
    """Filesystem path → engine reload.

    Config (the ``"filesystem"`` block of an ``init_transport`` body or a
    ``transport.filesystem`` block of an ``update_weights`` body):

        path:           str   (required for base / lora-load / lora-swap)
        require_marker: str   (optional, default 'STABLE')
    """

    backend_id = "filesystem"

    def __init__(self, engine_adapter: EngineAdapter, cfg: dict):
        self._engine = engine_adapter
        self._cfg = cfg or {}
        self._state: TransportState = "configured"
        self._transport_id: str = self._cfg.get("transport_id", "filesystem")

    @property
    def state(self) -> TransportState:
        return self._state

    async def init(self, ctx: InitCtx, cfg: dict) -> InitResult:
        # No setup needed for filesystem — degenerate one-shot.
        self._cfg = {**self._cfg, **(cfg or {})}
        self._transport_id = self._cfg.get("transport_id", self._transport_id)
        self._state = "ready"
        return InitResult(
            status="ok",
            transport_id=self._transport_id,
            ready=True,
            message="filesystem transport ready (no setup required)",
        )

    async def teardown(self) -> None:
        self._state = "configured"

    async def update_weights(
        self, req: UpdateWeightsRequest
    ) -> UpdateResult:
        fs = req.transport.get("filesystem") or {}
        path: Optional[str] = fs.get("path")
        require_marker: Optional[str] = fs.get(
            "require_marker", self._cfg.get("require_marker", "STABLE")
        )

        # ---- LoRA unload: no transport, no path ----------------------------
        if req.target.kind == "lora" and req.target.op == "unload":
            return await self._engine.remove_lora(name=req.target.name)

        if not path:
            raise ValueError(
                "filesystem.update_weights: 'transport.filesystem.path' is "
                "required (except for lora unload)"
            )

        if require_marker:
            marker = os.path.join(path, require_marker)
            if not os.path.exists(marker):
                raise FileNotFoundError(
                    f"filesystem transport: require_marker '{require_marker}' "
                    f"not found under {path!r}"
                )

        if req.target.kind == "base":
            self._state = "receiving"
            try:
                result = await self._engine.update_weights_from_disk(
                    path=path, version=req.version, target=req.target
                )
            finally:
                self._state = "ready"
            logger.info(
                f"[RL] filesystem.update_weights: base reload from {path} "
                f"(version={req.version})"
            )
            return result

        # target.kind == "lora", op in {load, swap}
        result = await self._engine.add_lora(name=req.target.name, source=path)
        logger.info(
            f"[RL] filesystem.update_weights: lora {req.target.op} "
            f"name={req.target.name} from {path}"
        )
        return result
