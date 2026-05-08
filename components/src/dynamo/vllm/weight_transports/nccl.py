# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NCCL weight transport (Phase 4).

Trainer + dynamo.vllm worker(s) form a NCCL process group at
``init_transport`` time; per-step ``update_weights`` triggers receive via
``collective_rpc("update_weight", ...)`` for each named parameter.

Phase 4 scope: vLLM only. The trainer side is responsible for driving the
broadcast itself; dynamo just exposes the receiver hook.
"""

from __future__ import annotations

import logging
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


class NcclTransport(WeightTransport):
    """NCCL collective broadcast → engine receive.

    Config (the ``"nccl"`` block of an ``init_transport`` body or a
    ``transport.nccl`` block of an ``update_weights`` body):

        group_name:           str   (required)
        init_method:          str   (e.g. "tcp://trainer:29500", required at init)
        trainer_world_size:   int   (required at init)
        inference_world_size: int   (required at init; usually == # workers)
        dtype:                str   (e.g. "bf16")

    For ``update_weights``:

        weight_names:         list[str]  (the iteration order of named params
                                          the trainer is broadcasting; required)
    """

    backend_id = "nccl"

    def __init__(self, engine_adapter: EngineAdapter, cfg: dict):
        self._engine = engine_adapter
        self._cfg = cfg or {}
        self._state: TransportState = "configured"
        self._transport_id: str = self._cfg.get("transport_id", "nccl")

    @property
    def state(self) -> TransportState:
        return self._state

    async def init(self, ctx: InitCtx, cfg: dict) -> InitResult:
        cfg = cfg or {}
        merged = {**self._cfg, **cfg}
        # vLLM's init_weight_transfer_engine takes:
        #   master_address, master_port, rank_offset, world_size
        # The trainer is rank 0; inference workers are rank_offset..world_size-1.
        for required in ("master_address", "master_port", "world_size"):
            if required not in merged:
                raise ValueError(
                    f"nccl transport: '{required}' is required in init_transport"
                )

        self._cfg = merged
        self._transport_id = merged.get("transport_id", self._transport_id)

        # Drive the worker-side bootstrap via vLLM's
        # `init_weight_transfer_engine` collective.
        try:
            init_info = {
                "master_address": str(merged["master_address"]),
                "master_port": int(merged["master_port"]),
                "rank_offset": int(merged.get("rank_offset", 1)),
                "world_size": int(merged["world_size"]),
            }
            await self._engine.engine_client.collective_rpc(
                "init_weight_transfer_engine",
                kwargs={"init_info": init_info},
            )
            self._state = "ready"
            return InitResult(
                status="ok",
                transport_id=self._transport_id,
                ready=True,
                message=(
                    f"nccl init_weight_transfer_engine ok "
                    f"(master={init_info['master_address']}:{init_info['master_port']}, "
                    f"world_size={init_info['world_size']})"
                ),
                extra={"init_info": init_info},
            )
        except Exception as exc:
            self._state = "failed"
            logger.error(f"[RL] nccl.init failed: {exc}")
            raise

    async def teardown(self) -> None:
        # vLLM doesn't expose an explicit destroy hook; engine teardown handles it.
        self._state = "configured"

    async def update_weights(
        self, req: UpdateWeightsRequest
    ) -> UpdateResult:
        # NCCL transport does not own LoRA hot-swap in this iteration; LoRA
        # adapters are tiny enough that filesystem stays the better path.
        if req.target.kind == "lora":
            raise NotImplementedError(
                "nccl transport: LoRA adapter transfer is deferred. Use "
                "transport.backend='filesystem' for LoRA in this iteration."
            )

        nccl = req.transport.get("nccl") or {}
        # The trainer must supply (names, dtype_names, shapes) so the worker
        # knows how big each `torch.empty(...)` receive buffer should be.
        names: Optional[list[str]] = nccl.get("names") or nccl.get("weight_names")
        dtype_names: Optional[list[str]] = nccl.get("dtype_names")
        shapes: Optional[list[list[int]]] = nccl.get("shapes")
        if not names:
            raise ValueError(
                "nccl.update_weights: 'transport.nccl.names' is required"
            )
        if not dtype_names or not shapes:
            raise ValueError(
                "nccl.update_weights: 'transport.nccl.dtype_names' and "
                "'transport.nccl.shapes' are required"
            )
        if len(dtype_names) != len(names) or len(shapes) != len(names):
            raise ValueError(
                f"nccl.update_weights: names/dtype_names/shapes length mismatch "
                f"({len(names)} / {len(dtype_names)} / {len(shapes)})"
            )

        update_info = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "is_checkpoint_format": bool(nccl.get("is_checkpoint_format", True)),
            "packed": bool(nccl.get("packed", False)),
        }

        self._state = "receiving"
        try:
            await self._engine.engine_client.collective_rpc(
                "update_weights",
                kwargs={"update_info": update_info},
            )
        finally:
            self._state = "ready"
        logger.info(
            f"[RL] nccl.update_weights: {len(names)} weights received "
            f"(version={req.version})"
        )
        return UpdateResult(
            status="ok",
            message=f"Updated {len(names)} weights via nccl",
            version=req.version,
            extra={"weights_received": len(names)},
        )
