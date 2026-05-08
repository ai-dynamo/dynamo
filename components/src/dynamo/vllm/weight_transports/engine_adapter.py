# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM-flavor engine adapter.

Wraps ``engine_client.collective_rpc(...)`` so each :class:`WeightTransport`
implementation can call a stable, engine-agnostic API. Future:
``SglangEngineAdapter`` will wrap ``tokenizer_manager.update_weights_from_*``
following the same Protocol.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import EngineAdapter, UpdateResult, WeightTarget

logger = logging.getLogger(__name__)


class VllmEngineAdapter(EngineAdapter):
    """vLLM-flavor :class:`EngineAdapter` backed by an ``engine_client``.

    All four ``update_weights_from_*`` paths route through ``collective_rpc``
    against the in-process worker(s); LoRA ops route through the engine's
    ``add_lora`` / ``remove_lora`` (or equivalent collective) calls.
    """

    backend_id = "vllm"

    def __init__(self, engine_client, *, lora_loader=None):
        self.engine_client = engine_client
        self._lora_loader = lora_loader  # optional callable for LoRA add path

    # ---- four canonical update paths ---------------------------------------

    async def update_weights_from_disk(
        self, *, path: str, version: str, target: WeightTarget
    ) -> UpdateResult:
        await self.engine_client.collective_rpc(
            "reload_weights",
            kwargs={"weights_path": path},
        )
        return UpdateResult(
            status="ok",
            message=f"Weights loaded from {path}",
            version=version,
        )

    async def update_weights_from_distributed(
        self,
        *,
        group: str,
        dtype: str,
        version: str,
        target: WeightTarget,
        weight_names: Optional[list[str]] = None,
    ) -> UpdateResult:
        # vLLM exposes per-name distributed update via the worker's
        # `update_weight_from_tensor` / `update_weight` collective. We loop
        # over weight_names so the trainer can drive the broadcast iteration.
        if not weight_names:
            raise ValueError(
                "update_weights_from_distributed: weight_names is required so "
                "the worker knows which named parameters to receive on the "
                "NCCL group."
            )
        for name in weight_names:
            await self.engine_client.collective_rpc(
                "update_weight",
                kwargs={"name": name, "dtype": dtype, "shape": None},
            )
        return UpdateResult(
            status="ok",
            message=f"Updated {len(weight_names)} weights via group '{group}'",
            version=version,
            extra={"weights_received": len(weight_names)},
        )

    async def update_weights_from_tensor(
        self, *, tensors: Any, version: str, target: WeightTarget
    ) -> UpdateResult:
        # Future hook for NIXL/MX paths (deferred).
        raise NotImplementedError(
            "update_weights_from_tensor is reserved for NIXL/ModelExpress "
            "transports; not implemented in Phase 1+4."
        )

    async def update_weights_from_ipc(
        self, *, handle: Any, version: str, target: WeightTarget
    ) -> UpdateResult:
        raise NotImplementedError(
            "update_weights_from_ipc is reserved for the colocated-trainer "
            "path; not implemented in Phase 1+4."
        )

    # ---- LoRA ops ----------------------------------------------------------

    async def add_lora(self, *, name: str, source: str) -> UpdateResult:
        if self._lora_loader is None:
            raise RuntimeError(
                "VllmEngineAdapter.add_lora called but no lora_loader was "
                "supplied at construction. Wire it from the handler."
            )
        result = await self._lora_loader(name=name, path=source)
        return UpdateResult(
            status=result.get("status", "ok"),
            message=result.get("message", ""),
            extra={k: v for k, v in result.items() if k not in ("status", "message")},
        )

    async def remove_lora(self, *, name: str) -> UpdateResult:
        if self._lora_loader is None:
            raise RuntimeError(
                "VllmEngineAdapter.remove_lora called but no lora_loader was "
                "supplied at construction. Wire it from the handler."
            )
        # The handler exposes both load and unload via the same `lora_loader`
        # callable, dispatched on a sentinel ``op`` field. We use the same
        # convention: invoke the unload helper if available; otherwise fall
        # through and let the caller handle.
        unloader = getattr(self, "_lora_unloader", None)
        if unloader is None:
            raise RuntimeError(
                "VllmEngineAdapter.remove_lora called but no lora_unloader "
                "was supplied. Wire it from the handler."
            )
        result = await unloader(name=name)
        return UpdateResult(
            status=result.get("status", "ok"),
            message=result.get("message", ""),
            extra={k: v for k, v in result.items() if k not in ("status", "message")},
        )

    # Convenience: handler wires both helpers in one shot.
    def bind_lora_helpers(self, *, loader, unloader):
        self._lora_loader = loader
        self._lora_unloader = unloader
