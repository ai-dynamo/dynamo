# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trait + types for the WeightTransferConfig API (vLLM-scoped, Phase 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol


PauseMode = Literal["keep", "wait", "abort"]
TransportState = Literal["configured", "ready", "receiving", "failed"]
TargetKind = Literal["base", "lora"]
LoraOp = Literal["load", "swap", "unload"]


@dataclass(frozen=True)
class WeightTarget:
    """What is being updated.

    * ``kind="base"``: the base model itself (full-FT reload).
    * ``kind="lora"``: a LoRA adapter; ``name`` is required and ``op`` selects
      between load/swap/unload.
    """

    kind: TargetKind
    name: Optional[str] = None
    op: Optional[LoraOp] = None

    @classmethod
    def from_dict(cls, body: dict) -> "WeightTarget":
        kind = body.get("kind")
        if kind not in ("base", "lora"):
            raise ValueError(
                f"WeightTarget.kind must be 'base' or 'lora', got {kind!r}"
            )
        if kind == "lora":
            name = body.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "WeightTarget.name is required when kind='lora'"
                )
            op = body.get("op")
            if op not in ("load", "swap", "unload"):
                raise ValueError(
                    f"WeightTarget.op must be 'load'|'swap'|'unload' when "
                    f"kind='lora', got {op!r}"
                )
            return cls(kind="lora", name=name, op=op)
        return cls(kind="base")


@dataclass
class UpdateWeightsRequest:
    """Single discriminated body for ``POST /v1/rl/update_weights``."""

    version: str
    target: WeightTarget
    transport: dict           # backend-specific block, validated by the transport impl
    pause_mode: PauseMode = "keep"
    clear_cache: bool = True

    @classmethod
    def from_dict(cls, body: dict) -> "UpdateWeightsRequest":
        version = body.get("version")
        if not isinstance(version, str) or not version:
            raise ValueError("update_weights: 'version' is required")
        target = WeightTarget.from_dict(body.get("target", {}) or {})
        transport = body.get("transport") or {}
        if target.kind == "base" or target.op != "unload":
            if not isinstance(transport, dict) or "backend" not in transport:
                raise ValueError(
                    "update_weights: 'transport.backend' is required "
                    "(except for lora unload)"
                )
        pause_mode = body.get("pause_mode", "keep")
        if pause_mode not in ("keep", "wait", "abort"):
            raise ValueError(
                f"update_weights: pause_mode must be 'keep'|'wait'|'abort', "
                f"got {pause_mode!r}"
            )
        clear_cache = bool(body.get("clear_cache", True))
        return cls(
            version=version,
            target=target,
            transport=transport,
            pause_mode=pause_mode,
            clear_cache=clear_cache,
        )


@dataclass
class InitCtx:
    """Constant context passed to every transport ``init`` call."""

    rank: int
    world_size: int
    served_model_name: str


@dataclass
class InitResult:
    status: str
    transport_id: str
    ready: bool
    message: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "status": self.status,
            "transport_id": self.transport_id,
            "ready": self.ready,
        }
        if self.message:
            out["message"] = self.message
        if self.extra:
            out.update(self.extra)
        return out


@dataclass
class UpdateResult:
    status: str
    message: str = ""
    version: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"status": self.status, "message": self.message}
        if self.version is not None:
            out["version"] = self.version
        if self.extra:
            out.update(self.extra)
        return out


class WeightTransport(Protocol):
    """One implementation per backend.

    Phase 1: ``FilesystemTransport``.
    Phase 4: ``NcclTransport``.
    """

    backend_id: str

    async def init(self, ctx: InitCtx, cfg: dict) -> InitResult: ...

    async def update_weights(
        self, req: UpdateWeightsRequest
    ) -> UpdateResult: ...

    async def teardown(self) -> None: ...

    @property
    def state(self) -> TransportState: ...


class EngineAdapter(Protocol):
    """Engine-flavor shim. One implementation per engine.

    Phase 1+4 ships :class:`VllmEngineAdapter` only. Future:
    ``SglangEngineAdapter`` drops in as one extra subclass without touching
    any :class:`WeightTransport` impl.
    """

    async def update_weights_from_disk(
        self, *, path: str, version: str, target: WeightTarget
    ) -> UpdateResult: ...

    async def update_weights_from_distributed(
        self,
        *,
        group: str,
        dtype: str,
        version: str,
        target: WeightTarget,
        weight_names: Optional[list[str]] = None,
    ) -> UpdateResult: ...

    async def update_weights_from_tensor(
        self, *, tensors: Any, version: str, target: WeightTarget
    ) -> UpdateResult: ...

    async def update_weights_from_ipc(
        self, *, handle: Any, version: str, target: WeightTarget
    ) -> UpdateResult: ...

    async def add_lora(self, *, name: str, source: str) -> UpdateResult: ...

    async def remove_lora(self, *, name: str) -> UpdateResult: ...
