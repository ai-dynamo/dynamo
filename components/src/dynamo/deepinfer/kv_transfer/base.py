# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefill -> decode KV transfer, one instance per DP rank (pull model).

The decode rank READs the prefill rank's KV block rows and its state slot,
then notifies the prefill side, which unpins them. Rank worker processes import
this module, so it must not import ``dynamo._core``, torch or DeepInfer.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from dynamo.deepinfer.disagg import DeepInferHandoff


@dataclass
class KVRegion:
    """One registered memory region: ``num_items`` rows of ``item_bytes``."""

    name: str  # "gpu_blocks" or "gpu_states"
    base_ptr: int
    num_items: int
    item_bytes: int
    # CUDA device ordinal, or -1 for host memory (the CPU fake).
    device_id: int
    # The tensor/array itself; LocalKVTransfer copies through it.
    buffer: Any = field(default=None, repr=False, compare=False)


@dataclass
class KVLayout:
    regions: list[KVRegion]
    # Covers model, block bytes, state layout, nextn, backtracking, YOCO and the
    # DeepInfer version; prefill and decode must match exactly.
    layout_hash: str

    def region(self, name: str) -> KVRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)


class KVTransfer(ABC):
    """Called only from the thread that owns the rank's engine."""

    def __init__(self, *, rank: int, engine_id: str, **options: Any) -> None:
        self.rank = rank
        self.engine_id = engine_id

    @abstractmethod
    def register(self, layout: KVLayout) -> None:
        """Register this rank's regions once, after the engine allocated them."""

    @abstractmethod
    def agent_info(self) -> dict[str, str]:
        """``{"agent_name": ..., "agent_metadata": <base64>}`` for the handoff."""

    @abstractmethod
    def start_read(
        self,
        req_id: str,
        handoff: DeepInferHandoff,
        local_block_ids: list[int],
        local_state_id: int,
    ) -> None:
        """Decode side: start pulling ``handoff``'s blocks and state slot.

        Raises when the read cannot start (unknown peer, layout mismatch);
        once it returns, ``poll`` reports ``req_id`` as done or failed.
        """

    @abstractmethod
    def poll(self) -> tuple[list[str], list[str], list[str]]:
        """Return ``(done, failed, released)`` request ids since the last poll.

        ``done``/``failed`` are decode-side reads; ``released`` are prefill-side
        requests whose KV the peer finished reading.
        """

    @abstractmethod
    def release_remote(self, handoff: DeepInferHandoff) -> None:
        """Decode side: tell the prefill side it may free ``handoff``'s KV.

        Call it once per handoff when its read finished (either way) or will
        never start. A completed read already notified the peer, so for those
        this is a no-op. May raise; callers treat it as best effort (the
        prefill lease frees the KV eventually).
        """

    @abstractmethod
    def shutdown(self) -> None:
        ...


_TRANSFERS = {
    "nixl": (".nixl", "NixlKVTransfer"),
    "local": (".local", "LocalKVTransfer"),
}


def make_kv_transfer(
    name: str, *, rank: int, engine_id: str, **options: Any
) -> Optional[KVTransfer]:
    """``"none"`` returns None; other backends are imported lazily."""
    if name == "none":
        return None
    try:
        module_name, cls_name = _TRANSFERS[name]
    except KeyError:
        raise ValueError(
            f"unknown kv transfer {name!r}; expected none or one of {sorted(_TRANSFERS)}"
        ) from None
    module = importlib.import_module(module_name, __package__)
    return getattr(module, cls_name)(rank=rank, engine_id=engine_id, **options)
