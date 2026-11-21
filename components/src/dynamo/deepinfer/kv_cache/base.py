# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable per-rank KV cache manager over DeepInfer's block allocator.

One manager per DP rank, owned by that rank's ``DynamoLLMEngine``. Every
method runs on the thread that owns the engine. Rank worker processes import
this module, so it must not import ``dynamo._core``, torch or DeepInfer.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Admission:
    """Result of :meth:`KVCacheManager.admit`."""

    # False means LATER: no room yet; the rank retries on a later iteration.
    admitted: bool


@dataclass
class CacheStats:
    used_blocks: int
    total_blocks: int


class KVCacheManager(ABC):
    """Admission, retention and eviction of KV for one DeepInfer engine.

    Constructed with the engine (a ``DynamoLLMEngine``) after DeepInfer built it;
    use ``engine.scheduler``, ``engine.block_manager``, ``engine.state_manager``,
    ``engine.cache_config`` and ``engine.driver_worker.cache_engine``.
    """

    def __init__(self, engine: Any, **options: Any) -> None:
        self.engine = engine

    @abstractmethod
    def admit(self, seq: Any, token_ids: np.ndarray) -> Admission:
        """Called after ``add_request(enqueue=False)``. Hand the seq to the
        scheduler, or return ``Admission(False)`` to be retried later.

        Must not allocate the prompt: DeepInfer allocates when it schedules
        the prefill, behind its 90% usage gate."""

    @abstractmethod
    def on_finished(self, seq: Any) -> None:
        """``LLMEngine.finalize_seq`` hook: retain what is cacheable, free the rest."""

    @abstractmethod
    def stats(self) -> CacheStats:
        ...


# name -> (module relative to this package, class), imported lazily.
_MANAGERS = {
    "none": (".noop", "NoopCacheManager"),
}


def make_cache_manager(name: str, engine: Any, **options: Any) -> KVCacheManager:
    try:
        module_name, cls_name = _MANAGERS[name]
    except KeyError:
        raise ValueError(
            f"unknown kv cache manager {name!r}; expected one of {sorted(_MANAGERS)}"
        ) from None
    module = importlib.import_module(module_name, __package__)
    return getattr(module, cls_name)(engine, **options)
