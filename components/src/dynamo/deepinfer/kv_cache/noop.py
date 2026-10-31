# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No prefix reuse and no KV events: DeepInfer's own allocation behaviour."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Admission, CacheStats, KVCacheManager


class NoopCacheManager(KVCacheManager):
    def admit(self, seq: Any, token_ids: np.ndarray) -> Admission:
        # DeepInfer allocates the prompt when it schedules the prefill, behind
        # its 90% usage gate; allocating here would bypass that gate.
        self.engine.scheduler.add_sequence(seq)
        return Admission(admitted=True)

    def on_finished(self, seq: Any) -> None:
        self.engine.scheduler.free_seq(seq)

    def stats(self) -> CacheStats:
        total = int(self.engine.cache_config.num_gpu_blocks)
        # Block 0 is DeepInfer's reserved null block and is never free.
        free = int(self.engine.block_manager.get_num_free_blocks())
        return CacheStats(used_blocks=max(0, total - 1 - free), total_blocks=total)
