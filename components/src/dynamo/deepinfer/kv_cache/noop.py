# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No prefix reuse and no KV events: DeepInfer's own allocation behaviour."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Admission, CacheStats, HeldKV, KVCacheManager


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

    def hold_for_transfer(self, seq: Any) -> HeldKV:
        engine = self.engine
        block_ids = list(seq.block_table)
        state_id = int(seq.state_id)
        engine.block_manager.pin_blocks(block_ids)
        if engine.state_manager is not None and state_id >= 0:
            engine.state_manager.pin_blocks([state_id])
        rid = seq.request_id
        engine.scheduler.free_seq(seq)
        engine.scheduler.running = [
            s for s in engine.scheduler.running if s.request_id != rid
        ]
        return HeldKV(
            request_id=rid,
            block_ids=block_ids,
            state_id=state_id,
            num_tokens=int(seq.get_prompt_len()),
            first_token=int(seq.last_output_token),
        )

    def release_hold(self, held: HeldKV) -> None:
        self.engine.block_manager.unpin_blocks(held.block_ids)
        if self.engine.state_manager is not None and held.state_id >= 0:
            self.engine.state_manager.unpin_blocks([held.state_id])
