# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Small compatibility hooks until vLLM ships the equivalent native APIs."""

from functools import wraps


def install_block_pool_hooks():
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

    if hasattr(BlockPool, "restore_prefix_index"):
        return

    def restore_prefix_index(self, records):
        if not self.enable_caching or self.cached_block_hash_to_block:
            raise ValueError("prefix restore requires an empty caching-enabled pool")
        seen = set()
        for block_id, key, num_tokens in records:
            if (
                not 0 < block_id < self.num_gpu_blocks
                or block_id in seen
                or len(key) <= 4
                or num_tokens <= 0
                or self.blocks[block_id].ref_cnt != 0
            ):
                raise ValueError("invalid restored prefix block")
            seen.add(block_id)
        observer, self.cache_observer = getattr(self, "cache_observer", None), None
        try:
            restored = []
            for block_id, key, num_tokens in records:
                block = self.blocks[block_id]
                self._insert_block_hash(
                    BlockHashWithGroupId(key), block, num_tokens=num_tokens
                )
                self.free_block_queue.remove(block)
                restored.append(block)
            self.free_block_queue.append_n(restored)
        finally:
            self.cache_observer = observer

    original_get = BlockPool.get_new_blocks
    original_remove = BlockPool._remove_cached_block_hashes
    original_insert = BlockPool._insert_block_hash
    original_reset = BlockPool.reset_prefix_cache

    @wraps(original_get)
    def get_new_blocks(self, count):
        if count > self.get_num_free_blocks():
            return original_get(self, count)
        # The native method's hash-removal callback invalidates cached blocks.
        # Invalidate all handed-out IDs before the caller can submit GPU work.
        blocks = original_get(self, count)
        if (observer := getattr(self, "cache_observer", None)) is not None:
            observer.before_blocks_reused([b.block_id for b in blocks])
        return blocks

    @wraps(original_remove)
    def remove(self, block):
        if (observer := getattr(self, "cache_observer", None)) is not None and (
            block.block_hash is not None
            or block.block_id in self.cached_block_hashes_by_block
        ):
            observer.before_blocks_reused([block.block_id])
        return original_remove(self, block)

    @wraps(original_insert)
    def insert(self, key, block, num_tokens):
        already_cached = self.cached_block_hash_to_block.contain(key, block.block_id)
        original_insert(self, key, block, num_tokens)
        if (
            not already_cached
            and (observer := getattr(self, "cache_observer", None)) is not None
        ):
            observer.on_block_cached(block.block_id, bytes(key), num_tokens)

    @wraps(original_reset)
    def reset(self):
        if (
            self.num_gpu_blocks - self.get_num_free_blocks() == 1
            and (observer := getattr(self, "cache_observer", None)) is not None
        ):
            observer.before_cache_reset()
        return original_reset(self)

    BlockPool.restore_prefix_index = restore_prefix_index
    BlockPool.get_new_blocks = get_new_blocks
    BlockPool._remove_cached_block_hashes = remove
    BlockPool._insert_block_hash = insert
    BlockPool.reset_prefix_cache = reset


def install_engine_hooks():
    from vllm.v1.engine.core import EngineCore

    if hasattr(EngineCore, "set_cache_recovery_hooks"):
        return
    original_resume = EngineCore.resume_scheduler

    def set_cache_recovery_hooks(self, before_resume, after_model_output):
        self._before_cache_resume = before_resume
        self._after_model_output = after_model_output
        if not getattr(self, "_cache_output_hook_installed", False):
            original_update = self.scheduler.update_from_output

            @wraps(original_update)
            def update(scheduler_output, model_output):
                output = original_update(scheduler_output, model_output)
                if self._after_model_output is not None:
                    self._after_model_output(scheduler_output, model_output)
                return output

            self.scheduler.update_from_output = update
            self._cache_output_hook_installed = True

    @wraps(original_resume)
    def resume(self):
        if (hook := getattr(self, "_before_cache_resume", None)) is not None:
            hook()
        return original_resume(self)

    EngineCore.set_cache_recovery_hooks = set_cache_recovery_hooks
    EngineCore.resume_scheduler = resume
