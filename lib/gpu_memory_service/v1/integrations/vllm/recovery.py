# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TP1 prefix recovery; native vLLM owns lookup, refcounts, and eviction."""

import logging
import os
from functools import wraps
from pathlib import Path

from gpu_memory_service.common.persistent_pool import PersistentAllocation
from gpu_memory_service.kv_recovery.backends.mmap import MmapKVRecoveryManager
from gpu_memory_service.kv_recovery.types import (
    KVBlockRecord,
    KVRecoveryManifest,
    KVTensorLayout,
    RecoveryResult,
)

logger = logging.getLogger(__name__)


class VllmKVRecoveryAdapter:
    def __init__(self, core):
        import vllm
        from vllm.v1.executor.uniproc_executor import UniProcExecutor

        config = core.vllm_config
        parallel = config.parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or core.async_scheduling
            or core.batch_queue is not None
            or core.is_pooling_model
            or config.speculative_config is not None
            or config.kv_transfer_config is not None
            or getattr(config, "lora_config", None) is not None
            or not isinstance(core.model_executor, UniProcExecutor)
        ):
            raise ValueError(
                "KV recovery requires TP/PP/DP1, sync scheduling and generation"
            )
        pool = core.scheduler.kv_cache_manager.block_pool
        if (
            not pool.enable_caching
            or len(core.scheduler.kv_cache_config.kv_cache_groups) != 1
        ):
            raise ValueError(
                "KV recovery requires prefix caching and one KV cache group"
            )
        self.core = core
        self.pool = pool
        self.session = None
        self.pending = {}
        self.version = vllm.__version__
        self.block_size = core.scheduler.kv_cache_config.kv_cache_groups[
            0
        ].kv_cache_spec.block_size
        control = Path(os.environ.get("GMS_SOCKET_DIR", "/tmp"))
        self.manager = MmapKVRecoveryManager(
            os.environ.get("DYN_KV_RECOVERY_PATH", str(control / "kv-recovery.mmap"))
        )
        pool.cache_observer = self
        core.set_cache_recovery_hooks(self.before_resume, self.after_model_output)

    def before_resume(self):
        if self.session is not None:
            return
        states = self.core.model_executor.collective_rpc("get_kv_recovery_state")
        if len(states) != 1 or not states[0]["tensors"]:
            raise RuntimeError("KV recovery requires complete attached rank state")
        state = states[0]
        allocations = tuple(PersistentAllocation(**a) for a in state["allocations"])
        tensors = tuple(KVTensorLayout(**t) for t in state["tensors"])
        manifest = KVRecoveryManifest(
            self.core.vllm_config.model_config.model,
            self.pool.num_gpu_blocks,
            self.block_size,
            str(self.core.vllm_config.cache_config.prefix_caching_hash_algo),
            tensors,
            self.version,
        )
        session = self.manager.recover(allocations, manifest)
        records = session.blocks()
        if not state["backing_recovered"] and records:
            session.clear()
            records = ()
            session.result = RecoveryResult.FRESH
            session.reason = "new physical backing"
        if any(
            int.from_bytes(b.block_hash[-4:], "big") != 0
            or b.num_tokens % self.block_size
            for b in records
        ):
            session.clear()
            records = ()
            session.result = RecoveryResult.FRESH
            session.reason = "invalid full-block inventory"
        # Bulk restore suppresses mirroring; only completed records are replayed.
        try:
            self.pool.restore_prefix_index(
                [(b.block_id, b.block_hash, b.num_tokens) for b in records]
            )
        except (ValueError, AssertionError):
            observer, self.pool.cache_observer = self.pool.cache_observer, None
            try:
                if not self.pool.reset_prefix_cache():
                    raise RuntimeError("cannot reset partial prefix recovery")
            finally:
                self.pool.cache_observer = observer
            session.clear()
            session.result = RecoveryResult.FRESH
            session.reason = "native restore rejected records"
            records = ()
        self.pending.clear()
        self.session = session
        logger.info(
            "KV_RECOVERY_READY result=%s blocks=%d reason=%s epoch=%d",
            session.result.value,
            len(records),
            session.reason,
            session.epoch,
        )

    def before_blocks_reused(self, block_ids):
        if self.session is not None:
            self.session.invalidate_blocks(block_ids)
        for block_id in block_ids:
            self.pending.pop(block_id, None)

    def on_block_cached(self, block_id, key, num_tokens):
        if (
            self.session is not None
            and num_tokens is not None
            and num_tokens % self.block_size == 0
        ):
            self.pending[block_id] = KVBlockRecord(block_id, key, num_tokens)

    def before_cache_reset(self):
        if self.session is not None:
            self.session.clear()
        self.pending.clear()

    def after_model_output(self, scheduler_output, model_output):
        if self.session is None or not self.pending:
            return
        records = [
            b
            for b in self.pending.values()
            if self.pool.blocks[b.block_id].block_hash == b.block_hash
        ]
        # First implementation deliberately uses a device barrier. The callback
        # marks host scheduling completion; publication also requires proof that
        # all GPU KV writes are finished. TP1/UniProc makes that proof local.
        self.core.model_executor.collective_rpc("synchronize_device")
        self.session.publish_blocks(records)
        self.pending.clear()


def register():
    if os.environ.get("DYN_KV_RECOVERY") != "true":
        return
    if os.environ.get("DYN_GMS_USE_V1") != "true":
        raise RuntimeError("KV recovery requires GMS V1")
    import vllm
    from gpu_memory_service.v1.integrations.vllm.recovery_compat import (
        install_block_pool_hooks,
        install_engine_hooks,
    )
    from vllm.v1.engine.core import EngineCore

    if vllm.__version__ != "0.30.0":
        raise RuntimeError("KV recovery compatibility is qualified for vLLM 0.30.0")
    install_block_pool_hooks()
    install_engine_hooks()
    if getattr(EngineCore, "_gms_kv_recovery_installed", False):
        return

    # vLLM can first load a plugin *inside* an already-running EngineCore
    # constructor. Attach at the sleep/resume boundaries, when the scheduler
    # exists, rather than relying on wrapping a constructor already entered.
    def ensure_adapter(core):
        if not hasattr(core, "_gms_kv_recovery_adapter"):
            core._gms_kv_recovery_adapter = VllmKVRecoveryAdapter(core)

    original_sleep = EngineCore.sleep
    original_resume = EngineCore.resume_scheduler

    @wraps(original_sleep)
    def sleep(self, *args, **kwargs):
        ensure_adapter(self)
        return original_sleep(self, *args, **kwargs)

    @wraps(original_resume)
    def resume(self):
        ensure_adapter(self)
        return original_resume(self)

    EngineCore.sleep = sleep
    EngineCore.resume_scheduler = resume
    EngineCore._gms_kv_recovery_installed = True
