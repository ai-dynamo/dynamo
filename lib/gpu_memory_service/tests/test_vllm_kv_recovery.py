# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native cache behavior, publication ordering, and the activation gate."""

from dataclasses import asdict
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm")

from gpu_memory_service.common.persistent_pool import PersistentAllocation
from gpu_memory_service.kv_recovery.types import KVRecoveryManifest, KVTensorLayout
from gpu_memory_service.v1.integrations.vllm.recovery import VllmKVRecoveryAdapter
from gpu_memory_service.v1.integrations.vllm.recovery_compat import (
    install_block_pool_hooks,
    install_engine_hooks,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.uniproc_executor import UniProcExecutor

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.fault_tolerance,
]

_KEY = BlockHashWithGroupId(b"x" * 32 + b"\0" * 4)


@pytest.fixture
def core(tmp_path, monkeypatch):
    install_block_pool_hooks()
    install_engine_hooks()
    monkeypatch.setenv("DYN_KV_RECOVERY_PATH", str(tmp_path / "metadata"))
    state = {
        "allocations": [asdict(PersistentAllocation("kv", 4096, "server", "GPU-0"))],
        "tensors": [
            asdict(KVTensorLayout("kv-0", "kv", 0, (8, 16), (16, 1), "float16"))
        ],
        "backing_recovered": True,
    }
    pool = BlockPool(8, True, 16)
    scheduler = NS(
        kv_cache_manager=NS(block_pool=pool),
        kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=NS(block_size=16))]),
        update_from_output=Mock(return_value={}),
        set_pause_state=Mock(),
    )
    config = NS(
        parallel_config=NS(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
        model_config=NS(model="model"),
        speculative_config=None,
        cache_config=NS(prefix_caching_hash_algo="sha256"),
        kv_transfer_config=None,
    )
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.scheduler = scheduler
    engine.model_executor = Mock(spec=UniProcExecutor)
    engine.model_executor.collective_rpc.return_value = [state]
    engine.async_scheduling = False
    engine.batch_queue = None
    engine.is_pooling_model = False
    return engine


def test_only_model_completed_blocks_are_recovered_into_native_pool(core):
    first = VllmKVRecoveryAdapter(core)
    first.before_resume()
    block = first.pool.get_new_blocks(1)[0]
    first.pool._insert_block_hash(_KEY, block, num_tokens=16)
    first.pool.free_blocks([block])
    # Candidate discovery alone must not publish uncompleted GPU writes.
    cold = first.manager.recover(
        [PersistentAllocation("kv", 4096, "server", "GPU-0")],
        KVRecoveryManifest(
            "model",
            8,
            16,
            "sha256",
            (KVTensorLayout("kv-0", "kv", 0, (8, 16), (16, 1), "float16"),),
            "0.30.0",
        ),
    )
    assert cold.blocks() == ()
    cold.close()
    first.session.close()
    # Start a fresh engine against the same metadata and publish after output.
    core.scheduler.kv_cache_manager.block_pool = BlockPool(8, True, 16)
    primary = VllmKVRecoveryAdapter(core)
    primary.before_resume()
    block = primary.pool.get_new_blocks(1)[0]
    primary.pool._insert_block_hash(_KEY, block, num_tokens=16)
    primary.pool.free_blocks([block])
    primary.after_model_output(None, None)
    primary.session.close()
    core.scheduler.kv_cache_manager.block_pool = BlockPool(8, True, 16)
    replacement = VllmKVRecoveryAdapter(core)
    try:
        replacement.before_resume()
        found = replacement.pool.cached_block_hash_to_block.get_one_block(_KEY)
        assert found is not None and found.block_id == block.block_id
        assert found.ref_cnt == 0
        assert replacement.pool.get_num_free_blocks() == 7
        # Uncached blocks are reused before the recovered eviction candidate.
        allocated = replacement.pool.get_new_blocks(6)
        assert found not in allocated
        assert replacement.pool.get_new_blocks(1) == [found]
    finally:
        replacement.session.close()


def test_failed_invalidation_prevents_handing_a_block_to_executor(core, monkeypatch):
    adapter = VllmKVRecoveryAdapter(core)
    adapter.before_resume()
    monkeypatch.setattr(
        adapter.session,
        "invalidate_blocks",
        Mock(side_effect=OSError("metadata unavailable")),
    )
    try:
        with pytest.raises(OSError, match="metadata unavailable"):
            adapter.pool.get_new_blocks(1)
    finally:
        adapter.session.close()


def test_recovery_failure_keeps_scheduler_paused(core):
    VllmKVRecoveryAdapter(core)
    core.model_executor.collective_rpc.side_effect = RuntimeError("missing backing")
    with pytest.raises(RuntimeError, match="missing backing"):
        core.resume_scheduler()
    core.scheduler.set_pause_state.assert_not_called()


def test_async_scheduling_is_rejected(core):
    core.async_scheduling = True
    with pytest.raises(ValueError, match="sync scheduling"):
        VllmKVRecoveryAdapter(core)


def test_failed_gpu_completion_does_not_publish_and_semantic_reset_clears_records(core):
    adapter = VllmKVRecoveryAdapter(core)
    adapter.before_resume()
    try:
        block = adapter.pool.get_new_blocks(1)[0]
        adapter.pool._insert_block_hash(_KEY, block, num_tokens=16)
        adapter.pool.free_blocks([block])
        core.model_executor.collective_rpc.side_effect = RuntimeError("GPU sync failed")
        with pytest.raises(RuntimeError, match="GPU sync failed"):
            adapter.after_model_output(None, None)
        assert adapter.session._read_blocks() == ()
        core.model_executor.collective_rpc.side_effect = None
        adapter.after_model_output(None, None)
        assert len(adapter.session._read_blocks()) == 1
        assert adapter.pool.reset_prefix_cache()
        assert adapter.session._read_blocks() == ()
    finally:
        adapter.session.close()
