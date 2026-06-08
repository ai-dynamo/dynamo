# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import itertools
import sys
import types
from types import SimpleNamespace

import pytest
from gpu_memory_service.integrations.common.kv_lease_client import KVLease
from gpu_memory_service.server.kv_leases import KVLeaseManager

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_kv_lease_pressure_logs_are_rate_limited(monkeypatch, caplog):
    import logging

    from gpu_memory_service.integrations.common import kv_lease_client

    kv_lease_client._LEASE_PRESSURE_LOG_STATE.clear()
    monkeypatch.setenv("GMS_KV_LEASE_PRESSURE_LOG_INTERVAL_S", "60")
    test_logger = logging.getLogger("test.gms.kv_lease_pressure")

    with caplog.at_level(logging.WARNING, logger=test_logger.name):
        kv_lease_client.log_lease_pressure(
            test_logger,
            "same-pressure-key",
            "lease pressure observed",
            requested=2,
        )
        kv_lease_client.log_lease_pressure(
            test_logger,
            "same-pressure-key",
            "lease pressure observed",
            requested=2,
        )

    records = [r for r in caplog.records if r.name == test_logger.name]
    assert len(records) == 1
    assert "requested=2" in records[0].getMessage()


class _InMemoryLeaseClient:
    def __init__(
        self,
        manager: KVLeaseManager,
        namespace: str,
        owner_id: str,
        total_blocks: int,
        reserved_blocks: list[int] | None = None,
    ) -> None:
        self.manager = manager
        self.namespace = namespace
        self.owner_id = owner_id
        self.manager.init_namespace(
            namespace, total_blocks, reserved_blocks=reserved_blocks or []
        )

    def acquire(
        self,
        count: int,
        *,
        preferred_blocks: list[int] | None = None,
        allow_partial: bool = False,
        strict_preferred: bool = False,
    ) -> list[KVLease]:
        records = self.manager.acquire(
            self.namespace,
            self.owner_id,
            count,
            preferred_blocks=preferred_blocks or [],
            allow_partial=allow_partial,
            strict_preferred=strict_preferred,
        )
        return [KVLease(r.block_id, r.generation, r.lease_epoch) for r in records]

    def seal(self, leases: list[KVLease]) -> None:
        if not leases:
            return
        self.manager.seal(
            self.namespace,
            self.owner_id,
            [lease.block_id for lease in leases],
            [lease.generation for lease in leases],
        )

    def release(self, leases: list[KVLease]) -> None:
        if not leases:
            return
        self.manager.release(
            self.namespace,
            self.owner_id,
            [lease.block_id for lease in leases],
            [lease.generation for lease in leases],
        )

    def free_count(self) -> int:
        return self.manager.count_free(self.namespace)


def _set_module(monkeypatch, name: str, module: types.ModuleType) -> None:
    monkeypatch.setitem(sys.modules, name, module)


def _install_fake_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    v1 = types.ModuleType("vllm.v1")
    core = types.ModuleType("vllm.v1.core")
    block_pool = types.ModuleType("vllm.v1.core.block_pool")

    class _Block:
        def __init__(self, block_id: int) -> None:
            self.block_id = block_id
            self.ref_cnt = 0
            self.is_null = block_id == 0

    class _Queue:
        def __init__(self, blocks):
            self.blocks = list(blocks)

        def get_all_free_blocks(self):
            return list(self.blocks)

        def remove(self, block):
            self.blocks.remove(block)

        def append_n(self, blocks):
            self.blocks.extend(blocks)

        def prepend_n(self, blocks):
            self.blocks = list(blocks) + self.blocks

    class BlockPool:
        def __init__(self, num_gpu_blocks: int, enable_caching: bool = False) -> None:
            self.num_gpu_blocks = num_gpu_blocks
            self.enable_caching = enable_caching
            self.blocks = [_Block(i) for i in range(num_gpu_blocks)]
            self.free_block_queue = _Queue(self.blocks[1:])
            self.metrics_collector = None

        def get_num_free_blocks(self) -> int:
            return len(self.free_block_queue.blocks)

        def get_new_blocks(self, num_blocks: int):
            if num_blocks > len(self.free_block_queue.blocks):
                raise ValueError("not enough blocks")
            blocks = self.free_block_queue.blocks[:num_blocks]
            for block in blocks:
                self.free_block_queue.remove(block)
                block.ref_cnt += 1
            return blocks

        def free_blocks(self, ordered_blocks, prepend: bool = False):
            freed_blocks = []
            for block in ordered_blocks:
                block.ref_cnt -= 1
                if block.ref_cnt == 0 and not block.is_null:
                    freed_blocks.append(block)
            if prepend:
                self.free_block_queue.prepend_n(freed_blocks)
            else:
                self.free_block_queue.append_n(freed_blocks)

        def _maybe_evict_cached_block(self, block):
            return None

    block_pool.BlockPool = BlockPool
    vllm.v1 = v1
    v1.core = core
    core.block_pool = block_pool
    _set_module(monkeypatch, "vllm", vllm)
    _set_module(monkeypatch, "vllm.v1", v1)
    _set_module(monkeypatch, "vllm.v1.core", core)
    _set_module(monkeypatch, "vllm.v1.core.block_pool", block_pool)
    return BlockPool


def test_vllm_block_pool_uses_global_gms_leases(monkeypatch):
    BlockPool = _install_fake_vllm(monkeypatch)
    from gpu_memory_service.integrations.vllm import install_kv_leases

    install_kv_leases._patched = False
    install_kv_leases._factory = None
    manager = KVLeaseManager()
    owners = itertools.count()

    def factory(total_blocks: int):
        return _InMemoryLeaseClient(
            manager, "vllm", f"vllm-{next(owners)}", total_blocks, [0]
        )

    assert install_kv_leases.install(factory=factory) is True
    first = BlockPool(5)
    second = BlockPool(5)

    first_blocks = first.get_new_blocks(2)
    second_blocks = second.get_new_blocks(2)
    assert [block.block_id for block in first_blocks] == [1, 2]
    assert [block.block_id for block in second_blocks] == [3, 4]
    assert first.get_num_free_blocks() == 0
    assert second.get_num_free_blocks() == 0

    first.free_blocks([first_blocks[0]])
    assert [block.block_id for block in second.get_new_blocks(1)] == [1]


def test_vllm_block_pool_free_blocks_accepts_prepend_keyword(monkeypatch):
    BlockPool = _install_fake_vllm(monkeypatch)
    from gpu_memory_service.integrations.vllm import install_kv_leases

    install_kv_leases._patched = False
    install_kv_leases._factory = None
    manager = KVLeaseManager()

    def factory(total_blocks: int):
        return _InMemoryLeaseClient(
            manager, "vllm-prepend", "vllm-owner", total_blocks, [0]
        )

    assert install_kv_leases.install(factory=factory) is True
    pool = BlockPool(5)

    blocks = pool.get_new_blocks(2)
    assert [block.block_id for block in blocks] == [1, 2]

    pool.free_blocks(blocks, prepend=True)
    assert [block.block_id for block in pool.get_new_blocks(2)] == [1, 2]


def test_vllm_block_pool_uses_nonstrict_hint_for_stale_local_head(monkeypatch):
    BlockPool = _install_fake_vllm(monkeypatch)
    from gpu_memory_service.integrations.vllm import install_kv_leases

    install_kv_leases._patched = False
    install_kv_leases._factory = None
    manager = KVLeaseManager()
    owners = itertools.count()
    calls = []

    class RecordingLeaseClient(_InMemoryLeaseClient):
        def acquire(
            self,
            count: int,
            *,
            preferred_blocks: list[int] | None = None,
            allow_partial: bool = False,
            strict_preferred: bool = False,
        ) -> list[KVLease]:
            calls.append(
                {
                    "owner_id": self.owner_id,
                    "preferred_blocks": list(preferred_blocks or []),
                    "strict_preferred": bool(strict_preferred),
                }
            )
            return super().acquire(
                count,
                preferred_blocks=preferred_blocks,
                allow_partial=allow_partial,
                strict_preferred=strict_preferred,
            )

    def factory(total_blocks: int):
        return RecordingLeaseClient(
            manager, "vllm-stale-head", f"vllm-{next(owners)}", total_blocks, [0]
        )

    assert install_kv_leases.install(factory=factory) is True
    first = BlockPool(8)
    second = BlockPool(8)

    assert [block.block_id for block in first.get_new_blocks(1)] == [1]
    calls.clear()

    assert [block.block_id for block in second.get_new_blocks(1)] == [2]
    assert calls == [
        {
            "owner_id": "vllm-1",
            "preferred_blocks": [1],
            "strict_preferred": False,
        }
    ]


def test_vllm_block_pool_falls_back_beyond_preferred_window(monkeypatch):
    BlockPool = _install_fake_vllm(monkeypatch)
    from gpu_memory_service.integrations.vllm import install_kv_leases

    install_kv_leases._patched = False
    install_kv_leases._factory = None
    manager = KVLeaseManager()
    owners = itertools.count()

    def factory(total_blocks: int):
        return _InMemoryLeaseClient(
            manager, "vllm-window", f"vllm-{next(owners)}", total_blocks, [0]
        )

    assert install_kv_leases.install(factory=factory) is True
    first = BlockPool(300)
    second = BlockPool(300)

    assert [block.block_id for block in first.get_new_blocks(256)] == list(
        range(1, 257)
    )

    assert [block.block_id for block in second.get_new_blocks(1)] == [257]


def test_vllm_allocate_slots_backpressures_when_shared_leases_exhausted(monkeypatch):
    BlockPool = _install_fake_vllm(monkeypatch)
    kv_cache_manager = types.ModuleType("vllm.v1.core.kv_cache_manager")

    class KVCacheManager:
        def __init__(self) -> None:
            self.block_pool = BlockPool(2)

        def allocate_slots(self, request, num_new_tokens: int, **kwargs):
            return self.block_pool.get_new_blocks(1)

    kv_cache_manager.KVCacheManager = KVCacheManager
    _set_module(monkeypatch, "vllm.v1.core.kv_cache_manager", kv_cache_manager)

    from gpu_memory_service.integrations.vllm import install_kv_leases

    install_kv_leases._patched = False
    install_kv_leases._factory = None
    manager = KVLeaseManager()
    owners = itertools.count()

    def factory(total_blocks: int):
        return _InMemoryLeaseClient(
            manager, "vllm-empty", f"vllm-{next(owners)}", total_blocks, [0]
        )

    assert install_kv_leases.install(factory=factory) is True
    first = KVCacheManager()
    second = KVCacheManager()

    assert [block.block_id for block in first.allocate_slots(SimpleNamespace(), 1)] == [
        1
    ]
    assert second.allocate_slots(SimpleNamespace(), 1) is None


def test_vllm_geometry_patch_reuses_existing_lease_blocks(monkeypatch):
    vllm = types.ModuleType("vllm")
    v1 = types.ModuleType("vllm.v1")
    core_pkg = types.ModuleType("vllm.v1.core")
    kv_cache_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")
    engine_pkg = types.ModuleType("vllm.v1.engine")
    engine_core = types.ModuleType("vllm.v1.engine.core")
    calls = []

    def original(vllm_config, kv_cache_specs, available_memory):
        calls.append(vllm_config.cache_config.num_gpu_blocks_override)
        return [SimpleNamespace(num_blocks=calls[-1])]

    kv_cache_utils.get_kv_cache_configs = original
    engine_core.get_kv_cache_configs = original
    vllm.v1 = v1
    v1.core = core_pkg
    v1.engine = engine_pkg
    core_pkg.kv_cache_utils = kv_cache_utils
    engine_pkg.core = engine_core
    _set_module(monkeypatch, "vllm", vllm)
    _set_module(monkeypatch, "vllm.v1", v1)
    _set_module(monkeypatch, "vllm.v1.core", core_pkg)
    _set_module(monkeypatch, "vllm.v1.core.kv_cache_utils", kv_cache_utils)
    _set_module(monkeypatch, "vllm.v1.engine", engine_pkg)
    _set_module(monkeypatch, "vllm.v1.engine.core", engine_core)

    from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv

    waits = []

    def existing_blocks(*, wait_ms=0):
        waits.append(wait_ms)
        return 123

    monkeypatch.setenv("GMS_VLLM_VMM_IPC_KV", "1")
    monkeypatch.setenv("GMS_VLLM_KV_GEOMETRY_WAIT_MS", "17")
    monkeypatch.setattr(install_vmm_ipc_kv, "_GEOMETRY_PATCH_INSTALLED", False)
    monkeypatch.setattr(
        install_vmm_ipc_kv, "_existing_shared_kv_blocks", existing_blocks
    )

    assert install_vmm_ipc_kv.install_geometry_patch() is True

    config = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=None))
    result = engine_core.get_kv_cache_configs(config, [], [-1])

    assert waits == [17]
    assert calls == [123]
    assert result[0].num_blocks == 123
    assert config.cache_config.num_gpu_blocks_override is None


def _install_fake_sglang(monkeypatch):
    torch = pytest.importorskip("torch")
    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    mem_cache = types.ModuleType("sglang.srt.mem_cache")
    allocator = types.ModuleType("sglang.srt.mem_cache.allocator")
    utils = types.ModuleType("sglang.srt.utils")

    def get_num_new_pages(*, seq_lens, page_size: int, prefix_lens=None, decode=False):
        if decode:
            return len(seq_lens)
        total = 0
        prefix_lens = prefix_lens or [0 for _ in seq_lens]
        for prefix, seq in zip(prefix_lens, seq_lens):
            total += max(
                0, (int(seq) + page_size - 1) // page_size - int(prefix) // page_size
            )
        return total

    class BaseTokenToKVPoolAllocator:
        def __init__(self, size: int, page_size: int = 1, device: str = "cpu") -> None:
            self.size = size
            self.page_size = page_size
            self.device = device
            self.need_sort = False
            self.free_pages = torch.arange(1, size // page_size + 1, dtype=torch.int64)

        def available_size(self):
            return len(self.free_pages) * self.page_size

        def merge_and_sort_free(self):
            self.free_pages = torch.sort(self.free_pages).values

    class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
        def available_size(self):
            return len(self.free_pages)

        def alloc(self, need_size: int):
            if need_size > len(self.free_pages):
                return None
            out = self.free_pages[:need_size]
            self.free_pages = self.free_pages[need_size:]
            return out

        def free(self, free_index):
            self.free_pages = torch.cat(
                (free_index.to(dtype=torch.int64), self.free_pages)
            )

        def clear(self):
            self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64)

    class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
        def alloc(self, need_size: int):
            return TokenToKVPoolAllocator.alloc(self, need_size // self.page_size)

        def alloc_extend(self, *args, **kwargs):
            return self.alloc(kwargs.get("num_new_pages", 0) * self.page_size)

        def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
            return self.alloc(len(seq_lens_cpu) * self.page_size)

        def free(self, free_index):
            pages = torch.unique(free_index // self.page_size)
            self.free_pages = torch.cat((pages.to(dtype=torch.int64), self.free_pages))

        def clear(self):
            self.free_pages = torch.arange(
                1, self.size // self.page_size + 1, dtype=torch.int64
            )

    allocator.BaseTokenToKVPoolAllocator = BaseTokenToKVPoolAllocator
    allocator.TokenToKVPoolAllocator = TokenToKVPoolAllocator
    allocator.PagedTokenToKVPoolAllocator = PagedTokenToKVPoolAllocator
    utils.get_num_new_pages = get_num_new_pages
    sglang.srt = srt
    srt.mem_cache = mem_cache
    srt.utils = utils
    mem_cache.allocator = allocator

    _set_module(monkeypatch, "sglang", sglang)
    _set_module(monkeypatch, "sglang.srt", srt)
    _set_module(monkeypatch, "sglang.srt.mem_cache", mem_cache)
    _set_module(monkeypatch, "sglang.srt.mem_cache.allocator", allocator)
    _set_module(monkeypatch, "sglang.srt.utils", utils)
    return torch, TokenToKVPoolAllocator


def test_sglang_token_allocator_uses_global_gms_leases(monkeypatch):
    torch, TokenToKVPoolAllocator = _install_fake_sglang(monkeypatch)
    from gpu_memory_service.integrations.sglang import install_kv_leases

    install_kv_leases._patched = False
    install_kv_leases._factory = None
    install_kv_leases._STATE.clear()
    manager = KVLeaseManager()
    owners = itertools.count()

    def factory(_allocator, total_pages: int):
        return _InMemoryLeaseClient(
            manager, "sglang", f"sglang-{next(owners)}", total_pages + 1, [0]
        )

    assert install_kv_leases.install(factory=factory) is True
    first = TokenToKVPoolAllocator(4)
    second = TokenToKVPoolAllocator(4)

    assert first.alloc(2).tolist() == [1, 2]
    assert second.alloc(2).tolist() == [3, 4]
    assert first.available_size() == 2
    assert second.available_size() == 2
    assert second.alloc(1) is None

    first.free(torch.tensor([1], dtype=torch.int64))
    assert second.alloc(1).tolist() == [1]


def test_sglang_shared_geometry_namespace_is_stable_across_local_profiles(monkeypatch):
    model_executor = types.ModuleType("sglang.srt.model_executor")
    mixin_mod = types.ModuleType(
        "sglang.srt.model_executor.model_runner_kv_cache_mixin"
    )
    pool_configurator = types.ModuleType("sglang.srt.model_executor.pool_configurator")

    class ModelRunnerKVCacheMixin:
        def __init__(self, max_total_num_tokens: int, *, gpu_id: int = 0) -> None:
            self._max_total_num_tokens = int(max_total_num_tokens)
            self.gpu_id = int(gpu_id)
            self.server_args = SimpleNamespace(
                page_size=16,
                mem_fraction_static=0.45,
                served_model_name="Qwen/Qwen3-0.6B",
                model_path="Qwen/Qwen3-0.6B",
            )

        def _resolve_memory_pool_config(self, pre_model_load_memory):
            return SimpleNamespace(
                max_total_num_tokens=self._max_total_num_tokens,
                max_running_requests=None,
                mem_fraction_static=None,
            )

        def _resolve_max_num_reqs(self, max_total_num_tokens: int) -> int:
            return int(max_total_num_tokens) // self.server_args.page_size

    class _Configurator:
        def calculate_pool_sizes_from_max_tokens(self, target_tokens, page_size):
            return SimpleNamespace(
                max_total_num_tokens=int(target_tokens),
                max_running_requests=None,
                mem_fraction_static=None,
            )

    def create_memory_pool_configurator(_runner):
        return _Configurator()

    mixin_mod.ModelRunnerKVCacheMixin = ModelRunnerKVCacheMixin
    pool_configurator.create_memory_pool_configurator = create_memory_pool_configurator
    model_executor.model_runner_kv_cache_mixin = mixin_mod
    model_executor.pool_configurator = pool_configurator
    _set_module(monkeypatch, "sglang.srt.model_executor", model_executor)
    _set_module(
        monkeypatch,
        "sglang.srt.model_executor.model_runner_kv_cache_mixin",
        mixin_mod,
    )
    _set_module(
        monkeypatch,
        "sglang.srt.model_executor.pool_configurator",
        pool_configurator,
    )

    from gpu_memory_service.integrations.common import kv_lease_client
    from gpu_memory_service.integrations.sglang import patches

    seen: list[tuple[int, int, str]] = []
    published: dict[str, int] = {}

    def fake_resolve(
        engine,
        device,
        *,
        total_blocks,
        namespace_suffix="kv",
        reserved_blocks=None,
        timeout_ms=None,
    ):
        del reserved_blocks, timeout_ms
        namespace = f"{engine}:gpu{device}:{namespace_suffix}"
        seen.append((int(total_blocks), int(device), namespace_suffix))
        published.setdefault(namespace, int(total_blocks))
        return namespace, published[namespace]

    monkeypatch.setenv("GMS_KV_LEASES", "1")
    monkeypatch.setenv("GMS_SGLANG_SHARED_KV", "1")
    monkeypatch.delenv("GMS_SGLANG_KV_LEASE_DEVICE", raising=False)
    monkeypatch.setenv("DYN_NAMESPACE", "test-dynamo-namespace")
    monkeypatch.setattr(
        kv_lease_client,
        "resolve_kv_lease_namespace_total_blocks",
        fake_resolve,
    )
    monkeypatch.setattr(patches, "_kv_pool_geometry_patched", False)

    patches.patch_shared_kv_pool_geometry()

    primary_config = ModelRunnerKVCacheMixin(
        1600, gpu_id=2
    )._resolve_memory_pool_config(None)
    shadow_config = ModelRunnerKVCacheMixin(800, gpu_id=2)._resolve_memory_pool_config(
        None
    )
    other_gpu_config = ModelRunnerKVCacheMixin(
        800, gpu_id=3
    )._resolve_memory_pool_config(None)

    assert primary_config.max_total_num_tokens == 1600
    assert shadow_config.max_total_num_tokens == 1600
    assert other_gpu_config.max_total_num_tokens == 800
    assert len({suffix for _blocks, _device, suffix in seen}) == 1
    assert seen[0][0] == 101
    assert seen[1][0] == 51
    assert seen[2][0] == 51
    assert [device for _blocks, device, _suffix in seen] == [2, 2, 3]
    assert "tokens" not in seen[0][2]
    assert "test-dynamo-namespace" in seen[0][2]
    assert "Qwen/Qwen3-0.6B" in seen[0][2]


def test_sglang_vmm_kv_pool_device_resolver_uses_positional_and_current_device(
    monkeypatch,
):
    from gpu_memory_service.integrations.sglang import install_vmm_ipc_kv

    class _Device:
        index = 5

    assert (
        install_vmm_ipc_kv._resolve_kv_pool_device((None,) * 6 + (_Device(),), {}) == 5
    )
    assert install_vmm_ipc_kv._resolve_kv_pool_device((), {"device": "cuda:3"}) == 3

    monkeypatch.delenv("GMS_SGLANG_KV_LEASE_DEVICE", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "4")
    assert install_vmm_ipc_kv._resolve_kv_pool_device((None,) * 6 + ("cuda",), {}) == 4


def _install_fake_trtllm(monkeypatch):
    tensorrt_llm = types.ModuleType("tensorrt_llm")
    runtime = types.ModuleType("tensorrt_llm.runtime")
    kvcache = types.ModuleType("tensorrt_llm.runtime.kv_cache_manager_v2")
    storage = types.ModuleType("tensorrt_llm.runtime.kv_cache_manager_v2._storage")
    core = types.ModuleType("tensorrt_llm.runtime.kv_cache_manager_v2._storage._core")

    class OutOfPagesError(Exception):
        pass

    class SlotId(int):
        pass

    class CachedCudaEvent:
        NULL = None

    class Slot:
        def __init__(self, slot_id, event=None) -> None:
            self.slot_id = SlotId(slot_id)
            self.event = event

    class _Mask:
        def __init__(self) -> None:
            self.values: set[int] = set()

        def set(self, slot_id) -> None:
            self.values.add(int(slot_id))

        def clear(self, slot_id) -> None:
            self.values.discard(int(slot_id))

    class SlotAllocator:
        def __init__(self, num_slots: int) -> None:
            self.num_slots = num_slots
            self._target_capacity = num_slots
            self._num_active_slots = 0
            self._recycled_slots: list[Slot] = []
            self._num_ready_recycled_slots = 0
            self._occupied_mask = _Mask()

        @property
        def num_free_slots(self):
            return self.num_slots - self._num_active_slots + len(self._recycled_slots)

        def _scrub_events(self):
            self._num_ready_recycled_slots = len(self._recycled_slots)

        def allocate(self):
            self._scrub_events()
            if self._recycled_slots:
                slot = self._recycled_slots.pop(0)
                self._occupied_mask.set(slot.slot_id)
                return slot
            if self._num_active_slots >= self.num_slots:
                raise OutOfPagesError("No free slots")
            slot = Slot(SlotId(self._num_active_slots), CachedCudaEvent.NULL)
            self._num_active_slots += 1
            self._occupied_mask.set(slot.slot_id)
            return slot

        def allocate_multiple(self, num_slots: int):
            return [self.allocate() for _ in range(num_slots)]

        def release(self, slot):
            self._occupied_mask.clear(slot.slot_id)
            self._recycled_slots.append(slot)
            self._num_ready_recycled_slots = len(self._recycled_slots)

    class GpuPoolGroup:
        def __init__(self, num_slots, slot_size_list, shared_phys_mem_pool) -> None:
            self._slot_allocator = SlotAllocator(num_slots)

    core.GpuPoolGroup = GpuPoolGroup
    core.SlotAllocator = SlotAllocator
    core.Slot = Slot
    core.SlotId = SlotId
    core.CachedCudaEvent = CachedCudaEvent
    core.OutOfPagesError = OutOfPagesError
    storage._core = core
    kvcache._storage = storage
    runtime.kv_cache_manager_v2 = kvcache
    tensorrt_llm.runtime = runtime

    _set_module(monkeypatch, "tensorrt_llm", tensorrt_llm)
    _set_module(monkeypatch, "tensorrt_llm.runtime", runtime)
    _set_module(monkeypatch, "tensorrt_llm.runtime.kv_cache_manager_v2", kvcache)
    _set_module(
        monkeypatch, "tensorrt_llm.runtime.kv_cache_manager_v2._storage", storage
    )
    _set_module(
        monkeypatch, "tensorrt_llm.runtime.kv_cache_manager_v2._storage._core", core
    )
    return core


def test_trtllm_v2_slot_allocator_uses_global_gms_leases(monkeypatch):
    core = _install_fake_trtllm(monkeypatch)
    from gpu_memory_service.integrations.trtllm import install_kv_leases_v2

    install_kv_leases_v2._patched = False
    install_kv_leases_v2._factory = None
    install_kv_leases_v2._ALLOC_STATE.clear()
    manager = KVLeaseManager()
    owners = itertools.count()

    def factory(_group, total_slots: int, _suffix: str):
        return _InMemoryLeaseClient(
            manager, "trtllm", f"trtllm-{next(owners)}", total_slots, []
        )

    assert install_kv_leases_v2.install(factory=factory) is True
    first = core.GpuPoolGroup(4, [128], None)._slot_allocator
    second = core.GpuPoolGroup(4, [128], None)._slot_allocator

    first_slots = first.allocate_multiple(2)
    second_slots = second.allocate_multiple(2)
    assert [int(slot.slot_id) for slot in first_slots] == [0, 1]
    assert [int(slot.slot_id) for slot in second_slots] == [2, 3]
    with pytest.raises(core.OutOfPagesError):
        first.allocate()

    first.release(first_slots[0])
    assert int(second.allocate().slot_id) == 0


def test_trtllm_gms_remote_code_cache_is_local(monkeypatch, tmp_path):
    model_config = types.ModuleType("tensorrt_llm._torch.model_config")
    model_config.HF_MODULES_CACHE = "/shared/hf/modules"

    @contextlib.contextmanager
    def original_lock(timeout=10):
        yield

    model_config.config_file_lock = original_lock

    tensorrt_llm = types.ModuleType("tensorrt_llm")
    torch_mod = types.ModuleType("tensorrt_llm._torch")
    hub = types.ModuleType("transformers.utils.hub")
    dynamic_modules = types.ModuleType("transformers.dynamic_module_utils")
    hub.HF_MODULES_CACHE = "/shared/hf/modules"
    dynamic_modules.HF_MODULES_CACHE = "/shared/hf/modules"

    _set_module(monkeypatch, "tensorrt_llm", tensorrt_llm)
    _set_module(monkeypatch, "tensorrt_llm._torch", torch_mod)
    _set_module(monkeypatch, "tensorrt_llm._torch.model_config", model_config)
    _set_module(monkeypatch, "transformers.utils.hub", hub)
    _set_module(monkeypatch, "transformers.dynamic_module_utils", dynamic_modules)
    monkeypatch.setenv("GMS_TRTLLM_REMOTE_CODE_CACHE_DIR", str(tmp_path / "modules"))

    from gpu_memory_service.integrations.trtllm import remote_code_cache

    remote_code_cache._patched = False
    remote_code_cache.patch_remote_code_cache()

    assert model_config.HF_MODULES_CACHE == str(tmp_path / "modules")
    assert hub.HF_MODULES_CACHE == str(tmp_path / "modules")
    assert dynamic_modules.HF_MODULES_CACHE == str(tmp_path / "modules")

    with model_config.config_file_lock():
        pass
