# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
import unittest


PYTHON_ROOT = Path(__file__).resolve().parents[1] / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


class _StubRequest:
    def __init__(
        self,
        request_id: int,
        *,
        context_current_position: int = 0,
        is_first_context_chunk: bool = True,
    ) -> None:
        self.request_id = request_id
        self.py_request_id = request_id
        self.is_first_context_chunk = is_first_context_chunk
        self.context_current_position = context_current_position
        self.context_chunk_size = 0
        self.context_remaining_length = 0
        self.max_beam_num_tokens = context_current_position + 1
        self.py_rewind_len = 0
        self.py_draft_tokens: list[int] = []


class _StubBatch:
    def __init__(self, *, context_requests=None, generation_requests=None) -> None:
        self.context_requests = context_requests or []
        self.generation_requests = generation_requests or []


class TrtllmIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "kvbm" or name.startswith("kvbm.") or name == "nixl"
        }

        for name in list(self._saved_modules):
            sys.modules.pop(name, None)

        nixl = types.ModuleType("nixl")
        nixl._api = object()

        trtllm_integration = types.SimpleNamespace(
            KvbmRequest=type("KvbmRequest", (), {}),
            KvbmBlockList=type("KvbmBlockList", (), {}),
            BlockState=type("BlockState", (), {}),
            BlockStates=type("BlockStates", (), {}),
            SlotUpdate=type("SlotUpdate", (), {}),
            PyTrtllmKvConnectorWorker=type("PyTrtllmKvConnectorWorker", (), {}),
            PyTrtllmKvConnectorLeader=type("PyTrtllmKvConnectorLeader", (), {}),
            SchedulerOutput=type("SchedulerOutput", (), {}),
        )

        core = types.ModuleType("kvbm._core")
        core.BlockManager = type("BlockManager", (), {})
        core.KvbmLeader = type("KvbmLeader", (), {})
        core.KvbmWorker = type("KvbmWorker", (), {})
        core._trtllm_integration = trtllm_integration

        sys.modules["nixl"] = nixl
        sys.modules["kvbm._core"] = core

    def tearDown(self) -> None:
        for name in list(sys.modules):
            if name == "kvbm" or name.startswith("kvbm.") or name == "nixl":
                sys.modules.pop(name, None)
        sys.modules.update(self._saved_modules)

    def test_rust_loader_uses_dedicated_trtllm_module(self) -> None:
        rust = importlib.import_module("kvbm.trtllm_integration.rust")

        self.assertEqual(rust.BlockManager.__name__, "BlockManager")
        self.assertEqual(rust.KvbmRequest.__name__, "KvbmRequest")
        self.assertEqual(
            rust.KvConnectorWorker.__name__, "PyTrtllmKvConnectorWorker"
        )
        self.assertEqual(
            rust.KvConnectorLeader.__name__, "PyTrtllmKvConnectorLeader"
        )
        self.assertIsNone(rust.create_primary_pool)

    def test_manager_tracks_request_lifecycle_and_indices(self) -> None:
        integration = importlib.import_module("kvbm.trtllm_integration")
        manager = integration.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=16,
            pp_layers=[0, 1],
            total_num_kv_heads_per_layer=[8, 8],
            max_seq_len=32,
            num_blocks=8,
        )

        request = _StubRequest(101, context_current_position=0)
        request.context_chunk_size = 6
        request.context_remaining_length = 6
        self.assertTrue(manager.prepare_context(request))
        self.assertTrue(manager.resize_context(request, 6))
        self.assertEqual(manager.get_cache_indices(101), [0, 1])
        self.assertEqual(
            manager.host_kv_cache_block_offsets[0][0][0][:4],
            [0, 1, -1, -1],
        )

        dst = [None]
        manager.copy_batch_block_offsets(dst, [101], beam_width=1, num_contexts=1, num_seqs=1)
        self.assertEqual(dst[0][:4], [0, 1, -1, -1])
        self.assertEqual(manager.get_block_ids_per_seq([101]), [[0, 1]])

        request.py_draft_tokens = [1, 2]
        self.assertTrue(manager.try_allocate_generation(request))
        self.assertTrue(manager.is_request_active(101))

        request.max_beam_num_tokens = 8
        manager.update_resources(_StubBatch(generation_requests=[request]))
        self.assertGreater(manager.get_kv_cache_stats().allocated_bytes, 0)

        manager.suspend_request(request)
        self.assertFalse(manager.is_request_active(101))

        manager.free_resources(request)
        self.assertEqual(manager.get_num_free_blocks(), 8)
        self.assertEqual(
            manager.host_kv_cache_block_offsets[0][0][0][:4],
            [-1, -1, -1, -1],
        )

    def test_manager_requires_explicit_tensor_export_wiring(self) -> None:
        manager_mod = importlib.import_module(
            "kvbm.trtllm_integration.kv_cache_manager"
        )
        manager = manager_mod.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=32,
            pp_layers=[0],
            total_num_kv_heads_per_layer=[8],
            max_seq_len=16,
            num_blocks=4,
        )

        with self.assertRaises(NotImplementedError):
            manager.get_buffers(0)

        with self.assertRaises(NotImplementedError):
            manager.get_unique_primary_pool()

    def test_manager_handles_non_first_context_chunks_and_missing_generation(self) -> None:
        integration = importlib.import_module("kvbm.trtllm_integration")
        manager = integration.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=16,
            pp_layers=[0, 1],
            total_num_kv_heads_per_layer=[8, 8],
            max_seq_len=32,
            num_blocks=8,
            num_pools=2,
            kv_cache_pool_pointers=[[10, 0], [20, 0]],
        )

        request = _StubRequest(301, context_current_position=0, is_first_context_chunk=True)
        request.context_chunk_size = 4
        request.context_remaining_length = 4
        self.assertTrue(manager.prepare_context(request))
        self.assertTrue(manager.resize_context(request, 4))

        later_chunk = _StubRequest(
            301,
            context_current_position=4,
            is_first_context_chunk=False,
        )
        later_chunk.context_chunk_size = 4
        later_chunk.context_remaining_length = 4
        self.assertTrue(manager.prepare_context(later_chunk))
        self.assertTrue(manager.resize_context(later_chunk, 4))
        self.assertEqual(manager.get_cache_indices(301), [0, 1])
        self.assertFalse(manager.try_allocate_generation(_StubRequest(999)))

        dst = [[None], [None]]
        manager.copy_batch_block_offsets(dst, [301], beam_width=1, num_contexts=1, num_seqs=1)
        self.assertEqual(dst[0][0][:4], [0, 1, -1, -1])
        self.assertEqual(dst[1][0][:4], [0, 1, -1, -1])
        self.assertEqual(manager.kv_cache_pool_pointers, [[10, 0], [20, 0]])
        self.assertEqual(manager.kv_cache_pool_mapping, [[0, 0], [0, 1]])

    def test_manager_export_resolution_and_impl_compat(self) -> None:
        integration = importlib.import_module("kvbm.trtllm_integration")

        class _PrimaryPool:
            def get_layer_view(self, layer_idx: int, *, kv_layout: str) -> str:
                return f"layer={layer_idx},layout={kv_layout}"

        manager = integration.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=16,
            pp_layers=[2, 4],
            total_num_kv_heads_per_layer=[8, 8, 8, 8, 8],
            max_seq_len=32,
            num_blocks=8,
            primary_pool=_PrimaryPool(),
            layer_buffers={1: "local-layer-1"},
            layer_offsets={2: 0, 4: 1},
        )

        self.assertEqual(manager.get_buffers(4), "local-layer-1")
        self.assertEqual(manager.impl.get_primary_pool_data(4), "local-layer-1")
        self.assertEqual(manager.get_buffers(2), "layer=0,layout=NHD")
        self.assertEqual(manager.get_buffers(0, kv_layout="HND"), "layer=0,layout=HND")
        with self.assertRaises(ValueError):
            manager.get_buffers(2, kv_layout="BAD")

    def test_manager_can_seed_dummy_requests(self) -> None:
        integration = importlib.import_module("kvbm.trtllm_integration")
        manager = integration.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=16,
            pp_layers=[0],
            total_num_kv_heads_per_layer=[8],
            max_seq_len=32,
            num_blocks=8,
        )
        draft_manager = integration.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=16,
            pp_layers=[0],
            total_num_kv_heads_per_layer=[8],
            max_seq_len=32,
            num_blocks=8,
            is_draft=True,
        )

        requests = manager.add_dummy_requests(
            [201, 202],
            token_nums=[3, 5],
            is_gen=True,
            max_num_draft_tokens=2,
            draft_kv_cache_manager=draft_manager,
        )

        self.assertEqual([request.py_request_id for request in requests], [201, 202])
        self.assertTrue(manager.is_request_active(201))
        self.assertTrue(draft_manager.is_request_active(202))

    def test_manager_auto_wires_native_primary_pool_when_available(self) -> None:
        class _NativePool:
            def layer_view(self, layer_idx: int) -> str:
                return f"native-layer-{layer_idx}"

        calls = []

        def _create_primary_pool(**kwargs):
            calls.append(kwargs)
            return _NativePool()

        sys.modules["kvbm._core"]._trtllm_integration.create_primary_pool = _create_primary_pool
        for name in (
            "kvbm.trtllm_integration",
            "kvbm.trtllm_integration.rust",
            "kvbm.trtllm_integration.kv_cache_manager",
        ):
            sys.modules.pop(name, None)

        integration = importlib.import_module("kvbm.trtllm_integration")
        manager = integration.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=16,
            pp_layers=[2, 4],
            total_num_kv_heads_per_layer=[8, 8, 8, 8, 8],
            max_seq_len=32,
            num_blocks=8,
            layer_offsets={2: 0, 4: 1},
        )

        self.assertEqual(manager.get_buffers(4), "native-layer-1")
        self.assertIs(manager.get_unique_primary_pool(), manager.primary_pool)
        self.assertEqual(
            calls,
            [
                {
                    "num_blocks": 8,
                    "num_layers": 2,
                    "kv_factor": 2,
                    "page_size": 4,
                    "inner_dim": 128,
                    "dtype": "float16",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
