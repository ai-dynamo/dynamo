# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import enum
import math
from pathlib import Path
import pickle
import sys
import types
import unittest


PYTHON_ROOT = Path(__file__).resolve().parents[1] / "python"
TRTLLM_ROOT = Path("/tmp/trtllm-latest/tensorrt_llm")
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
            if name == "kvbm"
            or name.startswith("kvbm.")
            or name == "nixl"
            or name == "msgpack"
            or name == "tensorrt_llm"
            or name.startswith("tensorrt_llm.")
            or name == "torch"
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
            TrtllmStateManager=None,
            PyTrtllmKvConnectorWorker=type("PyTrtllmKvConnectorWorker", (), {}),
            PyTrtllmKvConnectorLeader=type("PyTrtllmKvConnectorLeader", (), {}),
            SchedulerOutput=type("SchedulerOutput", (), {}),
            create_primary_pool=None,
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
            if (
                name == "kvbm"
                or name.startswith("kvbm.")
                or name == "nixl"
                or name == "msgpack"
                or name == "tensorrt_llm"
                or name.startswith("tensorrt_llm.")
                or name == "torch"
            ):
                sys.modules.pop(name, None)
        sys.modules.update(self._saved_modules)

    def _install_pinned_trtllm_disagg_stubs(self):
        tensorrt_llm = types.ModuleType("tensorrt_llm")
        tensorrt_llm.__path__ = [str(TRTLLM_ROOT)]
        tensorrt_llm.logger = types.SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        )

        class _DisaggregatedParams:
            def __init__(self, disagg_request_id=None, schedule_style=None) -> None:
                self.disagg_request_id = disagg_request_id
                self.schedule_style = schedule_style

        tensorrt_llm.DisaggregatedParams = _DisaggregatedParams
        sys.modules["tensorrt_llm"] = tensorrt_llm

        for package_name in (
            "tensorrt_llm._torch",
            "tensorrt_llm._torch.disaggregation",
            "tensorrt_llm._torch.disaggregation.base",
            "tensorrt_llm._torch.disaggregation.resource",
            "tensorrt_llm._torch.disaggregation.native",
            "tensorrt_llm._torch.disaggregation.native.mixers",
            "tensorrt_llm._torch.disaggregation.native.mixers.attention",
            "tensorrt_llm._torch.pyexecutor",
            "tensorrt_llm.runtime",
        ):
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package

        base_region = types.ModuleType("tensorrt_llm._torch.disaggregation.base.region")

        class _DataLayout(enum.Enum):
            HND = "HND"
            NHD = "NHD"

        class _DataRole(enum.IntFlag):
            KEY = 1
            VALUE = 2
            BLOCK_QUANT = 4

        class _RegionExtractorBase:
            pass

        class _MemRegionGroup:
            def __init__(self, ptrs, bytes_per_region) -> None:
                self.ptrs = ptrs
                self.bytes_per_region = bytes_per_region

        class _SpecRegion:
            def __init__(self, memory) -> None:
                self.memory = memory

        base_region.DataLayout = _DataLayout
        base_region.DataRole = _DataRole
        base_region.RegionExtractorBase = _RegionExtractorBase
        base_region.MemRegionGroup = _MemRegionGroup
        base_region.SpecRegion = _SpecRegion
        sys.modules["tensorrt_llm._torch.disaggregation.base.region"] = base_region

        resource_manager = types.ModuleType("tensorrt_llm._torch.pyexecutor.resource_manager")
        resource_manager.KVCacheManager = type("KVCacheManager", (), {})
        resource_manager.Role = type(
            "Role",
            (),
            {
                "KEY": "key",
                "VALUE": "value",
                "KEY_BLOCK_SCALE": "key_block_scale",
                "VALUE_BLOCK_SCALE": "value_block_scale",
            },
        )
        sys.modules["tensorrt_llm._torch.pyexecutor.resource_manager"] = resource_manager

        utils_mod = types.ModuleType("tensorrt_llm._utils")
        utils_mod.get_size_in_bytes = lambda elems, dtype: int(elems) * (2 if str(dtype) == "float16" else 4)
        sys.modules["tensorrt_llm._utils"] = utils_mod

        bindings_mod = types.ModuleType("tensorrt_llm.bindings")
        bindings_mod.DataType = type("DataType", (), {"NVFP4": object()})
        sys.modules["tensorrt_llm.bindings"] = bindings_mod

        kv_cache_manager_v2 = types.ModuleType("tensorrt_llm.runtime.kv_cache_manager_v2")
        kv_cache_manager_v2.CacheTier = type("CacheTier", (), {"GPU_MEM": "gpu_mem"})
        sys.modules["tensorrt_llm.runtime.kv_cache_manager_v2"] = kv_cache_manager_v2

        msgpack_mod = types.ModuleType("msgpack")
        msgpack_mod.packb = lambda data: pickle.dumps(data)
        msgpack_mod.unpackb = lambda data, strict_map_key=False: pickle.loads(data)
        sys.modules["msgpack"] = msgpack_mod

        auxiliary_mod = types.ModuleType("tensorrt_llm._torch.disaggregation.native.auxiliary")

        class _AuxBufferMeta:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

            def to_dict(self) -> dict:
                return dict(self.__dict__)

            @classmethod
            def from_dict(cls, data: dict):
                return cls(**data)

        auxiliary_mod.AuxBufferMeta = _AuxBufferMeta
        sys.modules["tensorrt_llm._torch.disaggregation.native.auxiliary"] = auxiliary_mod

        spec_mod = types.ModuleType("tensorrt_llm._torch.disaggregation.native.mixers.attention.spec")

        class _AttentionInfo:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

            def to_dict(self) -> dict:
                return dict(self.__dict__)

            @classmethod
            def from_dict(cls, data: dict):
                return cls(**data)

        spec_mod.AttentionInfo = _AttentionInfo
        sys.modules["tensorrt_llm._torch.disaggregation.native.mixers.attention.spec"] = spec_mod

        page_mod = _load_module(
            "tensorrt_llm._torch.disaggregation.resource.page",
            TRTLLM_ROOT / "_torch/disaggregation/resource/page.py",
        )
        utils_page_mod = _load_module(
            "tensorrt_llm._torch.disaggregation.resource.utils",
            TRTLLM_ROOT / "_torch/disaggregation/resource/utils.py",
        )
        kv_extractor_mod = _load_module(
            "tensorrt_llm._torch.disaggregation.resource.kv_extractor",
            TRTLLM_ROOT / "_torch/disaggregation/resource/kv_extractor.py",
        )
        rank_info_mod = _load_module(
            "tensorrt_llm._torch.disaggregation.native.rank_info",
            TRTLLM_ROOT / "_torch/disaggregation/native/rank_info.py",
        )
        return page_mod, utils_page_mod, kv_extractor_mod, rank_info_mod

    def _install_pinned_trtllm_transceiver_stubs(self):
        _, _, kv_extractor_mod, _ = self._install_pinned_trtllm_disagg_stubs()

        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(current_device=lambda: 0)
        sys.modules["torch"] = torch_mod

        communicator_mod = types.ModuleType("tensorrt_llm._torch.distributed.communicator")
        communicator_mod.Distributed = type("Distributed", (), {})
        sys.modules["tensorrt_llm._torch.distributed.communicator"] = communicator_mod

        kv_cache_transceiver_mod = types.ModuleType(
            "tensorrt_llm._torch.pyexecutor.kv_cache_transceiver"
        )
        kv_cache_transceiver_mod.KvCacheTransceiver = type("KvCacheTransceiver", (), {})
        sys.modules["tensorrt_llm._torch.pyexecutor.kv_cache_transceiver"] = (
            kv_cache_transceiver_mod
        )

        llm_request_mod = types.ModuleType("tensorrt_llm._torch.pyexecutor.llm_request")
        llm_request_mod.LlmRequest = type("LlmRequest", (), {})
        sys.modules["tensorrt_llm._torch.pyexecutor.llm_request"] = llm_request_mod

        resource_manager = sys.modules["tensorrt_llm._torch.pyexecutor.resource_manager"]
        resource_manager.KVCacheManagerV2 = type("KVCacheManagerV2", (), {})

        bindings_mod = sys.modules["tensorrt_llm.bindings"]
        bindings_mod.LlmRequestState = type(
            "LlmRequestState",
            (),
            {
                "DISAGG_CONTEXT_TRANS_IN_PROGRESS": "ctx_in_progress",
                "DISAGG_CONTEXT_COMPLETE": "ctx_complete",
                "DISAGG_GENERATION_TRANS_IN_PROGRESS": "gen_in_progress",
                "DISAGG_GENERATION_TRANS_COMPLETE": "gen_complete",
                "DISAGG_TRANS_ERROR": "transfer_error",
                "DISAGG_CONTEXT_WAIT_SCHEDULER": "ctx_wait",
                "CONTEXT_INIT": "ctx_init",
            },
        )

        executor_mod = types.ModuleType("tensorrt_llm.bindings.executor")

        class _ContextPhaseParams:
            def __init__(
                self,
                *,
                first_gen_tokens,
                req_id,
                opaque_state,
                draft_tokens,
                ctx_dp_rank,
                disagg_info_endpoint,
            ) -> None:
                self.first_gen_tokens = first_gen_tokens
                self.req_id = req_id
                self.opaque_state = opaque_state
                self.draft_tokens = draft_tokens
                self.ctx_dp_rank = ctx_dp_rank
                self.disagg_info_endpoint = disagg_info_endpoint

        executor_mod.ContextPhaseParams = _ContextPhaseParams
        sys.modules["tensorrt_llm.bindings.executor"] = executor_mod

        disagg_params_mod = types.ModuleType("tensorrt_llm.disaggregated_params")
        disagg_params_mod.DisaggScheduleStyle = type(
            "DisaggScheduleStyle", (), {"GENERATION_FIRST": "generation_first"}
        )
        sys.modules["tensorrt_llm.disaggregated_params"] = disagg_params_mod

        llm_args_mod = types.ModuleType("tensorrt_llm.llmapi.llm_args")
        llm_args_mod.CacheTransceiverConfig = type("CacheTransceiverConfig", (), {})
        sys.modules["tensorrt_llm.llmapi.llm_args"] = llm_args_mod

        mapping_mod = types.ModuleType("tensorrt_llm.mapping")
        mapping_mod.Mapping = type("Mapping", (), {})
        sys.modules["tensorrt_llm.mapping"] = mapping_mod

        base_transfer_mod = _load_module(
            "tensorrt_llm._torch.disaggregation.base.transfer",
            TRTLLM_ROOT / "_torch/disaggregation/base/transfer.py",
        )

        native_transfer_mod = types.ModuleType("tensorrt_llm._torch.disaggregation.native.transfer")

        @dataclass
        class _TransferWorkerConfig:
            kv_cache_manager: object
            device_id: int
            instance_name: str
            max_concurrent_sessions: int
            tx_timeout_s: float
            rx_timeout_s: float

        class _FakeTxSession:
            def __init__(self, req) -> None:
                self.req = req
                self.sent_slices = []
                self.sent_aux = False
                self.closed = False
                self.disagg_request_id = req.request_id

            def send(self, slice_):
                self.sent_slices.append(slice_)
                return object()

            def pack_aux(self, req) -> None:
                self.req = req

            def send_aux(self) -> None:
                self.sent_aux = True

            def is_completed(self) -> bool:
                return True

            def has_failed(self) -> bool:
                return False

            def wait_complete(self):
                return base_transfer_mod.WaitResult.COMPLETED

            def close(self) -> None:
                self.closed = True

        class _FakeRxSession:
            def __init__(self, req) -> None:
                self.req = req
                self.received_slices = []
                self.closed = False
                self.disagg_request_id = req.request_id

            def receive(self, slice_):
                self.received_slices.append(slice_)
                return object()

            def unpack_aux(self, req) -> None:
                req.py_first_gen_tokens = []
                req.py_draft_tokens = []

            def is_completed(self) -> bool:
                return True

            def has_failed(self) -> bool:
                return False

            def wait_complete(self, blocking: bool = False):
                del blocking
                return base_transfer_mod.WaitResult.COMPLETED

            def close(self) -> None:
                self.closed = True

        class _TransferWorker:
            instances = []

            def __init__(self, config: _TransferWorkerConfig) -> None:
                self.config = config
                self.page_table = kv_extractor_mod.build_page_table_from_manager(
                    config.kv_cache_manager
                )
                self.rank_info_server_endpoint = "ctx-endpoint"
                self.sender_endpoint = f"sender-{config.device_id}"
                self.tx_sessions = []
                self.rx_sessions = []
                self.rank_info_calls = []
                self.shutdown_calls = 0
                type(self).instances.append(self)

            def populate_instance_and_rank_info(self, *, endpoints, layer_num_per_pp) -> None:
                self.rank_info_calls.append(
                    {"endpoints": list(endpoints), "layer_num_per_pp": list(layer_num_per_pp)}
                )

            def create_tx_session(self, req):
                session = _FakeTxSession(req)
                self.tx_sessions.append(session)
                return session

            def create_rx_session(self, req):
                session = _FakeRxSession(req)
                self.rx_sessions.append(session)
                return session

            def has_all_peer_req_infos_for_send(self, rid: int) -> bool:
                del rid
                return True

            def shutdown(self) -> None:
                self.shutdown_calls += 1

        native_transfer_mod.TransferWorkerConfig = _TransferWorkerConfig
        native_transfer_mod.TransferWorker = _TransferWorker
        sys.modules["tensorrt_llm._torch.disaggregation.native.transfer"] = (
            native_transfer_mod
        )

        transceiver_mod = _load_module(
            "tensorrt_llm._torch.disaggregation.transceiver",
            TRTLLM_ROOT / "_torch/disaggregation/transceiver.py",
        )
        return transceiver_mod, _TransferWorker

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

    def test_rust_loader_requires_pinned_optional_symbols(self) -> None:
        del sys.modules["kvbm._core"]._trtllm_integration.TrtllmStateManager
        for name in (
            "kvbm.trtllm_integration",
            "kvbm.trtllm_integration.rust",
        ):
            sys.modules.pop(name, None)

        with self.assertRaises(AttributeError):
            importlib.import_module("kvbm.trtllm_integration.rust")

    def test_rust_loader_requires_dedicated_trtllm_extension_module(self) -> None:
        del sys.modules["kvbm._core"]._trtllm_integration
        for name in (
            "kvbm.trtllm_integration",
            "kvbm.trtllm_integration.rust",
        ):
            sys.modules.pop(name, None)

        with self.assertRaises(AttributeError):
            importlib.import_module("kvbm.trtllm_integration.rust")

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

    def test_manager_requires_pinned_request_fields(self) -> None:
        integration = importlib.import_module("kvbm.trtllm_integration")
        manager = integration.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=16,
            pp_layers=[0],
            total_num_kv_heads_per_layer=[8],
            max_seq_len=16,
            num_blocks=4,
        )

        request = types.SimpleNamespace(
            py_request_id=7,
            is_first_context_chunk=True,
            context_current_position=0,
            context_remaining_length=4,
            max_beam_num_tokens=1,
            py_rewind_len=0,
            py_draft_tokens=[],
        )

        with self.assertRaises(AttributeError):
            manager.prepare_context(request)

    def test_manager_export_resolution_and_impl_compat(self) -> None:
        integration = importlib.import_module("kvbm.trtllm_integration")

        class _PrimaryPool:
            def layer_view(self, layer_idx: int) -> str:
                return f"layer={layer_idx}"

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
        self.assertEqual(manager.get_buffers(2), "layer=0")
        self.assertEqual(manager.get_buffers(0, kv_layout="HND"), "layer=0")
        with self.assertRaises(ValueError):
            manager.get_buffers(2, kv_layout="BAD")

    def test_manager_requires_primary_pool_layer_view_symbol(self) -> None:
        integration = importlib.import_module("kvbm.trtllm_integration")

        class _PrimaryPool:
            def get_layer_view(self, layer_idx: int, *, kv_layout: str) -> str:
                return f"layer={layer_idx},layout={kv_layout}"

        manager = integration.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=16,
            pp_layers=[1],
            total_num_kv_heads_per_layer=[8, 8],
            max_seq_len=32,
            num_blocks=8,
            primary_pool=_PrimaryPool(),
        )

        with self.assertRaises(NotImplementedError):
            manager.get_buffers(1)

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
                    "device_id": 0,
                }
            ],
        )

    def test_manager_can_delegate_request_state_to_native_helper(self) -> None:
        class _NativeState:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.block_ids = {}
                self.slots = {}

            def is_request_active(self, request_id: int) -> bool:
                return request_id in self.block_ids

            def prepare_context(self, request_id: int, is_first_context_chunk: bool) -> bool:
                del is_first_context_chunk
                self.block_ids.setdefault(request_id, [])
                self.slots.setdefault(request_id, 0)
                return True

            def resize_context(
                self,
                request_id: int,
                context_current_position: int,
                num_tokens: int,
                num_extra_kv_tokens: int,
                is_first_context_chunk: bool,
            ) -> bool:
                del context_current_position
                del num_tokens
                del num_extra_kv_tokens
                del is_first_context_chunk
                self.block_ids[request_id] = [0, 1]
                return True

            def get_padded_block_row(self, request_id: int, bad_page_index: int):
                del bad_page_index
                return self.block_ids.get(request_id, []) + [-1, -1]

            def get_slot(self, request_id: int) -> int:
                return self.slots.setdefault(request_id, 0)

            def get_cache_indices(self, request_id: int):
                return self.block_ids[request_id]

            def get_batch_cache_indices(self, request_ids):
                return [self.block_ids[request_id] for request_id in request_ids]

            def get_num_free_blocks(self) -> int:
                return 6

            def get_num_available_tokens(
                self,
                token_num_upper_bound: int,
                max_num_draft_tokens: int,
                num_extra_kv_tokens: int,
            ) -> int:
                del max_num_draft_tokens
                del num_extra_kv_tokens
                return token_num_upper_bound

            def get_num_kv_blocks(self) -> int:
                return 8

            def free_resources(self, request_id: int) -> int:
                self.block_ids.pop(request_id, None)
                return self.slots.pop(request_id, 0)

            def shutdown(self) -> None:
                self.block_ids.clear()
                self.slots.clear()

        sys.modules["kvbm._core"]._trtllm_integration.TrtllmStateManager = _NativeState
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
            pp_layers=[0],
            total_num_kv_heads_per_layer=[8],
            max_seq_len=32,
            num_blocks=8,
        )

        request = _StubRequest(401, context_current_position=0)
        request.context_chunk_size = 4
        request.context_remaining_length = 4
        self.assertTrue(manager.prepare_context(request))
        self.assertTrue(manager.resize_context(request, 4))
        self.assertEqual(manager.get_cache_indices(401), [0, 1])
        self.assertEqual(
            manager.host_kv_cache_block_offsets[0][0][0][:4],
            [0, 1, -1, -1],
        )

    def test_manager_can_delegate_dummy_requests_and_stats_to_native_helper(self) -> None:
        class _NativeState:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.block_ids = {}
                self.slots = {}
                self.calls = []

            def is_request_active(self, request_id: int) -> bool:
                return request_id in self.block_ids

            def add_dummy_request(self, request_id: int, target_capacity: int) -> bool:
                self.calls.append((request_id, target_capacity))
                num_blocks = math.ceil(target_capacity / self.kwargs["tokens_per_block"])
                self.block_ids[request_id] = list(range(num_blocks))
                self.slots.setdefault(request_id, 0)
                return True

            def get_padded_block_row(self, request_id: int, bad_page_index: int):
                row = list(self.block_ids[request_id])
                row.extend([bad_page_index] * 8)
                return row

            def get_slot(self, request_id: int) -> int:
                return self.slots.setdefault(request_id, 0)

            def get_cache_indices(self, request_id: int):
                return self.block_ids[request_id]

            def get_num_free_blocks(self) -> int:
                return 6

            def shutdown(self) -> None:
                self.block_ids.clear()
                self.slots.clear()

        sys.modules["kvbm._core"]._trtllm_integration.TrtllmStateManager = _NativeState
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
            pp_layers=[0],
            total_num_kv_heads_per_layer=[8],
            max_seq_len=32,
            num_blocks=8,
        )

        requests = manager.add_dummy_requests(
            [501],
            token_nums=[3],
            is_gen=True,
            max_num_draft_tokens=2,
        )

        self.assertEqual([request.py_request_id for request in requests], [501])
        self.assertEqual(manager._native_state.calls, [(501, 6)])
        self.assertEqual(manager.get_cache_indices(501), [0, 1])
        self.assertEqual(
            manager.host_kv_cache_block_offsets[0][0][0][:4],
            [0, 1, -1, -1],
        )
        self.assertEqual(manager.get_kv_cache_stats().allocated_bytes, 4096)

    def test_manager_surfaces_native_helper_construction_errors(self) -> None:
        def _create_primary_pool(**kwargs):
            del kwargs
            raise RuntimeError("bad primary pool wiring")

        class _NativeState:
            def __init__(self, **kwargs) -> None:
                del kwargs
                raise RuntimeError("bad state wiring")

        sys.modules["kvbm._core"]._trtllm_integration.create_primary_pool = _create_primary_pool
        sys.modules["kvbm._core"]._trtllm_integration.TrtllmStateManager = _NativeState
        for name in (
            "kvbm.trtllm_integration",
            "kvbm.trtllm_integration.rust",
            "kvbm.trtllm_integration.kv_cache_manager",
        ):
            sys.modules.pop(name, None)

        integration = importlib.import_module("kvbm.trtllm_integration")
        with self.assertRaisesRegex(RuntimeError, "bad primary pool wiring"):
            integration.KvbmKVCacheManager(
                tokens_per_block=4,
                dtype="float16",
                head_dim=16,
                pp_layers=[0],
                total_num_kv_heads_per_layer=[8],
                max_seq_len=32,
                num_blocks=8,
            )

        sys.modules["kvbm._core"]._trtllm_integration.create_primary_pool = None
        for name in (
            "kvbm.trtllm_integration",
            "kvbm.trtllm_integration.rust",
            "kvbm.trtllm_integration.kv_cache_manager",
        ):
            sys.modules.pop(name, None)

        integration = importlib.import_module("kvbm.trtllm_integration")
        with self.assertRaisesRegex(RuntimeError, "bad state wiring"):
            integration.KvbmKVCacheManager(
                tokens_per_block=4,
                dtype="float16",
                head_dim=16,
                pp_layers=[0],
                total_num_kv_heads_per_layer=[8],
                max_seq_len=32,
                num_blocks=8,
            )

    def test_manager_reuses_slots_after_free_in_fallback_path(self) -> None:
        integration = importlib.import_module("kvbm.trtllm_integration")
        manager = integration.KvbmKVCacheManager(
            tokens_per_block=2,
            dtype="float16",
            head_dim=16,
            pp_layers=[0],
            total_num_kv_heads_per_layer=[8],
            max_seq_len=16,
            num_blocks=6,
        )

        first = _StubRequest(601, context_current_position=0)
        first.context_chunk_size = 4
        first.context_remaining_length = 4
        self.assertTrue(manager.prepare_context(first))
        self.assertTrue(manager.resize_context(first, 4))
        self.assertEqual(manager._request_slots[601], 0)

        manager.free_resources(first)
        self.assertEqual(
            manager.host_kv_cache_block_offsets[0][0][0][:4],
            [-1, -1, -1, -1],
        )

        second = _StubRequest(602, context_current_position=0)
        second.context_chunk_size = 2
        second.context_remaining_length = 2
        self.assertTrue(manager.prepare_context(second))
        self.assertTrue(manager.resize_context(second, 2))

        self.assertEqual(manager._request_slots[602], 0)
        self.assertEqual(
            manager.host_kv_cache_block_offsets[0][0][0][:4],
            [2, -1, -1, -1],
        )
        self.assertEqual(manager.get_cache_indices(602), [2])
        manager.shutdown()
        self.assertEqual(manager.get_num_free_blocks(), 6)
        self.assertEqual(
            manager.host_kv_cache_block_offsets[0][0][0][:4],
            [-1, -1, -1, -1],
        )

    def test_manager_shutdown_clears_native_state_and_owned_exports(self) -> None:
        class _NativeState:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.block_ids = {}
                self.slots = {}
                self.shutdown_calls = 0

            def is_request_active(self, request_id: int) -> bool:
                return request_id in self.block_ids

            def add_dummy_request(self, request_id: int, target_capacity: int) -> bool:
                num_blocks = math.ceil(target_capacity / self.kwargs["tokens_per_block"])
                self.block_ids[request_id] = list(range(num_blocks))
                self.slots.setdefault(request_id, 0)
                return True

            def get_padded_block_row(self, request_id: int, bad_page_index: int):
                row = list(self.block_ids.get(request_id, ()))
                row.extend([bad_page_index] * 8)
                return row

            def get_slot(self, request_id: int) -> int:
                return self.slots.setdefault(request_id, 0)

            def get_num_free_blocks(self) -> int:
                allocated = sum(len(block_ids) for block_ids in self.block_ids.values())
                return self.kwargs["num_blocks"] - allocated

            def shutdown(self) -> None:
                self.shutdown_calls += 1
                self.block_ids.clear()
                self.slots.clear()

        sys.modules["kvbm._core"]._trtllm_integration.TrtllmStateManager = _NativeState
        for name in (
            "kvbm.trtllm_integration",
            "kvbm.trtllm_integration.rust",
            "kvbm.trtllm_integration.kv_cache_manager",
        ):
            sys.modules.pop(name, None)

        integration = importlib.import_module("kvbm.trtllm_integration")
        primary_pool = object()
        manager = integration.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=16,
            pp_layers=[0],
            total_num_kv_heads_per_layer=[8],
            max_seq_len=32,
            num_blocks=8,
            primary_pool=primary_pool,
            layer_buffers={0: "layer-0"},
        )
        manager._iteration_events.append("evt")
        manager.add_dummy_requests([701], token_nums=[3], is_gen=True)

        self.assertIs(manager.get_unique_primary_pool(), primary_pool)
        self.assertTrue(manager.is_request_active(701))

        manager.impl.shutdown()

        self.assertEqual(manager._native_state.shutdown_calls, 1)
        self.assertFalse(manager.is_request_active(701))
        self.assertEqual(manager.get_num_free_blocks(), 8)
        self.assertEqual(manager.get_latest_events(), [])
        self.assertIsNone(manager.primary_pool)
        self.assertEqual(manager.layer_buffers, {})
        self.assertEqual(
            manager.host_kv_cache_block_offsets[0][0][0][:4],
            [-1, -1, -1, -1],
        )
        with self.assertRaises(NotImplementedError):
            manager.get_unique_primary_pool()

    def test_manager_shutdown_requires_native_helper_shutdown(self) -> None:
        class _NativeState:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def get_num_free_blocks(self) -> int:
                return self.kwargs["num_blocks"]

        sys.modules["kvbm._core"]._trtllm_integration.TrtllmStateManager = _NativeState
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
            pp_layers=[0],
            total_num_kv_heads_per_layer=[8],
            max_seq_len=32,
            num_blocks=8,
            primary_pool=object(),
        )

        with self.assertRaises(AttributeError):
            manager.shutdown()

    def test_manager_derives_rank_local_standard_geometry(self) -> None:
        calls = []

        class _NativePool:
            def layer_view(self, layer_idx: int) -> str:
                return f"standard-layer-{layer_idx}"

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
            tokens_per_block=8,
            dtype="float16",
            head_dim=16,
            pp_layers=[2, 3],
            total_num_kv_heads_per_layer=[12, 12, 12, 12],
            max_seq_len=64,
            num_blocks=16,
            device_id=3,
            world_size=4,
            tp_size=2,
            tp_rank=1,
            pp_size=2,
            pp_rank=1,
        )

        self.assertEqual(manager.num_kv_heads_per_layer, [6, 6])
        self.assertEqual(manager.kv_factor, 2)
        self.assertEqual(
            manager.get_worker_identity(),
            {
                "device_id": 3,
                "world_size": 4,
                "tp_size": 2,
                "tp_rank": 1,
                "pp_size": 2,
                "pp_rank": 1,
                "cache_mode": "standard",
                "kv_factor": 2,
            },
        )
        self.assertEqual(manager.get_buffers(3), "standard-layer-1")
        self.assertEqual(
            calls,
            [
                {
                    "num_blocks": 16,
                    "num_layers": 2,
                    "kv_factor": 2,
                    "page_size": 8,
                    "inner_dim": 96,
                    "dtype": "float16",
                    "device_id": 3,
                }
            ],
        )

    def test_manager_keeps_mla_latent_geometry_rank_local(self) -> None:
        calls = []

        class _NativePool:
            def layer_view(self, layer_idx: int) -> str:
                return f"mla-layer-{layer_idx}"

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
            head_dim=576,
            pp_layers=[0, 1],
            total_num_kv_heads_per_layer=[1, 1, 1, 1],
            max_seq_len=64,
            num_blocks=10,
            device_id=1,
            world_size=4,
            tp_size=4,
            tp_rank=2,
            pp_size=1,
            pp_rank=0,
            cache_mode="mla",
        )

        self.assertTrue(manager.is_mla_enable)
        self.assertEqual(manager.kv_factor, 1)
        self.assertEqual(manager.num_kv_heads_per_layer, [1, 1])
        self.assertEqual(manager.get_buffers(1), "mla-layer-1")
        self.assertEqual(
            manager.get_worker_identity(),
            {
                "device_id": 1,
                "world_size": 4,
                "tp_size": 4,
                "tp_rank": 2,
                "pp_size": 1,
                "pp_rank": 0,
                "cache_mode": "mla",
                "kv_factor": 1,
            },
        )
        self.assertEqual(
            calls,
            [
                {
                    "num_blocks": 10,
                    "num_layers": 2,
                    "kv_factor": 1,
                    "page_size": 4,
                    "inner_dim": 576,
                    "dtype": "float16",
                    "device_id": 1,
                }
            ],
        )

    def test_manager_passes_topology_into_native_state(self) -> None:
        constructed = []

        class _NativeState:
            def __init__(self, **kwargs) -> None:
                constructed.append(kwargs)

        sys.modules["kvbm._core"]._trtllm_integration.TrtllmStateManager = _NativeState
        for name in (
            "kvbm.trtllm_integration",
            "kvbm.trtllm_integration.rust",
            "kvbm.trtllm_integration.kv_cache_manager",
        ):
            sys.modules.pop(name, None)

        integration = importlib.import_module("kvbm.trtllm_integration")
        integration.KvbmKVCacheManager(
            tokens_per_block=8,
            dtype="float16",
            head_dim=64,
            pp_layers=[4, 5],
            total_num_kv_heads_per_layer=[8, 8, 8, 8, 8, 8],
            max_seq_len=128,
            num_blocks=20,
            device_id=2,
            world_size=4,
            tp_size=2,
            tp_rank=0,
            pp_size=2,
            pp_rank=1,
            cache_mode="standard",
        )

        self.assertEqual(
            constructed,
            [
                {
                    "tokens_per_block": 8,
                    "max_seq_len": 128,
                    "num_blocks": 20,
                    "max_blocks_per_seq": 20,
                    "max_num_sequences": 32,
                    "max_beam_width": 1,
                    "device_id": 2,
                    "world_size": 4,
                    "tp_size": 2,
                    "tp_rank": 0,
                    "pp_size": 2,
                    "pp_rank": 1,
                    "kv_factor": 2,
                    "cache_mode": "standard",
                }
            ],
        )

    def test_manager_rejects_invalid_topology(self) -> None:
        integration = importlib.import_module("kvbm.trtllm_integration")

        with self.assertRaises(ValueError):
            integration.KvbmKVCacheManager(
                tokens_per_block=4,
                dtype="float16",
                head_dim=16,
                pp_layers=[0],
                total_num_kv_heads_per_layer=[8],
                max_seq_len=16,
                num_blocks=4,
                world_size=8,
                tp_size=2,
                pp_size=2,
            )

    def test_manager_exposes_fake_v2_disagg_metadata(self) -> None:
        integration = importlib.import_module("kvbm.trtllm_integration")

        class _FakeTensor:
            def __init__(self, shape: tuple[int, ...], strides: tuple[int, ...], ptr: int) -> None:
                self.shape = shape
                self._strides = strides
                self._ptr = ptr

            def stride(self, dim: int) -> int:
                return self._strides[dim]

            def element_size(self) -> int:
                return 2

            def data_ptr(self) -> int:
                return self._ptr

        manager = integration.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=16,
            pp_layers=[2, 3],
            total_num_kv_heads_per_layer=[6, 6, 6, 6],
            max_seq_len=64,
            num_blocks=8,
            primary_pool=_FakeTensor(
                shape=(8, 2, 2, 4, 6, 16),
                strides=(1536, 768, 384, 96, 16, 1),
                ptr=4096,
            ),
            device_id=2,
            world_size=4,
            tp_size=2,
            tp_rank=1,
            pp_size=2,
            pp_rank=1,
        )

        self.assertEqual(manager.max_batch_size, 16)
        self.assertEqual(manager.mapping.rank, 3)
        self.assertEqual(manager.impl.layer_grouping, [[0, 1]])

        storage = manager.impl._storage
        init_config = manager.impl._init_config
        life_cycles = manager.impl._life_cycles
        pool = storage._levels[0].storage._pool_groups[0]._pools[0]

        self.assertEqual(init_config.tokens_per_block, 4)
        self.assertEqual(storage.num_life_cycles, 1)
        self.assertEqual(storage.get_pool_group_index(0), 0)
        self.assertEqual(pool.base_address, 4096)
        self.assertEqual(pool.slot_size, 3072)
        self.assertEqual(pool.num_slots, 8)
        self.assertEqual(pool.slot_address(3), 4096 + 3 * 3072)
        self.assertIsNone(life_cycles[0].window_size)
        self.assertEqual(
            storage._buffer_attr[(0, "key")],
            manager.get_disagg_storage_metadata()._buffer_attr[(0, "key")],
        )
        self.assertEqual(storage._buffer_attr[(0, "key")].offset, 0)
        self.assertEqual(storage._buffer_attr[(0, "key")].size, 768)
        self.assertEqual(storage._buffer_attr[(0, "value")].offset, 768)
        self.assertEqual(storage._buffer_attr[(1, "key")].offset, 1536)
        self.assertEqual(storage._buffer_attr[(1, "value")].offset, 2304)

    def test_manager_exposes_key_only_disagg_metadata_for_mla(self) -> None:
        integration = importlib.import_module("kvbm.trtllm_integration")

        class _FakeTensor:
            def __init__(self, shape: tuple[int, ...], strides: tuple[int, ...], ptr: int) -> None:
                self.shape = shape
                self._strides = strides
                self._ptr = ptr

            def stride(self, dim: int) -> int:
                return self._strides[dim]

            def element_size(self) -> int:
                return 2

            def data_ptr(self) -> int:
                return self._ptr

        manager = integration.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=576,
            pp_layers=[0, 1],
            total_num_kv_heads_per_layer=[1, 1],
            max_seq_len=64,
            num_blocks=8,
            cache_mode="mla",
            primary_pool=_FakeTensor(
                shape=(8, 2, 1, 4, 1, 576),
                strides=(4608, 2304, 2304, 576, 576, 1),
                ptr=8192,
            ),
        )

        storage = manager.impl._storage

        self.assertEqual(sorted(storage._buffer_attr), [(0, "key"), (1, "key")])
        self.assertEqual(storage._buffer_attr[(0, "key")].size, 4608)
        self.assertEqual(storage._buffer_attr[(1, "key")].offset, 4608)
        with self.assertRaises(NotImplementedError):
            manager.impl.get_indexer_k_cache_pool()

    def test_manager_supports_pinned_trtllm_page_table_builder(self) -> None:
        _, utils_mod, kv_extractor_mod, _ = self._install_pinned_trtllm_disagg_stubs()
        integration = importlib.import_module("kvbm.trtllm_integration")

        class _FakeTensor:
            def __init__(self, shape: tuple[int, ...], strides: tuple[int, ...], ptr: int) -> None:
                self.shape = shape
                self._strides = strides
                self._ptr = ptr

            def stride(self, dim: int) -> int:
                return self._strides[dim]

            def element_size(self) -> int:
                return 2

            def data_ptr(self) -> int:
                return self._ptr

        manager = integration.KvbmKVCacheManager(
            tokens_per_block=8,
            dtype="float16",
            head_dim=16,
            pp_layers=[4, 5],
            total_num_kv_heads_per_layer=[8, 8, 8, 8, 6, 6],
            max_seq_len=128,
            num_blocks=12,
            primary_pool=_FakeTensor(
                shape=(12, 2, 2, 8, 6, 16),
                strides=(3072, 1536, 768, 96, 16, 1),
                ptr=12288,
            ),
            device_id=2,
            world_size=4,
            tp_size=2,
            tp_rank=1,
            pp_size=2,
            pp_rank=1,
        )

        page_table = kv_extractor_mod.build_page_table_from_manager(manager)
        self.assertEqual(page_table.tokens_per_block, 8)
        self.assertEqual(len(page_table.layer_groups), 1)
        self.assertEqual(len(page_table.pool_groups), 1)
        self.assertEqual(utils_mod.get_total_slots(page_table), 12)
        self.assertEqual(utils_mod.get_total_pool_bytes(page_table), 12 * 6144)
        self.assertEqual(
            utils_mod.get_global_layer_ids(page_table.layer_groups[0]),
            [4, 5],
        )
        pool = page_table.pool_groups[0].pools[0]
        self.assertEqual(pool.base_address, 12288)
        self.assertEqual(pool.slot_bytes, 6144)
        self.assertEqual(pool.num_slots, 12)
        entries = page_table.layer_groups[0].pool_views[0].buffer_entries.tolist()
        self.assertEqual(
            entries,
            [
                (0, 1, 0, 1536),
                (0, 2, 1536, 1536),
                (1, 1, 3072, 1536),
                (1, 2, 4608, 1536),
            ],
        )

    def test_manager_supports_pinned_trtllm_rank_info_builder(self) -> None:
        _, _, _, rank_info_mod = self._install_pinned_trtllm_disagg_stubs()
        integration = importlib.import_module("kvbm.trtllm_integration")

        class _FakeTensor:
            def __init__(self, shape: tuple[int, ...], strides: tuple[int, ...], ptr: int) -> None:
                self.shape = shape
                self._strides = strides
                self._ptr = ptr

            def stride(self, dim: int) -> int:
                return self._strides[dim]

            def element_size(self) -> int:
                return 2

            def data_ptr(self) -> int:
                return self._ptr

        manager = integration.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=576,
            pp_layers=[0, 1],
            total_num_kv_heads_per_layer=[1, 1],
            max_seq_len=64,
            num_blocks=8,
            cache_mode="mla",
            primary_pool=_FakeTensor(
                shape=(8, 2, 1, 4, 1, 576),
                strides=(4608, 2304, 2304, 576, 576, 1),
                ptr=8192,
            ),
            world_size=4,
            tp_size=4,
            tp_rank=2,
            pp_size=1,
            pp_rank=0,
            device_id=7,
        )

        rank_info = rank_info_mod.RankInfo.from_kv_cache_manager(
            instance_name="kvbm-rank",
            kv_cache_manager=manager,
            device_id=7,
        )

        self.assertEqual(rank_info.instance_name, "kvbm-rank")
        self.assertEqual(rank_info.instance_rank, manager.mapping.rank)
        self.assertEqual(rank_info.device_id, 7)
        self.assertEqual(rank_info.attention.kv_heads_per_rank, 1)
        self.assertEqual(rank_info.attention.tokens_per_block, 4)
        self.assertEqual(rank_info.attention.dims_per_head, 576)
        self.assertTrue(rank_info.attention.is_mla)
        self.assertEqual(rank_info.layer_num_per_pp, [2])
        self.assertEqual(rank_info.page_table.tokens_per_block, 4)
        self.assertEqual(rank_info.page_table.pool_groups[0].pools[0].slot_bytes, 9216)
        round_trip = rank_info_mod.RankInfo.from_bytes(rank_info.to_bytes())
        self.assertEqual(round_trip.instance_name, "kvbm-rank")
        self.assertEqual(round_trip.page_table.pool_groups[0].pools[0].num_slots, 8)

    def test_manager_supports_pinned_trtllm_python_transceiver(self) -> None:
        transceiver_mod, transfer_worker_cls = self._install_pinned_trtllm_transceiver_stubs()
        integration = importlib.import_module("kvbm.trtllm_integration")

        class _FakeTensor:
            def __init__(self, shape: tuple[int, ...], strides: tuple[int, ...], ptr: int) -> None:
                self.shape = shape
                self._strides = strides
                self._ptr = ptr

            def stride(self, dim: int) -> int:
                return self._strides[dim]

            def element_size(self) -> int:
                return 2

            def data_ptr(self) -> int:
                return self._ptr

        manager = integration.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=16,
            pp_layers=[4, 5],
            total_num_kv_heads_per_layer=[8, 8, 8, 8, 6, 6],
            max_seq_len=128,
            num_blocks=12,
            primary_pool=_FakeTensor(
                shape=(12, 2, 2, 4, 6, 16),
                strides=(1536, 768, 384, 96, 16, 1),
                ptr=12288,
            ),
            device_id=2,
            world_size=4,
            tp_size=2,
            tp_rank=1,
            pp_size=2,
            pp_rank=1,
        )

        manager.add_dummy_requests([901], token_nums=[12])

        class _Dist:
            rank = 0
            tp_size = 2

            def broadcast(self, value, root):
                del root
                return value if value is not None else "broadcast-value"

            def allgather(self, value):
                return [value, value]

            def pp_allgather(self, value):
                return [value, value]

            def tp_allgather(self, value):
                return [list(value), list(value)]

        config = types.SimpleNamespace(
            kv_transfer_timeout_ms=7000,
            kv_transfer_sender_future_timeout_ms=3000,
        )
        transceiver = transceiver_mod.KvCacheTransceiverV2(
            manager.mapping,
            _Dist(),
            manager,
            config,
        )

        worker = transfer_worker_cls.instances[-1]
        self.assertEqual(worker.config.kv_cache_manager, manager)
        self.assertEqual(worker.config.device_id, 0)
        self.assertEqual(worker.config.max_concurrent_sessions, manager.max_batch_size * 20000)
        self.assertEqual(
            worker.rank_info_calls,
            [{"endpoints": ["sender-0", "sender-0"], "layer_num_per_pp": [2, 2]}],
        )

        request = types.SimpleNamespace(
            request_id=901,
            py_request_id=901,
            py_disaggregated_params=None,
            prompt_len=12,
            state=None,
        )
        transceiver.respond_and_send_async(request)
        self.assertEqual(request.state, "ctx_in_progress")
        self.assertEqual(request.context_phase_params.ctx_dp_rank, 0)
        self.assertEqual(request.context_phase_params.disagg_info_endpoint, "ctx-endpoint")
        self.assertEqual(
            worker.tx_sessions[0].sent_slices[0].block_ids_per_layer_groups,
            [[0, 1, 2]],
        )

        completed, failed = transceiver.check_context_transfer_status(1, mark_complete=True)
        self.assertEqual((completed, failed), ([901], []))
        self.assertEqual(request.state, "ctx_complete")

        transceiver.request_and_receive_async(request)
        self.assertEqual(request.state, "gen_in_progress")
        self.assertEqual(
            worker.rx_sessions[0].received_slices[0].block_ids_per_layer_groups,
            [[0, 1, 2]],
        )

        completed, failed = transceiver.check_gen_transfer_status(1)
        self.assertEqual((completed, failed), ([901], []))
        self.assertEqual(request.state, "gen_complete")
        self.assertTrue(transceiver.check_gen_transfer_complete())

        waiting_request = types.SimpleNamespace(
            request_id=902,
            py_request_id=902,
            py_disaggregated_params=None,
            prompt_len=4,
            state=None,
        )
        transceiver.prepare_context_requests([waiting_request])
        self.assertEqual(waiting_request.state, "ctx_init")
        self.assertEqual(
            transceiver.get_disaggregated_params(),
            {"ctx_dp_rank": 0, "ctx_info_endpoint": ["ctx-endpoint"]},
        )

        transceiver.shutdown()
        self.assertEqual(worker.shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main()
