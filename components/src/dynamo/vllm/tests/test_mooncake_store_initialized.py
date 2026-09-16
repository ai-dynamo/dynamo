# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned connector initialization, with only transport and topology replaced.

This test requires the exact installed vLLM build but no model, Mooncake process,
or GPU. It does not validate tensor registration, transfer, or inference.
"""

import asyncio
import copy
import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from dynamo.vllm import mooncake_store_runtime as runtime

pytestmark = [
    pytest.mark.integration,
    pytest.mark.vllm,
    pytest.mark.router,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize("other_kind", ["sliding_window", "mamba"])
@pytest.mark.parametrize("wrapped", [False, True], ids=["direct", "multi"])
def test_initialized_pinned_connector_exports_all_rank_contract(
    tmp_path, monkeypatch, other_kind, wrapped
):
    pytest.importorskip("torch")
    pytest.importorskip("vllm.config")
    try:
        runtime._verify_pinned_sources()
    except runtime.UnsupportedContract as exc:
        pytest.skip(str(exc))

    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
        MooncakeStoreConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        MambaManager,
        SlidingWindowManager,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
        SlidingWindowSpec,
    )
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    registry = importlib.import_module("vllm.v1.kv_cache_spec_registry")
    monkeypatch.setattr(registry, "_REGISTRY_KVCACHESPEC_LIST", {})
    monkeypatch.setattr(registry, "_REGISTRY_ROLE_MANAGERS", {})
    for spec, manager in (
        (FullAttentionSpec, FullAttentionManager),
        (SlidingWindowSpec, SlidingWindowManager),
        (MambaSpec, MambaManager),
    ):
        KVCacheSpecRegistry.register(spec, manager)

    stores = []

    def create_store():
        store = Mock()
        store.setup.return_value = 0
        stores.append(store)
        return store

    fake_mooncake = ModuleType("mooncake")
    fake_store = ModuleType("mooncake.store")
    fake_store.MooncakeDistributedStore = create_store
    fake_store.ReplicateConfig = SimpleNamespace
    fake_mooncake.store = fake_store
    monkeypatch.setitem(sys.modules, "mooncake", fake_mooncake)
    monkeypatch.setitem(sys.modules, "mooncake.store", fake_store)
    path = tmp_path / "mooncake.json"
    path.write_text(
        json.dumps(
            {
                "metadata_server": "http://127.0.0.1/metadata",
                "global_segment_size": "4gb",
                "local_buffer_size": "64mb",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "127.0.0.1:50051",
            }
        )
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(path))
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    worker_module = importlib.import_module(runtime._STORE_MODULE + ".worker")
    monkeypatch.setattr(
        worker_module, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        worker_module, "get_pcp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(
        worker_module, "get_dcp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(worker_module, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(worker_module, "LookupKeyServer", Mock())

    full = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=None)
    other = (
        SlidingWindowSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=None, sliding_window=32
        )
        if other_kind == "sliding_window"
        else MambaSpec(
            block_size=16, shapes=((1,),), dtypes=(None,), mamba_cache_mode="align"
        )
    )
    kv_cache = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full"], full),
            KVCacheGroupSpec(["other"], other),
        ],
    )
    child_config = {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "cache_prefix": "test-deployment",
            "enable_lookup": True,
        },
    }
    transfer_config = KVTransferConfig(
        **(
            {
                "kv_connector": "MultiConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {"connectors": [child_config]},
            }
            if wrapped
            else child_config
        )
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="test-model",
            use_mla=False,
            is_multimodal_model=False,
            enable_prompt_embeds=False,
            is_encoder_decoder=False,
            runner_type="generate",
            get_num_layers=lambda _: 2,
            get_total_num_kv_heads=lambda: 2,
        ),
        parallel_config=SimpleNamespace(
            rank=0,
            world_size=2,
            data_parallel_index=0,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            worker_extension_cls=runtime.WORKER_EXTENSION,
        ),
        cache_config=SimpleNamespace(
            block_size=16,
            num_gpu_blocks=10,
            prefix_match_unit=None,
            enable_prefix_caching=True,
            prefix_caching_hash_algo="sha256",
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False, scheduler_cls=None
        ),
        kv_events_config=SimpleNamespace(enable_kv_cache_events=True),
        speculative_config=None,
        kv_transfer_config=transfer_config,
    )
    transfer = importlib.import_module("vllm.distributed.kv_transfer")
    connectors = []
    replies = []
    try:
        for rank in range(2):
            rank_config = copy.copy(config)
            rank_config.parallel_config = copy.copy(config.parallel_config)
            rank_config.parallel_config.rank = rank
            monkeypatch.setattr(
                worker_module, "get_tensor_model_parallel_rank", lambda rank=rank: rank
            )
            cls = MultiConnector if wrapped else MooncakeStoreConnector
            connector = cls(rank_config, KVConnectorRole.WORKER, kv_cache)
            connectors.append(connector)
            monkeypatch.setattr(
                transfer, "get_kv_transfer_group", lambda connector=connector: connector
            )
            ext = runtime.MooncakeStoreWorkerExtension()
            ext.vllm_config = rank_config
            reply = ext.dynamo_mooncake_store_descriptor()
            assert "unsupported" not in reply, reply
            resolved = runtime._find_store_connector(connector).connector_worker
            for group, prefixes in zip(
                reply["descriptor"]["groups"],
                resolved._lookup_key_prefixes,
                strict=True,
            ):
                assert group["key_prefixes"] == sorted(prefixes)
                assert len(prefixes) == 2
            replies.append(reply)
        metadata = SimpleNamespace(set_engine_specific=Mock())
        engine = SimpleNamespace(collective_rpc=AsyncMock(return_value=replies))
        assert asyncio.run(
            runtime.publish_mooncake_store_runtime(
                metadata,
                engine,
                config,
                dp_range=(0, 1),
                event_span=16,
            )
        )
        key, value = metadata.set_engine_specific.call_args.args
        assert key == runtime.RUNTIME_KEY
        assert json.loads(value) == replies[0]["descriptor"]
    finally:
        for connector in connectors:
            connector.shutdown()
    assert len(stores) == 2
    for store in stores:
        store.setup.assert_called_once()
        store.close.assert_called_once()
