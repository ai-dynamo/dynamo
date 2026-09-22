# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import msgspec
import pytest
import pytest_asyncio
import torch
import zmq
import zmq.asyncio

from dynamo.llm import (
    KvEventPublisher,
    KvRouter,
    KvRouterConfig,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime
from dynamo.vllm import kv_cache_metadata_compat
from dynamo.vllm.cache_info import configure_kv_event_block_size

pytestmark = [
    pytest.mark.integration,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest_asyncio.fixture
async def runtime(monkeypatch):
    monkeypatch.setenv("DYN_ROUTER_MIN_INITIAL_WORKERS", "0")
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
    )
    try:
        yield runtime
    finally:
        runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_dcp_events_match_configured_block_size(monkeypatch, runtime):
    from vllm.distributed.kv_events import KVEventBatch
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.request import Request

    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    monkeypatch.setenv(kv_cache_metadata_compat._ACTIVATION_ENV, "1")
    monkeypatch.setattr(
        EngineCore,
        "get_kv_cache_group_metadata",
        getattr(EngineCore, "get_kv_cache_group_metadata", None),
        raising=False,
    )
    monkeypatch.delattr(EngineCore, "get_kv_cache_group_metadata")
    kv_cache_metadata_compat.register()
    monkeypatch.setattr(
        kv_cache_utils,
        "NONE_HASH",
        getattr(kv_cache_utils, "NONE_HASH", None),
        raising=False,
    )
    kv_cache_utils.init_none_hash(sha256)
    cache_config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["full"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16
                ),
            )
        ],
    )
    manager = KVCacheManager(
        cache_config,
        max_model_len=128,
        scheduler_block_size=32,
        hash_block_size=32,
        dcp_world_size=2,
        enable_kv_cache_events=True,
    )
    core = EngineCore.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        kv_cache_config=cache_config, kv_cache_manager=manager
    )
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(
            call_utility_async=AsyncMock(
                side_effect=lambda method: getattr(core, method)()
            )
        )
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16), additional_config={}
    )
    block_size = await configure_kv_event_block_size(engine, config)
    tokens = list(range(1, 97))
    request = Request(
        request_id="dcp-prefix",
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=kv_cache_utils.get_request_block_hasher(32, sha256),
    )
    assert manager.allocate_slots(request, num_new_tokens=len(tokens)) is not None
    batch = KVEventBatch(ts=0.0, events=manager.take_events(), data_parallel_rank=0)

    endpoint = runtime.endpoint("dcp.worker.generate")
    await endpoint.register_endpoint_instance()
    runtime_config = ModelRuntimeConfig()
    runtime_config.kv_event_publishing_enabled = True
    # Tensor registration supplies discovery metadata without loading a model.
    await register_model(
        ModelInput.Tensor,
        ModelType.TensorBased,
        endpoint,
        "cpu-dcp",
        worker_type=WorkerType.Aggregated,
        runtime_config=runtime_config,
    )
    with zmq.asyncio.Context() as context, context.socket(zmq.XPUB) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        publisher = KvEventPublisher(
            endpoint,
            kv_block_size=block_size,
            enable_local_indexer=True,
            zmq_endpoint=f"tcp://127.0.0.1:{port}",
            zmq_topic="kv-events",
        )
        try:
            router = KvRouter(endpoint, block_size, KvRouterConfig(use_kv_events=True))
            assert await asyncio.wait_for(socket.recv(), timeout=5) == b"\x01kv-events"
            await socket.send_multipart(
                [b"kv-events", (0).to_bytes(8, "big"), msgspec.msgpack.encode(batch)]
            )
            deadline = asyncio.get_running_loop().time() + 10
            while True:
                scores = await router.get_overlap_scores(tokens, include_shared=False)
                if any(
                    row["worker_id"] == endpoint.connection_id()
                    and row["dp_rank"] == 0
                    and row["device_blocks"] == 3
                    for row in scores["workers"]
                ):
                    break
                assert (
                    asyncio.get_running_loop().time() < deadline
                ), f"Expected three matching DCP prefix blocks, got {scores}"
                await asyncio.sleep(0.01)
            assert scores["block_size"] == 32
            assert scores["num_blocks"] == 3
            assert config.cache_config.block_size == 16
        finally:
            publisher.shutdown()
