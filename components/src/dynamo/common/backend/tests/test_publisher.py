# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Optional

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
)

from dynamo._core import Context  # noqa: E402
from dynamo.common.backend.engine import (  # noqa: E402
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.publisher import (  # noqa: E402
    ComponentSnapshot,
    PushSource,
    ZmqSource,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_source_descriptors_carry_payload_and_defaults():
    zmq = ZmqSource(endpoint="tcp://127.0.0.1:5557")
    assert (zmq.endpoint, zmq.topic, zmq.dp_rank) == ("tcp://127.0.0.1:5557", "", 0)

    seen: list[object] = []
    push = PushSource(on_ready=seen.append, dp_rank=2)
    push.on_ready("publisher")
    assert seen == ["publisher"]
    assert push.dp_rank == 2

    # ComponentSnapshot is the payload engines push into SnapshotPublisher.
    snap = ComponentSnapshot(
        kv_used_blocks=42, kv_total_blocks=100, gpu_cache_usage=0.42, dp_rank=0
    )
    assert snap.dp_rank == 0
    assert snap.kv_cache_hit_rate is None


class _MinimalEngine(LLMEngine):
    @classmethod
    async def from_args(cls, argv: Optional[list[str]] = None):
        raise NotImplementedError

    async def start(self, worker_id: int) -> EngineConfig:
        return EngineConfig(model="minimal")

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        yield {"token_ids": [], "index": 0, "finish_reason": "stop"}

    async def cleanup(self) -> None:
        pass


@pytest.mark.asyncio
async def test_abc_source_methods_default_to_empty_list():
    engine = _MinimalEngine()
    assert await engine.kv_event_sources() == []
    assert engine.component_metrics_dp_ranks() == []
    # attach_snapshot_publisher default is no-op
    engine.attach_snapshot_publisher(object())


@pytest.mark.asyncio
async def test_register_prometheus_default_is_noop():
    assert await _MinimalEngine().register_prometheus(metrics=object()) is None


@pytest.mark.asyncio
async def test_vllm_kv_event_sources_return_one_zmq_source_per_dp_rank(monkeypatch):
    mod = pytest.importorskip(
        "dynamo.vllm.llm_engine", reason="vLLM backend dependencies not installed"
    )
    from dynamo.common.constants import DisaggregationMode

    engine = mod.VllmLLMEngine.__new__(mod.VllmLLMEngine)
    engine.engine_args = SimpleNamespace(
        enable_prefix_caching=True,
        kv_events_config=SimpleNamespace(
            enable_kv_cache_events=True,
            endpoint="tcp://*:5557",
        ),
    )
    engine.disaggregation_mode = DisaggregationMode.AGGREGATED
    engine._vllm_config = object()
    engine._dp_range = (2, 3)

    monkeypatch.setattr(
        mod.ZmqEventPublisher,
        "offset_endpoint_port",
        staticmethod(
            lambda endpoint, data_parallel_rank: f"{endpoint}-{data_parallel_rank}"
        ),
    )

    sources = await engine.kv_event_sources()

    assert all(isinstance(source, ZmqSource) for source in sources)
    assert [(source.endpoint, source.dp_rank) for source in sources] == [
        ("tcp://127.0.0.1:5557-2", 2),
        ("tcp://127.0.0.1:5557-3", 3),
        ("tcp://127.0.0.1:5557-4", 4),
    ]

    engine.engine_args.enable_prefix_caching = False
    assert await engine.kv_event_sources() == []


@pytest.mark.asyncio
async def test_sglang_kv_event_sources_return_one_zmq_source_per_local_dp_rank(
    monkeypatch,
):
    mod = pytest.importorskip(
        "dynamo.sglang.llm_engine", reason="SGLang backend dependencies not installed"
    )

    engine = mod.SglangLLMEngine.__new__(mod.SglangLLMEngine)
    engine.server_args = SimpleNamespace(
        kv_events_config=json.dumps({"endpoint": "tcp://*:5557"}),
        dp_size=8,
        enable_dp_attention=True,
        nnodes=2,
        node_rank=1,
    )

    monkeypatch.setattr(mod, "get_local_ip_auto", lambda: "127.0.0.1")
    monkeypatch.setattr(
        mod.ZmqEventPublisher,
        "offset_endpoint_port",
        staticmethod(lambda _endpoint, dp_rank: f"tcp://*:{6000 + dp_rank}"),
    )

    sources = await engine.kv_event_sources()

    assert all(isinstance(source, ZmqSource) for source in sources)
    assert [(source.endpoint, source.dp_rank) for source in sources] == [
        ("tcp://127.0.0.1:6004", 4),
        ("tcp://127.0.0.1:6005", 5),
        ("tcp://127.0.0.1:6006", 6),
        ("tcp://127.0.0.1:6007", 7),
    ]


@pytest.mark.asyncio
async def test_trtllm_push_sources_wait_for_all_attention_dp_publishers(monkeypatch):
    mod = pytest.importorskip(
        "dynamo.trtllm.llm_engine",
        reason="TensorRT-LLM backend dependencies not installed",
    )

    started_threads: list[str] = []

    class FakeThread:
        def __init__(self, *, target, daemon, name):
            self.target = target
            self.daemon = daemon
            self.name = name

        def start(self):
            started_threads.append(self.name)

    engine = mod.TrtllmLLMEngine.__new__(mod.TrtllmLLMEngine)
    engine.publish_events_and_metrics = True
    engine._attention_dp_size = 3
    engine._kv_publishers = {}
    engine._kv_events_thread = None
    engine._kv_events_poll_loop = lambda: None

    monkeypatch.setattr(mod.threading, "Thread", FakeThread)

    sources = await engine.kv_event_sources()

    assert all(isinstance(source, PushSource) for source in sources)
    assert [source.dp_rank for source in sources] == [0, 1, 2]

    sources[0].on_ready("publisher-0")
    sources[2].on_ready("publisher-2")
    assert started_threads == []

    sources[1].on_ready("publisher-1")
    assert sorted(engine._kv_publishers.items()) == [
        (0, "publisher-0"),
        (1, "publisher-1"),
        (2, "publisher-2"),
    ]
    assert started_threads == ["trtllm-kv-events-poll"]

    engine.publish_events_and_metrics = False
    assert await engine.kv_event_sources() == []
