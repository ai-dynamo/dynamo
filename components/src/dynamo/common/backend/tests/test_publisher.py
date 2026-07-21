# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
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
from dynamo.common.constants import DisaggregationMode  # noqa: E402

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


# The reference SampleLLMEngine exercises the retained publisher/KV-event
# contract end-to-end without any backend framework installed. Per-backend
# metric extraction is covered by each backend's own unit tests.
@pytest.mark.asyncio
async def test_sample_engine_declares_dp_ranks_and_kv_event_source():
    from dynamo.common.backend.sample_engine import SampleLLMEngine

    engine = SampleLLMEngine.__new__(SampleLLMEngine)

    engine.disaggregation_mode = DisaggregationMode.AGGREGATED
    assert engine.component_metrics_dp_ranks() == [0]
    sources = await engine.kv_event_sources()
    assert len(sources) == 1
    assert isinstance(sources[0], PushSource)
    assert sources[0].dp_rank == 0

    # Encode workers host neither the component gauges nor a KV-event source.
    engine.disaggregation_mode = DisaggregationMode.ENCODE
    assert engine.component_metrics_dp_ranks() == []
    assert await engine.kv_event_sources() == []


def test_sample_engine_publish_loop_pushes_component_snapshot():
    """An idle publish tick pushes a ComponentSnapshot to the attached
    publisher — the same path real engines use to feed the snapshot gauge."""
    from dynamo.common.backend.sample_engine import SampleLLMEngine

    engine = SampleLLMEngine.__new__(SampleLLMEngine)
    engine._kv_used_blocks = 25
    engine._publish_stop = threading.Event()

    published: list[tuple[int, ComponentSnapshot]] = []

    def _publish(rank, snapshot):
        published.append((rank, snapshot))
        engine._publish_stop.set()  # one tick, then let the loop exit

    engine.attach_snapshot_publisher(SimpleNamespace(publish=_publish))
    assert engine._snapshot_publisher is not None

    class _AlwaysEmpty:
        def get(self, timeout):
            raise queue.Empty

    engine._publish_queue = _AlwaysEmpty()

    engine._publish_loop(publisher=None)

    assert published == [
        (
            0,
            ComponentSnapshot(
                kv_used_blocks=25,
                kv_total_blocks=1000,
                gpu_cache_usage=0.025,
                kv_cache_hit_rate=None,
                dp_rank=0,
            ),
        )
    ]
