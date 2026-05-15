# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncGenerator
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
    ComponentMetricsSource,
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

    component_snap = ComponentSnapshot(
        kv_used_blocks=42, kv_total_blocks=100, gpu_cache_usage=0.42, dp_rank=0
    )
    src = ComponentMetricsSource(snapshot=lambda: component_snap)
    assert src.snapshot() == component_snap
    assert src.dp_rank == 0


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
    assert engine.component_metrics_sources() == []


@pytest.mark.asyncio
async def test_register_prometheus_default_is_noop():
    assert await _MinimalEngine().register_prometheus(metrics=object()) is None
