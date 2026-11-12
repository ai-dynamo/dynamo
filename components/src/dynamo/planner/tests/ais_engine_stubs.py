# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stand-in for the native AIS engine model.

Tests that only read the config the Planner's perf model adapter builds get
the adapter from ``adapter_builder``, so no native engine model is created.
This is a plain module, not a conftest, so that tests of another component
can import it.
"""

from collections.abc import Callable

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.perf_model import ais_adapter
from dynamo.planner.core.perf_model.ais_adapter import PlannerEnginePerfModel
from dynamo.planner.core.perf_model.engine_query import WorkerType
from dynamo.planner.core.types import EngineCapabilities


class ReadyEngine:
    """Reports that it is ready; a test that asks it anything else fails."""

    @classmethod
    def best_available(cls, **kwargs):
        return cls()

    def diagnostics(self):
        return {"readiness": "ready"}


def engine_limits() -> EngineCapabilities:
    return EngineCapabilities(
        max_num_batched_tokens=1024, max_num_seqs=128, max_kv_tokens=100_000
    )


def adapter_builder(monkeypatch) -> Callable[..., PlannerEnginePerfModel]:
    """Install ``ReadyEngine``; return a function that builds the adapter for a config.

    With ``limits=False`` the adapter is built without reading the perf model,
    as when the workers have not reported their limits yet.
    """
    monkeypatch.setattr(ais_adapter, "AISCoreEnginePerfModel", ReadyEngine)

    def build(
        config: PlannerConfig, worker_type: WorkerType = "decode", limits: bool = True
    ) -> PlannerEnginePerfModel:
        return PlannerEnginePerfModel(
            worker_type=worker_type,
            config=config,
            capabilities=engine_limits() if limits else None,
        )

    return build
