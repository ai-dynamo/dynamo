# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted collector registration; requests never choose commands or imports."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import msgspec

from .collectors import PerfCollector, TorchCollector
from .models import Config, Plan, RuntimeSnapshot


class Collector(Protocol):
    evidence: dict

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...


Factory = Callable[[Config, RuntimeSnapshot, Plan, Path], list[Collector]]


class CollectorRegistry:
    """The same registry can be supplied to an embedded or standalone host."""

    def __init__(self) -> None:
        self.factories: dict[str, Factory] = {}

    def register(self, mode: str, factory: Factory) -> None:
        if mode in self.factories:
            raise ValueError(f"Collector mode already registered: {mode}")
        self.factories[mode] = factory

    def supports(self, mode: str) -> bool:
        return mode in self.factories

    def create(
        self, config: Config, runtime: RuntimeSnapshot, plan: Plan, directory: Path
    ) -> list[Collector]:
        return self.factories[plan.mode](config, runtime, plan, directory)


def torch_collectors(
    config: Config, runtime: RuntimeSnapshot, plan: Plan, directory: Path
) -> list[Collector]:
    collectors: list[Collector] = []
    if config.frontend_perf_enabled:
        collectors.append(
            PerfCollector(runtime.frontend_pid, runtime.frontend_epoch, directory)
        )
    workers = {worker.target_id: worker for worker in runtime.workers}
    for target in plan.targets:
        if target.target_id == "frontend":
            continue
        worker = workers[target.target_id]
        # vLLM incorporates this prefix in every engine-rank filename. An
        # immutable session-specific prefix prevents stale files from counting.
        if worker.backend == "vllm":
            options = {
                **worker.start_options,
                "profile_prefix": f"dynamo-{directory.name}",
            }
            worker = msgspec.structs.replace(worker, start_options=options)
        collectors.append(TorchCollector(worker, config))
    return collectors


def default_registry() -> CollectorRegistry:
    registry = CollectorRegistry()
    registry.register("torch_profile", torch_collectors)
    # An nsys provider must be explicitly installed and check prepared sessions.
    return registry
