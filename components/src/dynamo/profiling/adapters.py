# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted collector registration; requests never choose commands or imports."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .collectors import PerfCollector, TorchCollector, selected_workers
from .models import Config, Plan, RuntimeSnapshot
from .nsys import nsys_collectors


class Collector(Protocol):
    """One target's capture: records `evidence`; start/stop raise on failure."""

    evidence: dict

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...


# Builds the collectors for one session of a mode.
Factory = Callable[[Config, RuntimeSnapshot, Plan, Path], list[Collector]]


def torch_collectors(
    config: Config, runtime: RuntimeSnapshot, plan: Plan, directory: Path
) -> list[Collector]:
    # vLLM and SGLang incorporate this prefix in every engine-rank filename. An
    # immutable session-specific prefix prevents stale files from counting.
    workers = selected_workers(
        runtime, plan, {"vllm", "sglang"}, profile_prefix=f"dynamo-{directory.name}"
    )
    return (
        [PerfCollector(runtime.frontend_pid, runtime.frontend_epoch, directory)]
        if config.frontend_perf_enabled
        else []
    ) + [TorchCollector(worker, config) for worker in workers if worker is not None]


REGISTRY: dict[str, Factory] = {
    "torch_profile": torch_collectors,
    "nsys": nsys_collectors,
}
