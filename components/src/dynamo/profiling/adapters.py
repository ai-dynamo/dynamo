# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted collector registration; requests never choose commands or imports."""

from pathlib import Path

from .collectors import PerfCollector, TorchCollector, selected_workers
from .models import Config, Plan, RuntimeSnapshot
from .nsys import nsys_collectors


def torch_collectors(
    config: Config, runtime: RuntimeSnapshot, plan: Plan, directory: Path
) -> list:
    # vLLM incorporates this prefix in every engine-rank filename. An
    # immutable session-specific prefix prevents stale files from counting.
    workers = selected_workers(
        runtime, plan, "vllm", profile_prefix=f"dynamo-{directory.name}"
    )
    return (
        [PerfCollector(runtime.frontend_pid, runtime.frontend_epoch, directory)]
        if config.frontend_perf_enabled
        else []
    ) + [TorchCollector(worker, config) for worker in workers if worker is not None]


# Mode -> factory(config, runtime, plan, directory) returning collectors.
REGISTRY = {"torch_profile": torch_collectors, "nsys": nsys_collectors}
