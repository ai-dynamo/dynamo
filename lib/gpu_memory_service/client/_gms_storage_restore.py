# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

from gpu_memory_service.client._gms_storage_model import AllocationEntry

WORK_QUEUE_DEPTH_MULTIPLIER = 4


@dataclass
class RestorePipelineContext:
    """Mutable state shared across disk, copy, and Phase A restore stages."""

    worker_count: int
    use_streams: bool
    device: int
    work_q: queue.Queue[Optional[Tuple[AllocationEntry, torch.Tensor]]]
    va_events: Dict[str, threading.Event]
    streams: List[torch.cuda.Stream]
    cancel_event: threading.Event = field(default_factory=threading.Event)
    vas: Dict[str, int] = field(default_factory=dict)
    staged_srcs: List[torch.Tensor] = field(default_factory=list)
    copy_errors: List[BaseException] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def build(
        cls,
        allocations: List[AllocationEntry],
        worker_count: int,
        *,
        device: int,
        use_streams: bool,
        torch_module,
    ) -> "RestorePipelineContext":
        streams = (
            [torch_module.cuda.Stream(device=device) for _ in range(worker_count)]
            if use_streams
            else []
        )
        return cls(
            worker_count=worker_count,
            use_streams=use_streams,
            device=device,
            work_q=queue.Queue(maxsize=worker_count * WORK_QUEUE_DEPTH_MULTIPLIER),
            va_events={entry.allocation_id: threading.Event() for entry in allocations},
            streams=streams,
        )


@dataclass
class RestorePipelineResources:
    """Live restore pipeline resources that must be torn down together."""

    ctx: RestorePipelineContext
    disk_pool: ThreadPoolExecutor
    disk_futures: Dict[Future[int], str]
    copy_threads: List[threading.Thread]
    active: bool = True
