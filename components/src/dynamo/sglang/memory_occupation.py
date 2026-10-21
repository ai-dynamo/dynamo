# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang plugin that makes a redundant ``resume_memory_occupation`` safe."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_RESUME_MEMORY_OCCUPATION_TARGET = (
    "sglang.srt.managers.scheduler_components.weight_updater."
    "SchedulerWeightUpdaterManager.resume_memory_occupation"
)


def register_memory_occupation_plugin() -> None:
    """Resume only the memory regions the scheduler has actually released.

    SGLang's scheduler resume removes every requested tag from
    ``offload_tags`` without checking it, so a second resume (or one that names
    a tag that was never released) raises ``KeyError`` and kills the scheduler.
    Only the scheduler knows which regions are released, so the guard runs
    there as a hook rather than in Dynamo's request handlers.
    """
    from sglang.srt.constants import GPU_MEMORY_ALL_TYPES
    from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqOutput
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def around_resume_memory_occupation(original, manager, recv_req):
        requested = recv_req.tags or GPU_MEMORY_ALL_TYPES
        released = [tag for tag in requested if tag in manager.offload_tags]
        if not released:
            logger.warning(
                "Ignoring resume_memory_occupation for %s: not released",
                list(requested),
            )
            return ResumeMemoryOccupationReqOutput()
        if len(released) != len(requested):
            logger.warning(
                "resume_memory_occupation: skipping %s, not released",
                [tag for tag in requested if tag not in released],
            )
            recv_req.tags = released
        return original(manager, recv_req)

    HookRegistry.register(
        _RESUME_MEMORY_OCCUPATION_TARGET,
        around_resume_memory_occupation,
        HookType.AROUND,
    )
