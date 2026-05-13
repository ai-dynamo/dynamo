# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ThunderAgent-style program-level scheduler for Dynamo. **Experimental.**

A standalone routing service that wraps Dynamo's native ``KvRouter`` with
program lifecycle tracking (REASONING/ACTING/PAUSED) and tool-boundary
pause/resume.

Mirrors the integration pattern of ``dynamo.thompson_router`` (PR #8522): it is
a ``dynamo_worker`` that owns its own ``KvRouter`` instance and serves
``{namespace}.thunderagent_router.generate``. The frontend discovers this
endpoint and dispatches to it.

Differs from the original ThunderAgent (Python proxy):

1. Real token accounting from the chat-completions ``usage`` field, not a
   ``chars / 5`` estimator.
2. Engine-true capacity from the FPM event plane at sub-second cadence,
   not 5-second Prometheus polling.
3. Working-set projection with a ``pause_target`` setpoint, so pause
   cycles drive util back to a stable point instead of stalling under
   threshold.
4. Asymmetric ACTING-token weighting: full weight on the pause side,
   exponential decay on the resume side (mirrors upstream's
   ``remaining_capacity_with_decay``).
5. Soft pause via priority demotion before hard pause.
6. BFD load-balance on resume worker selection by default. An opt-in
   ``kv_aware_resume_enabled`` flag exists for the hard-override
   ablation; multi-worker benchmarks showed it concentrates load and
   costs throughput. See README section 4.

Workflow-profile-aware pause selection, subagent-aware lifecycle, and
fairness aging are tracked on the roadmap (README section 5).

Usage:
    python -m dynamo.thunderagent_router \\
        --endpoint dynamo.vllm.generate \\
        --router-block-size 64
"""

from dynamo.thunderagent_router.program_state import (
    Program,
    ProgramLifecycle,
    ProgramStatus,
    ProgramTable,
)
from dynamo.thunderagent_router.router import KvThunderAgentRouter, PauseDecision

__all__ = [
    "KvThunderAgentRouter",
    "PauseDecision",
    "Program",
    "ProgramLifecycle",
    "ProgramStatus",
    "ProgramTable",
]
