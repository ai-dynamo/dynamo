# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ThunderAgent program scheduler inside a Dynamo router service. **Experimental.**

A native re-implementation of the upstream ThunderAgent algorithm as a
``dynamo_worker`` that owns its own ``KvRouter`` instance and registers
as a model handler. v0's one mechanical change versus the upstream
reference is real-token accounting from chat-completions ``usage``
instead of the proxy's ``chars / 5`` estimator -- available to us
because the router runs in-path rather than out in front.

The substantial deviations from upstream (blended cost-function worker
selection, workflow-profile-aware pause, KV demote/prefetch) are
explicitly future work.

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
