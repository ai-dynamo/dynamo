# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ThunderAgent program scheduler inside a Dynamo router service. **Experimental.**

The scheduler algorithm is upstream ThunderAgent's. v0 makes two
mechanical changes:

1. Real-token accounting from chat-completions ``usage`` instead of a
   ``chars / 5`` estimator.
2. Multi-worker BFD packing on resume (upstream is single-backend).

The integration shape -- a ``dynamo_worker`` inside the request path
that owns its own ``KvRouter`` instance, rather than a Python OpenAI
proxy in front of the engine -- is what makes both of those possible
and unlocks the optional ``dynamo.agent.trace.v1`` event stream for
offline analysis.

An opt-in ``kv_aware_resume_enabled`` flag exists for the hard-override
ablation on resume worker selection. Default off; the override costs
6-11% spm vs BFD at 128 concurrent agents (README section 4).

The more substantial deviations from upstream (blended cost-function
worker selection, workflow-profile-aware pause, KV demote/prefetch) are
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
