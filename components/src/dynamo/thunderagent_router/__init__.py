# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ThunderAgent-style program-level scheduler for Dynamo.

A standalone routing service that wraps Dynamo's native ``KvRouter`` with
program lifecycle tracking (REASONING/ACTING/PAUSED), tool-boundary pause/
resume, and KV-aware resume placement.

Mirrors the integration pattern of ``dynamo.thompson_router`` (PR #8522): it is
a ``dynamo_worker`` that owns its own ``KvRouter`` instance and serves
``{namespace}.thunderagent_router.generate``. The frontend discovers this
endpoint and dispatches to it.

Differs from the original ThunderAgent (Python proxy) in v0:

1. Real token accounting from the request's preprocessed ``token_ids``;
2. KV-aware resume placement via ``KvRouter.best_worker(last_prefix)``;
3. Engine-true capacity from the FPM event plane (not Prometheus polling);
4. Soft pause via priority demotion before hard pause.

Items 4/5/7 from the differentiator list (workflow profile, subagent-aware
lifecycle, fairness aging) are deferred to follow-up PRs gated on ablation
results.

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
from dynamo.thunderagent_router.router import (
    KvThunderAgentRouter,
    PauseDecision,
)

__all__ = [
    "KvThunderAgentRouter",
    "PauseDecision",
    "Program",
    "ProgramLifecycle",
    "ProgramStatus",
    "ProgramTable",
]
