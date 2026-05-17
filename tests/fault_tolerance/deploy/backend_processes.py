# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Per-backend process-name matchers for use with the StallProcess /
# TerminateProcess / RankProcessCount primitives.
#
# Process tree inside a vLLM worker pod at TP=N:
#
#     VLLM::EngineCore           # the engine / coordinator
#       \_ VLLM::Worker_TP0_EP0   # rank 0
#       \_ VLLM::Worker_TP1_EP1   # rank 1
#       ...
#       \_ VLLM::Worker_TP(N-1)_EP(N-1)
#
# Targeting:
#   - "EngineCore" → break the whole pod's coordinator
#   - "Worker"     → break a TP rank (substring matches all VLLM::Worker_TP*)
#
# Pass these as ``process_name=VLLM.engine_core`` /
# ``process_name=VLLM.worker``. Use ``rank_index=0`` on
# StallProcess / TerminateProcess to pick a specific TP rank
# (sorted by pid).
#
# When we add other backends (trtllm, sglang) add a sibling namespace
# below with their respective matchers. Keep tests backend-agnostic by
# importing the constants rather than hard-coding strings.

from dataclasses import dataclass


@dataclass(frozen=True)
class _Backend:
    name: str
    engine_core: str  # the coordinator / leader process
    worker: str       # the TP rank / per-GPU worker process


VLLM = _Backend(
    name="vllm",
    engine_core="VLLM::EngineCore",
    worker="VLLM::Worker",
)
