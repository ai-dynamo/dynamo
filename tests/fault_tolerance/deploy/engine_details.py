# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-backend engine process details — the single seam for how inference
backends differ, so a ``StallProcess``-based repro is backend-agnostic.

A scenario refers to a *semantic target* (``main`` | ``engine`` | ``worker``)
and this module resolves it to the concrete in-pod **process-name substring**
that ``StallProcess`` / ``TerminateProcess`` match against (the ps cmdline),
per backend. It folds together knowledge that was scattered across
``scenarios.py`` ``WORKER_MAP`` (backend -> service names) and
``test_canary_rank_pause.py`` ``_RANK_PATTERNS`` (backend -> rank process names).

``DEATH_TARGET`` records which target, when stalled, deterministically drives
the engine to self-kill (-> instance_id churn) for that backend — backend-
specific because the death mechanism differs: vllm dies via the **Worker**'s
``sample_tokens`` RPC timeout (stalling the EngineCore only hangs at
``shm_broadcast``, upstream of the timeout — verified 2026-06-04); sglang runs
one ``scheduler`` process; the mocker is a single process.
"""

# backend -> {semantic target: process-name substring matched in the ps cmdline}
ENGINE_PROCESS_NAMES: dict[str, dict[str, str]] = {
    "vllm": {
        "main": "dynamo.vllm",
        "engine": "VLLM::EngineCore",
        "worker": "VLLM::Worker",
    },
    "sglang": {
        "main": "dynamo.sglang",
        # sglang has no separate engine/worker split — the scheduler runs the model
        "engine": "sglang::scheduler",
        "worker": "sglang::scheduler",
    },
    "trtllm": {
        "main": "dynamo.runtime",
        "engine": "TRTLLM:EngineCore",
        "worker": "mpi4py.futures.server",
    },
    "mocker": {
        # single process — no sub-engine/worker to freeze independently
        "main": "dynamo.mocker",
        "engine": "dynamo.mocker",
        "worker": "dynamo.mocker",
    },
}

# backend -> the semantic target whose stall deterministically self-kills the
# engine (instance_id churn). vllm = "worker" VERIFIED on real GPU 2026-06-04
# (Worker SIGSTOP -> sample_tokens RPC timeout -> EngineDeadError -> restart);
# the others are the documented expectation and need their own validation.
DEATH_TARGET: dict[str, str] = {
    "vllm": "worker",
    "sglang": "engine",
    "trtllm": "worker",
    "mocker": "main",
}

# backend -> {role: k8s service name} for disagg (mirrors scenarios.py WORKER_MAP).
WORKER_SERVICES: dict[str, dict[str, str]] = {
    "vllm": {"decode": "VllmDecodeWorker", "prefill": "VllmPrefillWorker"},
    "trtllm": {"decode": "TRTLLMDecodeWorker", "prefill": "TRTLLMPrefillWorker"},
    "sglang": {"decode": "decode", "prefill": "prefill"},
    "mocker": {"decode": "decode", "prefill": "prefill"},
}

# backend -> all rank/engine process-name substrings (folds in
# test_canary_rank_pause._RANK_PATTERNS).
RANK_PATTERNS: dict[str, tuple[str, ...]] = {
    "vllm": ("VLLM::EngineCore", "EngineCoreProc"),
    "trtllm": ("mpi4py.futures.server", "TRTLLM:EngineCore", "tensorrt_llm"),
    "sglang": ("sglang::scheduler",),
}


def process_name_for(backend: str, target: str) -> str:
    """Resolve a semantic target (``main`` | ``engine`` | ``worker``) to the
    backend's concrete process-name substring. Raises ValueError on an unknown
    backend or target so a mis-targeted stall fails loudly rather than no-op'ing."""
    b = (backend or "").lower()
    if b not in ENGINE_PROCESS_NAMES:
        raise ValueError(
            f"engine_details: unknown backend {backend!r} "
            f"(known: {sorted(ENGINE_PROCESS_NAMES)})"
        )
    targets = ENGINE_PROCESS_NAMES[b]
    if target not in targets:
        raise ValueError(
            f"engine_details: backend {b!r} has no target {target!r} "
            f"(known: {sorted(targets)})"
        )
    return targets[target]


def death_target_for(backend: str) -> str:
    """The semantic target whose stall self-kills the engine for this backend
    (default ``worker`` if the backend is unknown)."""
    return DEATH_TARGET.get((backend or "").lower(), "worker")
