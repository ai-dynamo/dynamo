# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for detecting pytest collection-time environment mutations."""

from __future__ import annotations

import os
from collections.abc import Mapping

WATCHED_ENV_PREFIXES = (
    "DYN_",
    "DYNAMO_",
    "CUDA_",
    "NCCL_",
    "HF_",
    "TRANSFORMERS_",
    "SGLANG_",
    "SGL_",
    "MC_",
    "VLLM_",
    "TENSORRT_",
    "TORCH_",
    "UCX_",
    "NIXL_",
    "OMPI_",
    "LLM_",
    "TLLM_",
    "TRT_LLM_",
    "TRTLLM_",
    "NVIDIA_",
    "NSYS_",
    "GENERATE_CU_",
    "OVERRIDE_",
    "TOKENIZERS_",
    "DISABLE_TORCH_",
    "PYTORCH_",
    "ENABLE_PERFECT_ROUTER",
    "FLA_",
    "NEMOTRON_",
)

# Exact keys to watch that fall outside WATCHED_ENV_PREFIXES. Empty by default:
# the "DYNAMO_" prefix is watched, so import-time leaks under it (e.g. the
# log-init env from #9724) are caught without enumerating individual keys.
# Nothing should mutate a watched var during collection -- legitimate harness
# vars (DYNAMO_MODELS_DIR, DYNAMO_GPU_PARALLEL_DOWNLOADS_READY) are set in
# fixtures, after the collection-finish snapshot, and the guard's own bypass
# flag is only ever read, never written, during collection.
WATCHED_ENV_KEYS: frozenset[str] = frozenset()

SENSITIVE_PATTERNS = (
    "TOKEN",
    "API_KEY",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTH",
)

# Backend frameworks mutate these as import-time side effects while their test
# modules are imported for broad marker collection. Keep this narrow until those
# imports are made side-effect free.
#   SGLANG_LOGGING_CONFIG_PATH / VLLM_CONFIGURE_LOGGING -- logging setup on import.
#   TLLM_LOG_LEVEL / OMPI_MCA_coll_ucc_enable -- set by `import tensorrt_llm`,
#   which the trtllm test modules import at module scope.
ALLOWED_COLLECTION_ENV_MUTATIONS = frozenset(
    {
        "SGLANG_LOGGING_CONFIG_PATH",
        "VLLM_CONFIGURE_LOGGING",
        "TLLM_LOG_LEVEL",
        "OMPI_MCA_coll_ucc_enable",
    }
)

COLLECTION_ENV_GUARD_DISABLE_ENV = "DYNAMO_ALLOW_COLLECTION_ENV_MUTATION"

EnvSnapshot = dict[str, str]
EnvChanges = dict[str, tuple[str | None, str | None]]


def collection_env_guard_disabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the collection env guard has been explicitly disabled."""
    env = os.environ if environ is None else environ
    return env.get(COLLECTION_ENV_GUARD_DISABLE_ENV, "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def snapshot_collection_env(environ: Mapping[str, str] | None = None) -> EnvSnapshot:
    """Snapshot watched env vars before pytest imports test modules."""
    env = os.environ if environ is None else environ
    return {
        key: value
        for key, value in env.items()
        if key.startswith(WATCHED_ENV_PREFIXES) or key in WATCHED_ENV_KEYS
    }


def diff_collection_env(
    before: Mapping[str, str],
    environ: Mapping[str, str] | None = None,
) -> EnvChanges:
    """Return watched env vars that changed since the collection snapshot."""
    after = snapshot_collection_env(environ)
    return {
        key: (before.get(key), after.get(key))
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
        and key not in ALLOWED_COLLECTION_ENV_MUTATIONS
    }


def format_collection_env_changes(
    changes: Mapping[str, tuple[str | None, str | None]],
) -> str:
    """Format a collection-time env mutation failure for pytest output."""
    lines = [
        "pytest collection mutated watched environment variables.",
        "",
        "Importing test modules should not change process-global environment. "
        "Move setup into main(), fixtures, or monkeypatch-scoped helpers.",
        f"Temporary bypass: set {COLLECTION_ENV_GUARD_DISABLE_ENV}=1.",
        "",
        "Changed variables:",
    ]
    for key, (before, after) in changes.items():
        lines.append(
            f"  {key}: {_display_env_value(key, before)} -> "
            f"{_display_env_value(key, after)}"
        )
    return "\n".join(lines)


def _display_env_value(key: str, value: str | None) -> str:
    if value is None:
        return "<unset>"
    if _is_sensitive_env_key(key):
        return "<redacted>"
    return repr(value)


def _is_sensitive_env_key(key: str) -> bool:
    return any(token in key.upper() for token in SENSITIVE_PATTERNS)
