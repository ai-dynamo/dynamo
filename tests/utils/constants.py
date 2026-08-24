# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test constants.

Centralize model identifiers and other shared constants for tests to
avoid importing from conftest and to keep values consistent.
"""

import os
from enum import IntEnum

from tests.utils.model_registry import (
    DEEPSEEK_AI_DEEPSEEK_R1_DISTILL_LLAMA_8B,
    DEFAULT_TEST_MODELS,
    OPENAI_GPT_OSS_20B,
    QWEN_QWEN3_EMBEDDING_4B,
    resolve_model_profile,
)

# Role-based defaults are intentionally resolved at import/collection time. A CI
# job can compare a candidate without editing every test, while the profile
# rejects models that lack the required runtime or architectural properties.
CROSS_BACKEND_SMOKE_MODEL = resolve_model_profile("cross_backend_smoke")
VLLM_SMOKE_MODEL = resolve_model_profile("vllm_smoke")
SGLANG_SMOKE_MODEL = resolve_model_profile("sglang_smoke")
TRTLLM_SMOKE_MODEL = resolve_model_profile("trtllm_smoke")
KV_TRANSFER_MODEL = resolve_model_profile("kv_transfer")

# Compatibility alias for existing tests. New tests should name the role they
# need rather than the current model family.
QWEN = CROSS_BACKEND_SMOKE_MODEL
LLAMA = DEEPSEEK_AI_DEEPSEEK_R1_DISTILL_LLAMA_8B  # on an l4 gpu, must limit --max-seq-len, otherwise it will not fit
GPT_OSS = OPENAI_GPT_OSS_20B
QWEN_EMBEDDING = QWEN_QWEN3_EMBEDDING_4B

TEST_MODELS = list(DEFAULT_TEST_MODELS)


# Default ports used by test payloads/scripts when not overridden.
# Tests that need xdist-safety should allocate real ports via fixtures and map
# these defaults to per-test ports at runtime.
class DefaultPort(IntEnum):
    FRONTEND = 8000
    SYSTEM1 = 8081
    SYSTEM2 = 8082


class DynamoPortRange(IntEnum):
    """Disjoint port bases for Dynamo CI allocations.

    These bases are intentionally spaced far enough apart to leave headroom
    for the allocator's 500-port random start offset plus sequential retries,
    so concurrent host-network test containers can allocate independently
    without colliding across frontend, serve, bootstrap, prefill, router,
    NIXL, and FPM workloads.
    """

    FRONTEND = 22500
    SERVE = 24000
    BOOTSTRAP = 24600
    PREFILL = 25200
    NIXL = 25800
    ROUTER = 27000
    FPM = 28500


# Env-driven defaults for specific test groups
# Allows overriding via environment variables
ROUTER_MODEL_NAME = resolve_model_profile(
    "cross_backend_smoke", override=os.environ.get("ROUTER_MODEL_NAME")
)
FAULT_TOLERANCE_MODEL_NAME = resolve_model_profile(
    "cross_backend_smoke", override=os.environ.get("FAULT_TOLERANCE_MODEL_NAME")
)
