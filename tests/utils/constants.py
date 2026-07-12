# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test constants.

Centralize model identifiers and other shared constants for tests to
avoid importing from conftest and to keep values consistent.
"""

import os
from enum import IntEnum

from tests.utils.model_registry import (
    DEFAULT_TEST_MODELS,
    DEEPSEEK_AI_DEEPSEEK_R1_DISTILL_LLAMA_8B,
    OPENAI_GPT_OSS_20B,
    QWEN_QWEN3_0_6B,
    QWEN_QWEN3_EMBEDDING_4B,
)

QWEN = QWEN_QWEN3_0_6B
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


# Env-driven defaults for specific test groups
# Allows overriding via environment variables
ROUTER_MODEL_NAME = os.environ.get("ROUTER_MODEL_NAME", QWEN)
FAULT_TOLERANCE_MODEL_NAME = os.environ.get("FAULT_TOLERANCE_MODEL_NAME", QWEN)
