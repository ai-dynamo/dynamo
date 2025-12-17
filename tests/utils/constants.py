# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test constants.

Centralize model identifiers and other shared constants for tests to
avoid importing from conftest and to keep values consistent.
"""

import os
from enum import IntEnum

QWEN = "Qwen/Qwen3-0.6B"
LLAMA = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # on an l4 gpu, must limit --max-seq-len, otherwise it will not fit
GPT_OSS = "openai/gpt-oss-20b"
QWEN_EMBEDDING = "Qwen/Qwen3-Embedding-4B"

TEST_MODELS = [
    QWEN,
    LLAMA,
    GPT_OSS,
    QWEN_EMBEDDING,
]


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
