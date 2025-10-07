# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test constants.

Centralize model identifiers and other shared constants for tests to
avoid importing from conftest and to keep values consistent.
"""

import os

QWEN = "Qwen/Qwen3-0.6B"
LLAMA = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # on an l4 gpu, must limit --max-seq-len, otherwise it will not fit
QWEN_EMBEDDING = "Qwen/Qwen3-Embedding-4B"

TEST_MODELS = [
    QWEN,
    LLAMA,
    QWEN_EMBEDDING,
]

# Env-driven defaults for specific test groups
# Allows overriding via environment variables
ROUTER_MODEL_NAME = os.environ.get("ROUTER_MODEL_NAME", QWEN)
FAULT_TOLERANCE_MODEL_NAME = os.environ.get("FAULT_TOLERANCE_MODEL_NAME", QWEN)
