# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants and utilities for trace converters."""

from prefix_data_generator.hasher import (
    texts_to_hashes_and_lengths as texts_to_hashes_and_lengths,
)

DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_BLOCK_SIZE = 64
