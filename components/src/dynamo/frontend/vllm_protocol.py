# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared imports for vLLM's frontend helpers and OpenAI protocol types."""

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionDefinition,
)
from vllm.renderers.chat_utils import load_chat_template, make_tool_call_id

__all__ = [
    "DeltaFunctionCall",
    "DeltaMessage",
    "DeltaToolCall",
    "FunctionDefinition",
    "load_chat_template",
    "make_tool_call_id",
]
