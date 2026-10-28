# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for vLLM's frontend helpers and OpenAI protocol types."""

from importlib.util import find_spec

# vLLM 0.32 moved the chat helpers to vllm.renderers.
if find_spec("vllm.renderers.chat_utils") is not None:
    from vllm.renderers.chat_utils import load_chat_template, make_tool_call_id
else:
    from vllm.entrypoints.chat_utils import load_chat_template, make_tool_call_id

try:
    _HAS_GENERATE_PROTOCOL = (
        find_spec("vllm.entrypoints.generate.base.protocol") is not None
    )
except ModuleNotFoundError:
    _HAS_GENERATE_PROTOCOL = False

if _HAS_GENERATE_PROTOCOL:
    # vLLM 0.29 moved the shared types out of the OpenAI engine package.
    from vllm.entrypoints.generate.base.protocol import (
        DeltaFunctionCall,
        DeltaMessage,
        DeltaToolCall,
        FunctionDefinition,
    )
else:
    # vLLM 0.27 and 0.28 expose the same types at the legacy path.
    from vllm.entrypoints.openai.engine.protocol import (
        DeltaFunctionCall,
        DeltaMessage,
        DeltaToolCall,
        FunctionDefinition,
    )

__all__ = [
    "DeltaFunctionCall",
    "DeltaMessage",
    "DeltaToolCall",
    "FunctionDefinition",
    "load_chat_template",
    "make_tool_call_id",
]
