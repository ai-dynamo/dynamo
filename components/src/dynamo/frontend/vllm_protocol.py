# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for vLLM's shared OpenAI protocol types."""

from importlib.util import find_spec

if find_spec("vllm.entrypoints.generate.base.protocol") is not None:
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
]
