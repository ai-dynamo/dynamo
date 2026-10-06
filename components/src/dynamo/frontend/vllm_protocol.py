# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol types shared by the released and MiniMax nightly vLLM layouts."""

try:
    from vllm.entrypoints.generate.base.protocol import (
        DeltaFunctionCall as DeltaFunctionCall,
    )
    from vllm.entrypoints.generate.base.protocol import DeltaMessage as DeltaMessage
    from vllm.entrypoints.generate.base.protocol import DeltaToolCall as DeltaToolCall
    from vllm.entrypoints.generate.base.protocol import (
        FunctionDefinition as FunctionDefinition,
    )
except ModuleNotFoundError as exc:
    # The nightly moved these types out of openai.engine. Only fall back for
    # that module move; missing dependencies inside the new module must fail.
    if exc.name not in {
        "vllm.entrypoints.generate",
        "vllm.entrypoints.generate.base",
        "vllm.entrypoints.generate.base.protocol",
    }:
        raise
    from vllm.entrypoints.openai.engine.protocol import (
        DeltaFunctionCall as DeltaFunctionCall,
    )
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage as DeltaMessage
    from vllm.entrypoints.openai.engine.protocol import DeltaToolCall as DeltaToolCall
    from vllm.entrypoints.openai.engine.protocol import (
        FunctionDefinition as FunctionDefinition,
    )
