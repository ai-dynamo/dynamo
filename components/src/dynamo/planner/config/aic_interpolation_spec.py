# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema for the profiler → planner AIC interpolation handoff.

When the profiler runs in rapid mode, it picks parallelism configs with
AIConfigurator but does NOT run interpolation itself. Instead it serialises
this ``AICInterpolationSpec`` onto the planner's ConfigMap. At bootstrap the
planner lazy-imports ``aiconfigurator`` and runs the interpolation in-process.
"""

from typing import Literal

from pydantic import BaseModel, Field

from dynamo.planner.config.parallelization import PickedParallelConfig


class AICInterpolationSpec(BaseModel):
    """Everything the planner needs to reproduce the rapid-mode AIC sweep.

    The picks come straight from AIC's picker DataFrame (via
    :func:`dynamo.profiler.utils.profile_common.picked_config_from_row`) so
    any AIC-valid pick is representable.
    """

    hf_id: str = Field(description="HuggingFace model id, e.g. Qwen/Qwen3-32B")
    system: str = Field(description="AIC system identifier, e.g. h200_sxm")
    backend: Literal["trtllm", "vllm", "sglang"]

    isl: int = Field(gt=0)
    osl: int = Field(gt=0)
    sweep_max_context_length: int = Field(gt=0)

    prefill_interpolation_granularity: int = Field(gt=0)
    decode_interpolation_granularity: int = Field(gt=0)

    prefill_pick: PickedParallelConfig
    decode_pick: PickedParallelConfig
