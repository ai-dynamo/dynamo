# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DynamoGraphDeployment output adapter for AISimulate recommendations."""

from dynamo.aisimulate.output.dgd.adapter import (
    DGDOutputAdapter,
    DGDOutputConfig,
    create_adapter,
)
from dynamo.aisimulate.output.dgd.materialization import (
    CandidateMaterializationError,
    DGDGenerationOptions,
    render_dgd,
)
from dynamo.aisimulate.output.dgd.materialization.render import DGDRenderer

__all__ = [
    "CandidateMaterializationError",
    "DGDGenerationOptions",
    "DGDOutputAdapter",
    "DGDOutputConfig",
    "DGDRenderer",
    "create_adapter",
    "render_dgd",
]
