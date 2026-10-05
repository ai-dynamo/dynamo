# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate-to-DGD materialization shared by every AISimulate DGD output.

This is the narrow, supported surface used by both the standalone ``dgd``
adapter and the live ``dgdr_run`` adapter. Import from here rather than
reaching into ``renderers`` or into either adapter's modules.
"""

from dynamo.aisimulate.output.dgd.materialization.errors import (
    CandidateMaterializationError,
)
from dynamo.aisimulate.output.dgd.materialization.options import DGDGenerationOptions
from dynamo.aisimulate.output.dgd.materialization.render import render_dgd

__all__ = [
    "CandidateMaterializationError",
    "DGDGenerationOptions",
    "render_dgd",
]
