# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DynamoGraphDeployment output adapter for AISimulate recommendations."""

from dynamo.aisimulate.output.dgd.renderers import (
    CandidateMaterializationError,
    DGDGenerationOptions,
    DGDRenderer,
    render_dgd,
)

__all__ = [
    "CandidateMaterializationError",
    "DGDGenerationOptions",
    "DGDOutputAdapter",
    "DGDOutputConfig",
    "DGDRenderer",
    "create_adapter",
    "render_dgd",
]


def __getattr__(name: str):
    # adapter.py imports the optional aisimulate package, which is only
    # installable on Python 3.11-3.13; load it lazily so the renderers
    # stay importable without it.
    if name in ("DGDOutputAdapter", "DGDOutputConfig", "create_adapter"):
        from dynamo.aisimulate.output.dgd import adapter

        return getattr(adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
