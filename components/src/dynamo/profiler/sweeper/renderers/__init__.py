# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy registry for Sweeper DGD renderers."""

from __future__ import annotations

import importlib
from typing import Any, Literal

from dynamo.profiler.sweeper.renderers.base import (
    CandidateLike,
    CandidateMaterializationError,
    DGDMaterializationOptions,
    validate_candidate_target,
)

DGDRenderer = Literal["aic", "direct"]
_RENDERER_MODULES: dict[str, str] = {
    "aic": "dynamo.profiler.sweeper.renderers.aic.renderer",
    "direct": "dynamo.profiler.sweeper.renderers.direct.renderer",
}


def _load_renderer(renderer: str) -> Any:
    module_name = _RENDERER_MODULES.get(renderer)
    if module_name is None:
        raise CandidateMaterializationError(f"unknown DGD renderer {renderer!r}")
    try:
        module = importlib.import_module(module_name)
        return module.render
    except (AttributeError, ModuleNotFoundError) as exc:
        raise RuntimeError(f"DGD renderer {renderer!r} is unavailable") from exc


def render_dgd(
    candidate: CandidateLike,
    workload: Any,
    options: DGDMaterializationOptions,
    *,
    candidate_index: int,
    renderer: DGDRenderer = "aic",
) -> str:
    """Render one evaluated Sweeper result with the selected implementation."""
    validate_candidate_target(candidate, options)
    render = _load_renderer(renderer)
    return render(
        candidate,
        workload,
        options,
        candidate_index=candidate_index,
    )


__all__ = [
    "CandidateMaterializationError",
    "DGDMaterializationOptions",
    "DGDRenderer",
    "render_dgd",
]
