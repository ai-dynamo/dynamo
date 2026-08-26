# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt the direct config-modifier materializer to the renderer contract."""

from __future__ import annotations

import importlib
from typing import Any

import yaml

from dynamo.profiler.sweeper.renderers.base import (
    CandidateLike,
    CandidateMaterializationError,
    DGDGenerationOptions,
    patch_dgd_manifest,
)


def _load_materializer() -> tuple[type[Exception], Any]:
    """Load Dynamo's config modifiers only when the direct renderer is selected."""
    try:
        materializer = importlib.import_module(
            "dynamo.profiler.sweeper.renderers.direct.materializer"
        )
        return (
            materializer.MaterializationError,
            materializer.materialize_dgd_from_candidate,
        )
    except (AttributeError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "the direct renderer requires a complete Dynamo Python installation"
        ) from exc


def render(
    candidate: CandidateLike,
    _workload: Any,
    options: DGDGenerationOptions,
    *,
    dgd_name: str,
) -> str:
    """Lower one Sweeper result directly through Dynamo's v1 config modifiers."""
    MaterializationError, materialize = _load_materializer()
    try:
        result = materialize(
            candidate.config,
            image=options.runtime_image,
            num_gpus_per_node=options.num_gpus_per_node,
        )
    except MaterializationError as exc:
        raise CandidateMaterializationError(str(exc)) from exc

    return patch_dgd_manifest(
        yaml.safe_dump(result.dgd, sort_keys=False),
        options,
        dgd_name=dgd_name,
    )
