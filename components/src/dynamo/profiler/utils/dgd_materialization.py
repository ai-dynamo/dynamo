# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize immutable DGD blueprints for profiler consumers."""

from __future__ import annotations

import copy
import logging
from enum import Enum
from typing import Any

from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS
from dynamo.profiler.utils.dgd_override import apply_dgd_overrides
from dynamo.profiler.utils.profile_common import inject_tolerations_into_dgd

logger = logging.getLogger(__name__)


class DGDMaterializationPurpose(str, Enum):
    """Profiler boundary that consumes an independently materialized DGD."""

    BENCHMARK_CANDIDATE = "benchmark candidate"
    INTERPOLATION = "interpolation"
    FINAL_OUTPUT = "final output"


def materialize_dgd(
    blueprint: Any,
    *,
    purpose: DGDMaterializationPurpose,
    override: dict[str, Any] | None = None,
    tolerations: list[dict[str, Any]] | None = None,
    runtime_backend: str | None = None,
    model_name_or_path: str | None = None,
) -> Any:
    """Return an independent DGD with all consumer-facing transforms applied.

    Transform order is fixed because DGD overrides are not necessarily
    idempotent: override, model runtime constraints, then tolerations. For a
    multi-document final configuration, only the last DGD document is
    materialized; preceding resources are copied unchanged. Callers must pass
    the clean blueprint rather than a previously materialized result.
    """
    if blueprint is None:
        return None

    materialized = copy.deepcopy(blueprint)
    if isinstance(materialized, list):
        if not materialized:
            return materialized
        materialized[-1] = _materialize_dgd_document(
            materialized[-1],
            purpose=purpose,
            override=override,
            tolerations=tolerations,
            runtime_backend=runtime_backend,
            model_name_or_path=model_name_or_path,
        )
        return materialized

    return _materialize_dgd_document(
        materialized,
        purpose=purpose,
        override=override,
        tolerations=tolerations,
        runtime_backend=runtime_backend,
        model_name_or_path=model_name_or_path,
    )


def _materialize_dgd_document(
    blueprint: Any,
    *,
    purpose: DGDMaterializationPurpose,
    override: dict[str, Any] | None,
    tolerations: list[dict[str, Any]] | None,
    runtime_backend: str | None,
    model_name_or_path: str | None,
) -> dict[str, Any]:
    if not isinstance(blueprint, dict):
        raise TypeError(f"{purpose.value} DGD blueprint must be an object")

    materialized = blueprint
    applied_transforms: list[str] = []

    if override:
        materialized = apply_dgd_overrides(materialized, override)
        applied_transforms.append("override")

    modifier = CONFIG_MODIFIERS.get(runtime_backend) if runtime_backend else None
    apply_runtime_constraints = getattr(
        modifier, "apply_model_runtime_constraints", None
    )
    if apply_runtime_constraints is not None:
        materialized = apply_runtime_constraints(
            materialized,
            model_name_or_path,
        )
        applied_transforms.append("runtime constraints")

    if tolerations:
        materialized = inject_tolerations_into_dgd(materialized, tolerations)
        applied_transforms.append("tolerations")

    logger.debug(
        "Materialized %s DGD with transforms: %s",
        purpose.value,
        ", ".join(applied_transforms) if applied_transforms else "none",
    )
    return materialized
