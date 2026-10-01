# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-owned DGD output adapter for AISimulate recommendations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from aisimulate.output_adapter import OUTPUT_ADAPTER_API_VERSION
from aisimulate.sweeper.config import SmartSearchConfig
from aisimulate.sweeper.result import SweepResult
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dynamo.profiler.sweeper.output import write_outputs
from dynamo.profiler.sweeper.renderers import (
    CandidateMaterializationError,
    DGDGenerationOptions,
    render_dgd,
)


class DGDOutputConfig(BaseModel):
    """Configuration accepted from the resolved top-level ``dgd`` section."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    name_prefix: str | None = None
    namespace: str | None = None
    renderer: Literal["aic", "direct"] = "aic"
    format: Literal["manifest", "kustomize"] = "manifest"
    runtime_image: str = Field(min_length=1)
    runtime_version_override: str | None = None
    num_gpus_per_node: int = Field(gt=0)

    @field_validator("name", "name_prefix")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value in {".", ".."} or Path(value).name != value:
            raise ValueError("must be a non-empty name without path separators")
        return value

    @model_validator(mode="after")
    def _validate_names(self) -> DGDOutputConfig:
        if self.name is not None and self.name_prefix is not None:
            raise ValueError("name and name_prefix are mutually exclusive")
        return self

    def validate_result(self, *, is_pareto: bool) -> None:
        """Require the naming form that matches the selected result view."""
        if is_pareto and self.name is not None:
            raise ValueError("dgd.name is only valid for scalar recommendations")
        if not is_pareto and self.name_prefix is not None:
            raise ValueError("dgd.name_prefix is only valid for Pareto recommendations")
        if is_pareto and self.name_prefix is None:
            raise ValueError("Pareto recommendations require dgd.name_prefix")
        if not is_pareto and self.name is None:
            raise ValueError("scalar recommendations require dgd.name")

    def generation_options(self) -> DGDGenerationOptions:
        """Lower adapter configuration to the renderer's stable options."""
        return DGDGenerationOptions(
            runtime_image=self.runtime_image,
            runtime_version_override=self.runtime_version_override,
            namespace=self.namespace,
            num_gpus_per_node=self.num_gpus_per_node,
        )

    @property
    def output_format(self) -> Literal["dgd", "kustomize"]:
        """Map the public adapter format onto the internal writer name."""
        return "dgd" if self.format == "manifest" else "kustomize"


def _workload(result: SweepResult) -> Any:
    """Recover the validated Sweeper workload recorded in result provenance."""
    return SmartSearchConfig.model_validate(result.provenance.config).workload


class DGDOutputAdapter:
    """Render final AISimulate recommendation selections as Dynamo artifacts."""

    name = "dgd"
    api_version = OUTPUT_ADAPTER_API_VERSION

    def write(
        self,
        config: Mapping[str, Any],
        *,
        result: SweepResult,
        output_dir: Path,
    ) -> Sequence[str | Path]:
        """Write the scalar winner or selected Pareto front into ``output_dir``."""
        resolved = DGDOutputConfig.model_validate(dict(config))
        candidates = result.selected_candidates
        if not candidates:
            raise CandidateMaterializationError("no feasible candidate found")

        is_pareto = bool(result.views.pareto_front)
        resolved.validate_result(is_pareto=is_pareto)
        selected = candidates if is_pareto else candidates[:1]
        prefix = resolved.name_prefix if is_pareto else resolved.name
        assert prefix is not None
        names = (
            [f"{prefix}-{index:03d}" for index in range(len(selected))]
            if is_pareto
            else [prefix]
        )
        workload = _workload(result)
        rendered = [
            render_dgd(
                candidate,
                workload,
                resolved.generation_options(),
                dgd_name=dgd_name,
                renderer=resolved.renderer,
            )
            for candidate, dgd_name in zip(selected, names, strict=True)
        ]
        artifacts = write_outputs(
            rendered,
            output_dir,
            stems=names,
            renderer=resolved.renderer,
            output=resolved.output_format,
        )
        return [*(Path(artifact["path"]) for artifact in artifacts), Path("index.json")]


def create_dgd_output_adapter() -> DGDOutputAdapter:
    """Create the Dynamo adapter discovered by AISimulate."""
    return DGDOutputAdapter()


__all__ = [
    "DGDOutputAdapter",
    "DGDOutputConfig",
    "create_dgd_output_adapter",
]
