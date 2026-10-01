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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dynamo.aisimulate.output.dgd.renderers import (
    CandidateMaterializationError,
    DGDGenerationOptions,
    render_dgd,
)
from dynamo.aisimulate.output.dgd.writers import write_outputs


class DGDOutputConfig(BaseModel):
    """Configuration accepted from the resolved top-level ``dgd`` section."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    namespace: str | None = None
    output_file: str = Field(min_length=1)
    renderer: Literal["aic", "direct"] = "aic"
    format: Literal["manifest", "kustomize"] = "manifest"
    runtime_image: str = Field(min_length=1)
    runtime_version_override: str | None = None
    num_gpus_per_node: int = Field(gt=0)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        literal = value.replace("{index}", "")
        if (
            value in {".", ".."}
            or Path(value).name != value
            or "{" in literal
            or "}" in literal
        ):
            raise ValueError(
                "must not contain path separators; only {index} may be templated"
            )
        return value

    @field_validator("output_file")
    @classmethod
    def _validate_output_file(cls, value: str) -> str:
        literal = value.replace("{index}", "")
        if (
            Path(value).name != value
            or Path(literal).suffix not in {".yaml", ".yml"}
            or "{" in literal
            or "}" in literal
        ):
            raise ValueError(
                "must be a YAML filename without path separators; only {index} may be templated"
            )
        return value

    def validate_result(self, *, is_pareto: bool) -> None:
        """Require the naming form that matches the selected result view."""
        if is_pareto and "{index}" not in self.name:
            raise ValueError("Pareto recommendations require {index} in dgd.name")
        if not is_pareto and "{index}" in self.name:
            raise ValueError(
                "{index} in dgd.name is only valid for Pareto recommendations"
            )
        if is_pareto and "{index}" not in self.output_file:
            raise ValueError(
                "Pareto recommendations require {index} in dgd.output_file"
            )
        if not is_pareto and "{index}" in self.output_file:
            raise ValueError(
                "{index} in dgd.output_file is only valid for Pareto recommendations"
            )

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
        names = (
            [
                resolved.name.format(index=f"{index:03d}")
                for index in range(len(selected))
            ]
            if is_pareto
            else [resolved.name]
        )
        output_files = (
            [
                resolved.output_file.format(index=f"{index:03d}")
                for index in range(len(selected))
            ]
            if is_pareto
            else [resolved.output_file]
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
            filenames=output_files,
            renderer=resolved.renderer,
            output=resolved.output_format,
        )
        return [*(Path(artifact["path"]) for artifact in artifacts), Path("index.json")]


def create_adapter() -> DGDOutputAdapter:
    """Create the Dynamo adapter discovered by AISimulate."""
    return DGDOutputAdapter()


__all__ = [
    "DGDOutputAdapter",
    "DGDOutputConfig",
    "create_adapter",
]
