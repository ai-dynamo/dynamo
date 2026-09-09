# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo's `dgd` output adapter for `aisimulate recommend --output dgd`
(DEP #14282). Renders and writes DGDs for the Candidates AISimulate
selects, reusing Dynamo's existing renderer/output pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from dynamo.profiler.sweeper.output import OutputFormat, write_outputs
from dynamo.profiler.sweeper.renderers import (
    CandidateMaterializationError,
    DGDGenerationOptions,
    DGDRenderer,
    render_dgd,
)

OUTPUT_ADAPTER_API_VERSION = 1  # must equal aisimulate.output_adapter.OUTPUT_ADAPTER_API_VERSION


class DgdOutputConfigError(ValueError):
    """The resolved `dgd:` config section is missing or malformed."""


def _dgd_names(dgd_config: Mapping[str, Any], candidate_count: int) -> list[str]:
    """Returns dgd_config["name"] for a single candidate, or
    "{name_prefix}-{index:03d}" per candidate for a Pareto front."""
    name = dgd_config.get("name")
    name_prefix = dgd_config.get("name_prefix")

    if candidate_count == 1 and name:
        return [name]
    if name_prefix:
        return [f"{name_prefix}-{index:03d}" for index in range(candidate_count)]
    if name:
        raise DgdOutputConfigError(
            "dgd.name is only valid for a single candidate; use dgd.name_prefix "
            f"for {candidate_count} candidates"
        )
    raise DgdOutputConfigError("dgd config must set either 'name' or 'name_prefix'")


def _generation_options(dgd_config: Mapping[str, Any]) -> DGDGenerationOptions:
    try:
        return DGDGenerationOptions(
            runtime_image=dgd_config["runtime_image"],
            num_gpus_per_node=dgd_config["num_gpus_per_node"],
            runtime_version_override=dgd_config.get("runtime_version_override"),
            namespace=dgd_config.get("namespace"),
        )
    except KeyError as exc:
        raise DgdOutputConfigError(f"dgd config missing required field: {exc}") from exc


def render_and_write_dgds(
    candidates: Sequence[Any],
    workload: Any,
    dgd_config: Mapping[str, Any],
    output_dir: Path,
) -> list[str]:
    """Render every selected Candidate and write them with the resolved
    `dgd:` config. Only calls already-shipped, already-tested Dynamo
    functions (render_dgd, write_outputs). Returns paths as plain strings,
    relative to output_dir -- matching what write_output_adapters' real
    validation expects (Path(raw_path), rejecting absolute paths and `..`).

    Raises DgdOutputConfigError for a malformed config section, or
    CandidateMaterializationError if a candidate cannot be rendered --
    matching the DEP's own stated principle: "Candidate-to-DGD mapping
    rejects unsupported or incomplete combinations instead of guessing."
    Both propagate out of write() below, where AISimulate's real
    write_output_adapters wraps any adapter exception into
    OutputAdapterExecutionError.
    """
    options = _generation_options(dgd_config)
    renderer: DGDRenderer = dgd_config.get("renderer", "aic")

    # DEP's example config spells this field "format: manifest|kustomize";
    # the shipped OutputFormat type uses "dgd"|"kustomize" (output/__init__.py).
    raw_format = dgd_config.get("format", "manifest")
    output_format: OutputFormat = "dgd" if raw_format == "manifest" else raw_format

    names = _dgd_names(dgd_config, len(candidates))

    rendered_dgds = [
        render_dgd(candidate, workload, options, dgd_name=name, renderer=renderer)
        for candidate, name in zip(candidates, names, strict=True)
    ]

    artifacts = write_outputs(
        rendered_dgds,
        output_dir,
        stems=names,
        renderer=renderer,
        output=output_format,
    )
    return [artifact["path"] for artifact in artifacts]


class DgdOutputAdapter:
    """Dynamo's `dgd` output adapter, registered under
    "aisimulate.output_adapters" (pyproject.toml). Matches the
    RecommendationOutputAdapter Protocol: `name`/`api_version` class
    attributes, and `write(config, *, result, output_dir)` returning a
    sequence of paths relative to output_dir.
    """

    name = "dgd"
    api_version = OUTPUT_ADAPTER_API_VERSION

    def write(
        self,
        config: Mapping[str, Any],
        *,
        result: Any,
        output_dir: Path,
    ) -> Sequence[str | Path]:
        # result.candidates/.workload are inferred, not confirmed against a
        # real SweepResult class -- verify these two attribute names first
        # if this adapter ever misbehaves against a real aisimulate build.
        candidates = result.candidates
        workload = getattr(result, "workload", None)
        return render_and_write_dgds(candidates, workload, config, output_dir)
