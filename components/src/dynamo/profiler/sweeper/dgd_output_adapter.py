# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo's `dgd` output adapter for `aisimulate recommend --output dgd`
(DEP #14282). Renders and writes DGDs for the Candidates AISimulate
selects, reusing Dynamo's existing renderer/output pipeline."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dynamo.profiler.sweeper.output import OutputFormat, write_outputs
from dynamo.profiler.sweeper.renderers import (
    CandidateMaterializationError,
    DGDGenerationOptions,
    DGDRenderer,
    render_dgd,
)

OUTPUT_ADAPTER_API_VERSION = (
    1  # must equal aisimulate.output_adapter.OUTPUT_ADAPTER_API_VERSION
)


class DgdOutputConfigError(ValueError):
    """The resolved `dgd:` config section is missing or malformed."""


def _validate_name_component(value: Any, field: str) -> str:
    """Reject a dgd.name/name_prefix that could escape output_dir."""
    if not value or not isinstance(value, str):
        raise DgdOutputConfigError(f"dgd.{field} must be a non-empty string")
    if (
        Path(value).is_absolute()
        or any(sep in value for sep in ("/", "\\"))
        or value in (".", "..")
    ):
        raise DgdOutputConfigError(
            f"dgd.{field} must be a single path component, not {value!r}"
        )
    return value


def _dgd_names(dgd_config: Mapping[str, Any], candidate_count: int) -> list[str]:
    """Returns dgd_config["name"] for a single candidate, or
    "{name_prefix}-{index:03d}" per candidate for a Pareto front."""
    name = dgd_config.get("name")
    name_prefix = dgd_config.get("name_prefix")

    if name is not None:
        name = _validate_name_component(name, "name")
    if name_prefix is not None:
        name_prefix = _validate_name_component(name_prefix, "name_prefix")

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


def _best_candidate(candidates: Sequence[Any]) -> Any:
    """Match Sweeper's scalar ranking: highest score, then fewer GPUs.

    Restores the selection rule from the deleted __main__.py's
    _best_candidate -- the adapter that replaced it rendered every
    candidate it was given instead of picking a winner for scalar mode.
    """
    return max(
        candidates, key=lambda candidate: (candidate.score, -candidate.used_gpus)
    )


def render_and_write_dgds(
    candidates: Sequence[Any],
    workload: Any,
    dgd_config: Mapping[str, Any],
    output_dir: Path,
) -> list[str]:
    """Render the selected Candidate(s) and write them with the resolved
    `dgd:` config.

    Scalar mode (dgd.name set, no dgd.name_prefix): reduces `candidates` to
    the single highest-scoring one (ties broken by fewer GPUs) before
    rendering -- matching the deleted CLI's `_best_candidate` rule. Pareto
    mode (dgd.name_prefix set): renders every candidate, skipping (not
    aborting on) any that fails to materialize -- matching the deleted
    `_render_pareto`'s per-candidate try/except, so one bad candidate
    doesn't lose the rest of the front.

    Raises DgdOutputConfigError for a malformed config section, or
    CandidateMaterializationError if *no* candidate could be rendered.
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

    if dgd_config.get("name") and not dgd_config.get("name_prefix"):
        # Scalar mode: one DGD, the highest-scoring candidate -- regardless
        # of how many candidates were actually handed to us.
        if not candidates:
            raise CandidateMaterializationError(
                "no candidates were selected for this run; nothing to pick a "
                "scalar winner from"
            )
        candidates = [_best_candidate(candidates)]

    names = _dgd_names(dgd_config, len(candidates))

    rendered_dgds: list[Any] = []
    rendered_names: list[str] = []
    failures: list[str] = []
    last_exc: CandidateMaterializationError | None = None
    for candidate, cand_name in zip(candidates, names, strict=True):
        try:
            rendered_dgds.append(
                render_dgd(
                    candidate, workload, options, dgd_name=cand_name, renderer=renderer
                )
            )
        except CandidateMaterializationError as exc:
            # Not "Pareto candidate": this loop runs in scalar mode too
            # (a single-candidate list after the reduction above), where
            # calling it Pareto would mislabel the failure.
            print(f"skipping candidate {cand_name}: {exc}", file=sys.stderr)
            failures.append(f"{cand_name}: {exc}")
            last_exc = exc
            continue
        rendered_names.append(cand_name)

    if not rendered_dgds:
        # Keep the real cause: both in the message (so it's visible even
        # without a traceback) and chained via `from`, rather than the
        # stderr-only print above being the last anyone sees of it.
        raise CandidateMaterializationError(
            "no candidate could be rendered: " + "; ".join(failures)
        ) from last_exc

    artifacts = write_outputs(
        rendered_dgds,
        output_dir,
        stems=rendered_names,
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
        # Confirmed against the real aisimulate.sweeper.result.SweepResult:
        # .candidates is the full retained ledger (including non-feasible
        # records with score=None), not the active selection, and SweepResult
        # carries no top-level `workload` attribute at all.
        candidates = result.selected_candidates
        workload = result.provenance.config["workload"]
        return render_and_write_dgds(candidates, workload, config, output_dir)
