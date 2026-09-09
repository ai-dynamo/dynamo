# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo's `dgd` output adapter for `aisimulate recommend --output dgd`.

Implements DEP #14282 (https://github.com/ai-dynamo/dynamo/issues/14282)
against the confirmed, real AISimulate output-adapter ABI
(aisimulate/src/aisimulate/output_adapter.py, PR against ai-dynamo/aisimulate):

    class RecommendationOutputAdapter(Protocol):
        name: str
        api_version: int

        def write(
            self,
            config: Mapping[str, Any],
            *,
            result: SweepResult,
            output_dir: Path,
        ) -> Sequence[str | Path]: ...

Registered under the entry-point group "aisimulate.output_adapters" --
confirmed via OUTPUT_ADAPTER_ENTRY_POINT_GROUP in the real module, not
guessed.

One thing this module cannot fully confirm yet: SweepResult's own fields.
The public "Sweeper Results" doc
(docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/ai-simulate/
sweeper/results) documents Candidate and ReplaySpec in detail but never
mentions a SweepResult wrapper class, and the real test suite for this ABI
(tests/test_unified_cli.py) exercises it via a hand-rolled
`_RecommendationResult` stand-in whose own fake adapter never reads
`result.candidates` at all -- so nothing publicly available yet proves the
real attribute names. `result.candidates` and `result.workload` below are
inferred (the test constructs its stand-in as
`_RecommendationResult([candidate])`, strongly suggesting `.candidates`
exists) and are the one thing to verify the moment the real SweepResult
class is available -- everything else in this module is built against
confirmed interfaces.
"""

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
    """Mirror __main__.py's own naming rule exactly: one fixed name for a
    scalar goal (candidate_count == 1), or "{name_prefix}-{index:03d}" per
    candidate for a Pareto front. Confirmed against the real CLI's
    _render_dgds/_dgd_name logic, and against the real review-thread example
    (`--set dgd.name=qwen`) and the real shipped test
    (`"dgd": {"name": "original", ...}` overridden to `"qwen"` via `--set`).
    """
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
    """Build DGDGenerationOptions from the resolved `dgd:` config section.
    Field names (runtime_image, runtime_version_override, num_gpus_per_node,
    namespace) are confirmed, already-shipped dataclass fields
    (renderers/base.py).
    """
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
    "aisimulate.output_adapters" (pyproject.toml). Matches the confirmed
    RecommendationOutputAdapter Protocol exactly: `name`/`api_version` class
    attributes, and `write(config, *, result, output_dir)` returning a
    sequence of paths relative to output_dir.
    """

    name = "dgd"
    api_version = OUTPUT_ADAPTER_API_VERSION

    def write(
        self,
        config: Mapping[str, Any],
        *,
        result: Any,  # SweepResult -- see module docstring re: unconfirmed shape
        output_dir: Path,
    ) -> Sequence[str | Path]:
        # INFERRED, not confirmed -- see module docstring.
        candidates = result.candidates
        workload = getattr(result, "workload", None)
        return render_and_write_dgds(candidates, workload, config, output_dir)
