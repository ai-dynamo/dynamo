# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy registry and run index for Sweeper output writers."""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from dynamo.profiler.sweeper.output.atomic import replace_text

OutputFormat = Literal["dgd", "kustomize"]
_WRITER_MODULES: dict[str, str] = {
    "dgd": "dynamo.profiler.sweeper.output.dgd.writer",
    "kustomize": "dynamo.profiler.sweeper.output.kustomize.writer",
}


def _load_writer(output: str) -> Any:
    module_name = _WRITER_MODULES.get(output)
    if module_name is None:
        raise ValueError(f"unknown output format {output!r}")
    try:
        module = importlib.import_module(module_name)
        return module.write
    except (AttributeError, ModuleNotFoundError) as exc:
        raise RuntimeError(f"output writer {output!r} is unavailable") from exc


def write_outputs(
    rendered_dgds: Sequence[str],
    output_dir: Path,
    *,
    stems: Sequence[str],
    renderer: str,
    output: OutputFormat,
) -> list[dict[str, str]]:
    """Write rendered DGDs with one selected output plugin and a run index."""
    if len(rendered_dgds) != len(stems):
        raise ValueError("rendered DGDs and output stems must have equal lengths")
    output_dir.mkdir(parents=True, exist_ok=True)
    write = _load_writer(output)
    artifacts: list[dict[str, str]] = []
    for rendered_dgd, stem in zip(rendered_dgds, stems, strict=True):
        artifact_path = write(
            rendered_dgd,
            output_dir,
            stem=stem,
        )
        artifacts.append({"path": str(artifact_path.relative_to(output_dir))})

    replace_text(
        output_dir / "index.json",
        json.dumps(
            {"renderer": renderer, "output": output, "artifacts": artifacts},
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return artifacts


__all__ = ["OutputFormat", "write_outputs"]
