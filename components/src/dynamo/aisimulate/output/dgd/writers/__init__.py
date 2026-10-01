# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy registry and run index for DGD artifact writers."""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from dynamo.aisimulate.output.dgd.writers.atomic import replace_text

OutputFormat = Literal["dgd", "kustomize"]
_WRITER_MODULES: dict[str, str] = {
    "dgd": "dynamo.aisimulate.output.dgd.writers.manifest",
    "kustomize": "dynamo.aisimulate.output.dgd.writers.kustomize",
}


def _load_writer(output: str) -> Any:
    module_name = _WRITER_MODULES.get(output)
    if module_name is None:
        raise ValueError(f"unknown output format {output!r}")
    try:
        module = importlib.import_module(module_name)
        return module
    except (AttributeError, ModuleNotFoundError) as exc:
        raise RuntimeError(f"output writer {output!r} is unavailable") from exc


def write_outputs(
    rendered_dgds: Sequence[str],
    output_dir: Path,
    *,
    filenames: Sequence[str],
    renderer: str,
    output: OutputFormat,
) -> list[dict[str, str]]:
    """Write rendered DGDs with one selected output plugin and a run index."""
    if len(rendered_dgds) != len(filenames):
        raise ValueError("rendered DGDs and output filenames must have equal lengths")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = _load_writer(output)
    artifacts: list[dict[str, str]] = []
    artifact_paths: list[Path] = []
    for rendered_dgd, filename in zip(rendered_dgds, filenames, strict=True):
        artifact_path = writer.write(
            rendered_dgd,
            output_dir,
            filename=filename,
        )
        artifact_paths.append(artifact_path)
        artifacts.append({"path": str(artifact_path.relative_to(output_dir))})

    finalize = getattr(writer, "finalize", None)
    if finalize is not None:
        final_path = finalize(output_dir, artifact_paths)
        artifacts.append({"path": str(final_path.relative_to(output_dir))})

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
