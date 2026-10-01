# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write rendered DGDs as one composable Kustomize source."""

from collections.abc import Sequence
from pathlib import Path

from dynamo.aisimulate.output.dgd.writers.atomic import replace_text


def write(rendered_dgd: str, output_dir: Path, *, filename: str) -> Path:
    """Write one DGD manifest and return its path."""
    artifact_path = output_dir / filename
    replace_text(artifact_path, rendered_dgd)
    return artifact_path


def finalize(output_dir: Path, resources: Sequence[Path]) -> Path:
    """Write one Kustomization referencing every selected DGD."""
    artifact_path = output_dir / "kustomization.yaml"
    resource_lines = "\n".join(f"  - {resource.name}" for resource in resources)
    replace_text(
        artifact_path,
        "apiVersion: kustomize.config.k8s.io/v1beta1\n"
        "kind: Kustomization\n"
        f"resources:\n{resource_lines}\n",
    )
    return artifact_path
