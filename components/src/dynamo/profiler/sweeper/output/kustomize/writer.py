# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a rendered DGD as a composable Kustomize source."""

from pathlib import Path

from dynamo.profiler.sweeper.output.atomic import replace_text

_KUSTOMIZATION = """apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deploy.yaml
"""


def write(rendered_dgd: str, output_dir: Path, *, stem: str) -> Path:
    """Write one Kustomize source and return its directory."""
    artifact_path = output_dir / stem
    artifact_path.mkdir(parents=True, exist_ok=True)
    replace_text(artifact_path / "kustomization.yaml", _KUSTOMIZATION)
    replace_text(artifact_path / "deploy.yaml", rendered_dgd)
    return artifact_path
