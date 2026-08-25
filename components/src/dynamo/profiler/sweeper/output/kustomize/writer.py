# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a rendered DGD as a composable Kustomize source."""

from pathlib import Path

_KUSTOMIZATION = """apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deploy.yaml
"""


def write(rendered_dgd: str, output_dir: Path, *, stem: str) -> Path:
    """Write one Kustomize source and return its directory."""
    artifact_path = output_dir / stem
    artifact_path.mkdir(parents=True, exist_ok=True)
    (artifact_path / "deploy.yaml").write_text(rendered_dgd, encoding="utf-8")
    (artifact_path / "kustomization.yaml").write_text(
        _KUSTOMIZATION,
        encoding="utf-8",
    )
    return artifact_path
