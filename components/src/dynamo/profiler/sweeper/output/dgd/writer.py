# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a rendered DGD as a YAML file."""

from pathlib import Path


def write(rendered_dgd: str, output_dir: Path, *, stem: str) -> Path:
    """Write one DGD manifest and return its path."""
    artifact_path = output_dir / f"{stem}.yaml"
    artifact_path.write_text(rendered_dgd, encoding="utf-8")
    return artifact_path
