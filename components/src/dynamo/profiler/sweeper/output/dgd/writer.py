# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a rendered DGD as a YAML file."""

from pathlib import Path

from dynamo.profiler.sweeper.output.atomic import replace_text


def write(rendered_dgd: str, output_dir: Path, *, stem: str) -> Path:
    """Write one DGD manifest and return its path."""
    artifact_path = output_dir / f"{stem}.yaml"
    replace_text(artifact_path, rendered_dgd)
    return artifact_path
