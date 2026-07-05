# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SWE-bench Pro to mini-SWE-agent compatibility adapter."""

from __future__ import annotations

PRO_IMAGE_REPOSITORY = "docker.io/jefzda/sweap-images"


def format_problem_statement(row: dict) -> str:
    """Match the public Pro scaffold's prompt construction exactly."""
    return (
        f"{row['problem_statement']}\n\n"
        f"Requirements:\n{row['requirements']}\n\n"
        f"New interfaces introduced:\n{row['interface']}"
    )


def docker_image(row: dict) -> str:
    """Use the dataset's authoritative tag instead of reconstructing it."""
    tag = row["dockerhub_tag"]
    if not isinstance(tag, str) or not tag:
        raise ValueError(
            f"missing dockerhub_tag for {row.get('instance_id', '<unknown>')}"
        )
    if tag != tag.strip() or ":" in tag or "/" in tag:
        raise ValueError(
            f"invalid dockerhub_tag for {row.get('instance_id', '<unknown>')}: {tag!r}"
        )
    return f"{PRO_IMAGE_REPOSITORY}:{tag}"


def adapt_row(row: dict) -> dict:
    """Return the fields mini-SWE-agent needs while preserving evaluator data."""
    return {
        **row,
        "problem_statement": format_problem_statement(row),
        "image_name": docker_image(row),
    }
