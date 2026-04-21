# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run ID and directory naming conventions for sweep runs."""

from __future__ import annotations

import hashlib

from sweep_core.models import AiperfDimension, DeployDimension


def image_tag(image: str, length: int = 8) -> str:
    """Stable short hash of a container image reference, for run IDs.

    Uses a truncated SHA1 rather than something like an image-tag parse
    because images are passed as full refs (``registry/repo:tag``) where
    only the ``:tag`` portion is human-meaningful but may collide across
    registries. A content-hash of the whole ref is collision-free and
    filesystem-safe.
    """
    if not image:
        return ""
    return hashlib.sha1(image.encode()).hexdigest()[:length]


def build_run_id(
    deploy: DeployDimension,
    aiperf: AiperfDimension,
    include_image_tag: bool = False,
) -> str:
    """Build a human-readable run ID from deploy + aiperf dimensions.

    Format:
      [img<hash>_]{tokenizer}_c{concurrency}_isl{isl}_w{workers}[_m{models}][_rps{rate}]

    The ``img<hash>_`` prefix is added only when ``include_image_tag`` is
    True (set by the planner when the sweep has more than one image) so
    existing single-image sweeps keep the shorter historic name.
    """
    base = f"{deploy.tokenizer}_c{aiperf.concurrency}_isl{aiperf.isl}_w{deploy.workers}"
    if deploy.num_models > 1:
        base += f"_m{deploy.num_models}"
    if aiperf.request_rate is not None:
        base += f"_rps{aiperf.request_rate}"
    if include_image_tag and deploy.image:
        return f"img{image_tag(deploy.image)}_{base}"
    return base
