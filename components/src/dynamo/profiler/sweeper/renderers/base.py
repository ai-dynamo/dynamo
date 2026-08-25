# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts and manifest handling for Sweeper DGD renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import yaml

_SUPPORTED_BACKENDS = frozenset({"sglang", "trtllm", "vllm"})


class CandidateLike(Protocol):
    """Public Sweeper result fields used by DGD renderers."""

    config: dict[str, Any]


class CandidateMaterializationError(ValueError):
    """A Sweeper result cannot be represented faithfully as a DGD."""


@dataclass(frozen=True)
class DGDMaterializationOptions:
    """Pinned deployment inputs that are outside the Sweeper search space."""

    backend: str
    backend_version: str
    backend_image: str
    dynamo_version: str
    num_gpus_per_node: int
    namespace: str | None = None
    name_prefix: str = "sweeper-dgd"

    def __post_init__(self) -> None:
        if self.backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend must be one of {', '.join(sorted(_SUPPORTED_BACKENDS))}"
            )
        required_text = {
            "backend_version": self.backend_version,
            "backend_image": self.backend_image,
            "dynamo_version": self.dynamo_version,
        }
        for field_name, value in required_text.items():
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.num_gpus_per_node < 1:
            raise ValueError("num_gpus_per_node must be positive")
        if not self.name_prefix.strip():
            raise ValueError("name_prefix must not be empty")


def validate_candidate_target(
    candidate: CandidateLike, options: DGDMaterializationOptions
) -> None:
    """Require evaluated performance data to match the pinned runtime target."""
    candidate_backend = candidate.config.get("backend")
    if candidate_backend != options.backend:
        raise CandidateMaterializationError(
            f"candidate backend {candidate_backend!r} does not match target backend "
            f"{options.backend!r}"
        )
    candidate_version = candidate.config.get("backend_version")
    if candidate_version != options.backend_version:
        raise CandidateMaterializationError(
            f"candidate backend_version {candidate_version!r} does not match target "
            f"backend version {options.backend_version!r}"
        )


def patch_dgd_manifest(
    rendered: str,
    options: DGDMaterializationOptions,
    *,
    candidate_index: int,
) -> str:
    """Apply Dynamo-owned identity and runtime fields to one rendered DGD."""
    documents = [document for document in yaml.safe_load_all(rendered) if document]
    dgds = [
        document
        for document in documents
        if isinstance(document, dict)
        and document.get("kind") == "DynamoGraphDeployment"
    ]
    if len(dgds) != 1:
        raise CandidateMaterializationError(
            "renderer output must contain exactly one DynamoGraphDeployment"
        )

    dgd = dgds[0]
    metadata = dgd.setdefault("metadata", {})
    metadata["name"] = f"{options.name_prefix}-{candidate_index:03d}"
    if options.namespace:
        metadata["namespace"] = options.namespace

    components = dgd.get("spec", {}).get("components")
    if not isinstance(components, list):
        raise CandidateMaterializationError(
            "rendered DGD does not define spec.components"
        )
    for component in components:
        if not isinstance(component, dict):
            raise CandidateMaterializationError(
                "rendered DGD contains a non-object component"
            )
        component["runtimeVersionOverride"] = options.dynamo_version

    return yaml.safe_dump_all(documents, sort_keys=False)
