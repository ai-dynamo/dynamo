# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts and manifest handling for Sweeper DGD renderers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

import yaml

_SUPPORTED_BACKENDS = frozenset({"sglang", "trtllm", "vllm"})
_RUNTIME_VERSION_PATTERN = re.compile(
    r"^(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})$"
)


def _runtime_version(runtime_image: str, override: str | None) -> str:
    if override is not None:
        return override.strip()

    image_without_digest = runtime_image.partition("@")[0]
    image_name = image_without_digest.rsplit("/", 1)[-1]
    _, separator, tag = image_name.rpartition(":")
    if separator and _RUNTIME_VERSION_PATTERN.fullmatch(tag):
        return tag
    raise ValueError(
        "runtime_image must have a canonical MAJOR.MINOR.PATCH tag when "
        "runtime_version_override is not set"
    )


class CandidateLike(Protocol):
    """Public Sweeper result fields used by DGD renderers."""

    config: dict[str, Any]


class CandidateMaterializationError(ValueError):
    """A Sweeper result cannot be represented faithfully as a DGD."""


@dataclass(frozen=True)
class DGDGenerationOptions:
    """Inputs that control how one Sweeper Candidate becomes a DGD."""

    runtime_image: str
    num_gpus_per_node: int
    runtime_version_override: str | None = None
    namespace: str | None = None

    def __post_init__(self) -> None:
        if not self.runtime_image.strip():
            raise ValueError("runtime_image must not be empty")
        if self.num_gpus_per_node < 1:
            raise ValueError("num_gpus_per_node must be positive")
        if (
            self.runtime_version_override is not None
            and not _RUNTIME_VERSION_PATTERN.fullmatch(
                self.runtime_version_override.strip()
            )
        ):
            raise ValueError(
                "runtime_version_override must be a canonical MAJOR.MINOR.PATCH version"
            )

        # AIC needs the Dynamo runtime version even when the DGD does not need an override.
        _runtime_version(self.runtime_image, self.runtime_version_override)

    @property
    def dynamo_runtime_version(self) -> str:
        """Return the Dynamo version declared by the override or image tag."""
        return _runtime_version(
            self.runtime_image,
            self.runtime_version_override,
        )


def validate_candidate(candidate: CandidateLike) -> None:
    """Require the Candidate fields needed by every DGD renderer."""
    candidate_backend = candidate.config.get("backend")
    if candidate_backend not in _SUPPORTED_BACKENDS:
        raise CandidateMaterializationError(
            f"candidate backend must be one of {', '.join(sorted(_SUPPORTED_BACKENDS))}, "
            f"got {candidate_backend!r}"
        )
    candidate_version = candidate.config.get("backend_version")
    if not isinstance(candidate_version, str) or not candidate_version.strip():
        raise CandidateMaterializationError(
            "candidate backend_version must be a non-empty string"
        )


def patch_dgd_manifest(
    rendered: str,
    options: DGDGenerationOptions,
    *,
    dgd_name: str,
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
    metadata["name"] = dgd_name
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
        if options.runtime_version_override is not None:
            component["runtimeVersionOverride"] = options.runtime_version_override

    return yaml.safe_dump_all(documents, sort_keys=False)
