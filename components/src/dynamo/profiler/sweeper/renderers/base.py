# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts and manifest handling for Sweeper DGD renderers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

import yaml

_SUPPORTED_BACKENDS = frozenset({"sglang", "trtllm", "vllm"})
_EVALUATION_CONTEXT_ANNOTATION = "nvidia.com/sweeper-evaluation-context"
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


def _materialize_dgd(*args: Any, **kwargs: Any) -> Any:
    """Load legacy runtime modifiers only when a manifest is finalized."""
    from dynamo.profiler.utils.dgd_materialization import materialize_dgd

    return materialize_dgd(*args, **kwargs)


def patch_dgd_manifest(
    rendered: str,
    candidate: CandidateLike,
    options: DGDGenerationOptions,
    *,
    dgd_name: str,
    evaluation_context: dict[str, Any] | None = None,
) -> str:
    """Apply shared v1 finalization and CLI-owned fields to one rendered DGD."""
    documents = [document for document in yaml.safe_load_all(rendered) if document]
    indexed_dgds = [
        (index, document)
        for index, document in enumerate(documents)
        if isinstance(document, dict)
        and document.get("kind") == "DynamoGraphDeployment"
    ]
    if len(indexed_dgds) != 1:
        raise CandidateMaterializationError(
            "renderer output must contain exactly one DynamoGraphDeployment"
        )

    dgd_index, dgd = indexed_dgds[0]

    # TODO(#13770 follow-up after #14040): Remove this explicit TRT-LLM patch
    # when required runtime rules run once during base rendering.
    if candidate.config.get("backend") == "trtllm":
        from dynamo.profiler.utils.config_modifiers.trtllm import (
            enable_trtllm_chunked_prefill,
        )

        dgd = enable_trtllm_chunked_prefill(dgd)

    # TODO(#13770 follow-up after #14040): Remove this materialize_dgd() call
    # once required runtime rules run during base rendering and optional patches
    # move into the common assembler.
    from dynamo.profiler.utils.dgd_materialization import DGDMaterializationPurpose

    try:
        dgd = _materialize_dgd(
            dgd,
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
            runtime_backend=candidate.config.get("backend"),
            model_name_or_path=candidate.config.get("model_name"),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CandidateMaterializationError(
            f"renderer output failed legacy DGD finalization: {exc}"
        ) from exc

    documents[dgd_index] = dgd
    metadata = dgd.setdefault("metadata", {})
    metadata["name"] = dgd_name
    if options.namespace:
        metadata["namespace"] = options.namespace
    if evaluation_context:
        annotations = metadata.setdefault("annotations", {})
        if not isinstance(annotations, dict):
            raise CandidateMaterializationError(
                "rendered DGD metadata.annotations must be an object"
            )
        annotations[_EVALUATION_CONTEXT_ANNOTATION] = json.dumps(
            evaluation_context,
            sort_keys=True,
            separators=(",", ":"),
        )

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
