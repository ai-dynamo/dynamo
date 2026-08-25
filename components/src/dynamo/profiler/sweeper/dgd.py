# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize one AI Simulate Sweeper candidate as a Dynamo deployment."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol

import yaml

_SUPPORTED_BACKENDS = frozenset({"sglang", "trtllm", "vllm"})


class CandidateLike(Protocol):
    """Public Sweeper candidate fields used by the Dynamo materializer."""

    config: dict[str, Any]


class CandidateMaterializationError(ValueError):
    """A candidate cannot be represented faithfully as a DGD."""


@dataclass(frozen=True)
class DGDMaterializationOptions:
    """Pinned deployment inputs that are outside the Sweeper search space."""

    backend: str
    backend_version: str
    backend_image: str
    dynamo_version: str
    num_gpus_per_node: int
    namespace: str | None = None
    name_prefix: str = "sweeper-candidate"

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


def _load_generator_api() -> tuple[Any, Any]:
    """Load the official Sweeper-to-generator bridge from AISimulate."""
    try:
        generator_api = importlib.import_module("aiconfigurator.generator.api")
        generator_request = importlib.import_module("aiconfigurator.generator.request")
        return (
            generator_request.from_sweeper_candidate,
            generator_api.generate_from_request,
        )
    except (AttributeError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "DGD materialization requires an AISimulate release containing "
            "the Sweeper candidate generator bridge (ai-dynamo/aisimulate#11)"
        ) from exc


def _validate_candidate_target(
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


def _generator_overrides(options: DGDMaterializationOptions) -> dict[str, Any]:
    k8s: dict[str, Any] = {
        "k8s_image": options.backend_image,
        "name_prefix": options.name_prefix,
    }
    if options.namespace:
        k8s["k8s_namespace"] = options.namespace
    return {
        "generator_dynamo_version": options.dynamo_version,
        "K8sConfig": k8s,
        "NodeConfig": {"num_gpus_per_node": options.num_gpus_per_node},
    }


def _patch_dgd_manifest(
    rendered: str,
    options: DGDMaterializationOptions,
    *,
    candidate_index: int,
) -> str:
    documents = [document for document in yaml.safe_load_all(rendered) if document]
    dgds = [
        document
        for document in documents
        if isinstance(document, dict)
        and document.get("kind") == "DynamoGraphDeployment"
    ]
    if len(dgds) != 1:
        raise CandidateMaterializationError(
            "generator output must contain exactly one DynamoGraphDeployment"
        )

    dgd = dgds[0]
    metadata = dgd.setdefault("metadata", {})
    metadata["name"] = f"{options.name_prefix}-{candidate_index:03d}"
    if options.namespace:
        metadata["namespace"] = options.namespace

    components = dgd.get("spec", {}).get("components")
    if not isinstance(components, list):
        raise CandidateMaterializationError(
            "generated DGD does not define spec.components"
        )
    for component in components:
        if not isinstance(component, dict):
            raise CandidateMaterializationError(
                "generated DGD contains a non-object component"
            )
        component["runtimeVersionOverride"] = options.dynamo_version

    return yaml.safe_dump_all(documents, sort_keys=False)


def materialize_candidate_dgd(
    candidate: CandidateLike,
    workload: Any,
    options: DGDMaterializationOptions,
    *,
    candidate_index: int,
) -> str:
    """Lower one evaluated Candidate through AISimulate's typed generator bridge.

    The Candidate's backend version identifies both its performance data and the
    engine CLI template. ``dynamo_version`` separately describes the Dynamo
    runtime carried by the target image and becomes ``runtimeVersionOverride``.
    """
    _validate_candidate_target(candidate, options)
    from_sweeper_candidate, generate_from_request = _load_generator_api()
    request = from_sweeper_candidate(
        candidate,
        workload=workload,
        deployment_target="dynamo-python",
        generator_overrides=_generator_overrides(options),
    )
    artifacts = generate_from_request(request)
    rendered = artifacts.get("k8s_deploy.yaml")
    if not rendered:
        raise CandidateMaterializationError(
            "AISimulate generator did not return k8s_deploy.yaml"
        )
    return _patch_dgd_manifest(
        rendered,
        options,
        candidate_index=candidate_index,
    )
