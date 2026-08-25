# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a Sweeper result through AIC's typed generator bridge."""

from __future__ import annotations

import importlib
from typing import Any

from dynamo.profiler.sweeper.renderers.base import (
    CandidateLike,
    CandidateMaterializationError,
    DGDMaterializationOptions,
    patch_dgd_manifest,
)


def _load_generator_api() -> tuple[Any, Any]:
    """Load the Sweeper-result bridge owned by the AIC generator."""
    try:
        generator_api = importlib.import_module("aiconfigurator.generator.api")
        generator_request = importlib.import_module("aiconfigurator.generator.request")
        return (
            generator_request.from_sweeper_candidate,
            generator_api.generate_from_request,
        )
    except (AttributeError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "the AIC renderer requires an aiconfigurator generator containing "
            "the Sweeper-result bridge (ai-dynamo/aisimulate#11)"
        ) from exc


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


def render(
    candidate: CandidateLike,
    workload: Any,
    options: DGDMaterializationOptions,
    *,
    candidate_index: int,
) -> str:
    """Lower one Sweeper result through AIC and Dynamo's v1 templates."""
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
            "AIC generator did not return k8s_deploy.yaml"
        )
    return patch_dgd_manifest(
        rendered,
        options,
        candidate_index=candidate_index,
    )
