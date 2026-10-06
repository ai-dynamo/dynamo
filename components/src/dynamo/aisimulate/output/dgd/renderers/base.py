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
_CONTEXT_LENGTH_ARG = {
    "sglang": "--context-length",
    "trtllm": "--max-seq-len",
    "vllm": "--max-model-len",
}
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


def _apply_router_config(dgd: dict[str, Any], candidate: CandidateLike) -> None:
    """Carry the concrete Dynamo router selection into the frontend."""
    adapters = candidate.config.get("adapters")
    if adapters is None:
        return
    if not isinstance(adapters, dict):
        raise CandidateMaterializationError("candidate adapters must be an object")
    router = adapters.get("dynamo.router")
    if router is None:
        return
    if not isinstance(router, dict):
        raise CandidateMaterializationError(
            "candidate adapters.dynamo.router must be an object"
        )

    policy = router.get("policy", router.get("mode"))
    router_mode = {"kv_router": "kv", "round_robin": "round-robin"}.get(policy)
    if router_mode is None:
        raise CandidateMaterializationError(
            f"candidate router policy must be kv_router or round_robin, got {policy!r}"
        )
    load_model = router.get("prefill_load_model", {"type": "none"})
    if not isinstance(load_model, dict) or not isinstance(load_model.get("type"), str):
        raise CandidateMaterializationError(
            "candidate router prefill_load_model must contain a string type"
        )
    if load_model["type"] != "none":
        raise CandidateMaterializationError(
            "DGD generation does not yet support router prefill_load_model.type "
            f"{load_model['type']!r} because the Candidate does not carry its runtime model"
        )

    env_values = {"DYN_ROUTER_MODE": router_mode}
    if policy == "kv_router":
        for field, env_name in (
            ("overlap_score_credit", "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT"),
            ("prefill_load_scale", "DYN_ROUTER_PREFILL_LOAD_SCALE"),
        ):
            value = router.get(field)
            if value is not None:
                env_values[env_name] = str(value)

        # Native SmartSearchConfig calls this router_temperature; the public
        # adapter config uses the canonical temperature name.
        temperature = (
            router["temperature"]
            if "temperature" in router
            else router.get("router_temperature")
        )
        if temperature is not None:
            env_values["DYN_ROUTER_TEMPERATURE"] = str(temperature)
        env_values["DYN_ROUTER_PREFILL_LOAD_MODEL"] = "none"

    from dynamo.profiler.utils.config import (
        get_main_container_dict,
        set_unique_env_value,
    )

    frontends = [
        component
        for component in dgd.get("spec", {}).get("components", [])
        if isinstance(component, dict) and component.get("type") == "frontend"
    ]
    if not frontends:
        raise CandidateMaterializationError(
            "renderer output has no frontend for the selected router configuration"
        )
    for frontend in frontends:
        container = get_main_container_dict(frontend)
        if container is None:
            raise CandidateMaterializationError(
                "renderer output frontend has no main container for router configuration"
            )
        env = container.get("env")
        for name, value in env_values.items():
            env = set_unique_env_value(env, name, value)
        container["env"] = env


def _apply_context_length(dgd: dict[str, Any], candidate: CandidateLike) -> None:
    """Preserve the engine context limit used to evaluate the Candidate."""
    context_length = candidate.config.get("context_length")
    if context_length is None:
        return
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise CandidateMaterializationError(
            f"candidate context_length must be a positive integer, got {context_length!r}"
        )
    if context_length < 1:
        raise CandidateMaterializationError(
            f"candidate context_length must be a positive integer, got {context_length!r}"
        )
    backend = candidate.config.get("backend")
    arg_name = _CONTEXT_LENGTH_ARG.get(backend)
    if arg_name is None:
        raise CandidateMaterializationError(
            f"cannot materialize context_length for backend {backend!r}"
        )

    from dynamo.profiler.utils.config import (
        break_arguments,
        get_main_container_dict,
        set_unique_argument_value,
    )

    found_worker = False
    for component in dgd.get("spec", {}).get("components", []):
        if not isinstance(component, dict) or component.get("type") not in {
            "worker",
            "prefill",
            "decode",
        }:
            continue
        found_worker = True
        container = get_main_container_dict(component)
        if container is None:
            raise CandidateMaterializationError(
                "renderer output worker has no main container for context_length"
            )
        command = container.get("command") or []
        raw_args = container.get("args") or []
        if (
            len(command) >= 2
            and command[0] in ("/bin/sh", "sh")
            and command[1] == "-c"
            and len(raw_args) == 1
        ):
            raise CandidateMaterializationError(
                "cannot safely materialize context_length into a shell-form worker command"
            )
        container["args"] = set_unique_argument_value(
            break_arguments(raw_args), arg_name, str(context_length)
        )
    if not found_worker:
        raise CandidateMaterializationError(
            "renderer output has no worker for candidate context_length"
        )


def patch_dgd_manifest(
    rendered: str,
    candidate: CandidateLike,
    options: DGDGenerationOptions,
    *,
    dgd_name: str,
    evaluation_context: dict[str, Any] | None = None,
) -> str:
    """Apply shared v1 finalization and adapter-owned fields to one rendered DGD."""
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

    _apply_router_config(dgd, candidate)
    _apply_context_length(dgd, candidate)

    documents[dgd_index] = dgd
    metadata = dgd.setdefault("metadata", {})
    metadata["name"] = dgd_name
    if options.namespace:
        metadata["namespace"] = options.namespace
    else:
        metadata.pop("namespace", None)
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
