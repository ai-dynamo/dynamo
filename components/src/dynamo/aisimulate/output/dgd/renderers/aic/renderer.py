# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a Sweeper result through AIC's typed generator bridge."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import yaml

from dynamo.aisimulate.output.dgd.renderers.base import (
    CandidateLike,
    CandidateMaterializationError,
    DGDGenerationOptions,
    patch_dgd_manifest,
)

# AIC requires a workload ISL/OSL. With the evaluated engine limits preserved,
# the only engine field it still derives from them is TensorRT-LLM's
# disaggregated KV-transfer buffer. A trace has no single request length, so
# AIC gets placeholder lengths and the buffer size derived from them is dropped,
# leaving TensorRT-LLM's default, which covers max_seq_len.
_TRACE_PLACEHOLDER_LENGTH = 1
_TRANSFER_BUFFER_ARG = "--trtllm.cache_transceiver_config.max_tokens_in_buffer"


def _load_generator_api() -> tuple[Any, Any]:
    """Load the Sweeper-result bridge shipped by AISimulate."""
    try:
        generator_api = importlib.import_module("aisimulate.generator.api")
        generator_request = importlib.import_module("aisimulate.generator.request")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "the AIC renderer requires the AISimulate generator"
        ) from exc

    return generator_request.from_sweeper_candidate, generator_api.generate_from_request


def _generator_overrides(
    options: DGDGenerationOptions, *, dgd_name: str
) -> dict[str, Any]:
    k8s: dict[str, Any] = {
        "k8s_image": options.runtime_image,
        "name_prefix": dgd_name,
    }
    if options.namespace:
        k8s["k8s_namespace"] = options.namespace
    return {
        "generator_dynamo_version": options.dynamo_runtime_version,
        "K8sConfig": k8s,
        "NodeConfig": {"num_gpus_per_node": options.num_gpus_per_node},
    }


def _workload_field(workload: Any, name: str) -> Any:
    if isinstance(workload, Mapping):
        return workload.get(name)
    return getattr(workload, name, None)


def _is_trace_workload(workload: Any) -> bool:
    return bool(
        _workload_field(workload, "trace_paths")
        or _workload_field(workload, "trace_path")
    ) and (
        _workload_field(workload, "isl") is None
        or _workload_field(workload, "osl") is None
    )


def _with_placeholder_lengths(workload: Any) -> dict[str, Any]:
    """Copy a trace workload with the placeholder lengths AIC requires.

    The keys are overwritten rather than supplied as ``SlaConfig`` overrides
    because AIC prefers a workload's present ``None`` values over overrides.
    """
    if isinstance(workload, Mapping):
        payload = dict(workload)
    elif hasattr(workload, "model_dump"):
        payload = workload.model_dump(mode="json")
    else:
        payload = dict(vars(workload))
    payload["isl"] = payload["osl"] = _TRACE_PLACEHOLDER_LENGTH
    return payload


def _drop_transfer_buffer_size(rendered: str) -> str:
    """Remove the KV-transfer buffer size AIC derived from placeholder lengths.

    Only the size goes; the transceiver backend stays, since TensorRT-LLM
    disables KV transfer without one.
    """
    from dynamo.profiler.utils.config import (
        break_arguments,
        get_main_container_dict,
        remove_all_argument_occurrences,
    )

    documents = [document for document in yaml.safe_load_all(rendered) if document]
    changed = False
    for document in documents:
        if not isinstance(document, dict):
            continue
        for component in (document.get("spec") or {}).get("components") or []:
            container = (
                get_main_container_dict(component)
                if isinstance(component, dict)
                else None
            )
            if container is None:
                continue
            args = break_arguments(container.get("args"))
            kept = remove_all_argument_occurrences(args, _TRANSFER_BUFFER_ARG)
            if kept != args:
                container["args"] = kept
                changed = True
    return yaml.safe_dump_all(documents, sort_keys=False) if changed else rendered


def render(
    candidate: CandidateLike,
    workload: Any,
    options: DGDGenerationOptions,
    *,
    dgd_name: str,
) -> str:
    """Lower one Sweeper result through AIC and Dynamo's v1 templates."""
    from_sweeper_candidate, generate_from_request = _load_generator_api()
    trace = _is_trace_workload(workload)
    request = from_sweeper_candidate(
        candidate,
        workload=_with_placeholder_lengths(workload) if trace else workload,
        deployment_target="dynamo-python",
        generator_overrides=_generator_overrides(options, dgd_name=dgd_name),
    )
    artifacts = generate_from_request(request)
    rendered = artifacts.get("k8s_deploy.yaml")
    if not rendered:
        raise CandidateMaterializationError(
            "AIC generator did not return k8s_deploy.yaml"
        )
    if trace:
        rendered = _drop_transfer_buffer_size(rendered)
    return patch_dgd_manifest(
        rendered,
        candidate,
        options,
        dgd_name=dgd_name,
    )
