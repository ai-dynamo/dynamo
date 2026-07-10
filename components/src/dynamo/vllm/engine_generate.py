# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM request adaptation for Dynamo's engine-native generate API."""

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from vllm.entrypoints.serve.disagg.mm_serde import decode_mm_kwargs_item
from vllm.inputs import TokensPrompt, mm_input
from vllm.multimodal.inputs import (
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.sampling_params import (
    RepetitionDetectionParams,
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)

from dynamo.common.utils.structural_tag import serialize_structural_tag

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineGenerateRequest:
    """Typed worker adapter for one engine-native generate request."""

    request: Dict[str, Any]
    vllm_tito: Dict[str, Any]

    @classmethod
    def from_request(cls, request: Dict[str, Any]) -> Optional["EngineGenerateRequest"]:
        vllm_tito = payload(request)
        return cls(request, vllm_tito) if vllm_tito is not None else None

    def build_prompt(self) -> Any:
        return _build_prompt(self.request, self.vllm_tito)

    def build_sampling_params(
        self,
        default_sampling_params: Dict[str, Any],
        model_max_len: int | None,
    ) -> SamplingParams:
        return _build_sampling_params(
            self.request,
            self.vllm_tito,
            default_sampling_params,
            model_max_len,
        )

    def priority(self, routing: Dict[str, Any]) -> int:
        if "priority" in self.vllm_tito:
            return int(self.vllm_tito["priority"])
        return -int(routing.get("priority", 0))

    def adapt_response_metadata(
        self,
        output: Dict[str, Any],
        routed_experts: Any,
        kv_transfer_params: Any,
    ) -> None:
        metadata: Dict[str, Any] = {}
        serialized = serialize_routed_experts(routed_experts)
        if serialized is not None:
            metadata["routed_experts"] = serialized
        if kv_transfer_params is not None:
            metadata["kv_transfer_params"] = kv_transfer_params
        if metadata:
            engine_data = output.setdefault("engine_data", {})
            if isinstance(engine_data, dict):
                engine_data.update(metadata)


def payload(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    extra_args = request.get("extra_args")
    if not isinstance(extra_args, dict):
        return None
    value = extra_args.get("vllm_tito")
    return value if isinstance(value, dict) else None


def priority(request: Dict[str, Any], routing: Dict[str, Any]) -> int:
    adapter = EngineGenerateRequest.from_request(request)
    if adapter is not None:
        return adapter.priority(routing)
    return -int(routing.get("priority", 0))


def serialize_routed_experts(routed_experts: Any) -> Optional[str]:
    if routed_experts is None:
        return None
    try:
        buffer = io.BytesIO()
        np.save(buffer, routed_experts)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        logger.warning(
            "Unable to encode routed experts for generate API", exc_info=True
        )
        return None


def merge_kv_transfer_params(caller: Any, framework: Any) -> Dict[str, Any] | Any:
    if caller is None:
        return framework
    if framework is None:
        return caller
    if not isinstance(caller, dict) or not isinstance(framework, dict):
        raise ValueError(
            "kv_transfer_params from both caller and framework must be objects"
        )
    duplicate = caller.keys() & framework.keys()
    if duplicate:
        raise ValueError(
            "caller and framework kv_transfer_params collide on: "
            + ", ".join(sorted(duplicate))
        )
    return {**caller, **framework}


def build_prompt(request: Dict[str, Any]) -> Any:
    vllm_tito = payload(request)
    if vllm_tito is None:
        raise ValueError("extra_args.vllm_tito is missing from token-native request")
    return _build_prompt(request, vllm_tito)


def _build_prompt(request: Dict[str, Any], vllm_tito: Dict[str, Any]) -> Any:
    token_ids = list(request.get("token_ids") or vllm_tito.get("token_ids") or [])
    cache_salt = vllm_tito.get("cache_salt")
    if cache_salt is None:
        raw_sampling = vllm_tito.get("sampling_params")
        if isinstance(raw_sampling, dict):
            # Prime passes request-level cache isolation through SamplingConfig
            # extra_body. vLLM consumes it on the prompt, not SamplingParams.
            cache_salt = raw_sampling.get("cache_salt")
    features = vllm_tito.get("features")
    if not isinstance(features, dict):
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        if cache_salt is not None:
            prompt["cache_salt"] = cache_salt
        return prompt

    mm_hashes = features.get("mm_hashes") or {}
    placeholders = features.get("mm_placeholders") or {}
    kwargs_data = features.get("kwargs_data")
    mm_placeholders = {
        modality: [
            PlaceholderRange(offset=int(item["offset"]), length=int(item["length"]))
            for item in ranges
        ]
        for modality, ranges in placeholders.items()
    }
    mm_kwargs: Dict[str, list[MultiModalKwargsItem | None]] = {}
    if isinstance(kwargs_data, dict):
        for modality, items in kwargs_data.items():
            mm_kwargs[modality] = [
                decode_mm_kwargs_item(item) if item is not None else None
                for item in items
            ]
    else:
        for modality, hashes in mm_hashes.items():
            mm_kwargs[modality] = [None] * len(hashes)

    return mm_input(
        prompt_token_ids=token_ids,
        mm_kwargs=MultiModalKwargsItems(mm_kwargs),
        mm_hashes=mm_hashes,
        mm_placeholders=mm_placeholders,
        cache_salt=cache_salt,
    )


def build_sampling_params(
    request: Dict[str, Any],
    default_sampling_params: Dict[str, Any],
    model_max_len: int | None,
) -> SamplingParams:
    vllm_tito = payload(request)
    if vllm_tito is None:
        raise ValueError("extra_args.vllm_tito is missing from token-native request")
    return _build_sampling_params(
        request,
        vllm_tito,
        default_sampling_params,
        model_max_len,
    )


def _build_sampling_params(
    request: Dict[str, Any],
    vllm_tito: Dict[str, Any],
    default_sampling_params: Dict[str, Any],
    model_max_len: int | None,
) -> SamplingParams:
    raw_params = vllm_tito.get("sampling_params") or {}
    if not isinstance(raw_params, dict):
        raise ValueError("extra_args.vllm_tito.sampling_params must be an object")

    provided = request.get("generate_sampling_fields")
    provided_fields = set(provided if isinstance(provided, list) else raw_params.keys())
    kwargs: Dict[str, Any] = {}
    base = SamplingParams()
    extension_maps: list[Dict[str, Any]] = []
    for extension_name in ("extra_args", "vllm_xargs"):
        extension = raw_params.get(extension_name)
        if extension is not None:
            if not isinstance(extension, dict):
                raise ValueError(f"sampling_params.{extension_name} must be an object")
            extension_maps.append(extension)

    extensions: Dict[str, Any] = {}
    for extension in extension_maps:
        duplicate = extensions.keys() & extension.keys()
        if duplicate:
            raise ValueError(
                "duplicate backend sampling extension(s): "
                + ", ".join(sorted(duplicate))
            )
        extensions.update(extension)
    caller_kv = vllm_tito.get("kv_transfer_params")
    if caller_kv is not None:
        if "kv_transfer_params" in extensions:
            raise ValueError(
                "kv_transfer_params appears in both the request and sampling extensions"
            )
        extensions["kv_transfer_params"] = caller_kv

    for key in provided_fields:
        if key in ("extra_args", "vllm_xargs", "cache_salt") or key not in raw_params:
            continue
        value = raw_params[key]
        if key == "structured_outputs" and isinstance(value, dict):
            value = StructuredOutputsParams(
                json=value.get("json"),
                regex=value.get("regex"),
                choice=value.get("choice"),
                grammar=value.get("grammar"),
                json_object=value.get("json_object"),
                disable_any_whitespace=value.get("disable_any_whitespace", False),
                disable_additional_properties=value.get(
                    "disable_additional_properties", False
                ),
                whitespace_pattern=value.get("whitespace_pattern"),
                structural_tag=serialize_structural_tag(value.get("structural_tag")),
            )
        elif key == "repetition_detection" and isinstance(value, dict):
            value = RepetitionDetectionParams(**value)
        elif key == "logit_bias" and isinstance(value, dict):
            value = {int(token_id): bias for token_id, bias in value.items()}
        if not hasattr(base, key):
            raise ValueError(f"unsupported sampling parameter for this vLLM: {key}")
        kwargs[key] = value

    if extensions:
        kwargs["extra_args"] = extensions
    sampling_params = SamplingParams(**kwargs)

    if "max_tokens" not in provided_fields and model_max_len is not None:
        input_length = len(request.get("token_ids") or [])
        dynamic_default = max(1, model_max_len - input_length)
        configured_default = default_sampling_params.get("max_tokens", dynamic_default)
        sampling_params.max_tokens = min(configured_default, dynamic_default)

    # Dynamo transports disjoint deltas; the HTTP layer folds or streams them.
    sampling_params.output_kind = RequestOutputKind.DELTA
    return sampling_params
