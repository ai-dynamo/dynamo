# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private, optional retained-decoder KV metadata for the Rust prefill router."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any

from dynamo.vllm.cache_info import DYNAMO_KV_CACHE_GROUP_COUNT_KEY
from dynamo.vllm.kv_connector_protocols import (
    KvConnectorProtocol,
    NixlConnectorProtocol,
    make_kv_connector_protocol,
)

RETAINED_KV_KEY = "dynamo_retained_kv"
_MAX_PARAMS_BYTES = 256 * 1024
_MAX_PROMPT_TOKENS = 1_048_576


@dataclass(frozen=True)
class RetainedKvCapture:
    ttl_seconds: float | None


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _request_envelope(request: dict[str, Any]) -> dict[str, Any] | None:
    extra = request.get("extra_args")
    if not isinstance(extra, dict) or "vllm_tito" in extra:
        return None
    envelope = extra.get(RETAINED_KV_KEY)
    if (
        not isinstance(envelope, dict)
        or type(envelope.get("version")) is not int
        or envelope["version"] != 1
        or envelope.get("capture") is not True
    ):
        return None
    return envelope


def retained_cache_salt(request: dict[str, Any]) -> str | None:
    envelope = _request_envelope(request)
    salt = envelope.get("cache_salt") if envelope is not None else None
    return salt if isinstance(salt, str) and 0 < len(salt) <= 4096 else None


def _exact_text_request(request: dict[str, Any], sampling_params, lora_request) -> bool:
    return (
        sampling_params.n == 1
        and lora_request is None
        and all(
            request.get(field) is None
            for field in (
                "prompt_embeds",
                "multi_modal_data",
                "multi_modal_uuids",
                "mm_routing_info",
                "encoder_result",
            )
        )
        and not (request.get("routing") or {}).get("lora_name")
        and all(
            (request.get("extra_args") or {}).get(field) is None
            for field in (
                "mm_kwargs_shm",
                "mm_kwargs_nixl",
                "mm_hashes",
                "mm_hashes_by_modality",
            )
        )
    )


def _profile(
    vllm_config: Any, protocol: KvConnectorProtocol
) -> RetainedKvCapture | None:
    if not isinstance(protocol, NixlConnectorProtocol):
        return None
    outer = getattr(vllm_config, "kv_transfer_config", None)
    selected = getattr(protocol._vllm_config, "kv_transfer_config", None)
    metadata = getattr(vllm_config, "additional_config", None)
    group_count = (
        metadata.get(DYNAMO_KV_CACHE_GROUP_COUNT_KEY)
        if isinstance(metadata, dict)
        else None
    )
    if (
        outer is None
        or selected is None
        or selected.kv_connector != "NixlConnector"
        or getattr(outer, "kv_load_failure_policy", None) != "recompute"
        or getattr(selected, "kv_load_failure_policy", None) != "recompute"
        or selected.get_from_extra_config("bidirectional_kv_xfer", False) is not True
        # Current NIXL can report ambiguous flat block failures for multiple
        # scheduler groups, even when recompute is requested.
        or type(group_count) is not int
        or group_count != 1
    ):
        return None
    ttl = selected.get_from_extra_config("decoder_kv_blocks_ttl", None)
    if ttl is not None and (not _finite_number(ttl) or ttl <= 0):
        return None
    return RetainedKvCapture(float(ttl) if ttl is not None else None)


def retained_kv_capture(
    request: dict[str, Any],
    vllm_config: Any,
    sampling_params,
    *,
    is_decode_only: bool,
    lora_request,
) -> RetainedKvCapture | None:
    if (
        not is_decode_only
        or _request_envelope(request) is None
        or not _exact_text_request(request, sampling_params, lora_request)
    ):
        return None
    config = getattr(vllm_config, "kv_transfer_config", None)
    # Other profiles remain ordinary inference; do not instantiate their adapters
    # solely for this optional optimization.
    if config is None or config.kv_connector != "NixlConnector":
        return None
    return _profile(vllm_config, make_kv_connector_protocol(vllm_config))


def _validated_params(params: Any) -> dict[str, Any] | None:
    if not isinstance(params, dict) or any(
        key in params for key in ("_remote_blocks_processed", "_p_side_truncated")
    ):
        return None
    groups = params.get("remote_block_ids")
    if (
        not isinstance(groups, (list, tuple))
        or not 0 < len(groups) <= 128
        or not any(groups)
        or sum(len(group) for group in groups if isinstance(group, (list, tuple)))
        > 65536
        or any(
            not isinstance(group, (list, tuple))
            or any(type(block) is not int or block < 0 for block in group)
            for group in groups
        )
    ):
        return None
    for name in ("remote_engine_id", "remote_request_id", "remote_host"):
        value = params.get(name)
        if not isinstance(value, str) or not 0 < len(value) <= 4096:
            return None
    port = params.get("remote_port")
    count = params.get("remote_num_tokens")
    if (
        type(port) is not int
        or not 1 <= port <= 65535
        or type(count) is not int
        or count <= 0
    ):
        return None
    for name in ("tp_size", "dcp_size", "pp_size"):
        value = params.get(name, 1)
        if type(value) is not int or value <= 0:
            return None
    generation = params.get("remote_registration_generation", 0)
    if type(generation) is not int or generation < 0:
        return None
    expiry = params.get("remote_blocks_expiry_time")
    if expiry is not None and (not _finite_number(expiry)):
        return None
    try:
        encoded = json.dumps(params, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return None
    if len(encoded) > _MAX_PARAMS_BYTES:
        return None
    return copy.deepcopy(params)


def retained_prefill_params(
    request: dict[str, Any],
    vllm_config: Any,
    protocol: KvConnectorProtocol | None,
    sampling_params,
    *,
    lora_request,
) -> dict[str, Any] | None:
    envelope = _request_envelope(request)
    if (
        envelope is None
        or protocol is None
        or not _exact_text_request(request, sampling_params, lora_request)
        or _profile(vllm_config, protocol) is None
    ):
        return None
    params = _validated_params(envelope.get("kv_transfer_params"))
    if params is not None:
        # The engine owns extent, layout, expiry, and failed-load recomputation.
        params.update(do_remote_decode=True, do_remote_prefill=False)
    return params


def retained_terminal_metadata(
    capture: RetainedKvCapture,
    prompt_token_ids: Any,
    kv_transfer_params: Any,
) -> dict[str, Any] | None:
    if (
        not isinstance(prompt_token_ids, (list, tuple))
        or not 0 < len(prompt_token_ids) <= _MAX_PROMPT_TOKENS
        or any(type(token) is not int or token < 0 for token in prompt_token_ids)
    ):
        return None
    params = _validated_params(kv_transfer_params)
    if params is None:
        return None
    metadata: dict[str, Any] = {
        "version": 1,
        "prompt_token_ids": list(prompt_token_ids),
        "kv_transfer_params": params,
    }
    if capture.ttl_seconds is not None:
        metadata["ttl_seconds"] = capture.ttl_seconds
    return metadata
