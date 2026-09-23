# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize TensorRT-LLM V2 multimodal KV event tokens.

TensorRT-LLM's V2 KV cache manager encodes the first token of a multimodal
item as its 32-byte content digest and later item tokens as
``vocab_size + item_offset``.  Its ``mm_keys`` list associates each maximal
multimodal run with both that digest and the frontend-provided routing UUID.

This module decodes the fixed NVIDIA/TensorRT-LLM#19529 event shape and
converts it to Dynamo's canonical integer-only routing tokens at the backend
ingress boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from dynamo.common.multimodal.routing_utils import pad_value_for_mm_hash

_DIGEST_HEX_LENGTH = 64
_ROUTING_UUID_HEX_LENGTH = 16


class UnsupportedMultimodalKvEvent(ValueError):
    """A digest-bearing event cannot be normalized without risking false hits."""


@dataclass(frozen=True)
class _MmRun:
    start: int
    end: int
    item_offset: int
    digest: str | None


def _validated_digest(value: Any) -> str:
    if not isinstance(value, str) or len(value) != _DIGEST_HEX_LENGTH:
        raise UnsupportedMultimodalKvEvent("invalid digest encoding")
    try:
        digest_bytes = bytes.fromhex(value)
    except ValueError as error:
        raise UnsupportedMultimodalKvEvent("invalid digest encoding") from error
    if len(digest_bytes) != _DIGEST_HEX_LENGTH // 2:
        raise UnsupportedMultimodalKvEvent("invalid digest encoding")
    return digest_bytes.hex()


def _validated_routing_uuid(value: Any) -> str:
    if not isinstance(value, str) or len(value) != _ROUTING_UUID_HEX_LENGTH:
        raise UnsupportedMultimodalKvEvent("invalid multimodal routing UUID")
    try:
        uuid_bytes = bytes.fromhex(value)
    except ValueError as error:
        raise UnsupportedMultimodalKvEvent("invalid multimodal routing UUID") from error
    if len(uuid_bytes) != _ROUTING_UUID_HEX_LENGTH // 2:
        raise UnsupportedMultimodalKvEvent("invalid multimodal routing UUID")
    return uuid_bytes.hex()


def _extract_runs(token_ids: list[Any], token_id_offset: int) -> list[_MmRun]:
    runs: list[_MmRun] = []
    token_index = 0
    while token_index < len(token_ids):
        token_id = token_ids[token_index]
        if isinstance(token_id, str):
            digest = _validated_digest(token_id)
            start = token_index
            token_index += 1
            item_offset = 1
            # In KVCM V2's synthetic cache-key stream, item offset zero is the
            # digest token. Continuations are vocab_size + item_offset, so the
            # first integer continuation is token_id_offset + 1.
            while (
                token_index < len(token_ids)
                and type(token_ids[token_index]) is int
                and token_ids[token_index] > token_id_offset
            ):
                if token_ids[token_index] != token_id_offset + item_offset:
                    raise UnsupportedMultimodalKvEvent(
                        "non-consecutive continuation tokens"
                    )
                token_index += 1
                item_offset += 1
            runs.append(_MmRun(start, token_index, 0, digest))
            continue

        if type(token_id) is int and token_id > token_id_offset:
            start = token_index
            start_offset = token_id - token_id_offset
            token_index += 1
            expected_offset = start_offset + 1
            while (
                token_index < len(token_ids)
                and type(token_ids[token_index]) is int
                and token_ids[token_index] > token_id_offset
            ):
                if token_ids[token_index] != token_id_offset + expected_offset:
                    raise UnsupportedMultimodalKvEvent(
                        "non-consecutive continuation tokens"
                    )
                token_index += 1
                expected_offset += 1
            runs.append(_MmRun(start, token_index, start_offset, None))
            continue

        token_index += 1
    return runs


def _validated_mm_key(mm_key: Any) -> tuple[str, int, str]:
    if not isinstance(mm_key, dict) or mm_key.get("type") != "mm_key":
        raise UnsupportedMultimodalKvEvent("invalid multimodal key")
    digest = _validated_digest(mm_key.get("hash"))
    routing_uuid = _validated_routing_uuid(mm_key.get("uuid"))
    start_offset = mm_key.get("start_offset")
    if type(start_offset) is not int or start_offset < 0:
        raise UnsupportedMultimodalKvEvent("invalid multimodal key offset")
    return digest, start_offset, routing_uuid


def normalize_kv_event_blocks(
    blocks: list[dict[str, Any]], token_id_offset: int | None
) -> tuple[list[list[int]], bool]:
    """Return integer token blocks and whether V2 MM normalization was applied.

    Validation covers every block before a caller publishes any part of the
    stored event. An exception means the event must be dropped atomically.
    ``mm_keys`` supplies item identity; the V2 token stream locates and sizes
    each run. Text tokens retain their integer representation.
    """

    raw_blocks: list[list[int | str]] = []
    has_digest = False
    has_continuation = False
    for block in blocks:
        raw_token_ids: list[int | str] = []
        for token in block["tokens"]:
            token_id = token["token_id"]
            if isinstance(token_id, str):
                normalized_token_id = token_id
                has_digest = True
            elif type(token_id) is int:
                normalized_token_id = token_id
            else:
                raise UnsupportedMultimodalKvEvent("invalid token ID type")
            has_continuation = has_continuation or (
                token_id_offset is not None
                and type(normalized_token_id) is int
                and normalized_token_id > token_id_offset
            )
            raw_token_ids.append(normalized_token_id)
        raw_blocks.append(raw_token_ids)

    if not has_digest and not has_continuation:
        return cast(list[list[int]], raw_blocks), False
    if token_id_offset is None or token_id_offset < 0:
        raise UnsupportedMultimodalKvEvent(
            "multimodal continuation-token offset is unavailable"
        )

    normalized_blocks: list[list[int]] = []
    for block, raw_token_ids in zip(blocks, raw_blocks, strict=True):
        runs = _extract_runs(raw_token_ids, token_id_offset)
        mm_keys = block.get("mm_keys")
        if not isinstance(mm_keys, list) or len(mm_keys) != len(runs):
            raise UnsupportedMultimodalKvEvent("multimodal run/key count mismatch")

        normalized = [
            token_id if type(token_id) is int else 0 for token_id in raw_token_ids
        ]
        for run, mm_key in zip(runs, mm_keys, strict=True):
            digest, start_offset, routing_uuid = _validated_mm_key(mm_key)
            if start_offset != run.item_offset:
                raise UnsupportedMultimodalKvEvent(
                    "multimodal key offset does not match token run"
                )
            if run.digest is not None and digest != run.digest:
                raise UnsupportedMultimodalKvEvent(
                    "digest token does not match multimodal key"
                )

            mm_hash = int(routing_uuid, 16)
            pad_value = pad_value_for_mm_hash(mm_hash)
            normalized[run.start : run.end] = [pad_value] * (run.end - run.start)

        normalized_blocks.append(normalized)

    return normalized_blocks, True
