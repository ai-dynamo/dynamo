# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned handoff metadata for BridgeKV-style heterogeneous PD.

BridgeKV lets a large source model perform prefill and a smaller, same-family
target model perform decode after translating the source KV cache. This module
defines the internal payload Dynamo carries under
``disaggregated_params["bridgekv"]`` from prefill to decode. It is not an
OpenAI-compatible request or response field.

The source and target cache shapes are recorded for connector validation and
observability. ``compatibility_key`` is the authoritative opaque key computed by
the stitcher loader; it must cover details that these shape fields do not, such
as RoPE variant and scaling, partial-rotary dimensions, KV memory layout, dtype
normalization, tokenizer identity, and model revision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

BRIDGEKV_DISAGGREGATED_KEY = "bridgekv"
BRIDGEKV_SCHEMA_VERSION = 1

_HANDOFF_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "stitcher_id",
        "stitcher_version",
        "stitcher_digest",
        "source_model",
        "target_model",
        "first_token_id",
        "prompt_len",
        "source_cache",
        "target_cache",
        "compatibility_key",
        "fallback_mode",
    }
)
_CACHE_SHAPE_REQUIRED_KEYS = frozenset(
    {"num_layers", "num_kv_heads", "head_dim", "dtype"}
)


class BridgeKVHandoffError(ValueError):
    """The BridgeKV handoff payload is malformed or incompatible."""


class BridgeKVFallbackMode(str, Enum):
    """Decode-side behavior when the translated cache cannot be installed."""

    LOCAL_PREFILL = "local_prefill"
    FAIL = "fail"


def _require_positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BridgeKVHandoffError(f"{path} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BridgeKVHandoffError(f"{path} must be a non-negative integer")
    return value


def _require_nonempty_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BridgeKVHandoffError(f"{path} must be a non-empty string")
    return value


def _require_optional_str(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_str(value, path)


@dataclass(frozen=True, slots=True)
class BridgeKVCacheShape:
    """One model's KV-cache shape for compatibility checks."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str

    def __post_init__(self) -> None:
        _require_positive_int(self.num_layers, "num_layers")
        _require_positive_int(self.num_kv_heads, "num_kv_heads")
        _require_positive_int(self.head_dim, "head_dim")
        _require_nonempty_str(self.dtype, "dtype")

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-compatible representation of this shape."""
        return {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
        }


@dataclass(frozen=True, slots=True)
class BridgeKVHandoff:
    """Prefill-to-decode metadata for one BridgeKV request."""

    stitcher_id: str
    stitcher_version: str
    source_model: str
    target_model: str
    first_token_id: int
    prompt_len: int
    source_cache: BridgeKVCacheShape
    target_cache: BridgeKVCacheShape
    compatibility_key: str
    fallback_mode: BridgeKVFallbackMode
    stitcher_digest: str | None = None
    schema_version: int = BRIDGEKV_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != BRIDGEKV_SCHEMA_VERSION
        ):
            raise BridgeKVHandoffError(
                "schema_version must be "
                f"{BRIDGEKV_SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        _require_nonempty_str(self.stitcher_id, "stitcher_id")
        _require_nonempty_str(self.stitcher_version, "stitcher_version")
        _require_optional_str(self.stitcher_digest, "stitcher_digest")
        _require_nonempty_str(self.source_model, "source_model")
        _require_nonempty_str(self.target_model, "target_model")
        _require_nonnegative_int(self.first_token_id, "first_token_id")
        _require_positive_int(self.prompt_len, "prompt_len")
        if not isinstance(self.source_cache, BridgeKVCacheShape):
            raise BridgeKVHandoffError("source_cache must be a BridgeKVCacheShape")
        if not isinstance(self.target_cache, BridgeKVCacheShape):
            raise BridgeKVHandoffError("target_cache must be a BridgeKVCacheShape")
        _require_nonempty_str(self.compatibility_key, "compatibility_key")
        if not isinstance(self.fallback_mode, BridgeKVFallbackMode):
            raise BridgeKVHandoffError("fallback_mode must be a BridgeKVFallbackMode")

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-compatible handoff payload."""
        return {
            "schema_version": self.schema_version,
            "stitcher_id": self.stitcher_id,
            "stitcher_version": self.stitcher_version,
            "stitcher_digest": self.stitcher_digest,
            "source_model": self.source_model,
            "target_model": self.target_model,
            "first_token_id": self.first_token_id,
            "prompt_len": self.prompt_len,
            "source_cache": self.source_cache.to_payload(),
            "target_cache": self.target_cache.to_payload(),
            "compatibility_key": self.compatibility_key,
            "fallback_mode": self.fallback_mode.value,
        }


def bridgekv_handoff_to_disaggregated_params(
    handoff: BridgeKVHandoff,
) -> dict[str, Any]:
    """Wrap ``handoff`` for the existing ``disaggregated_params`` envelope."""
    if not isinstance(handoff, BridgeKVHandoff):
        raise TypeError("handoff must be a BridgeKVHandoff")
    return {BRIDGEKV_DISAGGREGATED_KEY: handoff.to_payload()}


def extract_bridgekv_handoff(
    disaggregated_params: Mapping[str, Any] | None,
) -> BridgeKVHandoff | None:
    """Extract and validate a BridgeKV handoff.

    A missing ``bridgekv`` key means this request does not use BridgeKV and
    returns ``None``. A present key is parsed fail-closed: malformed fields or
    an unsupported schema version raise :class:`BridgeKVHandoffError` rather
    than silently falling back to incompatible decode behavior.
    """
    if disaggregated_params is None:
        return None
    if not isinstance(disaggregated_params, Mapping):
        raise BridgeKVHandoffError("disaggregated_params must be an object")
    if BRIDGEKV_DISAGGREGATED_KEY not in disaggregated_params:
        return None
    return bridgekv_handoff_from_payload(
        disaggregated_params[BRIDGEKV_DISAGGREGATED_KEY]
    )


def bridgekv_handoff_from_payload(payload: Any) -> BridgeKVHandoff:
    """Parse and validate a raw ``bridgekv`` payload."""
    raw = _require_mapping(payload, BRIDGEKV_DISAGGREGATED_KEY)
    _require_exact_keys(raw, _HANDOFF_REQUIRED_KEYS, BRIDGEKV_DISAGGREGATED_KEY)

    schema_version = _require_positive_int(
        raw["schema_version"], f"{BRIDGEKV_DISAGGREGATED_KEY}.schema_version"
    )
    if schema_version != BRIDGEKV_SCHEMA_VERSION:
        raise BridgeKVHandoffError(
            f"{BRIDGEKV_DISAGGREGATED_KEY}.schema_version must be "
            f"{BRIDGEKV_SCHEMA_VERSION}, got {schema_version}"
        )

    fallback_mode = _parse_fallback_mode(raw["fallback_mode"])
    return BridgeKVHandoff(
        schema_version=schema_version,
        stitcher_id=_require_nonempty_str(
            raw["stitcher_id"], f"{BRIDGEKV_DISAGGREGATED_KEY}.stitcher_id"
        ),
        stitcher_version=_require_nonempty_str(
            raw["stitcher_version"],
            f"{BRIDGEKV_DISAGGREGATED_KEY}.stitcher_version",
        ),
        stitcher_digest=_require_optional_str(
            raw["stitcher_digest"],
            f"{BRIDGEKV_DISAGGREGATED_KEY}.stitcher_digest",
        ),
        source_model=_require_nonempty_str(
            raw["source_model"], f"{BRIDGEKV_DISAGGREGATED_KEY}.source_model"
        ),
        target_model=_require_nonempty_str(
            raw["target_model"], f"{BRIDGEKV_DISAGGREGATED_KEY}.target_model"
        ),
        first_token_id=_require_nonnegative_int(
            raw["first_token_id"],
            f"{BRIDGEKV_DISAGGREGATED_KEY}.first_token_id",
        ),
        prompt_len=_require_positive_int(
            raw["prompt_len"], f"{BRIDGEKV_DISAGGREGATED_KEY}.prompt_len"
        ),
        source_cache=_parse_cache_shape(
            raw["source_cache"], f"{BRIDGEKV_DISAGGREGATED_KEY}.source_cache"
        ),
        target_cache=_parse_cache_shape(
            raw["target_cache"], f"{BRIDGEKV_DISAGGREGATED_KEY}.target_cache"
        ),
        compatibility_key=_require_nonempty_str(
            raw["compatibility_key"],
            f"{BRIDGEKV_DISAGGREGATED_KEY}.compatibility_key",
        ),
        fallback_mode=fallback_mode,
    )


def _parse_cache_shape(value: Any, path: str) -> BridgeKVCacheShape:
    raw = _require_mapping(value, path)
    _require_exact_keys(raw, _CACHE_SHAPE_REQUIRED_KEYS, path)
    dtype = _require_nonempty_str(raw["dtype"], f"{path}.dtype")
    return BridgeKVCacheShape(
        num_layers=_require_positive_int(raw["num_layers"], f"{path}.num_layers"),
        num_kv_heads=_require_positive_int(raw["num_kv_heads"], f"{path}.num_kv_heads"),
        head_dim=_require_positive_int(raw["head_dim"], f"{path}.head_dim"),
        dtype=dtype,
    )


def _parse_fallback_mode(value: Any) -> BridgeKVFallbackMode:
    path = f"{BRIDGEKV_DISAGGREGATED_KEY}.fallback_mode"
    if not isinstance(value, str):
        raise BridgeKVHandoffError(f"{path} must be a string")
    try:
        return BridgeKVFallbackMode(value)
    except ValueError as error:
        allowed = sorted(mode.value for mode in BridgeKVFallbackMode)
        raise BridgeKVHandoffError(
            f"{path} must be one of {allowed}, got {value!r}"
        ) from error


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeKVHandoffError(f"{path} must be an object")
    return value


def _require_exact_keys(
    raw: Mapping[str, Any], expected: frozenset[str], path: str
) -> None:
    actual = set(raw)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise BridgeKVHandoffError(f"{path} is missing required keys: {missing}")
    if unknown:
        raise BridgeKVHandoffError(f"{path} has unknown keys: {unknown}")


__all__ = [
    "BRIDGEKV_DISAGGREGATED_KEY",
    "BRIDGEKV_SCHEMA_VERSION",
    "BridgeKVCacheShape",
    "BridgeKVFallbackMode",
    "BridgeKVHandoff",
    "BridgeKVHandoffError",
    "bridgekv_handoff_from_payload",
    "bridgekv_handoff_to_disaggregated_params",
    "extract_bridgekv_handoff",
]
