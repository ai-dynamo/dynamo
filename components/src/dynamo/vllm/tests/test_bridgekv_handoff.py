# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the BridgeKV internal handoff contract.

The module under test is stdlib-only, so these tests run without a vLLM
installation and are intentionally not named ``test_vllm_*`` (that prefix
makes this directory's conftest skip collection when vLLM is unavailable).
"""

from __future__ import annotations

import copy
import json

import pytest

from dynamo.vllm.bridgekv import (
    BRIDGEKV_DISAGGREGATED_KEY,
    BRIDGEKV_SCHEMA_VERSION,
    BridgeKVCacheShape,
    BridgeKVFallbackMode,
    BridgeKVHandoff,
    BridgeKVHandoffError,
    bridgekv_handoff_from_payload,
    bridgekv_handoff_to_disaggregated_params,
    extract_bridgekv_handoff,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _shape(**overrides):
    values = {
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "dtype": "bfloat16",
    }
    values.update(overrides)
    return BridgeKVCacheShape(**values)


def _handoff(**overrides):
    values = {
        "stitcher_id": "llama3-8b-to-1b",
        "stitcher_version": "2026.11",
        "stitcher_digest": "sha256:abc123",
        "source_model": "meta-llama/Llama-3.1-8B-Instruct",
        "target_model": "meta-llama/Llama-3.2-1B-Instruct",
        "first_token_id": 9906,
        "prompt_len": 1024,
        "source_cache": _shape(),
        "target_cache": _shape(num_layers=16, num_kv_heads=8, head_dim=64),
        "compatibility_key": "sha256:def456",
        "fallback_mode": BridgeKVFallbackMode.LOCAL_PREFILL,
    }
    values.update(overrides)
    return BridgeKVHandoff(**values)


def _payload(**overrides):
    payload = _handoff().to_payload()
    payload.update(overrides)
    return payload


def test_round_trip_through_disaggregated_params_envelope():
    handoff = _handoff()

    envelope = bridgekv_handoff_to_disaggregated_params(handoff)

    assert envelope[BRIDGEKV_DISAGGREGATED_KEY] == handoff.to_payload()
    assert extract_bridgekv_handoff(envelope) == handoff


def test_payload_is_json_serializable():
    envelope = bridgekv_handoff_to_disaggregated_params(_handoff())

    encoded = json.dumps(envelope, sort_keys=True)

    assert extract_bridgekv_handoff(json.loads(encoded)) == _handoff()


def test_missing_key_means_non_bridgekv_request():
    assert extract_bridgekv_handoff(None) is None
    assert extract_bridgekv_handoff({}) is None
    assert extract_bridgekv_handoff({"kv_transfer_params": {}}) is None


def test_envelope_must_be_an_object():
    with pytest.raises(BridgeKVHandoffError, match="disaggregated_params"):
        extract_bridgekv_handoff([])  # type: ignore[arg-type]


def test_missing_required_field_fails_closed():
    payload = _payload()
    del payload["first_token_id"]

    with pytest.raises(BridgeKVHandoffError, match="missing required keys"):
        bridgekv_handoff_from_payload(payload)


def test_unknown_field_fails_closed():
    payload = _payload(unexpected="value")

    with pytest.raises(BridgeKVHandoffError, match="unknown keys"):
        bridgekv_handoff_from_payload(payload)


@pytest.mark.parametrize("field", ["stitcher_id", "source_model", "compatibility_key"])
def test_required_strings_must_be_nonempty(field):
    payload = _payload(**{field: "   "})

    with pytest.raises(BridgeKVHandoffError, match=field):
        bridgekv_handoff_from_payload(payload)


@pytest.mark.parametrize("field", ["first_token_id", "prompt_len"])
@pytest.mark.parametrize("value", [True, False])
def test_bool_is_not_accepted_as_integer(field, value):
    payload = _payload(**{field: value})

    with pytest.raises(BridgeKVHandoffError, match=field):
        bridgekv_handoff_from_payload(payload)


def test_first_token_must_be_nonnegative():
    with pytest.raises(BridgeKVHandoffError, match="first_token_id"):
        bridgekv_handoff_from_payload(_payload(first_token_id=-1))


def test_prompt_len_must_be_positive():
    with pytest.raises(BridgeKVHandoffError, match="prompt_len"):
        bridgekv_handoff_from_payload(_payload(prompt_len=0))


def test_unknown_schema_version_fails_closed():
    with pytest.raises(BridgeKVHandoffError, match="schema_version"):
        bridgekv_handoff_from_payload(
            _payload(schema_version=BRIDGEKV_SCHEMA_VERSION + 1)
        )


def test_schema_version_bool_is_rejected():
    with pytest.raises(BridgeKVHandoffError, match="schema_version"):
        bridgekv_handoff_from_payload(_payload(schema_version=True))


def test_cache_shape_must_be_an_object():
    with pytest.raises(BridgeKVHandoffError, match=r"source_cache must be an object"):
        bridgekv_handoff_from_payload(_payload(source_cache=[]))


def test_cache_shape_missing_field_fails_closed():
    source_cache = copy.deepcopy(_payload()["source_cache"])
    del source_cache["head_dim"]

    with pytest.raises(BridgeKVHandoffError, match="source_cache"):
        bridgekv_handoff_from_payload(_payload(source_cache=source_cache))


@pytest.mark.parametrize("field", ["num_layers", "num_kv_heads", "head_dim"])
@pytest.mark.parametrize("value", [0, -1, True, "128"])
def test_cache_shape_dimensions_must_be_positive_integers(field, value):
    target_cache = _handoff().target_cache.to_payload()
    target_cache[field] = value

    with pytest.raises(BridgeKVHandoffError, match=f"target_cache.{field}"):
        bridgekv_handoff_from_payload(_payload(target_cache=target_cache))


def test_cache_dtype_must_be_nonempty():
    target_cache = _handoff().target_cache.to_payload()
    target_cache["dtype"] = ""

    with pytest.raises(BridgeKVHandoffError, match="target_cache.dtype"):
        bridgekv_handoff_from_payload(_payload(target_cache=target_cache))


def test_fallback_mode_must_be_known():
    with pytest.raises(BridgeKVHandoffError, match="fallback_mode"):
        bridgekv_handoff_from_payload(_payload(fallback_mode="silent_wrong_answer"))


def test_fallback_mode_fail_round_trips():
    handoff = _handoff(fallback_mode=BridgeKVFallbackMode.FAIL)

    parsed = extract_bridgekv_handoff(bridgekv_handoff_to_disaggregated_params(handoff))

    assert parsed == handoff


def test_stitcher_digest_is_optional_but_key_is_required():
    handoff = _handoff(stitcher_digest=None)

    assert (
        extract_bridgekv_handoff(bridgekv_handoff_to_disaggregated_params(handoff))
        == handoff
    )

    payload = _payload()
    del payload["stitcher_digest"]
    with pytest.raises(BridgeKVHandoffError, match="missing required keys"):
        bridgekv_handoff_from_payload(payload)


def test_handoff_rejects_non_handoff_in_envelope_helper():
    with pytest.raises(TypeError, match="BridgeKVHandoff"):
        bridgekv_handoff_to_disaggregated_params({})  # type: ignore[arg-type]


def test_direct_handoff_construction_validates_fields():
    with pytest.raises(BridgeKVHandoffError, match="fallback_mode"):
        _handoff(fallback_mode="local_prefill")


def test_cache_shape_requires_positive_dimensions_on_construction():
    with pytest.raises(BridgeKVHandoffError, match="head_dim"):
        BridgeKVCacheShape(
            num_layers=16,
            num_kv_heads=8,
            head_dim=0,
            dtype="bfloat16",
        )
