# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.vllm.protocol_extensions import (
    CAPABILITY_KEY,
    FULL_VOCAB_COUNT,
    MAX_EXTENSION_BYTES,
    MAX_EXTENSION_DEPTH,
    MAX_EXTENSION_VALUES,
    PROMPT_LOGPROBS_CAPABILITY_KEY,
    SAMPLING_FIELDS,
    ProtocolExtensionError,
    apply_sampling_extensions,
    lower_sampling_extensions,
    prompt_logprobs_capability,
    prompt_logprobs_model_limit,
    prompt_logprobs_to_wire,
    protocol_capability,
    resolve_sampling_extensions,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize("limit", [20, 32, -1])
def test_full_vocab_wire_count_and_effective_validator_limit(limit):
    model = SimpleNamespace(max_logprobs=limit, get_vocab_size=lambda: 32)
    runtime = {PROMPT_LOGPROBS_CAPABILITY_KEY: prompt_logprobs_capability(model)}
    assert prompt_logprobs_model_limit(runtime) == limit
    for value in (None, 0, 1):
        assert prompt_logprobs_to_wire(value, runtime) == value
    if limit == 20:
        with pytest.raises(ProtocolExtensionError, match="limit"):
            prompt_logprobs_to_wire(-1, runtime)
    else:
        assert prompt_logprobs_to_wire(-1, runtime) == FULL_VOCAB_COUNT


@pytest.mark.parametrize("value", [-2, FULL_VOCAB_COUNT, 1 << 64, True, 1.0, "1"])
def test_invalid_public_prompt_counts_cannot_wrap_to_a_wire_directive(value):
    with pytest.raises(ProtocolExtensionError, match="expected"):
        prompt_logprobs_to_wire(value, {})


def test_legacy_marker_does_not_authorize_full_vocab_and_defaults_still_work():
    for runtime in ({}, {"vllm_inference_v1_generate": True}):
        assert prompt_logprobs_model_limit(runtime) is None
        assert prompt_logprobs_to_wire(1, runtime) == 1
        with pytest.raises(ProtocolExtensionError, match="not advertised"):
            prompt_logprobs_to_wire(-1, runtime)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("wire_count", "signed"),
        ("max_logprobs", -2),
        ("max_logprobs", True),
        ("vocab_size", 0),
        ("vocab_size", FULL_VOCAB_COUNT),
        ("vocab_size", 32.0),
    ],
)
def test_malformed_full_vocab_contract_never_uses_a_legacy_fallback(field, value):
    contract = prompt_logprobs_capability(
        SimpleNamespace(max_logprobs=-1, get_vocab_size=lambda: 32)
    )
    contract[field] = value
    runtime = {
        PROMPT_LOGPROBS_CAPABILITY_KEY: contract,
        "vllm_inference_v1_generate": True,
    }
    with pytest.raises(ProtocolExtensionError, match="malformed"):
        prompt_logprobs_to_wire(-1, runtime)
    assert prompt_logprobs_to_wire(0, runtime) == 0


def capability():
    return {
        CAPABILITY_KEY: {
            "schema_version": 1,
            "target": "vllm",
            "engine_version": "0.30.0",
            "sampling_fields": sorted(SAMPLING_FIELDS),
        }
    }


def envelope(fields):
    return {"backend_extensions": {"schema_version": 1, "vllm": fields}}


def test_current_frontend_worker_roundtrip():
    fields = {
        "allowed_token_ids": [0, 1],
        "bad_words_token_ids": [[2]],
        "logprob_token_ids": [],
    }
    extra = lower_sampling_extensions(fields, capability())
    assert extra == {**envelope(fields), "sampling_options": fields}
    assert resolve_sampling_extensions({"extra_args": extra}, SAMPLING_FIELDS) == fields


def test_legacy_wire_both_directions_and_ordinary_defaults():
    fields = {"allowed_token_ids": [0, 1]}
    # Current writer to an identified old vLLM worker: the old shape is unchanged.
    legacy = lower_sampling_extensions(fields, {"vllm_inference_v1_generate": True})
    assert legacy == {"sampling_options": fields}
    # Old writer to current worker: normalize immediately at the boundary.
    assert (
        resolve_sampling_extensions({"extra_args": legacy}, SAMPLING_FIELDS) == fields
    )
    assert lower_sampling_extensions({}, {}) == {}
    assert resolve_sampling_extensions({}, set()) == {}
    with pytest.raises(ProtocolExtensionError, match="no verified"):
        lower_sampling_extensions(fields, {})


@pytest.mark.parametrize("value", [None, [], [0], [0xFFFFFFFF]])
def test_omission_null_empty_and_zero_token_ids(value):
    expected = {} if value is None else {"allowed_token_ids": value}
    assert (
        resolve_sampling_extensions(
            {"extra_args": envelope({"allowed_token_ids": value})}, SAMPLING_FIELDS
        )
        == expected
    )


@pytest.mark.parametrize(
    "value", [False, 0, [False], [1.0], [-1], [0x100000000], ["1"], [[1]]]
)
def test_invalid_token_id_values_rejected(value):
    with pytest.raises(ProtocolExtensionError, match="token ID arrays"):
        resolve_sampling_extensions(
            {"extra_args": envelope({"allowed_token_ids": value})}, SAMPLING_FIELDS
        )


@pytest.mark.parametrize("legacy,canonical", [([2], None), (None, [2])])
def test_conflicts_with_old_representation_or_canonical_rejected(legacy, canonical):
    extra = envelope({"allowed_token_ids": [1]})
    if legacy is not None:
        extra["sampling_options"] = {"allowed_token_ids": legacy}
    request = {
        "extra_args": extra,
        "sampling_options": {"allowed_token_ids": canonical},
    }
    with pytest.raises(ProtocolExtensionError, match="conflict"):
        resolve_sampling_extensions(request, SAMPLING_FIELDS)


def test_agreeing_dual_write_is_idempotent_and_null_cannot_revive_legacy_value():
    fields = {"allowed_token_ids": [1]}
    extra = {**envelope(fields), "sampling_options": fields}
    assert resolve_sampling_extensions({"extra_args": extra}, SAMPLING_FIELDS) == fields
    extra["backend_extensions"]["vllm"] = {"allowed_token_ids": None}
    with pytest.raises(ProtocolExtensionError, match="conflicting"):
        resolve_sampling_extensions({"extra_args": extra}, SAMPLING_FIELDS)


@pytest.mark.parametrize(
    "extra",
    [
        {"backend_extensions": {"schema_version": 1, "sglang": {}}},
        {"backend_extensions": {"schema_version": 2, "vllm": {}}},
        {"backend_extensions": {"schema_version": True, "vllm": {}}},
        {"backend_extensions": []},
        envelope({"temperature": 0.5}),
        {"sampling_options": {"temperature": 0.5}},
    ],
)
def test_unknown_versions_backends_and_canonical_injection_rejected(extra):
    with pytest.raises(ProtocolExtensionError):
        resolve_sampling_extensions({"extra_args": extra}, SAMPLING_FIELDS)


def test_missing_engine_attribute_and_authoritative_capability_rejected():
    with pytest.raises(ProtocolExtensionError, match="installed engine"):
        resolve_sampling_extensions(
            {"extra_args": envelope({"allowed_token_ids": [1]})}, set()
        )
    runtime = capability()
    runtime[CAPABILITY_KEY]["sampling_fields"] = []
    runtime["vllm_inference_v1_generate"] = True
    with pytest.raises(ProtocolExtensionError, match="selected worker"):
        lower_sampling_extensions({"allowed_token_ids": [1]}, runtime)


def test_runtime_capability_is_resolved_from_actual_engine_attributes():
    params = SimpleNamespace(allowed_token_ids=None, logprob_token_ids=None)
    resolved = protocol_capability(params, "0.29.0")
    assert resolved["sampling_fields"] == ["allowed_token_ids", "logprob_token_ids"]
    assert resolved["engine_version"] == "0.29.0"


def test_native_empty_logprob_selection_normalization_does_not_drop_token_allowlist():
    params = SimpleNamespace(allowed_token_ids=None, logprob_token_ids=[3])
    apply_sampling_extensions(
        params,
        {"extra_args": envelope({"allowed_token_ids": [], "logprob_token_ids": []})},
    )
    assert params.allowed_token_ids == []
    assert params.logprob_token_ids is None


def test_size_depth_and_value_count_bounded_without_payload_in_error():
    values = ["private" * MAX_EXTENSION_BYTES, [0] * MAX_EXTENSION_VALUES]
    nested = 0
    for _ in range(MAX_EXTENSION_DEPTH + 1):
        nested = [nested]
    values.append(nested)
    for value in values:
        with pytest.raises(ProtocolExtensionError, match="limit exceeded") as error:
            resolve_sampling_extensions(
                {"extra_args": envelope({"allowed_token_ids": value})}, SAMPLING_FIELDS
            )
        assert "private" not in str(error.value)
