# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from dynamo.common.legacy_vllm import LegacyVllmRelease, LegacyVllmTargets
from dynamo.vllm.protocol_extensions import (
    CAPABILITY_KEY,
    ProtocolExtensionError,
    lower_sampling_extensions,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def declaration():
    return {
        "namespace": "release-a",
        "component": "worker",
        "endpoint": "generate",
        "model": "model-a",
        "worker_type": "aggregated",
        "dynamo_release": "1.4.0",
    }


def test_exact_scope_and_pinned_engine(declaration):
    target = LegacyVllmTargets.from_json(json.dumps([declaration]))
    scope = {
        name: value for name, value in declaration.items() if name != "dynamo_release"
    }
    scope["model_input"] = "tokens"
    assert target.resolve(**scope) is LegacyVllmRelease.DYNAMO_14
    assert target.resolve(**scope).engine_version == "0.26.0"
    assert LegacyVllmRelease.DYNAMO_15.engine_version == "0.28.0"
    for key in scope:
        assert target.resolve(**{**scope, key: "other"}) is None
    assert target.resolve(**{**scope, "worker_type": None}) is None
    assert LegacyVllmTargets().resolve(**scope) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("namespace", "*"),
        ("model", ""),
        ("endpoint", " generate"),
        ("component", "worker\n"),
        ("worker_type", "encode"),
        ("worker_type", None),
        ("dynamo_release", "1.4"),
        ("dynamo_release", "1.3.0"),
        ("dynamo_release", "1.5.0+local"),
        ("unknown", True),
    ],
)
def test_invalid_declarations_rejected(declaration, field, value):
    with pytest.raises(ValueError):
        LegacyVllmTargets.from_json(json.dumps([{**declaration, field: value}]))


def test_duplicate_scope_rejected_even_if_release_differs(declaration):
    for second in (declaration, {**declaration, "dynamo_release": "1.5.0"}):
        with pytest.raises(ValueError, match="duplicate"):
            LegacyVllmTargets.from_json(json.dumps([declaration, second]))
    with pytest.raises(ValueError):
        LegacyVllmTargets.from_json(json.dumps([declaration] * 129))
    with pytest.raises(ValueError):
        LegacyVllmTargets.from_json(" " * 65537)
    with pytest.raises(ValueError, match="duplicate"):
        LegacyVllmTargets.from_json('[{"namespace":"one","namespace":"two"}]')


def test_independent_hops_do_not_inherit_identity(declaration):
    prefill = {
        **declaration,
        "component": "prefill",
        "worker_type": "prefill",
        "dynamo_release": "1.5.0",
    }
    targets = LegacyVllmTargets.from_json(json.dumps([declaration, prefill]))
    assert (
        targets.resolve(
            "release-a", "prefill", "generate", "model-a", "prefill", "tokens"
        )
        is LegacyVllmRelease.DYNAMO_15
    )
    assert (
        targets.resolve(
            "release-a", "worker", "generate", "model-a", "prefill", "tokens"
        )
        is None
    )


@pytest.mark.parametrize("release", list(LegacyVllmRelease))
def test_explicit_legacy_writer_keeps_runtime_facts_unchanged(release):
    runtime = {}
    fields = {"allowed_token_ids": [0], "bad_words_token_ids": [[1, 2]]}
    assert lower_sampling_extensions(fields, runtime, legacy_target=release) == {
        "sampling_options": fields
    }
    assert runtime == {}
    assert lower_sampling_extensions({}, runtime, legacy_target=release) == {}
    with pytest.raises(ProtocolExtensionError, match="no verified"):
        lower_sampling_extensions(fields, runtime)


def test_14_selection_does_not_gain_support_from_generic_marker():
    fields = {"logprob_token_ids": [0]}
    with pytest.raises(ProtocolExtensionError, match="legacy vLLM release"):
        lower_sampling_extensions(
            fields,
            {"vllm_inference_v1_generate": True},
            legacy_target=LegacyVllmRelease.DYNAMO_14,
        )
    assert lower_sampling_extensions(
        fields, {}, legacy_target=LegacyVllmRelease.DYNAMO_15
    ) == {"sampling_options": fields}


@pytest.mark.parametrize(
    "capability",
    [
        None,
        {},
        {"schema_version": 2},
        {
            "schema_version": 1,
            "target": "vllm",
            "engine_version": "0.30.0",
            "sampling_fields": [],
        },
    ],
)
def test_advertised_capability_is_authoritative_even_if_null(capability):
    with pytest.raises(ProtocolExtensionError):
        lower_sampling_extensions(
            {"allowed_token_ids": [0]},
            {CAPABILITY_KEY: capability},
            legacy_target=LegacyVllmRelease.DYNAMO_14,
        )


def test_current_advertisement_wins_over_legacy_fallback():
    fields = {"logprob_token_ids": [0]}
    capability = {
        "schema_version": 1,
        "target": "vllm",
        "engine_version": "0.30.0",
        "sampling_fields": list(fields),
    }
    extra = lower_sampling_extensions(
        fields, {CAPABILITY_KEY: capability}, legacy_target=LegacyVllmRelease.DYNAMO_14
    )
    assert extra["backend_extensions"]["vllm"] == fields


def test_untyped_legacy_target_cannot_enable_fallback():
    with pytest.raises(ProtocolExtensionError, match="invalid explicit"):
        lower_sampling_extensions({"allowed_token_ids": [0]}, {}, legacy_target="1.4.0")
