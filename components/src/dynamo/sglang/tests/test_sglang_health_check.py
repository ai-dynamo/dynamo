# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sglang health-check payload shape tests.

Asserts the canary HEALTH_CHECK_KEY marker is layered onto the disagg payload
(which the prefill handler reads), absent from the decode/agg payload (which
no handler reads), and survives DYN_HEALTH_CHECK_PAYLOAD env overrides.
"""

import json
from types import SimpleNamespace

import pytest

from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.sglang.health_check import (
    SglangDisaggHealthCheckPayload,
    SglangEmbeddingHealthCheckPayload,
    SglangHealthCheckPayload,
)
from dynamo.sglang.protocol import EmbeddingRequest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.fault_tolerance,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_disagg_payload_has_marker():
    assert SglangDisaggHealthCheckPayload().to_dict()[HEALTH_CHECK_KEY] is True


def test_decode_payload_has_no_marker():
    # Decode/agg handler doesn't read the marker; payload stays unmarked.
    assert HEALTH_CHECK_KEY not in SglangHealthCheckPayload().to_dict()


@pytest.mark.parametrize("use_text_input", [True, False])
def test_embedding_payload_matches_request_schema(monkeypatch, use_text_input):
    monkeypatch.delenv("DYN_HEALTH_CHECK_PAYLOAD", raising=False)
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(tokenizer=SimpleNamespace(bos_token_id=42))
    )
    payload = SglangEmbeddingHealthCheckPayload(
        "embedding-model", engine, use_text_input=use_text_input
    ).to_dict()

    request = EmbeddingRequest(**payload)
    assert request.model == "embedding-model"
    assert request.input == ("Test" if use_text_input else [42])


def test_embedding_env_override(monkeypatch):
    override = {"model": "embedding-model", "input": "custom probe"}
    monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", json.dumps(override))

    payload = SglangEmbeddingHealthCheckPayload("embedding-model").to_dict()

    assert payload == override
    assert EmbeddingRequest(**payload).input == "custom probe"


def test_disagg_env_override_preserves_marker(monkeypatch):
    """DYN_HEALTH_CHECK_PAYLOAD must not drop the canary marker."""
    monkeypatch.setenv(
        "DYN_HEALTH_CHECK_PAYLOAD",
        json.dumps(
            {
                "token_ids": [1],
                "sampling_options": {"temperature": 0.0},
                "stop_conditions": {"max_tokens": 1},
            }
        ),
    )
    assert SglangDisaggHealthCheckPayload().to_dict()[HEALTH_CHECK_KEY] is True
