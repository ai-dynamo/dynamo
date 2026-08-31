# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for engine-owned conversation-affinity ADP routing helpers.

Wheel-independent: the ConversationParams import is guarded, so these run on any
TensorRT-LLM build (including rc19, which lacks the API)."""

import pytest

from dynamo.trtllm.conversation_affinity import (
    CONVERSATION_PARAMS_AVAILABLE,
    conversation_params_for,
    engine_conversation_affinity_enabled,
    session_id_from_request,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_session_id_from_request_absent_or_malformed():
    assert session_id_from_request({}) is None
    assert session_id_from_request({"agent_context": None}) is None
    assert session_id_from_request({"agent_context": "not-a-dict"}) is None
    assert session_id_from_request({"agent_context": {}}) is None
    assert session_id_from_request({"agent_context": {"session_id": ""}}) is None
    assert session_id_from_request({"agent_context": {"session_id": "   "}}) is None
    assert session_id_from_request({"agent_context": {"session_id": 123}}) is None


def test_session_id_from_request_valid():
    assert (
        session_id_from_request({"agent_context": {"session_id": "run-123:agent-0"}})
        == "run-123:agent-0"
    )
    # trims surrounding whitespace
    assert session_id_from_request({"agent_context": {"session_id": " s1 "}}) == "s1"


class _Cfg:
    def __init__(self, value):
        self.kv_cache_routing_conversation_affinity = value


class _Args:
    def __init__(self, adp_cfg):
        self.attention_dp_config = adp_cfg


class _Llm:
    def __init__(self, args):
        self.args = args


def test_engine_conversation_affinity_enabled_dict():
    assert (
        engine_conversation_affinity_enabled(
            _Llm(_Args({"kv_cache_routing_conversation_affinity": True}))
        )
        is True
    )
    assert (
        engine_conversation_affinity_enabled(
            _Llm(_Args({"kv_cache_routing_conversation_affinity": False}))
        )
        is False
    )
    assert engine_conversation_affinity_enabled(_Llm(_Args({}))) is False


def test_engine_conversation_affinity_enabled_model():
    assert engine_conversation_affinity_enabled(_Llm(_Args(_Cfg(True)))) is True
    assert engine_conversation_affinity_enabled(_Llm(_Args(_Cfg(False)))) is False
    # only a genuine boolean True enables it
    assert engine_conversation_affinity_enabled(_Llm(_Args(_Cfg(1)))) is False


def test_engine_conversation_affinity_enabled_missing():
    assert engine_conversation_affinity_enabled(_Llm(_Args(None))) is False
    assert engine_conversation_affinity_enabled(_Llm(None)) is False
    assert engine_conversation_affinity_enabled(object()) is False


def test_conversation_params_for_none_id_is_none():
    assert conversation_params_for(None) is None


@pytest.mark.skipif(
    not CONVERSATION_PARAMS_AVAILABLE,
    reason="ConversationParams requires TensorRT-LLM newer than 1.3.0rc20",
)
def test_conversation_params_for_builds_when_available():
    params = conversation_params_for("run-123:agent-0")
    assert params is not None
    assert params.conversation_id == "run-123:agent-0"
