# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for omni/types.py Protocol definitions.

No GPU, no vllm_omni — pure structural typing checks.
"""

import json

import pytest

from dynamo.vllm.omni.types import OmniInterStageRequest, StageConnector, StageEngine

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


class _MockEngine:
    def generate(self, prompt, request_id="", *, sampling_params_list=None):
        async def _gen():
            yield {}

        return _gen()


class _MockConnector:
    def put(self, from_stage, to_stage, put_key, data):
        return True, 0, {}

    def cleanup(self, request_id):
        pass


def test_stage_engine_protocol_satisfied():
    assert isinstance(_MockEngine(), StageEngine)


def test_stage_connector_protocol_satisfied():
    assert isinstance(_MockConnector(), StageConnector)


def test_missing_generate_not_stage_engine():
    assert not isinstance(object(), StageEngine)


def test_missing_put_not_stage_connector():
    class NoCleanup:
        def put(self, f, t, r, p):
            return True, 0, {}

    assert not isinstance(NoCleanup(), StageConnector)


def test_missing_cleanup_not_stage_connector():
    class NoPut:
        def cleanup(self, rid):
            pass

    assert not isinstance(NoPut(), StageConnector)


# ── OmniInterStageRequest ──────────────────────────────────


class TestOmniInterStageRequest:
    def test_roundtrip_empty_refs(self):
        req = OmniInterStageRequest(
            request_id="req-1",
            original_prompt={"prompt": "hello", "height": 1024, "width": 1024},
        )
        recovered = OmniInterStageRequest.from_dict(req.to_dict())
        assert recovered.request_id == "req-1"
        assert recovered.original_prompt == {
            "prompt": "hello",
            "height": 1024,
            "width": 1024,
        }
        assert recovered.stage_connector_refs == {}

    def test_roundtrip_with_refs(self):
        req = OmniInterStageRequest(
            request_id="req-2",
            original_prompt={"prompt": "a cat"},
            stage_connector_refs={0: {"name": "abc-shm", "size": 9000}},
        )
        recovered = OmniInterStageRequest.from_dict(req.to_dict())
        assert recovered.stage_connector_refs[0] == {"name": "abc-shm", "size": 9000}

    def test_int_keys_preserved_after_json_roundtrip(self):
        """JSON serializes dict keys as strings — from_dict must convert back to int."""
        req = OmniInterStageRequest(
            request_id="req-3",
            original_prompt=None,
            stage_connector_refs={0: "ref0", 1: "ref1"},
        )
        # Simulate JSON round-trip (Dynamo network boundary)
        as_json = json.loads(json.dumps(req.to_dict()))
        recovered = OmniInterStageRequest.from_dict(as_json)
        assert 0 in recovered.stage_connector_refs
        assert 1 in recovered.stage_connector_refs
        assert isinstance(list(recovered.stage_connector_refs.keys())[0], int)
