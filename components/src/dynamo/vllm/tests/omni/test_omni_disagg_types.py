# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for omni/types.py Protocol definitions.

No GPU, no vllm_omni — pure structural typing checks.
"""

import json

import pytest

try:
    from dynamo.vllm.omni.types import OmniInterStageRequest, StageEngine, StageOutput
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


class _MockEngine:
    engine = None

    def generate(self, prompt, request_id="", *, sampling_params_list=None):
        async def _gen():
            yield {}

        return _gen()

    def get_tokenizer(self):
        return None


def test_stage_engine_protocol_satisfied():
    assert isinstance(_MockEngine(), StageEngine)


def test_missing_generate_not_stage_engine():
    assert not isinstance(object(), StageEngine)


# ── StageOutput ───────────────────────────────────────────


class TestStageOutput:
    def test_unknown_keys_are_dropped(self):
        out = StageOutput.model_validate(
            {"shm_meta": {"name": "x"}, "unknown_key": "noise"}
        )
        assert out.shm_meta == {"name": "x"}
        assert not hasattr(out, "unknown_key")

    def test_to_next_stage_request_excludes_finished_and_error(self):
        out = StageOutput.model_validate(
            {
                "stage_connector_refs": {"0": {"name": "x"}},
                "finished": True,
                "error": None,
            }
        )
        req = out.to_next_stage_request("req-1")
        assert "finished" not in req
        assert "error" not in req
        assert req["request_id"] == "req-1"

    def test_to_next_stage_request_excludes_shm_meta(self):
        """shm_meta is final-stage → router only; must not be forwarded to next stage."""
        out = StageOutput.model_validate({"shm_meta": {"name": "x"}})
        req = out.to_next_stage_request("req-2")
        assert "shm_meta" not in req

    def test_to_next_stage_request_passes_stage_connector_refs(self):
        out = StageOutput.model_validate(
            {
                "original_prompt": {"prompt": "hi"},
                "stage_connector_refs": {"0": {"ref": "abc"}},
            }
        )
        req = out.to_next_stage_request("req-3")
        assert req["original_prompt"] == {"prompt": "hi"}
        assert req["stage_connector_refs"] == {"0": {"ref": "abc"}}
        assert req["request_id"] == "req-3"


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
