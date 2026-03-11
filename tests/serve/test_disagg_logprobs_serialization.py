# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for disaggregated logprobs serialization round-trip.

TRT-LLM PR #11727 adds first_gen_log_probs to DisaggregatedParams with
integer token-ID dict keys ({4710: Logprob(...)}). The Dynamo Rust
transport layer (pythonize 0.23 → serde_json::Value) requires string map
keys. These tests verify the codec correctly converts between TRT-LLM's
internal format and a JSON-safe transport format.

Mirrors TRT-LLM's TestFirstGenLogProbsSerializeRoundtrip in
tests/unittest/disaggregated/test_openai_disagg_service.py.
"""

import dataclasses
import json

import pytest

try:
    from tensorrt_llm.executor.result import Logprob

    from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsCodec
except ImportError as e:
    pytest.skip(f"tensorrt_llm import failed: {e}", allow_module_level=True)


def _to_asdict_format(logprob_dicts):
    """Convert [{int: Logprob}] to [{int: dict}] to match what
    dataclasses.asdict() produces in the production flow."""
    return [
        {tid: dataclasses.asdict(lp) for tid, lp in pos.items()}
        for pos in logprob_dicts
    ]


@pytest.mark.pre_merge
@pytest.mark.trtllm
@pytest.mark.gpu_0
@pytest.mark.unit
class TestDisaggLogprobsSerializationRoundtrip:
    """Roundtrip tests for first_gen_log_probs serialize/deserialize."""

    def test_none_passthrough(self):
        params = {"first_gen_log_probs": None}
        DisaggregatedParamsCodec.serialize_first_gen_log_probs(params)
        assert params["first_gen_log_probs"] is None
        DisaggregatedParamsCodec.deserialize_first_gen_log_probs(params)
        assert params["first_gen_log_probs"] is None

    def test_missing_field_noop(self):
        params = {"request_type": "context_only"}
        DisaggregatedParamsCodec.serialize_first_gen_log_probs(params)
        assert "first_gen_log_probs" not in params
        DisaggregatedParamsCodec.deserialize_first_gen_log_probs(params)
        assert "first_gen_log_probs" not in params

    def test_single_token_roundtrip(self):
        original = [{4710: Logprob(logprob=-2.3256, rank=1)}]
        # In production, dataclasses.asdict() converts Logprob → dict before
        # serialize is called. Mimic that here.
        params = {"first_gen_log_probs": _to_asdict_format(original)}

        DisaggregatedParamsCodec.serialize_first_gen_log_probs(params)

        # Verify serialized format: list of lists of dicts (no int dict keys)
        serialized = params["first_gen_log_probs"]
        assert isinstance(serialized, list)
        assert isinstance(serialized[0], list)
        assert serialized[0][0]["token_id"] == 4710
        assert serialized[0][0]["logprob"] == pytest.approx(-2.3256)
        assert serialized[0][0]["rank"] == 1

        # Verify JSON-safe (this is the actual failure point — int dict keys
        # cause pythonize 0.23 depythonize to fail with dict_key_not_string)
        json.dumps(params)

        # Round-trip back
        DisaggregatedParamsCodec.deserialize_first_gen_log_probs(params)
        recovered = params["first_gen_log_probs"]
        assert len(recovered) == 1
        assert 4710 in recovered[0]
        assert isinstance(recovered[0][4710], Logprob)
        assert recovered[0][4710].logprob == pytest.approx(-2.3256)
        assert recovered[0][4710].rank == 1

    def test_multi_token_topk_roundtrip(self):
        original = [
            {
                100: Logprob(logprob=-0.1, rank=1),
                200: Logprob(logprob=-2.3, rank=2),
                300: Logprob(logprob=-5.0, rank=3),
            },
            {
                400: Logprob(logprob=-0.05, rank=1),
                500: Logprob(logprob=-3.7, rank=2),
            },
        ]
        params = {"first_gen_log_probs": _to_asdict_format(original)}

        DisaggregatedParamsCodec.serialize_first_gen_log_probs(params)
        json.dumps(params)  # Must be JSON-safe
        DisaggregatedParamsCodec.deserialize_first_gen_log_probs(params)

        recovered = params["first_gen_log_probs"]
        assert len(recovered) == 2
        assert set(recovered[0].keys()) == {100, 200, 300}
        assert set(recovered[1].keys()) == {400, 500}
        for orig_pos, rec_pos in zip(original, recovered, strict=True):
            for tid in orig_pos:
                assert rec_pos[tid].logprob == pytest.approx(orig_pos[tid].logprob)
                assert rec_pos[tid].rank == orig_pos[tid].rank

    def test_rank_none_preserved(self):
        original = _to_asdict_format([{42: Logprob(logprob=-1.0, rank=None)}])
        params = {"first_gen_log_probs": original}

        DisaggregatedParamsCodec.serialize_first_gen_log_probs(params)
        DisaggregatedParamsCodec.deserialize_first_gen_log_probs(params)

        assert params["first_gen_log_probs"][0][42].rank is None

    def test_empty_list_passthrough(self):
        params = {"first_gen_log_probs": []}
        DisaggregatedParamsCodec.serialize_first_gen_log_probs(params)
        assert params["first_gen_log_probs"] == []
