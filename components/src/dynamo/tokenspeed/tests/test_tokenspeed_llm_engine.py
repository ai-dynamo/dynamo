# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest

import dynamo.tokenspeed.llm_engine as tokenspeed_llm_engine
from dynamo.llm.exceptions import InvalidArgument
from dynamo.tokenspeed.llm_engine import (
    TokenspeedLLMEngine,
    _completion_delta_output,
    _validate_single_choice_sampling,
    build_sampling_params,
    convert_output_to_chunk,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class _FakeAsyncLLM:
    def __init__(self, outputs=None, context_len=2048):
        self.outputs = outputs or []
        self.context_len = context_len
        self.requests = []
        self.aborted = []

    async def generate_request(self, obj):
        self.requests.append(obj)
        for output in self.outputs:
            yield output

    def abort_request(self, rid):
        self.aborted.append(rid)


class _FakeContext:
    def __init__(self, request_id="req-1"):
        self._request_id = request_id

    def id(self):
        return self._request_id


class _FakeGenerateReqInput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _server_args(**overrides):
    values = {
        "model": "token-speed-model",
        "served_model_name": "served-token-speed-model",
        "max_model_len": 4096,
        "block_size": 16,
        "max_total_tokens": None,
        "chunked_prefill_size": None,
        "max_prefill_tokens": None,
        "max_num_seqs": None,
        "stream_output": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_start_launches_tokenspeed_subprocesses_directly(monkeypatch):
    async_llm = _FakeAsyncLLM(context_len=2048)
    server_args = _server_args()
    engine = TokenspeedLLMEngine(server_args)
    kills = []

    def launch(subprocess_server_args):
        assert subprocess_server_args is server_args
        assert subprocess_server_args.stream_output is True
        return async_llm, {
            "max_total_num_tokens": 160,
            "chunked_prefill_size": 32,
            "max_num_seqs": 4,
        }

    monkeypatch.setattr(
        tokenspeed_llm_engine, "_launch_tokenspeed_subprocesses", launch
    )
    monkeypatch.setattr(
        tokenspeed_llm_engine,
        "_kill_tokenspeed_process_tree",
        lambda: kills.append(1),
    )

    config = asyncio.run(engine.start(7))

    assert engine.async_llm is async_llm
    assert config.model == "token-speed-model"
    assert config.served_model_name == "served-token-speed-model"
    assert config.context_length == 2048
    assert config.kv_cache_block_size == 16
    assert config.total_kv_blocks == 10
    assert config.max_num_seqs == 4
    assert config.max_num_batched_tokens == 32

    asyncio.run(engine.cleanup())
    assert kills == [1]
    assert engine._tokenspeed_process_tree_owned is False


def test_start_rejects_non_serving_tokenspeed_rank(monkeypatch):
    engine = TokenspeedLLMEngine(_server_args())
    kills = []

    monkeypatch.setattr(
        tokenspeed_llm_engine,
        "_launch_tokenspeed_subprocesses",
        lambda _server_args: (None, {}),
    )
    monkeypatch.setattr(
        tokenspeed_llm_engine,
        "_kill_tokenspeed_process_tree",
        lambda: kills.append(1),
    )

    with pytest.raises(RuntimeError, match="rank 0"):
        asyncio.run(engine.start(0))

    assert kills == [1]
    assert engine.async_llm is None
    assert engine.scheduler_info == {}
    assert engine._tokenspeed_process_tree_owned is False


def test_start_cleans_up_tokenspeed_process_tree_when_launch_fails(monkeypatch):
    engine = TokenspeedLLMEngine(_server_args())
    kills = []

    def launch(_server_args):
        raise RuntimeError("scheduler failed")

    monkeypatch.setattr(
        tokenspeed_llm_engine, "_launch_tokenspeed_subprocesses", launch
    )
    monkeypatch.setattr(
        tokenspeed_llm_engine,
        "_kill_tokenspeed_process_tree",
        lambda: kills.append(1),
    )

    with pytest.raises(RuntimeError, match="scheduler failed"):
        asyncio.run(engine.start(0))

    assert kills == [1]
    assert engine._tokenspeed_process_tree_owned is False


def test_generate_streams_from_async_llm(monkeypatch):
    async_llm = _FakeAsyncLLM(
        outputs=[
            {"output_ids": [11, 12, 99], "meta_info": {"completion_tokens": 1}},
            {
                "output_ids": [100],
                "meta_info": {
                    "finish_reason": "length",
                    "prompt_tokens": 2,
                    "completion_tokens": 2,
                },
            },
        ]
    )
    engine = TokenspeedLLMEngine(_server_args())
    engine.async_llm = async_llm
    engine._model_max_len = 16
    monkeypatch.setattr(
        tokenspeed_llm_engine,
        "_generate_req_input_cls",
        lambda: _FakeGenerateReqInput,
    )

    async def collect_chunks():
        chunks = []
        async for chunk in engine.generate(
            {
                "token_ids": [11, 12],
                "sampling_options": {"temperature": 0.0},
                "stop_conditions": {"max_tokens": 2},
            },
            _FakeContext("generate-1"),
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect_chunks())

    assert async_llm.requests[0].input_ids == [11, 12]
    assert async_llm.requests[0].sampling_params["temperature"] == 0.0
    assert async_llm.requests[0].sampling_params["max_new_tokens"] == 2
    assert async_llm.requests[0].stream is True
    assert async_llm.requests[0].rid == "generate-1"
    assert chunks[0]["token_ids"] == [99]
    assert chunks[1]["token_ids"] == [100]
    assert chunks[1]["finish_reason"] == "length"
    assert "generate-1" not in engine._active_rids_by_context


def test_abort_forwards_active_request_ids():
    async_llm = _FakeAsyncLLM()
    engine = TokenspeedLLMEngine(_server_args())
    engine.async_llm = async_llm
    engine._active_rids_by_context["ctx-1"] = ["rid-a", "rid-b"]

    asyncio.run(engine.abort(_FakeContext("ctx-1")))

    assert async_llm.aborted == ["rid-a", "rid-b"]


def test_cleanup_kills_tokenspeed_process_tree_once(monkeypatch):
    engine = TokenspeedLLMEngine(_server_args())
    engine.async_llm = _FakeAsyncLLM()
    engine.scheduler_info = {"max_num_seqs": 4}
    engine._tokenspeed_process_tree_owned = True
    kills = []
    monkeypatch.setattr(
        tokenspeed_llm_engine,
        "_kill_tokenspeed_process_tree",
        lambda: kills.append(1),
    )

    asyncio.run(engine.cleanup())
    asyncio.run(engine.cleanup())

    assert kills == [1]
    assert engine.async_llm is None
    assert engine.scheduler_info == {}
    assert engine._tokenspeed_process_tree_owned is False


def test_build_sampling_params_maps_dynamo_request():
    params = build_sampling_params(
        {
            "token_ids": [1, 2, 3],
            "sampling_options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 20,
                "min_p": 0.1,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.4,
                "repetition_penalty": 1.1,
                "seed": 123,
                "n": 1,
                "guided_decoding": {
                    "choice": ["yes", "no"],
                },
            },
            "stop_conditions": {
                "max_tokens": 17,
                "min_tokens": 2,
                "ignore_eos": True,
                "stop_token_ids_hidden": [7, 8],
                "stop_token_ids": [8, 9],
            },
        },
        model_max_len=100,
    )

    assert params["temperature"] == 0.2
    assert params["top_p"] == 0.9
    assert params["top_k"] == 20
    assert params["min_p"] == 0.1
    assert params["frequency_penalty"] == 0.3
    assert params["presence_penalty"] == 0.4
    assert params["repetition_penalty"] == 1.1
    assert params["seed"] == 123
    assert params["n"] == 1
    assert params["max_new_tokens"] == 17
    assert params["min_new_tokens"] == 2
    assert params["ignore_eos"] is True
    assert params["stop_token_ids"] == [7, 8, 9]
    assert params["regex"] == "(yes|no)"


def test_build_sampling_params_maps_json_guided_decoding():
    params = build_sampling_params(
        {
            "token_ids": [1],
            "sampling_options": {
                "guided_decoding": {
                    "json": {"type": "object"},
                },
            },
        },
        model_max_len=10,
    )

    assert json.loads(params["json_schema"]) == {"type": "object"}


def test_build_sampling_params_maps_grammar_guided_decoding():
    params = build_sampling_params(
        {
            "token_ids": [1],
            "sampling_options": {
                "guided_decoding": {
                    "grammar": 'root ::= "x"',
                },
            },
        },
        model_max_len=10,
    )

    assert params["ebnf"] == 'root ::= "x"'


def test_build_sampling_params_maps_structural_tag_guided_decoding():
    params = build_sampling_params(
        {
            "token_ids": [1],
            "sampling_options": {
                "guided_decoding": {
                    "structural_tag": {"begin": "<a>", "schema": {"type": "object"}},
                },
            },
        },
        model_max_len=10,
    )

    assert json.loads(params["structural_tag"]) == {
        "begin": "<a>",
        "schema": {"type": "object"},
    }


def test_build_sampling_params_rejects_multiple_guided_constraints():
    with pytest.raises(InvalidArgument, match="one constraint"):
        build_sampling_params(
            {
                "token_ids": [1],
                "sampling_options": {
                    "guided_decoding": {
                        "json": {"type": "object"},
                        "choice": ["yes", "no"],
                    },
                },
            },
            model_max_len=10,
        )


def test_build_sampling_params_uses_dynamic_max_tokens():
    params = build_sampling_params({"token_ids": [1, 2, 3]}, model_max_len=10)

    assert params["max_new_tokens"] == 7


def test_convert_output_to_chunk_maps_finish_reason_and_usage():
    class FinishReason:
        def to_json(self):
            return {"type": "length", "length": 2}

    chunk = convert_output_to_chunk(
        {
            "index": 1,
            "output_ids": [11, 12],
            "meta_info": {
                "finish_reason": FinishReason(),
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "cached_tokens": 1,
            },
        }
    )

    assert chunk == {
        "index": 1,
        "token_ids": [11, 12],
        "finish_reason": "length",
        "completion_usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
    }


def test_convert_output_to_chunk_normalizes_abort_finish_reason():
    chunk = convert_output_to_chunk(
        {
            "output_ids": [],
            "meta_info": {
                "finish_reason": "abort_request",
                "prompt_tokens": 1,
                "completion_tokens": 0,
            },
        }
    )

    assert chunk["finish_reason"] == "cancelled"


def test_validate_single_choice_sampling_rejects_n_greater_than_one():
    with pytest.raises(InvalidArgument, match="n=2"):
        _validate_single_choice_sampling(
            {"token_ids": [1], "sampling_options": {"n": 2}}
        )


def test_completion_delta_output_strips_first_chunk_prompt_echo():
    out = {
        "output_ids": [10, 11, 12, 99],
        "meta_info": {"completion_tokens": 1},
    }

    delta_out, emitted = _completion_delta_output(out, 0)

    assert emitted == 1
    assert delta_out["output_ids"] == [99]
    assert out["output_ids"] == [10, 11, 12, 99]


def test_completion_delta_output_preserves_later_token_delta():
    out = {
        "output_ids": [100],
        "meta_info": {"completion_tokens": 2},
    }

    delta_out, emitted = _completion_delta_output(out, 1)

    assert emitted == 2
    assert delta_out["output_ids"] == [100]
