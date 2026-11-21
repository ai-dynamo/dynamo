# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np
import pytest

from dynamo.deepinfer.protocol import Output
from dynamo.deepinfer.request import (
    build_sampling_params,
    collect_outputs,
    map_finish_reason,
)
from dynamo.deepinfer.testing import fake_deepinfer as fake

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_sampling_options_map_to_deepinfer_fields():
    params, logprobs = build_sampling_params(
        {
            "token_ids": [1, 2, 3],
            "sampling_options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 20,
                "seed": 42,
                "n": 1,
                # Defaults of fields DeepInfer lacks are accepted.
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
                "best_of": 1,
                "guided_decoding": {"backend": "xgrammar"},
            },
            "stop_conditions": {"max_tokens": 17, "ignore_eos": True, "stop": ["x"]},
            "output_options": {"logprobs": 3},
        },
        max_model_len=100,
    )
    assert params == {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "seq_seed": 42,
        "max_tokens": 17,
        "ignore_eos": True,
        "top_logprobs": 3,
    }
    assert logprobs == 3
    fake.SamplingParams(**params)  # every key is a SamplingParams argument


def test_max_tokens_defaults_to_remaining_context():
    params, logprobs = build_sampling_params({"token_ids": [1] * 30}, max_model_len=100)
    assert params == {"max_tokens": 70}
    assert logprobs is None


@pytest.mark.parametrize("top_k", [0, -1])
def test_disabled_top_k_is_omitted(top_k):
    params, _ = build_sampling_params(
        {"token_ids": [1], "sampling_options": {"top_k": top_k}}, max_model_len=10
    )
    assert "top_k" not in params


def test_logprobs_zero_means_chosen_token_only():
    params, logprobs = build_sampling_params(
        {"token_ids": [1], "output_options": {"logprobs": 0}}, max_model_len=10
    )
    assert logprobs == 0 and "top_logprobs" not in params


@pytest.mark.parametrize(
    "request_, match",
    [
        ({"token_ids": [1], "sampling_options": {"n": 2}}, "n=1"),
        ({"token_ids": []}, "empty"),
        ({"token_ids": [1] * 10}, "max_model_len"),
    ],
)
def test_unservable_requests_raise(request_, match):
    with pytest.raises(ValueError, match=match):
        build_sampling_params(request_, max_model_len=10)


@pytest.mark.parametrize(
    "section, fields",
    [
        ("sampling_options", {"guided_decoding": {"json": {"type": "object"}}}),
        ("sampling_options", {"guided_decoding": {"structural_tag": {}}}),
        ("sampling_options", {"presence_penalty": 0.5}),
        ("sampling_options", {"frequency_penalty": -0.1}),
        ("sampling_options", {"repetition_penalty": 1.1}),
        ("sampling_options", {"min_p": 0.05}),
        ("sampling_options", {"use_beam_search": True}),
        ("sampling_options", {"best_of": 2}),
        ("stop_conditions", {"min_tokens": 5}),
        ("stop_conditions", {"max_thinking_tokens": 64}),
        ("output_options", {"prompt_logprobs": 0}),
    ],
)
def test_unsupported_fields_are_rejected_not_dropped(section, fields):
    name = next(iter(fields))
    with pytest.raises(ValueError, match=f"does not support {name}"):
        build_sampling_params({"token_ids": [1], section: fields}, max_model_len=10)


def test_min_tokens_with_ignore_eos_is_served():
    params, _ = build_sampling_params(
        {
            "token_ids": [1],
            "stop_conditions": {"min_tokens": 5, "max_tokens": 5, "ignore_eos": True},
        },
        max_model_len=10,
    )
    assert params == {"max_tokens": 5, "ignore_eos": True}


@pytest.mark.parametrize(
    "reason, expected",
    [
        ("stop", "stop"),
        ("length", "length"),
        ("abort", "cancelled"),
        ("prompt_length_exceeded", "error"),
        ("insufficient_system_resource", "error"),
        ("ve_error", "error"),
    ],
)
def test_finish_reason_mapping(reason, expected):
    assert map_finish_reason(reason) == expected


@dataclass
class _Ptr:
    rid: str
    token: int
    finished: bool = False
    reason: str = ""
    logprob: float = -0.5
    top_ids: list[int] = field(default_factory=list)
    top_lps: list[float] = field(default_factory=list)


class _RO:
    """RequestOutput stand-in: static getters over a handle table."""

    table: dict[int, _Ptr] = {}

    @classmethod
    def get_request_id(cls, p):
        return cls.table[p].rid

    @classmethod
    def get_token_id(cls, p):
        return cls.table[p].token

    @classmethod
    def get_finished(cls, p):
        return cls.table[p].finished

    @classmethod
    def get_finish_reason(cls, p):
        return cls.table[p].reason

    @classmethod
    def get_token_logprobs(cls, p):
        return cls.table[p].logprob

    @classmethod
    def get_top_token_ids(cls, p):
        return cls.table[p].top_ids

    @classmethod
    def get_top_logprobs(cls, p):
        return cls.table[p].top_lps


def test_collect_outputs_merges_filters_and_maps():
    _RO.table = {
        1: _Ptr("a", 10, top_ids=[10, 11, 12], top_lps=[-0.5, -1.0, -2.0]),
        2: _Ptr(
            "a",
            11,
            finished=True,
            reason="length",
            top_ids=[11, 3, 4],
            top_lps=[-0.1, -1, -2],
        ),
        # aborted while waiting: repeats the last prompt token, which is dropped
        3: _Ptr("b", 99, finished=True, reason="abort", logprob=1.0),
        4: _Ptr("c", 7, finished=True, reason="prompt_length_exceeded", logprob=1.0),
        5: _Ptr("unknown", 5),
        6: _Ptr("d", 8),
    }
    into: dict[str, Output] = {"d": Output(request_id="d", token_ids=[1])}
    live = {"a": 2, "b": 0, "c": 0, "d": 0}
    collect_outputs(np.arange(1, 7, dtype=np.uint64), _RO, live, into)
    assert set(into) == {"a", "b", "c", "d"}
    a = into["a"]
    assert a.token_ids == [10, 11] and a.finish_reason == "length"
    assert a.logprobs == [-0.5, -0.5]
    assert a.top_token_ids == [[10, 11], [11, 3]]
    assert a.top_logprobs == [[-0.5, -1.0], [-0.1, -1]]
    assert into["b"].token_ids == [] and into["b"].finish_reason == "cancelled"
    assert into["c"].token_ids == [] and into["c"].finish_reason == "error"
    assert into["d"].token_ids == [1, 8] and into["d"].top_token_ids == []


def test_step_output_handles_die_at_next_step():
    engine = fake.LLMEngine(fake.EngineArgs(model="m", chunked_prefill=True))
    engine.add_request("u", "r", {"max_tokens": 4}, np.array([2, 3, 4], dtype=np.int32))
    ptrs = engine.step()
    assert fake.RequestOutput.get_request_id(int(ptrs[0])) == "r"
    engine.step()
    with pytest.raises(KeyError):
        fake.RequestOutput.get_request_id(int(ptrs[0]))
    engine.close()


def test_prompt_bytes_passes_packed_token_ids_through():
    import numpy as np

    from dynamo.deepinfer.request import prompt_array, prompt_bytes

    ids = [0, 1, 128000, 2**31 - 1]
    packed = np.asarray(ids, dtype="<i4").tobytes()
    assert prompt_bytes(packed) == prompt_bytes(ids)
    assert prompt_array(prompt_bytes(memoryview(packed))).tolist() == ids


def test_build_sampling_params_counts_packed_token_ids():
    import numpy as np
    import pytest

    from dynamo.deepinfer.request import build_sampling_params

    packed = np.arange(10, dtype="<i4").tobytes()
    with pytest.raises(ValueError, match="10 tokens"):
        build_sampling_params(
            {"token_ids": packed, "stop_conditions": {"max_tokens": 4}}, 10
        )
