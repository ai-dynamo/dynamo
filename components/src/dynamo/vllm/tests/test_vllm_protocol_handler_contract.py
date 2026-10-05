# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native protocol comparison through the complete production sampling builder.

Requires Dynamo's real bindings and the installed vLLM backend, without a model
or GPU. Unlike the isolated extension-helper tests, this covers interactions with
canonical sampling/output options and post-extension normalization in the worker.
It does not establish HTTP, generation, disaggregation, or N-2 release parity.
"""

import pytest
import vllm
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.handlers import build_sampling_params, build_sampling_params_openai
from dynamo.vllm.protocol_extensions import (
    CAPABILITY_KEY,
    lower_sampling_extensions,
    protocol_capability,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
]


def internal_request(fields, wire):
    capability = protocol_capability(SamplingParams(), vllm.__version__)
    extra = lower_sampling_extensions(fields, {CAPABILITY_KEY: capability})
    if wire == "legacy":
        extra.pop("backend_extensions", None)
    return {
        "token_ids": [1, 2],
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 16},
        "output_options": {"logprobs": 0},
        "extra_args": extra,
    }


@pytest.mark.parametrize("endpoint", ["chat", "completion"])
@pytest.mark.parametrize("wire", ["legacy", "dual"])
@pytest.mark.parametrize("field", ["allowed_token_ids", "logprob_token_ids"])
@pytest.mark.parametrize("value", [None, [], [0], [1, 2]])
def test_complete_worker_sampling_builder_matches_native(endpoint, wire, field, value):
    # Keep version-specific native imports out of engine-free marker collection.
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    fields = {field: value}
    if endpoint == "chat":
        native = ChatCompletionRequest(
            model="contract-probe",
            messages=[{"role": "user", "content": "Hello"}],
            logprobs=True,
            top_logprobs=0,
            **fields,
        ).to_sampling_params(16, {})
    else:
        native = CompletionRequest(
            model="contract-probe", prompt="Hello", logprobs=0, **fields
        ).to_sampling_params(16, {})
    actual = build_sampling_params(internal_request(fields, wire), {})
    assert getattr(actual, field) == getattr(native, field)
    # This happens after extension application: nonempty explicit token IDs must
    # clear natural top-k logprobs, otherwise engine validation rejects the pair.
    assert actual.logprobs == native.logprobs
    assert actual.max_tokens == native.max_tokens
    assert actual.detokenize is False
    assert actual.output_kind == RequestOutputKind.DELTA


@pytest.mark.parametrize("wire", ["legacy", "dual"])
@pytest.mark.parametrize("value", [None, [], [[0]], [[1, 2], [3]]])
def test_bad_word_token_ids_survive_complete_builder(wire, value):
    actual = build_sampling_params(
        internal_request({"bad_words_token_ids": value}, wire), {}
    )
    expected = SamplingParams()._bad_words_token_ids if value is None else value
    assert actual._bad_words_token_ids == expected


@pytest.mark.parametrize("wire", ["legacy", "dual"])
def test_canonical_collision_is_a_typed_worker_error(wire):
    request = internal_request({"allowed_token_ids": [0]}, wire)
    request["sampling_options"]["allowed_token_ids"] = [1]
    with pytest.raises(InvalidArgument, match="conflicts with canonical"):
        build_sampling_params(request, {})


@pytest.mark.parametrize("value", [False, -1, "1", [True], [-1]])
def test_malformed_wire_is_rejected_by_complete_worker_builder(value):
    request = internal_request({}, "dual")
    request["extra_args"] = {
        "backend_extensions": {
            "schema_version": 1,
            "vllm": {"allowed_token_ids": value},
        }
    }
    with pytest.raises(InvalidArgument, match="token ID arrays"):
        build_sampling_params(request, {})


@pytest.mark.parametrize("field", ["best_of", "use_beam_search", "length_penalty"])
@pytest.mark.parametrize("value", [False, 0, 1])
@pytest.mark.parametrize("path", ["token", "openai"])
def test_unimplemented_canonical_sampling_fields_fail_closed(field, value, path):
    # A new upstream attribute is a review signal, not automatic permission to
    # treat a formerly unsupported feature as implemented.
    assert not hasattr(SamplingParams(), field)
    request = internal_request({}, "dual")
    request["sampling_options"][field] = value
    with pytest.raises(InvalidArgument, match=f"`{field}`.*adapter"):
        if path == "token":
            build_sampling_params(request, {})
        else:
            build_sampling_params_openai({field: value}, {})


def test_legacy_optional_fields_with_null_keep_default_requests_operable():
    request = internal_request({}, "legacy")
    request["sampling_options"].update(
        best_of=None, use_beam_search=None, length_penalty=None
    )
    actual = build_sampling_params(request, {})
    assert actual.max_tokens == 16
    actual = build_sampling_params_openai(
        {"best_of": None, "use_beam_search": None, "length_penalty": None}, {}
    )
    assert actual.detokenize is True


def test_unknown_canonical_wire_key_is_not_echoed_in_error():
    request = internal_request({}, "dual")
    request["sampling_options"]["private-client-marker"] = 1
    with pytest.raises(InvalidArgument, match="sampling_options") as caught:
        build_sampling_params(request, {})
    assert "private-client-marker" not in str(caught.value)


@pytest.mark.parametrize("prompt_logprobs", [0, 1, -1])
def test_prompt_logprobs_defaults_match_native_sampling_builder(prompt_logprobs):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    native = ChatCompletionRequest(
        model="contract-probe",
        messages=[{"role": "user", "content": "Hello"}],
        prompt_logprobs=prompt_logprobs,
    ).to_sampling_params(16, {})
    request = internal_request({}, "dual")
    # The Rust internal unsigned representation reserves MAX_U32 for full vocab.
    request["output_options"] = {
        "prompt_logprobs": 0xFFFFFFFF if prompt_logprobs == -1 else prompt_logprobs
    }
    actual = build_sampling_params(request, {})
    assert actual.prompt_logprobs == native.prompt_logprobs
    assert actual.skip_reading_prefix_cache == native.skip_reading_prefix_cache
