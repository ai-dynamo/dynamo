# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the production extension applicator to installed native vLLM adapters.

These CPU tests exercise real protocol models and SamplingParams, not HTTP serve
or generation. The HTTP conformance suite must establish the remaining behavior.
"""

import pytest
import vllm
from vllm import SamplingParams

from dynamo.vllm.protocol_extensions import (
    CAPABILITY_KEY,
    SAMPLING_FIELDS,
    apply_sampling_extensions,
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


@pytest.mark.parametrize("endpoint", ["chat", "completion"])
@pytest.mark.parametrize("field", ["allowed_token_ids", "logprob_token_ids"])
@pytest.mark.parametrize("value", [None, [], [0], [1, 2]])
def test_extension_normalization_matches_native_adapter(endpoint, field, value):
    # Keep version-specific native imports out of engine-free marker collection.
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    if endpoint == "chat":
        native_request = ChatCompletionRequest(
            model="contract-probe",
            messages=[{"role": "user", "content": "Hello"}],
            logprobs=True,
            **{field: value},
        )
    else:
        native_request = CompletionRequest(
            model="contract-probe", prompt="Hello", logprobs=0, **{field: value}
        )
    native = native_request.to_sampling_params(16, {})
    actual = SamplingParams()
    runtime = {CAPABILITY_KEY: protocol_capability(actual, vllm.__version__)}
    extra = lower_sampling_extensions({field: value}, runtime)
    apply_sampling_extensions(actual, {"extra_args": extra})
    assert getattr(actual, field) == getattr(native, field)


def test_installed_engine_supports_advertised_sampling_subset():
    actual = SamplingParams()
    capability = protocol_capability(actual, vllm.__version__)
    assert set(capability["sampling_fields"]) == SAMPLING_FIELDS
    fields = {"bad_words_token_ids": [[0], [1, 2]]}
    extra = lower_sampling_extensions(fields, {CAPABILITY_KEY: capability})
    apply_sampling_extensions(actual, {"extra_args": extra})
    # Token-ID bad words are a Dynamo extension, not a native OpenAI request
    # field. This asserts engine lowering only, not public protocol parity.
    assert actual._bad_words_token_ids == fields["bad_words_token_ids"]


@pytest.mark.parametrize("selection", [None, [], [0], [1, 2]])
@pytest.mark.parametrize("logprobs", [None, False, True])
def test_native_chat_requires_logprobs_for_nonempty_selection(selection, logprobs):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.exceptions import VLLMValidationError

    payload = {
        "model": "contract-probe",
        "messages": [{"role": "user", "content": "Hello"}],
        "logprob_token_ids": selection,
        "logprobs": logprobs,
    }
    if selection and not logprobs:
        with pytest.raises(VLLMValidationError, match="logprobs"):
            ChatCompletionRequest(**payload)
    else:
        ChatCompletionRequest(**payload)


@pytest.mark.parametrize("selection", [None, [], [0], [1, 2]])
@pytest.mark.parametrize("logprobs", [None, 0, 1])
def test_native_completion_allows_zero_logprobs(selection, logprobs):
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.exceptions import VLLMValidationError

    payload = {
        "model": "contract-probe",
        "prompt": "Hello",
        "logprob_token_ids": selection,
        "logprobs": logprobs,
    }
    if selection and logprobs is None:
        with pytest.raises(VLLMValidationError, match="logprobs"):
            CompletionRequest(**payload)
    else:
        CompletionRequest(**payload)
