# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from dynamo.frontend.delta_tokenizer import DeltaConfig
from dynamo.frontend.toktier_session import create_delta_tokenizer

try:
    from dynamo.frontend.sglang_prepost import preprocess_chat_request
except ImportError:
    pytest.skip("requires SGLang frontend dependencies", allow_module_level=True)

pytestmark = [pytest.mark.post_merge, pytest.mark.integration, pytest.mark.gpu_0]

_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def test_sglang_rendered_prompt_pipeline_parity(delta_staged_tokenizer):
    reference, directory = delta_staged_tokenizer
    reference.chat_template = _TEMPLATE
    manager = create_delta_tokenizer(reference, directory, "qwen", DeltaConfig())
    assert manager is not None
    request = {
        "messages": [
            {"role": "user", "content": "Check the source and explain.\n" * 8192}
        ]
    }
    options = {
        "tokenizer": reference,
        "tool_call_parser_name": None,
        "reasoning_parser_name": None,
    }
    with manager:
        for _ in range(5):
            baseline = preprocess_chat_request(copy.deepcopy(request), **options)
            result = preprocess_chat_request(
                copy.deepcopy(request),
                **options,
                delta_tokenizer=manager,
                delta_session_id="session",
                delta_scope="tenant",
            )
            assert result.prompt_token_ids == baseline.prompt_token_ids
            manager.wait_for_idle()
            request["messages"].extend(
                [
                    {"role": "assistant", "content": "The result is correct."},
                    {"role": "user", "content": "Check the next part."},
                ]
            )
        assert manager.stats().get("incremental", 0) > 0
        assert manager.stats()["mismatches"] == 0


def test_missing_scope_and_unsupported_template_options_use_original_pipeline(
    delta_staged_tokenizer,
):
    reference, directory = delta_staged_tokenizer
    manager = create_delta_tokenizer(reference, directory, "qwen", DeltaConfig())
    assert manager is not None
    request = {"messages": [{"role": "user", "content": "Check this result."}]}
    options = {
        "tokenizer": reference,
        "tool_call_parser_name": None,
        "reasoning_parser_name": None,
    }
    with manager:
        expected = preprocess_chat_request(copy.deepcopy(request), **options)
        result = preprocess_chat_request(
            copy.deepcopy(request), **options, delta_tokenizer=manager
        )
        assert result.prompt_token_ids == expected.prompt_token_ids
        assert manager.stats()["entries"] == 0
        request["chat_template_kwargs"] = {"return_dict": True}
        expected = preprocess_chat_request(copy.deepcopy(request), **options)
        result = preprocess_chat_request(
            copy.deepcopy(request),
            **options,
            delta_tokenizer=manager,
            delta_session_id="session",
            delta_scope="tenant",
        )
        assert result.prompt_token_ids == expected.prompt_token_ids
        assert manager.stats()["entries"] == 0
