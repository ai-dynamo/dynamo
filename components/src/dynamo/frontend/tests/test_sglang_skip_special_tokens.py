# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit decode flags survive projection and local frontend decoding."""

import asyncio

import pytest
from _routed_engine_fakes import FakeRoutedEngine

from dynamo.frontend.sglang_prepost import SglangStreamingPostProcessor
from dynamo.frontend.sglang_processor import SglangProcessor, _build_dynamo_preproc

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class SpecialTokenTokenizer:
    chat_template = ""

    def apply_chat_template(self, messages, **kwargs):
        return [1]

    def decode(self, token_ids, *, skip_special_tokens):
        return "".join(
            "<special>" if token == 2 else "A"
            for token in token_ids
            if token != 2 or not skip_special_tokens
        )


@pytest.mark.parametrize("requested", [None, True, False])
@pytest.mark.parametrize("parsers", ["none", "tool", "reasoning", "both"])
def test_skip_policy_matches_projection_and_postprocessor(requested, parsers):
    # FRONTEND.3 / FRONTEND.6: parser delimiters must remain visible even
    # when callers ask to skip special tokens; preserve the existing policy.
    tool = object() if parsers in ("tool", "both") else None
    reasoning = object() if parsers in ("reasoning", "both") else None
    request = {} if requested is None else {"skip_special_tokens": requested}
    expected = False if parsers != "none" else requested is not False
    projected = _build_dynamo_preproc(
        request, [1], "test", None, tool_call_parser=tool, reasoning_parser=reasoning
    )
    post = SglangStreamingPostProcessor(
        tokenizer=SpecialTokenTokenizer(),
        tool_call_parser=tool,
        reasoning_parser=reasoning,
        skip_special_tokens=requested,
    )
    assert projected["output_options"]["skip_special_tokens"] is expected
    assert post._decode_ids([2]) == ("" if expected else "<special>")


@pytest.mark.parametrize("requested", [None, True, False])
def test_generator_honors_explicit_skip_special_tokens(requested):
    # FRONTEND.3 / FRONTEND.6: exercise the caller wiring, not just the helper.
    engine = FakeRoutedEngine(items=[{"token_ids": [2, 3], "finish_reason": "length"}])
    processor = SglangProcessor(SpecialTokenTokenizer(), engine, None, None, None)
    request = {"model": "test", "messages": [{"role": "user", "content": "Hi"}]}
    if requested is not None:
        request["skip_special_tokens"] = requested

    async def collect():
        return [item async for item in processor.generator(request)]

    output = asyncio.run(collect())
    content = "".join(
        choice["delta"].get("content", "")
        for item in output
        for choice in item.get("data", {}).get("choices", [])
    )
    assert content == ("<special>A" if requested is False else "A")
    assert engine.requests[0]["output_options"]["skip_special_tokens"] is (
        requested is not False
    )


@pytest.mark.parametrize("requested", [True, False])
def test_explicit_skip_does_not_change_matched_stop_trimming(requested):
    # FRONTEND.5 / FRONTEND.6: displaying special tokens does not expose
    # a confirmed stopping suffix, which is controlled separately.
    post = SglangStreamingPostProcessor(
        tokenizer=SpecialTokenTokenizer(),
        tool_call_parser=None,
        reasoning_parser=None,
        eos_token_ids=[2],
        skip_special_tokens=requested,
    )
    choice = post.process_output(
        {"token_ids": [3, 2], "finish_reason": "stop", "stop_terminated": True}
    )
    assert choice["delta"]["content"] == "A"
    assert choice["finish_reason"] == "stop"
