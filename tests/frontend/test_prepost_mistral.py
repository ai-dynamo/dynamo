#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit test for StreamingPostProcessor with Mistral reasoning + tool calling."""

# mypy seems to be running both sides of the HAS_VLLM if statement
# mypy: ignore-errors

import json

import pytest

from .common import check_module_available

HAS_VLLM = check_module_available("vllm")
if HAS_VLLM:
    from mistral_common.tokens.tokenizers.base import SpecialTokens
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionToolsParam,
    )
    from vllm.entrypoints.openai.engine.protocol import FunctionDefinition
    from vllm.outputs import CompletionOutput
    from vllm.reasoning.mistral_reasoning_parser import MistralReasoningParser
    from vllm.sampling_params import SamplingParams
    from vllm.tokenizers.mistral import MistralTokenizer
    from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

    from dynamo.frontend.prepost import StreamingPostProcessor
else:
    # Fake some types so that `pre-commit` passes
    class MistralTokenizer:
        pass

    class CompletionOutput:
        def __init__(*args, **kwargs):
            pass


pytestmark = [
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,  # "Hardware"
    pytest.mark.pre_merge,  # "Lifecyle"
    pytest.mark.unit,  # "Test Type"
    pytest.mark.skipif(not HAS_VLLM, reason="requires vllm"),
]

# ---------------------------------------------------------------------------
# Mock MistralTokenizer
# ---------------------------------------------------------------------------
# Token IDs from unit_test_4.txt
TOOL_CALLS_TOKEN_ID = 9
EOS_TOKEN_ID = 2
BOS_TOKEN_ID = 1
# Arbitrary IDs for think tokens (not present in this test's output, but
# needed to initialise MistralReasoningParser).
THINK_START_TOKEN_ID = 7
THINK_END_TOKEN_ID = 8


class _InnerTokenizer:
    """Mimics the inner ``tokenizer.tokenizer`` accessed by MistralReasoningParser."""

    def get_special_token(self, token):
        # vLLM 0.17.0 renamed get_control_token -> get_special_token
        return self._token_lookup(token)

    def get_control_token(self, token):
        # kept for older vLLM compat
        return self._token_lookup(token)

    def _token_lookup(self, token):
        return {
            SpecialTokens.begin_think: THINK_START_TOKEN_ID,
            SpecialTokens.end_think: THINK_END_TOKEN_ID,
        }.get(token)


class MockMistralTokenizer(MistralTokenizer):
    """Lightweight MistralTokenizer subclass for testing.

    Passes ``isinstance(tok, MistralTokenizer)`` without needing model files.
    """

    def __new__(cls):
        # Bypass MistralTokenizer.__init__ (needs model artefacts).
        return object.__new__(cls)

    def __init__(self):
        self.version = 11
        self._vocab_dict = {"[TOOL_CALLS]": TOOL_CALLS_TOKEN_ID}
        self.tokenizer = _InnerTokenizer()
        self._special_tokens = ["[TOOL_CALLS]"]

    def __bool__(self):
        # Needed because MistralReasoningParser does ``if not self.model_tokenizer``
        # which triggers __len__ → vocab_size on the real MistralTokenizer.
        return True

    def get_vocab(self):
        return dict(self._vocab_dict)

    @property
    def all_special_tokens(self):
        return self._special_tokens


# ---------------------------------------------------------------------------
# Test data from unit_test_4.txt (stream_interval=1, Mistral format)
#
# Output: [TOOL_CALLS]search_gutenberg_books{"search_terms": ["James Joyce"]}
# No reasoning tokens at all — the model jumps straight to tool calls.
# ---------------------------------------------------------------------------
OUTPUTS_INTERVAL_1 = [
    CompletionOutput(
        index=0,
        text="[TOOL_CALLS]",
        token_ids=[9],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="search",
        token_ids=[8928],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="_g",
        token_ids=[11898],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="uten",
        token_ids=[8318],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="berg",
        token_ids=[6415],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="_",
        token_ids=[1095],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="books",
        token_ids=[32493],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="",
        token_ids=[32],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text='{"',
        token_ids=[19227],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="search",
        token_ids=[8928],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="_",
        token_ids=[1095],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="terms",
        token_ids=[62244],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text='":',
        token_ids=[2811],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=' ["',
        token_ids=[12161],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="James",
        token_ids=[31872],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" Joyce",
        token_ids=[58617],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text='"]',
        token_ids=[4964],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="}",
        token_ids=[1125],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="",
        token_ids=[2],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    ),
]

# ---------------------------------------------------------------------------
# Test data from unit_test_5.txt (stream_interval=20, Mistral format)
#
# Only 2 chunks: [TOOL_CALLS] alone, then the entire function name + JSON
# arguments + EOS in a single CompletionOutput with finish_reason=stop.
# ---------------------------------------------------------------------------
OUTPUTS_INTERVAL_20 = [
    CompletionOutput(
        index=0,
        text="[TOOL_CALLS]",
        token_ids=[9],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text='search_gutenberg_books{"search_terms": ["James Joyce books"]}',
        token_ids=[
            8928,
            11898,
            8318,
            6415,
            1095,
            32493,
            32,
            19227,
            8928,
            1095,
            62244,
            2811,
            12161,
            31872,
            58617,
            12796,
            4964,
            1125,
            2,
        ],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    ),
]

PROMPT_TOKEN_IDS = [
    1,
    5,
    1091,
    19227,
    4994,
    2811,
    1429,
    5165,
    1897,
    1429,
    5165,
    2811,
    16753,
    2391,
    2811,
    1429,
    8928,
    11898,
    8318,
    6415,
    1095,
    32493,
    1897,
    1429,
    14653,
    2811,
    1429,
    8483,
    1394,
    12796,
    1294,
    1278,
    13217,
    111317,
    6415,
    11329,
    1897,
    1429,
    26204,
    2811,
    16753,
    4994,
    2811,
    1429,
    6371,
    1897,
    1429,
    48649,
    2811,
    16753,
    8928,
    1095,
    62244,
    2811,
    16753,
    4994,
    2811,
    1429,
    5477,
    1897,
    1429,
    11089,
    2811,
    16753,
    4994,
    2811,
    1429,
    3607,
    50666,
    1429,
    14653,
    2811,
    1429,
    2525,
    1307,
    6123,
    6856,
    1317,
    3081,
    12796,
    1034,
    47579,
    1429,
    15760,
    2811,
    12161,
    8928,
    1095,
    62244,
    4964,
    2821,
    27028,
    6,
    3,
    7493,
    1584,
    1278,
    26864,
    1307,
    2269,
    7456,
    58617,
    12796,
    1063,
    13516,
    1278,
    9519,
    1317,
    6123,
    1046,
    4,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tokenizer():
    return MockMistralTokenizer()


@pytest.fixture
def request_for_sampling():
    """Construct a ChatCompletionRequest matching the Mistral test spec."""
    return ChatCompletionRequest.model_construct(
        messages=[
            {
                "content": "What are the titles of some James Joyce books? "
                "Use the tool to search.",
                "role": "user",
            }
        ],
        model="mistralai/Ministral-3-3B-Reasoning-2512",
        tools=[
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="search_gutenberg_books",
                    description="Search for books in the Project Gutenberg library",
                    parameters={
                        "type": "object",
                        "properties": {
                            "search_terms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search terms to find books",
                            }
                        },
                        "required": ["search_terms"],
                    },
                ),
            )
        ],
        tool_choice="auto",
        include_reasoning=True,
        stream=False,
        n=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        temperature=None,
        top_p=None,
        skip_special_tokens=True,
        chat_template_kwargs=None,
        reasoning_effort=None,
        parallel_tool_calls=True,
    )


@pytest.fixture
def sampling_params():
    return SamplingParams(
        n=1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=None,
        stop=[],
        stop_token_ids=[],
        include_stop_str_in_output=False,
        ignore_eos=False,
        max_tokens=100000,
        min_tokens=0,
        logprobs=None,
        prompt_logprobs=None,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
    )


@pytest.fixture
def processor(tokenizer, request_for_sampling, sampling_params):
    tool_parser = MistralToolParser(tokenizer)
    return StreamingPostProcessor(
        tokenizer=tokenizer,
        request_for_sampling=request_for_sampling,
        sampling_params=sampling_params,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        tool_parser=tool_parser,
        reasoning_parser_class=MistralReasoningParser,
        chat_template_kwargs={"reasoning_effort": None},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _collect_results(processor, outputs):
    """Run all outputs through process_output and collect non-None results."""
    results = []
    for output in outputs:
        result = processor.process_output(output)
        if result is not None:
            results.append(result)
    return results


def _collect_reasoning(results):
    """Extract and join all reasoning_content from results."""
    parts = []
    for r in results:
        rc = r.get("delta", {}).get("reasoning_content")
        if rc is not None:
            parts.append(rc)
    return "".join(parts)


def _collect_tool_calls(results):
    """Merge all streamed tool_call deltas into complete tool calls."""
    merged: dict[int, dict] = {}
    for r in results:
        tc_list = r.get("delta", {}).get("tool_calls")
        if not tc_list:
            continue
        for tc in tc_list:
            idx = tc["index"]
            if idx not in merged:
                merged[idx] = {
                    "id": tc.get("id"),
                    "type": tc.get("type"),
                    "function": {
                        "name": tc.get("function", {}).get("name"),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    },
                }
            else:
                existing = merged[idx]
                if tc.get("id") and not existing["id"]:
                    existing["id"] = tc["id"]
                if tc.get("type") and not existing["type"]:
                    existing["type"] = tc["type"]
                fn = tc.get("function", {})
                if fn.get("name") and not existing["function"]["name"]:
                    existing["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    existing["function"]["arguments"] += fn["arguments"]
    return [merged[k] for k in sorted(merged)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.vllm
def test_mistral_tool_call(processor):
    """Mistral tool call with no reasoning.

    The model output is:
        [TOOL_CALLS]search_gutenberg_books{"search_terms": ["James Joyce"]}
    with no [THINK]...[/THINK] reasoning block.

    The tool parser should extract the tool call correctly, not leak the
    tool-call markup as plain content.
    """
    results = _collect_results(processor, OUTPUTS_INTERVAL_1)
    tool_calls = _collect_tool_calls(results)

    # -- tool calls must be parsed correctly --------------------------------
    assert len(tool_calls) == 1, (
        f"Expected 1 tool call but got {len(tool_calls)}. "
        "Tool-call markup was likely emitted as plain content."
    )
    tc = tool_calls[0]
    assert tc["function"]["name"] == "search_gutenberg_books"
    assert json.loads(tc["function"]["arguments"]) == {
        "search_terms": ["James Joyce"],
    }
    assert tc["id"] is not None
    assert tc["type"] == "function"

    # -- no reasoning content should be present -----------------------------
    reasoning = _collect_reasoning(results)
    assert reasoning == "", f"Unexpected reasoning content: {reasoning!r}"

    # -- [TOOL_CALLS] markup should not appear in content -------------------
    all_content = "".join(r.get("delta", {}).get("content", "") for r in results)
    assert (
        "[TOOL_CALLS]" not in all_content
    ), f"Raw [TOOL_CALLS] markup leaked into content: {all_content!r}"

    # -- finish reason: remaps "stop" → "tool_calls" per openai-openapi.
    finish_reasons = [r["finish_reason"] for r in results if r.get("finish_reason")]
    assert finish_reasons == ["tool_calls"]


@pytest.mark.vllm
def test_mistral_tool_call_interval_20(
    tokenizer, request_for_sampling, sampling_params
):
    """stream_interval=20: function name + args + EOS in a single chunk.

    Only 2 CompletionOutput objects:
      1. [TOOL_CALLS] alone
      2. search_gutenberg_books{"search_terms": ["James Joyce books"]}
         with finish_reason=stop

    The tool call and finish_reason arrive together.  The processor must
    still emit the parsed tool call and the finish_reason.
    """
    tool_parser = MistralToolParser(tokenizer)
    proc = StreamingPostProcessor(
        tokenizer=tokenizer,
        request_for_sampling=request_for_sampling,
        sampling_params=sampling_params,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        tool_parser=tool_parser,
        reasoning_parser_class=MistralReasoningParser,
        chat_template_kwargs={"reasoning_effort": None},
    )

    results = _collect_results(proc, OUTPUTS_INTERVAL_20)
    tool_calls = _collect_tool_calls(results)

    # -- tool calls must be parsed correctly --------------------------------
    assert len(tool_calls) == 1, (
        f"Expected 1 tool call but got {len(tool_calls)}. "
        "Tool-call markup was likely emitted as plain content."
    )
    tc = tool_calls[0]
    assert tc["function"]["name"] == "search_gutenberg_books"
    assert json.loads(tc["function"]["arguments"]) == {
        "search_terms": ["James Joyce books"],
    }
    assert tc["id"] is not None
    assert tc["type"] == "function"

    # -- no reasoning content should be present -----------------------------
    reasoning = _collect_reasoning(results)
    assert reasoning == "", f"Unexpected reasoning content: {reasoning!r}"

    # -- [TOOL_CALLS] markup should not appear in content -------------------
    all_content = "".join(r.get("delta", {}).get("content", "") for r in results)
    assert (
        "[TOOL_CALLS]" not in all_content
    ), f"Raw [TOOL_CALLS] markup leaked into content: {all_content!r}"

    # -- finish reason: remaps "stop" → "tool_calls" per openai-openapi.
    finish_reasons = [r["finish_reason"] for r in results if r.get("finish_reason")]
    assert finish_reasons == ["tool_calls"]
