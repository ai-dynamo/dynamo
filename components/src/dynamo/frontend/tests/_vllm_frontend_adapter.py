#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for the shared FRONTEND fixtures (not a test module).

Builds a real ``prepost.py::StreamingPostProcessor`` and replays a case's
model_text through ``process_output``, mirroring how vLLM's output_processor
emits incremental (token_ids, text) deltas. ``with_tools`` selects the
FRONTEND.4 path (hermes tool parser) vs the FRONTEND.6 fast plain-text path
(no parser configured)."""

from types import SimpleNamespace

from frontend_fixture_cases import load_tools
from vllm.reasoning import ReasoningParserManager
from vllm.sampling_params import SamplingParams
from vllm.tool_parsers import ToolParserManager

from dynamo.frontend.prepost import StreamingPostProcessor, _prepare_request

MODEL = "Qwen/Qwen3-0.6B"


def build_postprocessor(
    tokenizer, *, with_tools: bool, with_reasoning: bool = False
) -> StreamingPostProcessor:
    request = {"model": MODEL, "messages": [{"role": "user", "content": "go"}]}
    tool_parser_class = None
    if with_tools:
        tool_parser_class = ToolParserManager.get_tool_parser("hermes")
        request["tools"] = [{"type": "function", "function": t} for t in load_tools()]
        request["tool_choice"] = "auto"
    request_for_sampling, tool_parser, _, _, _ = _prepare_request(
        request, tokenizer=tokenizer, tool_parser_class=tool_parser_class
    )
    reasoning_parser_class = (
        ReasoningParserManager.get_reasoning_parser("qwen3") if with_reasoning else None
    )
    return StreamingPostProcessor(
        tokenizer=tokenizer,
        request_for_sampling=request_for_sampling,
        sampling_params=SamplingParams(),
        prompt_token_ids=[],
        tool_parser=tool_parser,
        reasoning_parser_class=reasoning_parser_class,
        chat_template_kwargs={},
    )


def replay(
    tokenizer,
    model_text: str,
    batch_size: int | None,
    *,
    with_tools: bool,
    with_reasoning: bool = False,
) -> list[dict]:
    post = build_postprocessor(
        tokenizer, with_tools=with_tools, with_reasoning=with_reasoning
    )
    token_ids = tokenizer.encode(model_text, add_special_tokens=False)
    if batch_size is None:  # single-chunk: whole response + finish in one call
        batch_size = len(token_ids) or 1

    choices: list[dict] = []
    previous_text = ""
    for i in range(0, len(token_ids), batch_size):
        batch = token_ids[i : i + batch_size]
        is_last = i + batch_size >= len(token_ids)
        cumulative = tokenizer.decode(
            token_ids[: i + len(batch)], skip_special_tokens=False
        )
        delta_text = cumulative[len(previous_text) :]
        previous_text = cumulative
        output = SimpleNamespace(
            index=0,
            token_ids=batch,
            text=delta_text,
            finish_reason="stop" if is_last else None,
            logprobs=None,
        )
        choice = post.process_output(output)
        if choice:
            choices.append(choice)
    return choices
