#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""SGLang adapter for the shared FRONTEND fixtures (not a test module).

Builds a real ``sglang_prepost.py::SglangStreamingPostProcessor`` and replays a
case's model_text through ``process_output``. SGLang takes token_ids and
detokenizes internally, so this feeds token batches directly (no incremental
detok needed, unlike the vLLM adapter). ``with_tools`` selects the FRONTEND.4
path (hermes function-call parser) vs the FRONTEND.6 fast plain-text path."""

from frontend_fixture_cases import load_tools
from sglang.srt.entrypoints.openai.protocol import Function as SglangFunction
from sglang.srt.entrypoints.openai.protocol import Tool as SglangTool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from dynamo.frontend.sglang_prepost import SglangStreamingPostProcessor


def build_postprocessor(tokenizer, *, with_tools: bool) -> SglangStreamingPostProcessor:
    parser = None
    if with_tools:
        tools = [
            SglangTool(
                type="function",
                function=SglangFunction(name=t["name"], parameters=t["parameters"]),
            )
            for t in load_tools()
        ]
        parser = FunctionCallParser(tools=tools, tool_call_parser="hermes")
    return SglangStreamingPostProcessor(
        tokenizer=tokenizer,
        tool_call_parser=parser,
        reasoning_parser=None,
    )


def replay(
    tokenizer, model_text: str, batch_size: int | None, *, with_tools: bool
) -> list[dict]:
    post = build_postprocessor(tokenizer, with_tools=with_tools)
    token_ids = tokenizer.encode(model_text, add_special_tokens=False)
    if batch_size is None:  # single-chunk: whole response + finish in one call
        batch_size = len(token_ids) or 1

    choices: list[dict] = []
    for i in range(0, len(token_ids), batch_size):
        batch = token_ids[i : i + batch_size]
        is_last = i + batch_size >= len(token_ids)
        choice = post.process_output(
            {"token_ids": batch, "finish_reason": "stop" if is_last else None}
        )
        if choice:
            choices.append(choice)
    return choices
