# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts against the vLLM APIs installed in the runtime image."""

import importlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from dynamo.common.guided_output import has_named_parameter

reasoning = pytest.importorskip("vllm.reasoning")
async_llm = pytest.importorskip("vllm.v1.engine.async_llm")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
publisher = importlib.import_module("dynamo.vllm.guided_output")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize("parser_name", ["step3", "deepseek_r1", "qwen3"])
def test_installed_reasoner_and_generate_api_match_published_capability(
    parser_name: str,
) -> None:
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(
            tokenizers.models.WordLevel(
                {"[UNK]": 0, "<think>": 1, "</think>": 2}, unk_token="[UNK]"
            )
        ),
        unk_token="[UNK]",
        additional_special_tokens=["<think>", "</think>"],
    )

    class Engine:
        generate = async_llm.AsyncLLM.generate

        def get_tokenizer(self) -> Any:
            return tokenizer

    published = {}
    runtime = SimpleNamespace(
        set_engine_specific=lambda key, value: published.update(json.loads(value))
    )
    config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(
            reasoning_parser=parser_name, enable_in_reasoning=False
        ),
        model_config=SimpleNamespace(skip_tokenizer_init=False),
    )
    publisher.publish_guided_output_capability(runtime, Engine(), config)
    named = has_named_parameter(Engine().generate, "reasoning_ended")
    expected = "explicit_gate" if named and parser_name != "qwen3" else "unsupported"
    assert published["mode"] == expected
    if expected == "explicit_gate":
        assert published["end_token_ids"] == [2]
        assert published["end_text"] == "</think>"
