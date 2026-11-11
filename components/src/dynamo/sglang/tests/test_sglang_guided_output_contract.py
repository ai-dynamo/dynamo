# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts against the SGLang APIs installed in the runtime image."""

import importlib
import json
from types import SimpleNamespace

import pytest

from dynamo.common.guided_output import has_named_parameter

sglang = pytest.importorskip("sglang")
reasoning = pytest.importorskip("sglang.srt.parser.reasoning_parser")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
publisher = importlib.import_module("dynamo.sglang.guided_output")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.sglang,
    pytest.mark.pre_merge,
]


def test_installed_qwen3_detector_and_engine_api_match_published_capability() -> None:
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
        async_generate = sglang.Engine.async_generate
        tokenizer_manager = SimpleNamespace(tokenizer=tokenizer)

    published = {}
    runtime = SimpleNamespace(
        set_engine_specific=lambda key, value: published.update(json.loads(value))
    )
    publisher.publish_guided_output_capability(
        runtime,
        Engine(),
        SimpleNamespace(reasoning_parser="qwen3", skip_tokenizer_init=False),
    )
    named = has_named_parameter(Engine().async_generate, "require_reasoning")
    current_reasoner = has_named_parameter(reasoning.ReasoningParser, "tokenizer")
    expected = "explicit_gate" if named and current_reasoner else "unsupported"
    assert published["mode"] == expected
    if expected == "explicit_gate":
        assert published["end_token_ids"] == [2]
        assert published["end_text"] == "</think>"
