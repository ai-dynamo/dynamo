#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.model_handler.api_inference.glm52_openai import (
    GLM52OpenAIChatCompletionsHandler,
)
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler


def main() -> None:
    config = MODEL_CONFIG_MAPPING["zai-org/GLM-5.2-FC"]
    assert config.model_name == "zai-org/GLM-5.2"
    assert config.model_handler is GLM52OpenAIChatCompletionsHandler

    handler = config.model_handler(
        model_name=config.model_name,
        temperature=0,
        registry_name="zai-org/GLM-5.2-FC",
        is_fc_model=True,
    )
    assert not isinstance(handler, OSSHandler)

    captured: dict = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(), 0.0

    handler.generate_with_backoff = fake_generate
    inference_data = {
        "message": [{"role": "user", "content": "Use the tool."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    handler._query_FC(inference_data)
    assert captured["messages"] == inference_data["message"]
    assert captured["tools"] == inference_data["tools"]
    assert captured["tool_choice"] == "auto"
    assert captured["max_tokens"] == 64000
    assert "prompt" not in captured

    function = SimpleNamespace(name="get_weather", arguments='{"city":"Paris"}')
    tool_call = SimpleNamespace(id="call-1", type="function", function=function)
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=usage,
    )
    parsed = handler._parse_query_response_FC(response)
    assert parsed["model_responses"] == [{"get_weather": '{"city":"Paris"}'}]
    assert parsed["tool_call_ids"] == ["call-1"]
    assert parsed["input_token"] == 11
    assert parsed["output_token"] == 7
    print("GLM-5.2 native Chat Completions adapter validation: PASS")


if __name__ == "__main__":
    main()
