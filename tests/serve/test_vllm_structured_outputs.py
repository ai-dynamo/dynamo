# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured constraints must affect output, not merely be accepted by HTTP.

Whitespace modifier enforcement remains outside this regression because the
pinned reference uses server-level xgrammar settings (vLLM issue #48765).
"""

import json

import pytest

from tests.utils import vllm_protocol

protocol_server = vllm_protocol.protocol_server
generate = vllm_protocol.generate

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(3.4),
    pytest.mark.post_merge,
    pytest.mark.model("Qwen/Qwen3-0.6B"),
    pytest.mark.requested_vllm_kv_cache_bytes(1_073_741_824),
    pytest.mark.timeout(235),
]


def test_structured_outputs_constrain_chat_and_completion(protocol_server):
    """Guard shared admission and semantic delivery, in unary and streaming modes.

    The parent Dynamo returns 400 for the first nested constraint; unchanged direct
    vLLM must satisfy every case. Completion uses an already-finished reasoning
    prefix, because this reasoner activates constraints only after </think>.
    Whitespace inside valid JSON is allowed; exact string constraints are not trimmed.
    """
    base, evidence = protocol_server
    schema = {
        "type": "object",
        "properties": {"answer": {"const": "violet"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    constraints = [
        ("json", {"json": schema}),
        ("json-string", {"json": json.dumps(schema)}),
        ("json-object", {"json_object": True}),
        (
            "structural-tag",
            {
                "structural_tag": json.dumps(
                    {
                        "type": "structural_tag",
                        "format": {"type": "json_schema", "json_schema": schema},
                    }
                )
            },
        ),
        ("choice", {"choice": ["violet"]}),
        ("regex", {"regex": "violet"}),
        ("grammar", {"grammar": 'root ::= "violet"'}),
    ]
    for endpoint in ("chat", "completion"):
        path = "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"
        payload = {
            "model": "structured-output-test",
            "temperature": 0,
            "max_tokens": 64,
            "stream": False,
        }
        if endpoint == "chat":
            payload.update(
                messages=[{"role": "user", "content": "Answer with the value seven."}],
                chat_template_kwargs={"enable_thinking": False},
            )
        else:
            payload["prompt"] = (
                "<|im_start|>user\nAnswer with the value seven.<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        baseline = generate(base, path, payload, evidence / (endpoint + "-baseline"))
        assert (
            "violet" not in baseline
        ), "The control must distinguish constrained output"
        for stream in (False, True):
            for name, constraint in constraints:
                result = generate(
                    base,
                    path,
                    {**payload, "stream": stream, "structured_outputs": constraint},
                    evidence / f"{endpoint}-{name}-{stream}",
                )
                if name == "json-object":
                    assert isinstance(json.loads(result), dict), result
                elif name in ("json", "json-string", "structural-tag"):
                    assert json.loads(result) == {"answer": "violet"}, result
                else:
                    assert result == "violet", result
        if endpoint == "chat":
            # response_format replaces a same-kind constraint before its content
            # is validated. Text format instead leaves the explicit regex intact.
            overrides = [
                (
                    "replace-json",
                    {"json": "{"},
                    {
                        "type": "json_schema",
                        "json_schema": {"name": "answer", "schema": schema},
                    },
                ),
                ("replace-object", {"json_object": False}, {"type": "json_object"}),
                ("text-regex", {"regex": "violet"}, {"type": "text"}),
            ]
            for stream in (False, True):
                for name, constraint, response_format in overrides:
                    result = generate(
                        base,
                        path,
                        {
                            **payload,
                            "stream": stream,
                            "structured_outputs": constraint,
                            "response_format": response_format,
                        },
                        evidence / f"chat-{name}-{stream}",
                    )
                    if name == "replace-json":
                        assert json.loads(result) == {"answer": "violet"}, result
                    elif name == "replace-object":
                        assert isinstance(json.loads(result), dict), result
                    else:
                        assert result == "violet", result
