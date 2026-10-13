# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completion response formats constrain output and reject conflicting guidance."""

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


def test_completion_response_format_controls_output(protocol_server):
    """Guard completion-format admission, constraint effects and nested overrides.

    The prompt requests seven while schemas require violet. Legacy structural tags
    constrain only triggered regions; the Markdown fence must actually be emitted
    before its JSON constraint counts as enforced. Null/text retain unconstrained
    generation, and text must not erase a separately supplied regex constraint.
    The shared helper retains raw unary/SSE evidence and rejects error streams.
    """
    base, evidence = protocol_server
    path = "/v1/completions"
    payload = {
        "model": "structured-output-test",
        "prompt": (
            "<|im_start|>user\nReturn the answer seven as JSON inside <answer> and </answer>."
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
    }
    control = generate(base, path, payload, evidence / "format-baseline")
    assert "violet" not in control, control
    schema = {
        "type": "object",
        "properties": {"answer": {"const": "violet"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    json_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": schema},
    }
    tag_format = {
        "type": "structural_tag",
        "format": {"type": "json_schema", "json_schema": schema},
    }
    legacy_tag = {
        "type": "structural_tag",
        "structures": [{"begin": "```json\n", "schema": schema, "end": "\n```"}],
        "triggers": ["```json"],
    }
    cases = [
        ("json", json_format, None, "json"),
        ("object", {"type": "json_object"}, None, "object"),
        ("tag", tag_format, None, "json"),
        ("legacy-tag", legacy_tag, None, "tagged-json"),
        ("null", None, None, "unconstrained"),
        ("text", {"type": "text"}, None, "unconstrained"),
        ("json-override", json_format, {"json": "{"}, "json"),
        ("object-override", {"type": "json_object"}, {"json_object": False}, "object"),
        (
            "tag-override",
            tag_format,
            {
                "structural_tag": json.dumps(
                    {
                        "type": "structural_tag",
                        "format": {
                            "type": "json_schema",
                            "json_schema": {"const": "seven"},
                        },
                    }
                )
            },
            "json",
        ),
        ("text-keeps-regex", {"type": "text"}, {"regex": "violet"}, "regex"),
    ]
    for stream in (False, True):
        for name, response_format, nested, expected in cases:
            request = {**payload, "stream": stream, "response_format": response_format}
            if nested is not None:
                request["structured_outputs"] = nested
            result = generate(base, path, request, evidence / f"format-{name}-{stream}")
            if expected == "unconstrained":
                assert result and "violet" not in result, result
            elif expected == "regex":
                assert result == "violet", result
            elif expected == "object":
                assert isinstance(json.loads(result), dict), result
            else:
                if expected == "tagged-json":
                    assert result.startswith("```json\n") and result.endswith(
                        "\n```"
                    ), result
                    result = result[len("```json\n") : -len("\n```")]
                assert json.loads(result) == {"answer": "violet"}, result
