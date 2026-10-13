# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export token controls already enforced at runtime without changing admission."""

import json

import pytest
from transformers import AutoTokenizer

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


def _root_property(document, node, name):
    """Find a request-root property through these exports' acyclic local refs/allOf.

    Return None for an absent property. Do not descend into arbitrary properties:
    a same-named field inside messages or tools does not describe this request root.
    Broken local refs raise KeyError instead of becoming evidence of absence.
    """
    if name in node.get("properties", {}):
        return node["properties"][name]
    if "$ref" in node:
        reference = node["$ref"]
        assert reference.startswith("#/"), reference
        resolved = document
        for part in reference[2:].split("/"):
            resolved = resolved[part.replace("~1", "/").replace("~0", "~")]
        return _root_property(document, resolved, name)
    for part in node.get("allOf", []):
        found = _root_property(document, part, name)
        if found is not None:
            return found
    return None


def test_token_controls_are_effective_and_declared(protocol_server):
    """Separate existing token-control effects from missing OpenAPI declarations.

    First prove restriction and stopping with real inference on both transports;
    then require each admitted root field in that server's live schema. On the
    schema-deficient parent, inference passes but the declaration assertion fails.
    This does not certify count limits, coercion, or error/empty-content parity.
    Immediate stops may be an empty string or unary chat null; preserve raw bodies
    and check only that the stop token is not emitted, not response-shape equality.
    """
    base, evidence = protocol_server
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca"
    )
    digit = tokenizer.encode("7", add_special_tokens=False)
    assert len(digit) == 1 and tokenizer.decode(digit) == "7", digit
    spec = json.loads((evidence / "openapi.json").read_text())
    paths = ["/v1/chat/completions", "/v1/completions"]
    for path in paths:
        chat = path == paths[0]
        endpoint = "chat" if chat else "completion"
        instruction = "Reply with exactly the word hello."
        payload = {
            "model": "structured-output-test",
            "temperature": 0,
            "seed": 0,
            "max_tokens": 4,
            "stream": False,
        }
        if chat:
            payload.update(
                messages=[{"role": "user", "content": instruction}],
                chat_template_kwargs={"enable_thinking": False},
            )
        else:
            payload["prompt"] = (
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        baseline = generate(base, path, payload, evidence / f"{endpoint}-ids-control")
        assert baseline != "7777", baseline
        for stream in (False, True):
            restricted = {**payload, "stream": stream, "allowed_token_ids": digit}
            result = generate(
                base,
                path,
                restricted,
                evidence / f"{endpoint}-ids-allowed-{stream}",
                expected_finish="length",
            )
            assert result == "7777", result
            result = generate(
                base,
                path,
                {**restricted, "stop_token_ids": digit},
                evidence / f"{endpoint}-ids-stop-{stream}",
            )
            assert result in ("", None), result
    for path in paths:
        root = spec["paths"][path]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]
        for name in ("allowed_token_ids", "stop_token_ids"):
            field = _root_property(spec, root, name)
            assert field is not None, f"{path} accepts {name} but does not declare it"
            arms = field.get("anyOf", [field])
            assert any(
                "array" in arm.get("type", [])
                and arm.get("items", {}).get("type") == "integer"
                for arm in arms
            ), field
