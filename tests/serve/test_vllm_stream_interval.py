# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-request stream intervals preserve grouping, final flushing and unary text."""

import json

import pytest
from transformers import AutoTokenizer

from tests.utils import vllm_protocol
from tests.utils.openapi_root import root_property as _root_property

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


def test_stream_interval_controls_chunk_grouping(protocol_server):
    """Guard interval admission, delivery and final flushing, not just aggregate text.

    Twelve forced one-character tokens expose intervals 3 and 20 independently of
    model obedience. The parent rejects the first explicit interval; the unchanged
    reference must satisfy every case. Queues may combine adjacent emissions, so
    nonfinal text prefixes must lie on the first-token-plus-interval boundaries,
    not have an exact frame count. Final flushing must preserve all twelve tokens.
    Null/omission retain defaults; unary output is unaffected; invalid intervals
    reject with JSON before streaming. Live schemas must declare the root field.
    """
    base, evidence = protocol_server
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca"
    )
    digit = tokenizer.encode("7", add_special_tokens=False)
    assert len(digit) == 1 and tokenizer.decode(digit) == "7", digit
    spec = json.loads((evidence / "openapi.json").read_text())
    cases = [
        ("omitted", {}, 200),
        ("grouped", {"stream_interval": 3}, 200),
        ("final-flush", {"stream_interval": 20}, 200),
        ("one", {"stream_interval": 1}, 200),
        ("null", {"stream_interval": None}, 200),
        ("zero", {"stream_interval": 0}, 400),
        ("negative", {"stream_interval": -1}, 400),
        ("fractional", {"stream_interval": 1.5}, 400),
    ]
    for chat in (True, False):
        endpoint = "chat" if chat else "completion"
        path = "/v1/chat/completions" if chat else "/v1/completions"
        instruction = "Reply with exactly the word hello."
        payload = {
            "model": "structured-output-test",
            "temperature": 0,
            "seed": 0,
            "max_tokens": 12,
            "allowed_token_ids": digit,
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
        for stream in (False, True):
            for name, fields, status in cases:
                capture = evidence / f"{endpoint}-interval-{name}-{stream}"
                result = generate(
                    base,
                    path,
                    {**payload, "stream": stream, **fields},
                    capture,
                    expected_status=status,
                    expected_finish="length",
                )
                if status != 200:
                    continue
                assert result == "7" * 12, result
                if not stream:
                    continue
                body = (capture / "response.body").read_text()
                total = 0
                interval = fields.get("stream_interval") or 1
                for line in body.splitlines():
                    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                        continue
                    for choice in json.loads(line[5:].strip()).get("choices", []):
                        assert choice["index"] == 0, choice
                        content = choice.get("delta", {}) if chat else choice
                        text = content.get("content" if chat else "text") or ""
                        if text:
                            total += len(text)
                            assert total == 12 or (total - 1) % interval == 0, body
                assert total == 12, body
        root = spec["paths"][path]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]
        assert _root_property(spec, root, "stream_interval") is not None
