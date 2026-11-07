# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Impossible minimum output budgets fail before HTTP streaming starts."""

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


def test_minimum_tokens_rejected_before_streaming(protocol_server):
    """Reject impossible output budgets as HTTP errors, not HTTP 200 SSE errors.

    Equal/zero controls force a known token to prove valid requests still run.
    Test explicit limits and impossible minima above the fixture's context cap,
    including omitted/null maxima. Chat must use max_completion_tokens over the
    legacy max_tokens value in both directions. Error-envelope equality and the
    separate completion default-limit difference are not asserted here.
    """
    base, evidence = protocol_server
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca"
    )
    digit = tokenizer.encode("7", add_special_tokens=False)
    assert len(digit) == 1 and tokenizer.decode(digit) == "7", digit
    instruction = "Reply with exactly the word hello."
    for path, inputs in (
        (
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": instruction}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        (
            "/v1/completions",
            {
                "prompt": f"<|im_start|>user\n{instruction}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            },
        ),
    ):
        endpoint = "chat" if path == "/v1/chat/completions" else "completion"
        forced = {"allowed_token_ids": digit, "max_tokens": 4}
        cases = [
            ("zero", {**forced, "min_tokens": 0}, "7777"),
            ("equal", {**forced, "min_tokens": 4}, "7777"),
            ("explicit-overflow", {**forced, "min_tokens": 5}, None),
            ("omitted-over-context", {"min_tokens": 513}, None),
            ("null-over-context", {"min_tokens": 513, "max_tokens": None}, None),
        ]
        if endpoint == "chat":
            cases.extend(
                [
                    (
                        "primary-allows",
                        {**forced, "max_completion_tokens": 8, "min_tokens": 6},
                        "77777777",
                    ),
                    (
                        "primary-rejects",
                        {
                            **forced,
                            "max_tokens": 8,
                            "max_completion_tokens": 4,
                            "min_tokens": 6,
                        },
                        None,
                    ),
                ]
            )
        for stream in (False, True):
            for name, fields, expected in cases:
                result = generate(
                    base,
                    path,
                    {
                        "model": "structured-output-test",
                        "temperature": 0,
                        "seed": 0,
                        "stream": stream,
                        **inputs,
                        **fields,
                    },
                    evidence / f"{endpoint}-minimum-{name}-{stream}",
                    expected_status=400 if expected is None else 200,
                    expected_finish="length",
                )
                assert result == expected, result
