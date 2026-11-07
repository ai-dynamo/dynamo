# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit caller stops remain active when implicit EOS termination is disabled."""

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


def test_ignore_eos_preserves_explicit_user_stops(protocol_server):
    """Ignore implicit EOS without discarding caller stop strings or token IDs.

    Force digit7 or the pinned tokenizer's EOS, so a lost stop yields four tokens
    and a length finish instead of an immediate stop. An EOS-only control proves
    ignore_eos itself works; an explicitly requested EOS stop must still win.
    Check both endpoints/transports against the unchanged reference. Empty unary
    chat content may be null; raw bodies retain that separate response difference.
    """
    base, evidence = protocol_server
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca"
    )
    digit = tokenizer.encode("7", add_special_tokens=False)
    eos = [tokenizer.eos_token_id]
    eos_text = tokenizer.decode(eos, skip_special_tokens=False)
    assert len(digit) == 1 and tokenizer.decode(digit) == "7", digit
    assert eos_text, eos
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
        for stream in (False, True):
            payload = {
                "model": "structured-output-test",
                "temperature": 0,
                "seed": 0,
                "max_tokens": 4,
                "stream": stream,
                "ignore_eos": True,
                "skip_special_tokens": False,
                **inputs,
            }
            for name, allowed, stop, expected in (
                ("eos-control", eos, {}, eos_text * 4),
                ("digit-control", digit, {}, "7777"),
                ("user-string", digit, {"stop": "7"}, None),
                ("user-id", digit, {"stop_token_ids": digit}, None),
                ("explicit-eos", eos, {"stop_token_ids": eos}, None),
            ):
                result = generate(
                    base,
                    path,
                    {**payload, "allowed_token_ids": allowed, **stop},
                    evidence / f"{endpoint}-ignore-eos-{name}-{stream}",
                    expected_finish="stop" if stop else "length",
                )
                if expected is None:
                    assert result in ("", None), result
                else:
                    assert result == expected, result
