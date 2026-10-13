# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Word exclusions must suppress a spelling demonstrated by an unconstrained control."""

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


def test_bad_words_exclude_baseline_words(protocol_server):
    """Prove word exclusions reach sampling on both endpoints and transports.

    The control must emit a banned spelling before the constraint is tested;
    otherwise ordinary model output could masquerade as enforcement. Empty lists
    preserve the control and duplicates must not erase the exclusion. Reusing the
    same pinned real-server fixture keeps direct vLLM and Dynamo comparable.
    An empty entry must reject before streaming headers, not open an error stream.
    """
    base, evidence = protocol_server
    instruction = "Reply with exactly the word hello."
    for chat in (True, False):
        endpoint = "chat" if chat else "completion"
        path = "/v1/chat/completions" if chat else "/v1/completions"
        payload = {
            "model": "structured-output-test",
            "temperature": 0,
            "seed": 0,
            "max_tokens": 16,
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
        baseline = generate(base, path, payload, evidence / f"{endpoint}-words-control")
        excluded = ["Hello", "hello"]
        assert any(word in baseline for word in excluded), baseline
        for stream in (False, True):
            for name, words in [
                ("exclude", excluded),
                ("duplicates", excluded + excluded),
                ("empty", []),
            ]:
                result = generate(
                    base,
                    path,
                    {**payload, "stream": stream, "bad_words": words},
                    evidence / f"{endpoint}-words-{name}-{stream}",
                )
                if words:
                    assert result and result != baseline, result
                    assert all(word not in result for word in excluded), result
                else:
                    assert result == baseline, result
            generate(
                base,
                path,
                {**payload, "stream": stream, "bad_words": [""]},
                evidence / f"{endpoint}-words-empty-entry-{stream}",
                expected_status=400,
            )
