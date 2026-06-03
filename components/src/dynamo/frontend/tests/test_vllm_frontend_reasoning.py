#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""FE.process_output.9 reasoning<->tool orchestration — vLLM, over the shared case set.

Both a qwen3 reasoning parser and a hermes tool parser are active; process_output
must route <think>...</think> to reasoning_content and tool markup to tool_calls
without cross-leak. Runs the SAME cases as test_sglang_frontend_reasoning.py."""

import _vllm_frontend_adapter as adapter
import pytest
from frontend_fixture_cases import assert_case, load_cases, normalize, params
from transformers import AutoTokenizer

# Needs vllm packages (gpu_1 container), but does not allocate GPU VRAM.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.xpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),
]

MODEL = "Qwen/Qwen3-0.6B"
CASES = load_cases("frontend_reasoning")

# Known vLLM gaps surfaced by the shared cases (strict xfail -> flips to a
# failure the moment vLLM is fixed). sglang handles these cases.
_KNOWN_GAPS = {
    (
        "reasoning_then_multiple_tool_calls",
        10,
    ): "vLLM leaks the first of multiple post-reasoning tool calls into content (recovers only the last) -- the reasoning-end + multi-tool buffered path in StreamingPostProcessor. sglang recovers both.||tool_calls: [search_books(Joyce)] only; get_weather's <tool_call> markup leaked into content (reasoning_content was correct)",
}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


class TestVllmFrontendReasoning:  # FE.process_output.9 — reasoning <-> tool-call orchestration (shared case set)
    @pytest.mark.parametrize("case,batch_size", params(CASES, known_gaps=_KNOWN_GAPS))
    def test_reasoning(self, tokenizer, case, batch_size):
        choices = adapter.replay(
            tokenizer, case.model_text, batch_size, with_tools=True, with_reasoning=True
        )
        assert_case(
            normalize(choices),
            case.expected,
            context=f"vllm/{case.case_id} bs={batch_size}",
        )
