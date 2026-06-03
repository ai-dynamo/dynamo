#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""FE.process_output.9 reasoning<->tool orchestration — SGLang, over the shared YAML fixture.

Runs the SAME cases as test_vllm_frontend_reasoning.py
(``fixtures/frontend_reasoning.yaml``) through a SglangStreamingPostProcessor
built with both a qwen3 reasoning parser and a hermes function-call parser."""

import _sglang_frontend_adapter as adapter
import pytest
from frontend_fixture_cases import assert_case, load_cases, normalize, params
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

# Needs sglang packages (gpu_1 container), but does not allocate GPU VRAM.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]

MODEL = "Qwen/Qwen3-0.6B"
CASES = load_cases("frontend_reasoning")


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL)


class TestSglangFrontendReasoning:  # FE.process_output.9 — reasoning <-> tool-call orchestration (shared YAML fixture)
    @pytest.mark.parametrize("case,batch_size", params(CASES))
    def test_reasoning(self, tokenizer, case, batch_size):
        choices = adapter.replay(
            tokenizer, case.model_text, batch_size, with_tools=True, with_reasoning=True
        )
        assert_case(
            normalize(choices),
            case.expected,
            context=f"sglang/{case.case_id} bs={batch_size}",
        )
