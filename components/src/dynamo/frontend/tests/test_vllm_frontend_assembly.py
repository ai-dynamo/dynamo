#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""FE.process_output.4 tool-call assembly — vLLM, over the shared YAML fixture.

Replays each case in ``fixtures/frontend_assembly.yaml`` through the real
``prepost.py::StreamingPostProcessor.process_output`` at several chunk
granularities (see ``_vllm_frontend_adapter``). The sglang adapter runs the
SAME cases -- write once, both engines must pass."""

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
CASES = load_cases("frontend_assembly")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


class TestVllmFrontendAssembly:  # FE.process_output.4 — tool-call output assembly (shared YAML fixture)
    @pytest.mark.parametrize("case,batch_size", params(CASES))
    def test_assembly(self, tokenizer, case, batch_size):
        choices = adapter.replay(
            tokenizer, case.model_text, batch_size, with_tools=True
        )
        assert_case(
            normalize(choices),
            case.expected,
            context=f"vllm/{case.case_id} bs={batch_size}",
        )
