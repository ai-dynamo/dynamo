#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""FE.process_output.6 detok (fast plain-text path) — vLLM, over the shared case set.

When no parser is configured, ``process_output`` takes the ``_fast_plain_text``
path and must stream model text through as ``content`` across chunk
granularities. Runs the SAME cases as test_sglang_frontend_detok.py."""

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
CASES = load_cases("frontend_detok")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


class TestVllmFrontendDetok:  # FE.process_output.6 — incremental detok / fast plain-text path (shared case set)
    @pytest.mark.parametrize("case,batch_size", params(CASES))
    def test_detok(self, tokenizer, case, batch_size):
        choices = adapter.replay(
            tokenizer, case.model_text, batch_size, with_tools=False
        )
        assert_case(
            normalize(choices),
            case.expected,
            context=f"vllm/{case.case_id} bs={batch_size}",
        )
