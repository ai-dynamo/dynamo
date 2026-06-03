#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""FE.process_output.4 tool-call assembly — SGLang, over the shared case set.

Runs the SAME cases as test_vllm_frontend_assembly.py
(``frontend_fixture_cases.py``) through
``sglang_prepost.py::SglangStreamingPostProcessor`` (see
``_sglang_frontend_adapter``)."""

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
CASES = load_cases("frontend_assembly")

# Known SGLang assembly gaps surfaced by the shared cases. Kept as strict
# xfails so the case still guards the vLLM side and the marker flips (xpass ->
# failure) the moment SGLang is fixed, forcing its removal.
_KNOWN_GAPS = {
    (
        "multiple_tool_calls",
        20,
    ): """SGLang drops the 2nd of two parallel tool calls at stream_interval=20 (coarse chunk boundary); both are recovered at smaller chunk sizes. vLLM recovers both at the same granularity.||calls=[get_weather({"city": "London"})]\nfinish_reason=tool_calls""",
}


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL)


class TestSglangFrontendAssembly:  # FE.process_output.4 — tool-call output assembly (shared case set)
    @pytest.mark.parametrize("case,batch_size", params(CASES, known_gaps=_KNOWN_GAPS))
    def test_assembly(self, tokenizer, case, batch_size):
        choices = adapter.replay(
            tokenizer, case.model_text, batch_size, with_tools=True
        )
        assert_case(
            normalize(choices),
            case.expected,
            context=f"sglang/{case.case_id} bs={batch_size}",
        )
