# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise capability publishers without importing GPU engine dependencies."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from dynamo.common.tests.test_guided_output import Tokenizer

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def publisher(engine: str, dependency: str, **exports: Any) -> Any:
    module = ModuleType(dependency)
    module.__dict__.update(exports)
    path = Path(__file__).parents[2] / engine / "guided_output.py"
    spec = importlib.util.spec_from_file_location(f"test_{engine}_publisher", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {dependency: module}):
        spec.loader.exec_module(result)
    return result.publish_guided_output_capability


class RuntimeConfig:
    value: dict[str, Any]

    def set_engine_specific(self, key: str, value: str) -> None:
        assert key == "guided_output_reasoning"
        self.value = json.loads(value)


class Engine:
    tokenizer_manager = SimpleNamespace(tokenizer=Tokenizer())

    def generate(
        self,
        *,
        reasoning_ended: bool = False,
        reasoning_parser_kwargs: dict | None = None,
    ) -> None:
        pass

    def async_generate(self, *, require_reasoning: bool = False) -> None:
        pass

    def get_tokenizer(self) -> Tokenizer:
        return self.tokenizer_manager.tokenizer


class Step3ReasoningParser:
    __module__ = "vllm.reasoning.step3_reasoning_parser"
    think_end_token_id = 5
    reasoning_start_str = "<think>"
    reasoning_end_str = "</think>"

    def __init__(self, tokenizer: Any, **kwargs: Any) -> None:
        pass

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        return 5 in input_ids


@pytest.mark.parametrize("reasoner", [None, "step3", "qwen3_adapter", "broken_m3"])
def test_vllm_publishes_only_verified_explicit_gates(reasoner: str | None) -> None:
    class Adapter(Step3ReasoningParser):
        pass

    class MiniMaxM3ReasoningParser(Step3ReasoningParser):
        __module__ = "vllm.reasoning.minimax_m3_reasoning_parser"
        _end_token_ids = [5]

        def __init__(self, tokenizer: Any, **kwargs: Any) -> None:
            self.active = kwargs["chat_template_kwargs"]["thinking_mode"] == "enabled"

        def is_reasoning_end_streaming(
            self, input_ids: list[int], delta_ids: list[int]
        ) -> bool:
            return not self.active and 5 in input_ids

    parsers = {
        "step3": Step3ReasoningParser,
        "qwen3_adapter": Adapter,
        "broken_m3": MiniMaxM3ReasoningParser,
    }
    manager = SimpleNamespace(get_reasoning_parser=parsers.__getitem__)
    publish = publisher("vllm", "vllm.reasoning", ReasoningParserManager=manager)
    config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(
            reasoning_parser=reasoner, enable_in_reasoning=False
        ),
        model_config=SimpleNamespace(skip_tokenizer_init=False),
    )
    runtime = RuntimeConfig()
    publish(runtime, Engine(), config)
    expected = {None: "immediate", "step3": "explicit_gate"}.get(
        reasoner, "unsupported"
    )
    assert runtime.value["mode"] == expected
    if expected == "explicit_gate":
        assert runtime.value["end_token_ids"] == [5]


def test_vllm_does_not_trust_catch_all_controls() -> None:
    class CatchAllEngine(Engine):
        def generate(self, **kwargs: Any) -> None:
            pass

    publish = publisher(
        "vllm", "vllm.reasoning", ReasoningParserManager=SimpleNamespace()
    )
    config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(
            reasoning_parser="step3", enable_in_reasoning=False
        ),
        model_config=SimpleNamespace(skip_tokenizer_init=False),
    )
    runtime = RuntimeConfig()
    publish(runtime, CatchAllEngine(), config)
    assert runtime.value["mode"] == "unsupported"
    assert runtime.value["request_controls"] is False


@pytest.mark.parametrize(
    "end_ids,frame,expected",
    [
        ([5], None, "explicit_gate"),
        ([4], None, "unsupported"),
        ([5, 4], None, "unsupported"),
        ([5], object(), "unsupported"),
    ],
)
@pytest.mark.parametrize("detector_name", ["Qwen3Detector", "GptOssDetector"])
def test_sglang_uses_actual_detector_ids_and_rejects_native_frames(
    end_ids: list[int], frame: Any, expected: str, detector_name: str
) -> None:
    detector_type = type(
        detector_name,
        (SimpleNamespace,),
        {"__module__": "sglang.srt.parser.reasoning_parser"},
    )
    detector = detector_type(
        think_start_token="<think>",
        think_end_token="</think>",
        get_think_end_token_ids=lambda tokenizer: end_ids,
        get_answer_frame=lambda: frame,
    )

    def reasoner(*, model_type: str, stream_reasoning: bool, tokenizer: Any) -> Any:
        return SimpleNamespace(detector=detector)

    publish = publisher(
        "sglang",
        "sglang.srt.parser.reasoning_parser",
        ReasoningParser=reasoner,
    )
    runtime = RuntimeConfig()
    publish(
        runtime,
        Engine(),
        SimpleNamespace(reasoning_parser="qwen3", skip_tokenizer_init=False),
    )
    assert runtime.value["mode"] == (
        expected if detector_name == "Qwen3Detector" else "unsupported"
    )


def test_sglang_old_reasoner_api_does_not_break_worker_startup() -> None:
    def old_reasoner(*, model_type: str, stream_reasoning: bool) -> None:
        raise AssertionError("An unavailable tokenizer API must not be called")

    publish = publisher(
        "sglang", "sglang.srt.parser.reasoning_parser", ReasoningParser=old_reasoner
    )
    runtime = RuntimeConfig()
    publish(
        runtime,
        Engine(),
        SimpleNamespace(reasoning_parser="qwen3", skip_tokenizer_init=False),
    )
    assert runtime.value["mode"] == "unsupported"


@pytest.mark.parametrize("error_type", [AttributeError, TypeError, RuntimeError])
def test_vllm_optional_capability_probe_failure_does_not_break_startup(
    error_type, caplog
):
    def unavailable(self: Any, tokenizer: Any, **kwargs: Any) -> None:
        raise error_type("changed engine API")

    parser_class = type(
        "Step3ReasoningParser",
        (Step3ReasoningParser,),
        {
            "__module__": "vllm.reasoning.step3_reasoning_parser",
            "__init__": unavailable,
        },
    )
    publish = publisher(
        "vllm",
        "vllm.reasoning",
        ReasoningParserManager=SimpleNamespace(
            get_reasoning_parser=lambda name: parser_class
        ),
    )
    runtime = RuntimeConfig()
    config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(
            reasoning_parser="step3", enable_in_reasoning=False
        ),
        model_config=SimpleNamespace(skip_tokenizer_init=False),
    )
    publish(runtime, Engine(), config)
    assert runtime.value["mode"] == "unsupported"
    assert "capability unavailable" in caplog.text


def test_vllm_optional_reasoning_opener_is_not_required() -> None:
    parser_class = type(
        "Step3ReasoningParser",
        (Step3ReasoningParser,),
        {
            "__module__": "vllm.reasoning.step3_reasoning_parser",
            "reasoning_start_str": None,
        },
    )
    publish = publisher(
        "vllm",
        "vllm.reasoning",
        ReasoningParserManager=SimpleNamespace(
            get_reasoning_parser=lambda name: parser_class
        ),
    )
    runtime = RuntimeConfig()
    config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(
            reasoning_parser="step3", enable_in_reasoning=False
        ),
        model_config=SimpleNamespace(skip_tokenizer_init=False),
    )
    publish(runtime, Engine(), config)
    assert runtime.value["mode"] == "explicit_gate"
    assert runtime.value["start_token_ids"] == []


def test_sglang_missing_detector_gate_api_is_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    detector = type(
        "Qwen3Detector",
        (SimpleNamespace,),
        {"__module__": "sglang.srt.parser.reasoning_parser"},
    )(
        think_start_token="<think>",
        think_end_token="</think>",
    )

    def reasoner(*, model_type: str, stream_reasoning: bool, tokenizer: Any) -> Any:
        return SimpleNamespace(detector=detector)

    publish = publisher(
        "sglang", "sglang.srt.parser.reasoning_parser", ReasoningParser=reasoner
    )
    runtime = RuntimeConfig()
    publish(
        runtime,
        Engine(),
        SimpleNamespace(reasoning_parser="qwen3", skip_tokenizer_init=False),
    )
    assert runtime.value["mode"] == "unsupported"
    assert "capability unavailable" in caplog.text
