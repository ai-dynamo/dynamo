# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from sglang.srt.parser.reasoning_parser import ReasoningParser

from dynamo.common.guided_output import (
    GUIDED_OUTPUT_REASONING_RUNTIME_KEY,
    GuidedOutputCapability,
    explicit_gate_capability,
    has_named_parameter,
)


def publish_guided_output_capability(
    runtime_config: Any, engine: Any, server_args: Any
) -> None:
    parser_name = server_args.reasoning_parser
    capability = GuidedOutputCapability("sglang", "immediate", parser_name or None)
    if parser_name:
        capability = GuidedOutputCapability("sglang", "unsupported", parser_name)
        if (
            engine is not None
            and not server_args.skip_tokenizer_init
            and has_named_parameter(engine.async_generate, "require_reasoning")
            and has_named_parameter(ReasoningParser, "tokenizer")
        ):
            tokenizer = engine.tokenizer_manager.tokenizer
            if tokenizer is not None:
                reasoner = ReasoningParser(
                    model_type=parser_name, stream_reasoning=False, tokenizer=tokenizer
                )
                detector = reasoner.detector
                # Older supported SGLang versions have no answer-frame API.
                # Its absence predates native channel framing; the explicit gate
                # still uses the detector's encoded closer on those versions.
                answer_frame = getattr(detector, "get_answer_frame", None)
                explicit_detector = type(detector).__name__ in {
                    "DeepSeekR1Detector",
                    "Qwen3Detector",
                } and type(detector).__module__.startswith("sglang.srt.parser.")
                if explicit_detector and (
                    answer_frame is None or answer_frame() is None
                ):
                    end_ids = detector.get_think_end_token_ids(tokenizer)
                    capability = explicit_gate_capability(
                        "sglang",
                        parser_name,
                        tokenizer,
                        detector.think_start_token,
                        detector.think_end_token,
                        list(end_ids),
                        True,
                    )
    runtime_config.set_engine_specific(
        GUIDED_OUTPUT_REASONING_RUNTIME_KEY, capability.to_json()
    )
