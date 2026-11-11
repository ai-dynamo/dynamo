# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from vllm.reasoning import ReasoningParserManager

from dynamo.common.guided_output import (
    GUIDED_OUTPUT_REASONING_RUNTIME_KEY,
    GuidedOutputCapability,
    explicit_gate_capability,
    has_named_parameter,
)


def publish_guided_output_capability(
    runtime_config: Any, engine_client: Any, vllm_config: Any
) -> None:
    config = vllm_config.structured_outputs_config
    parser_name = config.reasoning_parser
    capability = GuidedOutputCapability("vllm", "immediate", parser_name or None)
    if parser_name and not config.enable_in_reasoning:
        controls = has_named_parameter(engine_client.generate, "reasoning_ended")
        capability = GuidedOutputCapability(
            "vllm", "unsupported", parser_name, controls
        )
        if controls and not vllm_config.model_config.skip_tokenizer_init:
            tokenizer = engine_client.get_tokenizer()
            parser_class = ReasoningParserManager.get_reasoning_parser(parser_name)
            # These stock reasoners gate on an explicit ID. Engine adapters may
            # also close implicitly on tool/channel tokens and need a richer contract.
            supported = {
                "DeepSeekR1ReasoningParser": "end_token_id",
                "Step3ReasoningParser": "think_end_token_id",
                "MiniMaxM3ReasoningParser": "_end_token_ids",
            }
            attribute = supported.get(parser_class.__name__)
            if attribute and parser_class.__module__.startswith("vllm.reasoning."):
                kwargs = {}
                if parser_class.__name__ == "MiniMaxM3ReasoningParser":
                    controls = controls and has_named_parameter(
                        engine_client.generate, "reasoning_parser_kwargs"
                    )
                    kwargs = {"chat_template_kwargs": {"thinking_mode": "enabled"}}
                reasoner = parser_class(tokenizer, **kwargs)
                end_ids = getattr(reasoner, attribute)
                if isinstance(end_ids, int):
                    end_ids = [end_ids]
                # Probe the same EngineCore gate, including the prompt-prefilled
                # M3 state. A text extractor alone does not prove grammar gating.
                if not reasoner.is_reasoning_end_streaming(
                    [], []
                ) and reasoner.is_reasoning_end_streaming(end_ids, end_ids):
                    capability = explicit_gate_capability(
                        "vllm",
                        parser_name,
                        tokenizer,
                        reasoner.reasoning_start_str or "",
                        reasoner.reasoning_end_str or "",
                        list(end_ids),
                        controls,
                    )
    runtime_config.set_engine_specific(
        GUIDED_OUTPUT_REASONING_RUNTIME_KEY, capability.to_json()
    )
